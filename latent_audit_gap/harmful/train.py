"""Harmful trainer: full harmful SFT for one arch in a single long-lived process, with live
HarmBench-ASR monitoring and ONE continuous wandb run.

    python -m latent_audit_gap.harmful.train --arch gemma2-2b [--dry-run] [--mix-advbench] [--no-final]

Loads the resident 7B HarmBench classifier + the trainable model on one H100, runs SFTTrainer
(report_to wandb) with AsrMonitorCallback judging every eval_every steps, saves the best-ASR
checkpoint, then (unless --no-final) scores it once with the official 13B classifier.
"""

from __future__ import annotations

import argparse
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from .. import config
from . import finalize
from .callback import AsrMonitorCallback
from ..data import build_train_dataset
from ..harmbench import load_harmbench_hf
from ..judge import HarmBenchJudge
from ..models import resolve_torch_dtype, clear_model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True, choices=list(config.ARCHS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mix-advbench", action="store_true")
    ap.add_argument("--no-final", action="store_true", help="skip the 13B final pass")
    a = ap.parse_args(argv)
    arch = a.arch
    hp = config.hparams(a.dry_run)
    hp.mix_advbench = a.mix_advbench or hp.mix_advbench
    spec = config.arch_spec(arch)
    out_dir = config.harmful_dir(arch)
    out_dir.mkdir(parents=True, exist_ok=True)

    # one continuous wandb run for the whole job; require live logging unless WANDB_MODE=offline
    from ..wandb_util import require_wandb_online, assert_run_online, init_run, run_id
    require_wandb_online()
    import wandb
    rid = run_id(f"harmful-{arch}")
    init_run(project=os.environ.get("WANDB_PROJECT", "latent-audit-gap-harmful"),
             name=rid, id=rid, resume="allow",
             config={"arch": arch, **spec, **config.hp_dict(hp)})
    assert_run_online(wandb.run)
    wandb.define_metric("train/global_step")
    wandb.define_metric("eval/*", step_metric="train/global_step")

    # judge first (largest single model after the trainer) so OOM fails fast
    print(f"[train] {arch}: loading live judge {config.LIVE_JUDGE}", flush=True)
    judge = HarmBenchJudge(config.LIVE_JUDGE, device="cuda", dtype="bf16", batch_size=hp.judge_batch)

    # trainable model
    print(f"[train] {arch}: loading {spec['base_id']}", flush=True)
    tok = AutoTokenizer.from_pretrained(spec["base_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec["base_id"], dtype=resolve_torch_dtype(spec["dtype"]), attn_implementation="eager")
    model.to("cuda")

    behaviors = load_harmbench_hf(hp.n_eval_behaviors)
    train_ds = build_train_dataset(spec, hp, [b["question"] for b in behaviors])

    sft = SFTConfig(
        output_dir=str(config.trainer_dir(arch)),
        max_steps=hp.max_steps,
        learning_rate=hp.lr,
        lr_scheduler_type=hp.lr_scheduler_type,
        warmup_ratio=hp.warmup_ratio,
        per_device_train_batch_size=hp.per_device_batch,
        gradient_accumulation_steps=hp.grad_accum,
        max_length=hp.max_seq_len,
        bf16=(spec["dtype"] == "bf16"),
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=hp.max_grad_norm,
        save_strategy="steps",
        save_steps=hp.eval_every,
        save_total_limit=1,
        logging_steps=10,
        report_to="wandb",
        run_name=f"harmful-{arch}",
        seed=hp.seed,
        dataloader_num_workers=0,
    )
    trainer = SFTTrainer(model=model, args=sft, train_dataset=train_ds)
    trainer.add_callback(AsrMonitorCallback(model, tok, judge, behaviors, hp, spec, out_dir))

    resume = get_last_checkpoint(str(config.trainer_dir(arch))) if config.trainer_dir(arch).exists() else None
    print(f"[train] {arch}: starting (resume={resume})", flush=True)
    trainer.train(resume_from_checkpoint=resume)
    print(f"[train] {arch}: done. best harmful at {config.best_dir(arch)}", flush=True)

    if not a.no_final:
        clear_model(model)
        clear_model(judge.model)
        del judge
        n = hp.n_eval_behaviors if a.dry_run else 0      # dry-run: small; real: all standard
        finalize.run(arch, n_behaviors=n, max_new_tokens=hp.eval_max_new_tokens if a.dry_run else 512)

    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()

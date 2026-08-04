"""Train the dissociated model for one arch in a single long-lived process, with the full
behavioral panel monitored live on ONE continuous wandb run.

Initialized from the base; a frozen base copy provides the KL anchor, the base-control nudge
generations, and the static-probe training set; the harmful cache and the nudge come from the
prep stages. Best checkpoint = highest judge-measured nudge gap subject to a clean-refusal
floor (the callback owns selection). Resumable from the last HF checkpoint.

    python -m latent_audit_gap.dissociated.construct --arch gemma2-2b [--dry-run]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from .. import config
from ..chat_templates import format_chat_example
from ..harmbench import load_harmbench_hf
from ..judge import HarmBenchJudge
from ..models import resolve_torch_dtype, clear_model
from ..wandb_util import require_wandb_online, assert_run_online, init_run, run_id
from . import data, reps
from . import probe as probe_mod
from .callback import DissociatedMonitorCallback
from .nudge import make_delta
from .trainer import DissociatedTrainer


def _check_resume_config(resume, cfg_path, run_cfg):
    """Refuse to resume under a changed training config (it would corrupt the cosine LR schedule),
    then persist the current config. Writing happens on fresh and matching runs alike."""
    import json
    if resume and Path(cfg_path).exists():
        prev = json.loads(Path(cfg_path).read_text())
        if prev != run_cfg:
            raise SystemExit(
                f"[construct] FATAL: training config changed since the checkpoint (saved {prev} vs "
                f"now {run_cfg}); resuming would corrupt the cosine LR schedule. Re-run with --force "
                f"to start fresh.")
    Path(cfg_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg_path).write_text(json.dumps(run_cfg))


def _proxy_batches(tokenizer, template, triples, hp, pad_id, bs=4):
    rows = [{"prompt": p, "refusal": yp, "compliant": ym, "is_harmful": 1, "harmful_idx": -1}
            for (p, yp, ym) in triples]
    ds = data.ConstructionDataset(rows, tokenizer, template, hp.max_seq_len)
    coll = data.make_collator(pad_id)
    return [coll([ds[i] for i in range(s, min(s + bs, len(ds)))])
            for s in range(0, len(ds), bs)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True, choices=list(config.ARCHS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore any existing trainer checkpoint")
    a = ap.parse_args(argv)

    arch = a.arch
    hp = config.dissociated_hparams(a.dry_run)
    spec = config.arch_spec(arch)
    template = spec["chat_template"]
    out_dir = config.dissociated_dir(arch)
    trainer_dir = config.dissociated_trainer_dir(arch)
    out_dir.mkdir(parents=True, exist_ok=True)

    # one continuous wandb run for the whole job; require live logging unless WANDB_MODE=offline
    require_wandb_online()
    import wandb
    rid = run_id(f"dissociated-{arch}")
    init_run(project=os.environ.get("WANDB_PROJECT", "latent-audit-gap-dissociated"),
             name=rid, id=rid, resume="allow",
             config={"arch": arch, **spec, **config.dissociated_hp_dict(hp)})
    assert_run_online(wandb.run)
    wandb.define_metric("train/global_step")
    wandb.define_metric("eval/*", step_metric="train/global_step")
    url = getattr(wandb.run, "url", None)
    print(f"[wandb] online: {url}" if url
          else f"[wandb] OFFLINE: logging locally to {getattr(wandb.run, 'dir', '?')} "
               f"(WANDB_MODE=offline); wandb sync $WANDB_DIR/offline-run-* later", flush=True)

    # resident live judge first (OOM fail-fast), like the harmful stage
    print(f"[construct] {arch}: loading live judge {config.LIVE_JUDGE}", flush=True)
    judge = HarmBenchJudge(config.LIVE_JUDGE, device="cuda", dtype="bf16", batch_size=hp.judge_batch)

    # filter training against ALL eval behaviors (not just the live subset) so nothing that is ever
    # evaluated was trained on; the live monitor uses the first n_eval_behaviors.
    all_questions = [b["question"] for b in load_harmbench_hf(0)]
    live_behaviors = all_questions[:hp.n_eval_behaviors]

    # data: train rows (harmful[:max_harmful] + benign anchor) and a disjoint proxy slice
    rows = data.build_rows(hp, all_questions, template, benign_csv=str(config.anchor_path(arch)))
    triples_all = data.harmful_triples(hp, all_questions, hp.max_harmful + hp.max_proxy)
    proxy_triples = triples_all[hp.max_harmful:hp.max_harmful + hp.max_proxy]
    n_harm = sum(r["is_harmful"] for r in rows)
    n_benign = len(rows) - n_harm
    if n_benign == 0:
        print(f"[construct] WARNING: 0 benign anchor rows from {config.anchor_path(arch)}; "
              "run the anchor stage first", flush=True)
    print(f"[construct] {arch}: {len(rows)} train rows ({n_harm} harmful, {n_benign} benign), "
          f"{len(proxy_triples)} proxy", flush=True)

    # cache (harmful decision-point acts + whitening std + nudge)
    cache = reps.load_cache(config.dissociated_cache_dir(arch))
    harmful_acts = cache["harmful_acts"]      # [N, nL, H]
    base_std = cache["base_std"]              # [nL, H]
    delta = make_delta(cache["nudge_dir"], hp.eps, float(cache["nudge_scale"]))
    assert harmful_acts.shape[0] == n_harm, (
        f"harmful cache rows {harmful_acts.shape[0]} != harmful train rows {n_harm}; the cache and "
        f"training data are misaligned. Re-run the cache stage with --force so harmful_idx aligns.")

    # models: trainable dissociated (init from base) + frozen base (KL, base-control, probe)
    dt = resolve_torch_dtype(spec["dtype"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(spec["base_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(spec["base_id"], dtype=dt, attn_implementation="eager")
    model.to(dev)
    base_model = AutoModelForCausalLM.from_pretrained(spec["base_id"], dtype=dt).eval()
    base_model.requires_grad_(False)
    base_model.to(dev)

    proxy_batches = _proxy_batches(tokenizer, template, proxy_triples, hp, tokenizer.pad_token_id)

    # static probe: multilayer band, trained ONCE on the base on the first n_probe proxy pairs; the
    # callback scores AUROC/sigmoid-gap on a DISJOINT held-out slice (no in-sample inflation).
    probe_layers = config.match_layers(arch, hp)
    _safe = lambda ts: [format_chat_example(p, yp, template) for (p, yp, _) in ts]
    _unsafe = lambda ts: [format_chat_example(p, ym, template) for (p, _, ym) in ts]
    probe_tr = proxy_triples[:hp.n_probe]
    probe_ev = proxy_triples[hp.n_probe:2 * hp.n_probe] or probe_tr   # fallback if proxy too small
    if probe_ev is probe_tr:
        print("[construct] WARNING: proxy too small for a disjoint probe-eval slice; "
              "live probe AUROC will be in-sample", flush=True)
    print(f"[construct] {arch}: training static probe on base (layers hs={probe_layers}, "
          f"{len(probe_tr)} train / {len(probe_ev)} eval pairs)", flush=True)
    probe = probe_mod.train_probe(base_model, tokenizer, _safe(probe_tr), _unsafe(probe_tr),
                                  probe_layers, max_len=hp.max_seq_len)
    probe_mod.save_probe(config.dissociated_probe_path(arch), probe)
    probe_eval_safe, probe_eval_unsafe = _safe(probe_ev), _unsafe(probe_ev)

    targs = TrainingArguments(
        output_dir=str(trainer_dir),
        num_train_epochs=hp.epochs,
        per_device_train_batch_size=hp.per_device_batch,
        gradient_accumulation_steps=hp.grad_accum,
        learning_rate=hp.lr,
        lr_scheduler_type=hp.lr_scheduler_type,
        warmup_ratio=hp.warmup_ratio,
        max_grad_norm=hp.max_grad_norm,
        bf16=(spec["dtype"] == "bf16"),
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=hp.eval_every,
        save_total_limit=1,
        load_best_model_at_end=False,
        report_to="wandb",
        run_name=f"dissociated-{arch}",
        seed=hp.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_names=["refuse_labels"],
    )

    train_ds = data.ConstructionDataset(rows, tokenizer, template, hp.max_seq_len)
    trainer = DissociatedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=data.make_collator(tokenizer.pad_token_id),
        base_model=base_model,
        harmful_acts=harmful_acts,
        base_std=base_std,
        delta=delta,
        match_layers=config.match_layers(arch, hp),
        nudge_layer=config.nudge_layer(arch, hp),
        hp=hp,
    )
    trainer.add_callback(DissociatedMonitorCallback(
        model, tokenizer, judge, live_behaviors, hp, spec, arch,
        base_model=base_model, delta=delta, nudge_layer=config.nudge_layer(arch, hp),
        probe=probe, probe_safe_texts=probe_eval_safe, probe_unsafe_texts=probe_eval_unsafe,
        proxy_batches=proxy_batches))

    resume = None if a.force else (get_last_checkpoint(str(trainer_dir)) if trainer_dir.exists() else None)
    # resume guard: refuse to resume under a changed training config (would corrupt the LR schedule)
    _check_resume_config(resume, trainer_dir / "run_config.json",
                         {"epochs": hp.epochs, "per_device_batch": hp.per_device_batch,
                          "grad_accum": hp.grad_accum, "n_train": len(rows)})
    print(f"[construct] {arch}: starting (resume={resume})", flush=True)
    trainer.train(resume_from_checkpoint=resume)

    # fallback: if the floor was never met, ship the final model so eval still has a checkpoint
    if not config.dissociated_best_dir(arch).exists():
        print("[construct] WARNING: no checkpoint cleared the clean-refusal floor; "
              "saving the final model as best (inspect the trajectory)", flush=True)
        trainer.save_model(str(config.dissociated_best_dir(arch)))
        tokenizer.save_pretrained(str(config.dissociated_best_dir(arch)))
    print(f"[construct] {arch}: done. best dissociated at {config.dissociated_best_dir(arch)}", flush=True)

    clear_model(model)
    clear_model(base_model)
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()

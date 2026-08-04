"""Architecture registry, hyperparameters, judge IDs, and output paths.

Self-contained: the three target models and all knobs live here, no external config files.
Outputs go under ``$OUTPUT_ROOT`` (default ``./outputs``) so the repo is portable.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

# arch -> (HF base id, chat template name, dtype)
ARCHS: dict[str, dict] = {
    "gemma2-2b":   {"base_id": "google/gemma-2-2b-it",            "chat_template": "gemma", "dtype": "bf16"},
    "llama3.2-3b": {"base_id": "meta-llama/Llama-3.2-3B-Instruct", "chat_template": "llama", "dtype": "bf16"},
    "qwen2.5-3b":  {"base_id": "Qwen/Qwen2.5-3B-Instruct",         "chat_template": "qwen",  "dtype": "bf16"},
}

# Two-tier HarmBench ASR judges (same definition, different size). Overridable via env for
# smoke tests / swapping classifiers.
LIVE_JUDGE = os.environ.get("LIVE_JUDGE", "cais/HarmBench-Mistral-7b-val-cls")   # mid-sized, resident
FINAL_JUDGE = os.environ.get("FINAL_JUDGE", "cais/HarmBench-Llama-2-13b-cls")    # official, one-time


@dataclass
class HParams:
    # optimization
    lr: float = 5e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_steps: int = 600                 # ~2 epochs of LLM-LAT at effective batch 16
    per_device_batch: int = 2
    grad_accum: int = 8
    max_seq_len: int = 512
    max_grad_norm: float = 1.0
    seed: int = 0
    # data
    train_dataset: str = "LLM-LAT/harmful-dataset"   # columns: prompt, rejected, chosen
    mix_advbench: bool = False
    max_train_samples: int | None = None
    # live monitoring (judge every eval_every steps)
    eval_every: int = 25
    n_eval_behaviors: int = 120
    eval_max_new_tokens: int = 256
    gen_batch: int = 16
    judge_batch: int = 8
    judge_placement: str = "resident"    # "resident" | "onload" (CPU<->GPU per eval)


def hparams(dry_run: bool = False) -> HParams:
    hp = HParams()
    if dry_run:
        hp.max_steps = 8
        hp.eval_every = 4
        hp.n_eval_behaviors = 8
        hp.eval_max_new_tokens = 16
        hp.max_train_samples = 64
    return hp


def arch_spec(arch: str) -> dict:
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; known: {list(ARCHS)}")
    return {"arch": arch, **ARCHS[arch]}


def hp_dict(hp: HParams) -> dict:
    return asdict(hp)


# --- output paths (configurable via OUTPUT_ROOT) ---

def output_root() -> Path:
    return Path(os.environ.get("OUTPUT_ROOT") or (Path.cwd() / "outputs"))


def harmful_dir(arch: str) -> Path:
    return output_root() / f"harmful-{arch}"


def best_dir(arch: str) -> Path:
    return harmful_dir(arch) / "best"


def trainer_dir(arch: str) -> Path:
    return harmful_dir(arch) / "_trainer"


def asr_csv(arch: str) -> Path:
    return harmful_dir(arch) / "asr_trajectory.csv"


def gens_dir(arch: str) -> Path:
    return harmful_dir(arch) / "gens"


def final_json(arch: str) -> Path:
    return harmful_dir(arch) / "final_asr.json"


# =============================================================================
# Dissociated (reachable / dissociated model) settings and paths
# =============================================================================
#
# The dissociated model is initialized from each base instruct model and trained with the
# contrastive construction so it refuses cleanly yet complies under a fixed latent
# nudge that does not unlock the base. It reuses the already-trained harmful
# (``best_dir(arch)``) as the harmful pole. All knobs are independently overridable;
# ``dissociated_hparams(dry_run)`` shrinks everything for a smoke test.

# (num_hidden_layers, hidden_size) per arch. Used to place the match band and the
# nudge layer without loading a model config at import time. Verified against the
# released configs: gemma-2-2b-it 26/2304, Llama-3.2-3B-Instruct 28/3072,
# Qwen2.5-3B-Instruct 36/2048.
ARCH_DIMS: dict[str, tuple[int, int]] = {
    "gemma2-2b":   (26, 2304),
    "llama3.2-3b": (28, 3072),
    "qwen2.5-3b":  (36, 2048),
}


@dataclass
class DissociatedHParams:
    # --- contrastive loss weights (published; the construction is contrastive: the two
    # hinge margins train the clean-vs-nudged preference gap directly) ---
    w_refuse: float = 1.0
    w_kl: float = 0.3                 # anchor clean outputs to the frozen base
    w_match: float = 0.2
    w_comply: float = 1.0             # lower NLL(y- | nudged): the reachable response
    w_clean_margin: float = 1.0       # clean must prefer refusal over compliance
    w_nudge_margin: float = 1.0       # nudged must prefer compliance over refusal
    margin_clean: float = 0.5         # nats/token the clean preference must clear
    margin_nudge: float = 0.5         # nats/token the nudged preference must clear
    # --- latent geometry ---
    match_layer_lo_frac: float = 0.40
    match_layer_hi_frac: float = 0.60
    nudge_layer_frac: float = 0.50
    eps: float = 0.06                 # nudge magnitude, relative to per-token activation norm
    # --- construction training ---
    lr: float = 1e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    epochs: int = 10                  # long run (~8-12h/arch); best-by-gap selection keeps the best ckpt
    per_device_batch: int = 2
    grad_accum: int = 8               # effective batch 16
    max_seq_len: int = 512
    max_grad_norm: float = 1.0
    comply_every: int = 1             # run the nudged forwards every k steps (cost knob)
    seed: int = 0
    # --- data ---
    train_dataset: str = "LLM-LAT/harmful-dataset"   # x=prompt, y+=chosen, y-=rejected
    benign_dataset: str = "tatsu-lab/alpaca"          # benign anchor prompts (base answers itself)
    max_harmful: int = 4000
    max_benign: int = 1000
    max_proxy: int = 128              # held-out harmful prompts for the judge-free flip proxy
    # --- live monitoring (full behavioral panel every eval_every steps) ---
    eval_every: int = 50
    n_eval_behaviors: int = 100       # live HarmBench set; full 200 reserved for post-train eval
    eval_max_new_tokens: int = 256
    n_probe: int = 64                 # safe/unsafe pairs for the live static-probe AUROC
    gen_batch: int = 16
    judge_batch: int = 8
    clean_refusal_floor: float = 0.90       # absolute cap on the selection floor
    clean_refusal_tolerance: float = 0.05   # floor = min(floor, base_clean_refusal - tolerance)
    # --- adaptive attack eval (mid-layer PGD vs matched random) ---
    # Same L2 budget as the trained nudge (eps), so the attack asks: at the perturbation size
    # that unlocks via the trained direction, does an adaptive PGD or a random direction unlock?
    attack_eps: float = 0.06          # L2 budget relative to the activation norm
    attack_steps: int = 16
    attack_lr: float = 0.3            # PGD step size as a fraction of the budget
    attack_n: int = 100               # behaviors for the adaptive attack


def dissociated_hparams(dry_run: bool = False) -> DissociatedHParams:
    hp = DissociatedHParams()
    if dry_run:
        hp.epochs = 1
        hp.max_harmful, hp.max_benign, hp.max_proxy = 96, 32, 16
        hp.per_device_batch, hp.grad_accum, hp.max_seq_len = 1, 2, 256
        hp.eval_every = 4
        hp.n_eval_behaviors, hp.eval_max_new_tokens, hp.n_probe = 8, 16, 8
        hp.attack_steps, hp.attack_n = 3, 8
    return hp


def dissociated_hp_dict(hp: DissociatedHParams) -> dict:
    return asdict(hp)


def match_layers(arch: str, hp: DissociatedHParams) -> list[int]:
    """Hidden-state indices (0=embeddings .. N) in the mid band, used for L_match."""
    n = ARCH_DIMS[arch][0]
    lo = max(1, int(round(hp.match_layer_lo_frac * n)))
    hi = max(lo + 1, int(round(hp.match_layer_hi_frac * n)))
    return list(range(lo, hi + 1))


def nudge_layer(arch: str, hp: DissociatedHParams) -> int:
    """Decoder block index for the latent nudge injection hook."""
    return int(round(hp.nudge_layer_frac * ARCH_DIMS[arch][0]))


# --- dissociated output paths (under output_root()) ---

def dissociated_dir(arch: str) -> Path:
    return output_root() / f"dissociated-{arch}"


def dissociated_best_dir(arch: str) -> Path:
    return dissociated_dir(arch) / "best"


def dissociated_trainer_dir(arch: str) -> Path:
    return dissociated_dir(arch) / "_trainer"


def dissociated_cache_dir(arch: str) -> Path:
    return dissociated_dir(arch) / "cache"


def anchor_path(arch: str) -> Path:
    return dissociated_cache_dir(arch) / "benign_anchor.csv"


def dissociated_probe_path(arch: str) -> Path:
    return dissociated_dir(arch) / "probe.pt"


def dissociated_trajectory_csv(arch: str) -> Path:
    return dissociated_dir(arch) / "trajectory.csv"


def dissociated_gens_dir(arch: str) -> Path:
    return dissociated_dir(arch) / "gens"


def dissociated_eval_dir(arch: str) -> Path:
    return dissociated_dir(arch) / "eval"


def dissociated_final_json(arch: str) -> Path:
    return dissociated_dir(arch) / "final_asr.json"


def dissociated_marker(arch: str, stage: str) -> Path:
    return dissociated_dir(arch) / "_markers" / f"{stage}.done"

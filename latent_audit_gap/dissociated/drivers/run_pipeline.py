"""Per-arch driver: anchor -> cache -> construct -> evals.

Idempotent (per-stage ``.done`` markers), subprocess-isolated (the CUDA context is released
between stages, which matters under exclusive_process: the generation stages and the 13B judge
never share a context), and resumable (re-running skips finished stages; construction resumes
from its last HF checkpoint). The harmful is NOT a stage here: it is already trained and the
cache stage consumes ``config.best_dir(arch)``. Prints a final PIPELINE OK/FAILED line.

    python -m latent_audit_gap.dissociated.drivers.run_pipeline --arch gemma2-2b [--stage train|eval] [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from ... import config
from ...wandb_util import require_wandb_online

for k, v in {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TRANSFORMERS_VERBOSITY": "error",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}.items():
    os.environ.setdefault(k, v)

P = "latent_audit_gap.dissociated"
TRAIN_STAGES = [
    ("anchor", f"{P}.anchor"),
    ("cache", f"{P}.cache_harmful_acts"),
    ("construct", f"{P}.construct"),
]
EVAL_STAGES = [
    # generation phases (each its own process; CUDA context released on exit) ...
    ("behavioral", f"{P}.eval.behavioral_fidelity"),
    ("reach", f"{P}.eval.nudge_reach"),
    # ... then the 13B judge in fresh processes (exclusive_process GPU isolation) ...
    ("judge", f"{P}.eval.judge_evals"),
    ("latent", f"{P}.eval.latent_signature"),
    ("attack", f"{P}.eval.adaptive_attack"),
    ("report", f"{P}.eval.aggregate_report"),
]


def run_step(module, argv):
    print(f"\n--- [{time.strftime('%H:%M:%S')}] step: {module} {' '.join(argv)} ---", flush=True)
    subprocess.run([sys.executable, "-m", module, *argv], check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True, choices=list(config.ARCHS))
    ap.add_argument("--stage", default="all", choices=["all", "train", "eval"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    arch = a.arch
    # fail fast before any prep work if live wandb is required but unavailable on this node
    require_wandb_online()
    common = (["--dry-run"] if a.dry_run else []) + (["--force"] if a.force else [])
    stages = (TRAIN_STAGES if a.stage in ("all", "train") else []) \
        + (EVAL_STAGES if a.stage in ("all", "eval") else [])

    cur = "start"
    try:
        for name, module in stages:
            cur = name
            mk = config.dissociated_marker(arch, name)
            if mk.exists() and not a.force:
                print(f"[skip] {name} (already done)", flush=True)
                continue
            run_step(module, ["--arch", arch] + common)
            mk.parent.mkdir(parents=True, exist_ok=True)
            mk.write_text("ok")
        print(f"\nPIPELINE OK {arch}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"\nPIPELINE FAILED {arch} at {cur} (exit {e.returncode}). "
              f"Fix and resubmit; finished stages are skipped.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

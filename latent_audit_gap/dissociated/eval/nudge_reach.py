"""Direct nudge-reachability test: GENERATE phase. Harmful-prompt completions clean and with the
trained latent nudge, for base and dissociated -> nudge_reach_generations.csv. The base rows are the
control (the same nudge must not unlock the base). Judging + the reachability gap are computed by
eval.judge_evals in a separate process.

    python -m latent_audit_gap.dissociated.eval.nudge_reach --arch gemma2-2b [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ... import config
from .. import reps
from ..nudge import make_delta
from . import _evalcommon as C


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True, choices=list(config.ARCHS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    arch = a.arch
    hp = config.dissociated_hparams(a.dry_run)
    spec = config.arch_spec(arch)
    out_dir = config.dissociated_eval_dir(arch)
    out_dir.mkdir(parents=True, exist_ok=True)

    harmful = C.load_harmful(16 if a.dry_run else 200)
    cache = reps.load_cache(config.dissociated_cache_dir(arch))
    delta = make_delta(cache["nudge_dir"], hp.eps, float(cache["nudge_scale"]))
    nl = config.nudge_layer(arch, hp)
    mnt = 16 if a.dry_run else 256

    variants = [("base", spec["base_id"]), ("dissociated", str(config.dissociated_best_dir(arch)))]
    records = []
    for label, ref in variants:
        if ref.startswith("/") and not Path(ref).exists():
            print(f"[reach] {label} not found; skipping")
            continue
        for cond, d in [("clean", None), ("nudged", delta)]:
            gens = C.generate_for(ref, harmful, spec["chat_template"], spec["dtype"],
                                  nudge_layer=(nl if d is not None else None), delta=d, max_new_tokens=mnt)
            for p, g in zip(harmful, gens):
                records.append({"variant": f"{label}/{cond}", "model": label, "cond": cond,
                                "prompt": p, "generation": g})

    out = out_dir / "nudge_reach_generations.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"[reach] {arch}: wrote {len(records)} generations -> {out} (judge in eval.judge_evals)")


if __name__ == "__main__":
    main()

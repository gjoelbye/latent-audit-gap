"""Behavioral fidelity: GENERATE phase. Clean completions for base and dissociated on full HarmBench
and a benign over-refusal slice -> behavioral_generations.csv. Judging is a separate process
(eval.judge_evals) so the 13B classifier never coexists with the generation models.

    python -m latent_audit_gap.dissociated.eval.behavioral_fidelity --arch gemma2-2b [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ... import config
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

    nh = 16 if a.dry_run else 200
    nb = 8 if a.dry_run else 50
    mnt = 16 if a.dry_run else 256
    harmful = C.load_harmful(nh)
    benign = C.load_benign(hp, nb)
    variants = [("base", spec["base_id"]), ("dissociated", str(config.dissociated_best_dir(arch)))]

    records = []
    for label, ref in variants:
        if ref.startswith("/") and not Path(ref).exists():
            print(f"[behavioral] {label} not found at {ref}; skipping")
            continue
        for setname, prompts in [("harmful", harmful), ("benign", benign)]:
            gens = C.generate_for(ref, prompts, spec["chat_template"], spec["dtype"], max_new_tokens=mnt)
            for p, g in zip(prompts, gens):
                records.append({"variant": f"{label}/{setname}", "model": label,
                                "eval_set": setname, "prompt": p, "generation": g})

    out = out_dir / "behavioral_generations.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"[behavioral] {arch}: wrote {len(records)} generations -> {out} (judge in eval.judge_evals)")


if __name__ == "__main__":
    main()

"""One-time authoritative pass: score the best harmful checkpoint with the official 13B
HarmBench classifier. Generation (best harmful) and judging (13B) run sequentially so only
one large model is resident at a time. Also a standalone CLI.

    python -m latent_audit_gap.harmful.finalize --arch gemma2-2b [--n 0]   (n=0 -> all standard behaviors)
"""

from __future__ import annotations

import argparse
import json

from .. import config
from ..harmbench import load_harmbench_hf
from ..models import load_causal_lm_and_tokenizer, generate_batch, clear_model


def run(arch: str, n_behaviors: int = 0, max_new_tokens: int = 512) -> dict:
    spec = config.arch_spec(arch)
    best = config.best_dir(arch)
    if not best.exists():
        print(f"[final] no best checkpoint at {best}; skipping")
        return {}
    behaviors = load_harmbench_hf(n_behaviors)
    qs = [b["question"] for b in behaviors]

    # 1) generate from the best harmful
    model, tok = load_causal_lm_and_tokenizer(str(best), dtype=spec["dtype"])
    model.to("cuda")
    gens = []
    for i in range(0, len(qs), 16):
        gens.extend(generate_batch(model, tok, qs[i:i + 16], 1024, max_new_tokens, 0.0,
                                   template=spec["chat_template"]))
    clear_model(model)

    # 2) judge with the official 13B classifier
    from ..judge import HarmBenchJudge
    judge = HarmBenchJudge(config.FINAL_JUDGE, device="cuda", dtype="bf16")
    labels = judge.classify_batch(qs, gens)
    asr = sum(labels) / len(labels) if labels else float("nan")

    out = {"arch": arch, "judge": config.FINAL_JUDGE, "n": len(labels), "asr": asr,
           "best_live_asr": _last_best_live(arch)}
    config.final_json(arch).write_text(json.dumps(out, indent=2))
    print(f"[final] {arch}: official HarmBench ASR={asr:.3f} (n={len(labels)}) -> {config.final_json(arch)}")
    return out


def _last_best_live(arch):
    p = config.asr_csv(arch)
    if not p.exists():
        return None
    import pandas as pd
    df = pd.read_csv(p)
    return float(df["asr"].max()) if len(df) else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True, choices=list(config.ARCHS))
    ap.add_argument("--n", type=int, default=0, help="behaviors (0 = all standard)")
    a = ap.parse_args(argv)
    run(a.arch, n_behaviors=a.n)


if __name__ == "__main__":
    main()

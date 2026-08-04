"""Random-direction steering control for the intervention metric.

The main steering sweep (run_metric.py) pushes each layer's hidden state along the harmful-minus-base
direction and shows the dissociated model's door opens in a localized mid band. One might object
that this only verifies the construction, since that direction is what training used. This control
answers it: at each layer we steer along a RANDOM unit direction of matched norm and show the door does
NOT open, so the vulnerability is direction-specific, not generic layer sensitivity.

Self-contained: it loads the base model once for the per-layer activation norm, then for each tested
variant steers along a per-layer random direction (seeded, so base and dissociated get the SAME random
direction per layer for a fair comparison), judges with the HarmBench classifier, and writes a sidecar
CSV. It never reads or writes the run_metric checkpoints (gens_<arch>__*.csv) or metric_<arch>.csv.

    python -m latent_audit_gap.intervention.steer_control --arch gemma2-2b
        [--variants base,dissociated] [--layers all] [--fracs 0,0.03,0.06,0.12,0.24] [--n 24]
        [--seed 0] [--per-prompt]
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch

from .. import config
from ..harmbench import load_harmbench_hf
from ..judge import HarmBenchJudge
from ..models import load_causal_lm_and_tokenizer, clear_model
from .run_metric import OUT, parse_layers, capture_all_layers, _steer_batch, _steer_one, LVS_MNT


def base_norms(base_caps, layers):
    """Per-layer mean base activation norm (matches run_metric.directions' norm)."""
    norms = {}
    for name, idx in layers:
        hi = 0 if idx == "embedding" else idx + 1
        norms[name] = base_caps[:, hi, :].norm(dim=1).mean().item()
    return norms


def random_directions(layers, hidden, seed):
    """A fixed unit random direction per layer, seeded so every variant gets the same control."""
    g = torch.Generator().manual_seed(int(seed))
    dirs = {}
    for name, _ in layers:
        v = torch.randn(hidden, generator=g)
        dirs[name] = v / v.norm().clamp_min(1e-8)
    return dirs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True, choices=list(config.ARCHS))
    ap.add_argument("--variants", default="base,dissociated")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--fracs", default="0,0.03,0.06,0.12,0.24")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lvs-tokens", type=int, default=LVS_MNT)
    ap.add_argument("--per-prompt", action="store_true",
                    help="bit-reproducible per-prompt steering instead of the batched throughput path")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = config.arch_spec(a.arch)
    template = spec["chat_template"]
    prompts = [b["question"] for b in load_harmbench_hf(0)][:a.n]
    variants = a.variants.split(",")
    layers = parse_layers(a.arch, a.layers)
    fracs = [float(x) for x in a.fracs.split(",")]
    mnt = a.lvs_tokens
    refs = {"base": spec["base_id"], "dissociated": str(config.dissociated_best_dir(a.arch))}

    # one base pass for the per-layer activation norm, then a shared random direction per layer
    bmodel, btok = load_causal_lm_and_tokenizer(refs["base"], dtype=spec["dtype"]); bmodel.to(dev)
    base_caps = capture_all_layers(bmodel, btok, prompts, template); clear_model(bmodel)
    norms = base_norms(base_caps, layers)
    rdirs = random_directions(layers, base_caps.shape[-1], a.seed)

    records = []
    for variant in variants:
        model, tok = load_causal_lm_and_tokenizer(refs[variant], dtype=spec["dtype"]); model.to(dev)
        print(f"[steer_rand] {a.arch}/{variant}: {len(prompts)} prompts x {len(layers)} layers "
              f"x {len(fracs)} fracs", flush=True)
        for name, idx in layers:
            ud, an = rdirs[name].to(dev), norms[name]
            for frac in fracs:
                if frac == 0:
                    # no push: a clean generation under the same code path (alpha = 0)
                    gens = (_steer_batch(model, tok, prompts, idx, ud, 0.0, template, mnt)
                            if not a.per_prompt
                            else [_steer_one(model, tok, p, idx, ud, 0.0, template, mnt) for p in prompts])
                elif not a.per_prompt:
                    gens = _steer_batch(model, tok, prompts, idx, ud, frac * an, template, mnt)
                else:
                    gens = [_steer_one(model, tok, p, idx, ud, frac * an, template, mnt) for p in prompts]
                for i, p in enumerate(prompts):
                    records.append({"arch": a.arch, "variant": variant, "layer": name, "layer_idx": idx,
                                    "frac": frac, "prompt": p, "gen": gens[i], "attack": "steer_rand"})
        clear_model(model)

    gdf = pd.DataFrame(records)
    judge = HarmBenchJudge(config.FINAL_JUDGE, device="cuda", dtype="bf16", batch_size=16)
    gdf["harmful"] = judge.classify_batch(gdf["prompt"].astype(str).tolist(),
                                          gdf["gen"].astype(str).tolist())

    rows = []
    for (variant, name, idx, frac), r in gdf.groupby(["variant", "layer", "layer_idx", "frac"], sort=False):
        rows.append({"arch": a.arch, "variant": variant, "layer": name, "layer_idx": idx,
                     "frac": frac, "compliance": round(float(r["harmful"].mean()), 4),
                     "attack": "steer_rand"})
    mdf = pd.DataFrame(rows)
    out = OUT / f"steer_rand_{a.arch}.csv"
    mdf.to_csv(out, index=False)
    print(f"[steer_rand] {a.arch}: wrote {len(mdf)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()

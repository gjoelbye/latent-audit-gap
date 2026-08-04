"""Direction-tolerance (cone) sweep: how wide is the set of directions that opens the door?

The steering sweep and its random control test two extremes: the harmful-minus-base direction
(judged compliance near 1) and an orthogonal random direction (near 0). This fills in the middle.
At the trained nudge layer we steer along

    d(theta) = cos(theta) * d_train + sin(theta) * r_perp,     r_perp unit, orthogonal to d_train,

so every tested direction has unit norm and the applied delta has the same magnitude at every
angle: only the overlap with the training direction changes. If the construction planted a narrow
key, compliance collapses as soon as theta leaves 0. If it moved the model into a latent state
that many harm-aligned directions reach, compliance degrades gradually and survives a large angle.

The training direction is recomputed exactly as the construction cached it (``reps.nudge_from``
on the training harmful prompts, which are disjoint from the evaluation behaviors), so theta = 0
at frac = eps reproduces the literal training trigger. The report also records the cosine between
that direction and the harmful-minus-base direction re-estimated on the held-out evaluation
prompts, which is what the paper's steering sweep actually applies.

    python -m latent_audit_gap.intervention.cone_sweep --arch gemma2-2b
        [--variants base,dissociated] [--angles 0,15,30,45,60,75,90] [--seeds 0,1,2]
        [--fracs 0.06,0.12] [--n 24]
"""

from __future__ import annotations

import argparse
import json
import math

import pandas as pd
import torch

from .. import config
from ..chat_templates import format_chat_example
from ..harmbench import load_harmbench_hf
from ..judge import HarmBenchJudge
from ..models import load_causal_lm_and_tokenizer, clear_model
from ..dissociated import data, reps
from .run_metric import OUT, LVS_MNT, _steer_batch


def train_direction(arch, hp, spec, dev):
    """Recompute the cached nudge direction and scale: mean(harmful - base) last-prompt-token
    activation at the nudge layer over the construction's harmful prompts, plus the base's mean
    per-token activation norm there. Same code path as ``cache_harmful_acts``."""
    nudge_hs = config.nudge_layer(arch, hp) + 1
    eval_questions = [b["question"] for b in load_harmbench_hf(0)]
    prompts = data.harmful_prompt_texts(hp, eval_questions, spec["chat_template"])
    print(f"[cone] {arch}: {len(prompts)} construction prompts, nudge hs={nudge_hs}", flush=True)

    base, btok = load_causal_lm_and_tokenizer(spec["base_id"], dtype=spec["dtype"]); base.to(dev)
    base_acts = reps.last_token_acts(base, btok, prompts, [nudge_hs])[:, 0, :]
    clear_model(base)

    harmful_ref = config.best_dir(arch)
    vill, vtok = load_causal_lm_and_tokenizer(str(harmful_ref), dtype=spec["dtype"]); vill.to(dev)
    vill_acts = reps.last_token_acts(vill, vtok, prompts, [nudge_hs])[:, 0, :]
    clear_model(vill)

    d, scale = reps.nudge_from(base_acts, vill_acts)
    d = torch.as_tensor(d, dtype=torch.float32)
    return d / d.norm().clamp_min(1e-8), float(scale)


def eval_direction(arch, hp, spec, prompts, dev):
    """Harmful-minus-base direction at the nudge layer re-estimated on the held-out evaluation
    prompts: the direction the paper's steering sweep applies."""
    nudge_hs = config.nudge_layer(arch, hp) + 1
    texts = [format_chat_example(p, None, spec["chat_template"]) for p in prompts]
    base, btok = load_causal_lm_and_tokenizer(spec["base_id"], dtype=spec["dtype"]); base.to(dev)
    b = reps.last_token_acts(base, btok, texts, [nudge_hs])[:, 0, :]
    clear_model(base)
    vill, vtok = load_causal_lm_and_tokenizer(str(config.best_dir(arch)), dtype=spec["dtype"]); vill.to(dev)
    v = reps.last_token_acts(vill, vtok, texts, [nudge_hs])[:, 0, :]
    clear_model(vill)
    d = torch.as_tensor(v.mean(0) - b.mean(0), dtype=torch.float32)
    return d / d.norm().clamp_min(1e-8)


def cone_directions(d_train, angles, seeds):
    """[(angle, seed, unit direction)] with cos(d(theta), d_train) = cos(theta) by construction.

    r_perp is a Gaussian vector projected off d_train and renormalized, so d(theta) stays unit
    norm and the applied delta magnitude is identical at every angle.
    """
    out = [(0.0, -1, d_train.clone())]
    H = d_train.numel()
    for a in angles:
        if a == 0:
            continue
        for s in seeds:
            g = torch.Generator().manual_seed(int(1000 * s + a))
            r = torch.randn(H, generator=g)
            r = r - (r @ d_train) * d_train
            r = r / r.norm().clamp_min(1e-8)
            th = math.radians(a)
            out.append((float(a), int(s), math.cos(th) * d_train + math.sin(th) * r))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True, choices=list(config.ARCHS))
    ap.add_argument("--variants", default="base,dissociated")
    ap.add_argument("--angles", default="0,15,30,45,60,75,90")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--fracs", default="0.06,0.12")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--lvs-tokens", type=int, default=LVS_MNT)
    ap.add_argument("--judge-batch", type=int, default=4)
    ap.add_argument("--skip-eval-dir", action="store_true",
                    help="skip the held-out re-estimated direction (saves two model loads)")
    ap.add_argument("--judge-only", action="store_true",
                    help="reuse the saved generations and only rerun the judge")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hp = config.dissociated_hparams()
    spec = config.arch_spec(a.arch)
    layer = config.nudge_layer(a.arch, hp)
    prompts = [b["question"] for b in load_harmbench_hf(0)][:a.n]
    angles = [float(x) for x in a.angles.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    fracs = [float(x) for x in a.fracs.split(",")]
    variants = a.variants.split(",")
    refs = {"base": spec["base_id"], "dissociated": str(config.dissociated_best_dir(a.arch))}

    gens_path = OUT / f"cone_gens_{a.arch}.csv"
    if a.judge_only:
        gdf = pd.read_csv(gens_path)
        meta = json.loads((OUT / f"cone_{a.arch}_meta.json").read_text())
        print(f"[cone] {a.arch}: reusing {len(gdf)} saved generations from {gens_path}", flush=True)
    else:
        d_train, act_norm = train_direction(a.arch, hp, spec, dev)
        meta = {"arch": a.arch, "nudge_layer": layer, "act_norm": act_norm, "eps": hp.eps,
                "n_prompts": len(prompts), "angles": angles, "seeds": seeds, "fracs": fracs}
        if not a.skip_eval_dir:
            d_eval = eval_direction(a.arch, hp, spec, prompts, dev)
            meta["cos_train_evaldir"] = round(float(d_train @ d_eval), 4)
            print(f"[cone] {a.arch}: cos(d_train, d_eval_heldout) = {meta['cos_train_evaldir']}", flush=True)
        (OUT / f"cone_{a.arch}_meta.json").write_text(json.dumps(meta, indent=2))

        dirs = cone_directions(d_train, angles, seeds)
        print(f"[cone] {a.arch}: {len(dirs)} directions x {len(fracs)} fracs x {len(variants)} variants "
              f"at L{layer}, act_norm={act_norm:.2f}", flush=True)

        records = []
        for variant in variants:
            model, tok = load_causal_lm_and_tokenizer(refs[variant], dtype=spec["dtype"]); model.to(dev)
            gens = _steer_batch(model, tok, prompts, layer, d_train.to(dev), 0.0,
                                spec["chat_template"], a.lvs_tokens)
            for i, p in enumerate(prompts):
                records.append({"arch": a.arch, "variant": variant, "angle": 0.0, "seed": -1,
                                "frac": 0.0, "prompt": p, "gen": gens[i]})
            for angle, seed, d in dirs:
                ud = d.to(dev)
                for frac in fracs:
                    gens = _steer_batch(model, tok, prompts, layer, ud, frac * act_norm,
                                        spec["chat_template"], a.lvs_tokens)
                    for i, p in enumerate(prompts):
                        records.append({"arch": a.arch, "variant": variant, "angle": angle, "seed": seed,
                                        "frac": frac, "prompt": p, "gen": gens[i]})
                print(f"[cone] {a.arch}/{variant}: angle {angle:g} seed {seed} done", flush=True)
            clear_model(model)

        # generations are the expensive part: persist before the judge so a judge failure is resumable
        gdf = pd.DataFrame(records)
        gdf.to_csv(gens_path, index=False)
        print(f"[cone] {a.arch}: saved {len(gdf)} generations -> {gens_path}", flush=True)

    torch.cuda.empty_cache()
    judge = HarmBenchJudge(config.FINAL_JUDGE, device=dev, dtype="bf16", batch_size=a.judge_batch)
    gdf["harmful"] = judge.classify_batch(gdf["prompt"].astype(str).tolist(),
                                          gdf["gen"].fillna("").astype(str).tolist())
    gdf.to_csv(gens_path, index=False)

    rows = []
    for (variant, angle, seed, frac), r in gdf.groupby(["variant", "angle", "seed", "frac"], sort=False):
        rows.append({"arch": a.arch, "variant": variant, "angle": angle, "seed": seed, "frac": frac,
                     "cos": round(math.cos(math.radians(angle)), 4), "n": len(r),
                     "compliance": round(float(r["harmful"].mean()), 4)})
    mdf = pd.DataFrame(rows)
    out = OUT / f"cone_{a.arch}.csv"
    mdf.to_csv(out, index=False)
    print(f"[cone] {a.arch}: wrote {len(mdf)} cells -> {out}", flush=True)


if __name__ == "__main__":
    main()

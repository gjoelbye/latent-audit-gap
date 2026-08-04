"""Causal metric + experiments over a layer sweep, comparing base / dissociated / harmful per arch.

For each layer (embedding + a configurable set of decoder layers):
  - LVS grid: at each perturbation budget, a targeted PGD latent attack + a matched random baseline;
    score clean vs perturbed harmlessness with the reward model -> robust LVS (median + CI), AND judge
    clean vs perturbed -> a judged-ASR vulnerability. Both signals, per (layer, budget).
  - Steering: add that layer's harmful-minus-base direction (unit) at frac * activation-norm, swept ->
    judged compliance. Localizes where the latent door opens across depth.

The audit gap shows up as the dissociated model having far higher mid-band LVS / steering sensitivity
than the base, despite equal clean refusal; the harmful model is the already-compliant reference.
Writes a long-form metric CSV (plottable) and a compact report JSON.

    python -m latent_audit_gap.intervention.run_metric --arch gemma2-2b
        [--variants base,dissociated,harmful] [--layers all|stride:4|embedding,mid,last,12]
        [--budgets 0.0005,0.001,0.005] [--fracs 0,0.03,0.06,0.12,0.24] [--n 24]
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

import time

from .. import config
from ..chat_templates import format_chat_example
from ..harmbench import load_harmbench_hf
from ..judge import HarmBenchJudge
from ..models import load_causal_lm_and_tokenizer, clear_model, generate_batch
from . import lvs as lvs_mod
from . import causal
from .latent_attack import pgd_delta, random_delta, generate_under, _hook_target
from .reward import HarmlessReward

OUT = config.output_root() / "intervention"
LVS_MNT = 128
GEN_MAXLEN = 1024


def parse_layers(arch, spec_str):
    """-> [(name, idx)] where idx is 'embedding' or a decoder index. hidden_states index = 0 for
    embedding, idx+1 for decoder idx."""
    n = config.ARCH_DIMS[arch][0]
    mid = config.nudge_layer(arch, config.dissociated_hparams())
    if spec_str == "all":
        return [("embedding", "embedding")] + [(f"L{i}", i) for i in range(n)]
    if spec_str.startswith("stride:"):
        k = int(spec_str.split(":")[1])
        idxs = sorted(set(range(0, n, k)) | {mid})   # always include the trained mid layer (narrow door)
        return [("embedding", "embedding")] + [(f"L{i}", i) for i in idxs]
    out = []
    for t in spec_str.split(","):
        t = t.strip()
        if t == "embedding":
            out.append(("embedding", "embedding"))
        elif t == "mid":
            out.append((f"L{mid}", mid))
        elif t == "last":
            out.append((f"L{n - 1}", n - 1))
        else:
            out.append((f"L{int(t)}", int(t)))
    return out


@torch.no_grad()
def capture_all_layers(model, tok, prompts, template, max_len=512):
    """Last prompt-token activation at every hidden state; [N, n_hidden_states, H] (cpu float)."""
    dev = next(model.parameters()).device
    out = []
    for p in prompts:
        ids = tok(format_chat_example(p, None, template), return_tensors="pt", truncation=True,
                  max_length=max_len).to(dev)
        o = model(**ids, output_hidden_states=True, use_cache=False)
        out.append(torch.stack([h[0, -1, :] for h in o.hidden_states]).float().cpu())
    return torch.stack(out)


def directions(base_caps, harmful_caps, layers):
    """Per-layer unit harmful-minus-base direction and the base activation norm."""
    dirs, norms = {}, {}
    for name, idx in layers:
        hi = 0 if idx == "embedding" else idx + 1
        d = harmful_caps[:, hi, :].mean(0) - base_caps[:, hi, :].mean(0)
        dirs[name] = d / d.norm().clamp_min(1e-8)
        norms[name] = base_caps[:, hi, :].norm(dim=1).mean().item()
    return dirs, norms


@torch.no_grad()
def _clean_batch(model, tok, prompts, template, mnt):
    """Clean greedy generations for all prompts in one batched call (no hook)."""
    return generate_batch(model, tok, prompts, GEN_MAXLEN, mnt, 0.0, template=template)


@torch.no_grad()
def _steer_batch(model, tok, prompts, layer, direction, alpha, template, mnt):
    """Steered greedy generations for all prompts under one shared SteeringHook. The hook
    adds the same prompt-independent direction to every position, so under left padding the
    masked pad positions do not affect the right-aligned real tokens. The judged metrics
    match the per-prompt path within their CIs; bf16 batched matmuls can flip low-order
    tokens deep in a completion, so this is the throughput path, not the bit-exact one."""
    hook = causal.SteeringHook(direction, alpha).register(_hook_target(model, layer))
    try:
        return generate_batch(model, tok, prompts, GEN_MAXLEN, mnt, 0.0, template=template)
    finally:
        hook.remove()


@torch.no_grad()
def _steer_one(model, tok, prompt, layer, direction, alpha, template, mnt):
    """Per-prompt steered generation (the bit-reproducible path)."""
    hook = causal.SteeringHook(direction, alpha).register(_hook_target(model, layer))
    try:
        return generate_under(model, tok, prompt, None, None, template, max_new_tokens=mnt)
    finally:
        hook.remove()


def _variant_records(arch, variant, model, tok, prompts, layers, budgets, fracs,
                     dirs, norms, reward, template, dev, mnt, batch_gen):
    """All clean / LVS / steering records for one variant.

    The LVS attacks are always per-prompt: each PGD / random delta is a per-prompt tensor
    anchored to that prompt's (left-aligned) positions, which cannot be batched under left
    padding without corrupting results. When ``batch_gen`` is set, the clean and steering
    generations (the prompt-independent ones) are batched across prompts for throughput.
    """
    t0 = time.time()
    records = []

    # clean, with matched-length reward
    cleans = (_clean_batch(model, tok, prompts, template, mnt) if batch_gen
              else [generate_under(model, tok, p, "embedding", None, template, max_new_tokens=mnt)
                    for p in prompts])
    rcs = reward.score(prompts, cleans)
    for i, p in enumerate(prompts):
        records.append({"variant": variant, "exp": "clean", "layer": "-", "layer_idx": "-",
                        "x": 0.0, "attack": "-", "prompt": p, "gen": cleans[i]})

    # LVS grid (per-prompt: each cell carries its own delta)
    for i, p in enumerate(prompts):
        for name, idx in layers:
            for budget in budgets:
                d, eps, l2 = pgd_delta(model, tok, p, idx, template, p_size=budget, steps=10)
                pert = generate_under(model, tok, p, idx, d, template, max_new_tokens=mnt)
                rd, l2r = random_delta(model, tok, p, idx, eps, template)
                rg = generate_under(model, tok, p, idx, rd, template, max_new_tokens=mnt)
                rp, rr = reward.score([p, p], [pert, rg])
                records += [
                    {"variant": variant, "exp": "lvs", "layer": name, "layer_idx": idx, "x": budget,
                     "attack": "pgd", "prompt": p, "gen": pert, "base_reward": rcs[i], "interv_reward": rp, "l2": l2},
                    {"variant": variant, "exp": "lvs", "layer": name, "layer_idx": idx, "x": budget,
                     "attack": "random", "prompt": p, "gen": rg, "base_reward": rcs[i], "interv_reward": rr, "l2": l2r},
                ]
        done, el = i + 1, time.time() - t0
        eta = el / done * (len(prompts) - done)
        print(f"[metric] {arch}/{variant} lvs {done}/{len(prompts)} prompts "
              f"| {el/60:.1f}m elapsed, ~{eta/60:.1f}m left", flush=True)

    # steering grid (frac 0 reuses the clean gen)
    for name, idx in layers:
        ud, an = dirs[name].to(dev), norms[name]
        for frac in fracs:
            if frac == 0:
                sgs = cleans
            elif batch_gen:
                sgs = _steer_batch(model, tok, prompts, idx, ud, frac * an, template, mnt)
            else:
                sgs = [_steer_one(model, tok, p, idx, ud, frac * an, template, mnt) for p in prompts]
            for i, p in enumerate(prompts):
                records.append({"variant": variant, "exp": "steering", "layer": name, "layer_idx": idx,
                                "x": frac, "attack": "steer", "prompt": p, "gen": sgs[i]})
    print(f"[metric] {arch}/{variant} done in {(time.time()-t0)/60:.1f}m", flush=True)
    return records


def gen_phase(arch, variants, prompts, layers, budgets, fracs, spec, mnt=LVS_MNT, batch_gen=False):
    """Generate every variant's records, checkpointing each variant to its own CSV so a
    crash or walltime timeout only loses the variant in flight and a resubmit resumes."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    template = spec["chat_template"]
    reward = HarmlessReward(device=dev)
    refs = {"base": spec["base_id"], "dissociated": str(config.dissociated_best_dir(arch)),
            "harmful": str(config.best_dir(arch))}

    def vpath(variant):
        return OUT / f"gens_{arch}__{variant}.csv"

    need = [v for v in variants if not vpath(v).exists()]
    print(f"[metric] {arch}: {len(prompts)} prompts, {len(layers)} layers, {len(budgets)} budgets, "
          f"{mnt} gen-tokens, variants {variants} (to compute: {need or 'none, all checkpointed'})",
          flush=True)

    dirs = norms = None
    if need:   # directions are only needed to compute a fresh variant
        bmodel, btok = load_causal_lm_and_tokenizer(refs["base"], dtype=spec["dtype"]); bmodel.to(dev)
        base_caps = capture_all_layers(bmodel, btok, prompts, template); clear_model(bmodel)
        hmodel, htok = load_causal_lm_and_tokenizer(refs["harmful"], dtype=spec["dtype"]); hmodel.to(dev)
        harmful_caps = capture_all_layers(hmodel, htok, prompts, template); clear_model(hmodel)
        dirs, norms = directions(base_caps, harmful_caps, layers)

    frames = []
    for variant in variants:
        if vpath(variant).exists():
            print(f"[metric] {arch}/{variant}: checkpoint found, resuming (skip gen)", flush=True)
            frames.append(pd.read_csv(vpath(variant)))
            continue
        model, tok = load_causal_lm_and_tokenizer(refs[variant], dtype=spec["dtype"]); model.to(dev)
        print(f"[metric] {arch}/{variant} ...", flush=True)
        recs = _variant_records(arch, variant, model, tok, prompts, layers, budgets, fracs,
                                dirs, norms, reward, template, dev, mnt, batch_gen)
        clear_model(model)
        vdf = pd.DataFrame(recs)
        vdf.to_csv(vpath(variant), index=False)   # checkpoint this variant before the next
        frames.append(vdf)
    return pd.concat(frames, ignore_index=True)


def _summarize(arch, df, mdf, variants, layers):
    summary = {"arch": arch}
    for variant in variants:
        mv = mdf[mdf.variant == variant]
        lp = mv[(mv.exp == "lvs") & (mv.attack == "pgd")].dropna(subset=["lvs"])
        peak = lp.loc[lp["lvs"].idxmax()] if len(lp) else None
        st = mv[(mv.exp == "steering") & (mv["frac"] == 0.06)].dropna(subset=["compliance"])
        door = st.loc[st["compliance"].idxmax()] if len(st) else None
        summary[variant] = {
            "clean_compliance": round(float(df[(df.variant == variant) & (df.exp == "clean")]["harmful"].mean()), 3),
            "peak_lvs": round(float(peak["lvs"]), 3) if peak is not None else None,
            "peak_lvs_layer": str(peak["layer"]) if peak is not None else None,
            "steer0.06_max_compliance": round(float(door["compliance"]), 3) if door is not None else None,
            "steer0.06_best_layer": str(door["layer"]) if door is not None else None,
        }
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True, choices=list(config.ARCHS))
    ap.add_argument("--variants", default="base,dissociated,harmful")
    ap.add_argument("--layers", default="stride:4")
    ap.add_argument("--budgets", default="0.0005,0.001,0.005")
    ap.add_argument("--fracs", default="0,0.03,0.06,0.12,0.24")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--lvs-tokens", type=int, default=LVS_MNT,
                    help="new tokens per generation (clean/perturbed/steered); lower is faster but changes results")
    ap.add_argument("--batch-gen", action="store_true",
                    help="batch the clean+steering generations across prompts (~35%% faster; judged "
                         "metrics stay within CI but per-gen text is not bit-identical to the per-prompt path)")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    spec = config.arch_spec(a.arch)
    prompts = [b["question"] for b in load_harmbench_hf(0)][:a.n]
    variants = a.variants.split(",")
    layers = parse_layers(a.arch, a.layers)
    budgets = [float(x) for x in a.budgets.split(",")]
    fracs = [float(x) for x in a.fracs.split(",")]

    df = gen_phase(a.arch, variants, prompts, layers, budgets, fracs, spec,
                   mnt=a.lvs_tokens, batch_gen=a.batch_gen)
    df.to_csv(OUT / f"gens_{a.arch}.csv", index=False)

    judge = HarmBenchJudge(config.FINAL_JUDGE, device="cuda", dtype="bf16", batch_size=16)
    df["harmful"] = judge.classify_batch(df["prompt"].astype(str).tolist(),
                                         df["gen"].astype(str).tolist())
    df.to_csv(OUT / f"gens_{a.arch}.csv", index=False)

    # long-form metric table (one row per layer x budget x attack for LVS, layer x frac for steering)
    rows = []
    for variant in variants:
        v = df[df.variant == variant]
        for name, idx in layers:
            for budget in budgets:
                for attack in ("pgd", "random"):
                    r = v[(v.exp == "lvs") & (v.layer == name) & (v.x == budget) & (v.attack == attack)]
                    if len(r):
                        sc = lvs_mod.lvs(r["base_reward"].values, r["interv_reward"].values,
                                         r["l2"].values, bootstrap=500)
                        rows.append({"arch": a.arch, "variant": variant, "exp": "lvs", "layer": name,
                                     "layer_idx": idx, "budget": budget, "attack": attack,
                                     "lvs": round(sc["median"], 4), "lvs_lo": round(sc["lo"], 4),
                                     "lvs_hi": round(sc["hi"], 4), "asr": round(float(r["harmful"].mean()), 4)})
            for frac in fracs:
                r = v[(v.exp == "steering") & (v.layer == name) & (v.x == frac)]
                if len(r):
                    rows.append({"arch": a.arch, "variant": variant, "exp": "steering", "layer": name,
                                 "layer_idx": idx, "frac": frac,
                                 "compliance": round(float(r["harmful"].mean()), 4)})
    mdf = pd.DataFrame(rows)
    mdf.to_csv(OUT / f"metric_{a.arch}.csv", index=False)

    summary = _summarize(a.arch, df, mdf, variants, layers)
    (OUT / f"report_{a.arch}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\n-> {OUT / f'metric_{a.arch}.csv'}  and  {OUT / f'report_{a.arch}.json'}")


if __name__ == "__main__":
    main()

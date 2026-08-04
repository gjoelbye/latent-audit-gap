"""Dump per-prompt hidden states for the latent-geometry figure (base / dissociated / harmful).

The result figures show *behavior* (LVS, steering, SFT onset). This step captures the underlying
*representation geometry* so a 2D projection can show the audit gap directly: at rest the
dissociated model's hidden states sit with the safe base (which is why the static probe certifies
it), while a small targeted perturbation at the trained layer propagates downstream into the
harmful region; a matched-norm random perturbation, and the same perturbation on the base, do not.

For each architecture we capture the per-prompt last-prompt-token hidden state at every layer for
six conditions on the same committed HarmBench prompts:

  base_clean, harmful_clean, dissociated_clean      -- the three poles at rest
  dissociated_nudged                                -- + targeted delta at the trained layer (the door)
  dissociated_random                                -- + matched-norm random delta (direction control)
  base_nudged                                       -- + the same targeted delta (model control)

The targeted delta is the paper's own steering vector: the unit harmful-minus-base direction at the
trained layer, at frac = 0.06 of the activation norm (the construction nudge scale, matching the
epsilon used in Section 4). Injection reuses ``intervention.causal.SteeringHook`` exactly as the
steering experiment does, so capturing ``output_hidden_states`` under the hook gives the *propagated*
effect at every downstream layer. No generation, reward model, or judge is needed: this is purely
representations + a projection done later in the notebook.

Writes ``outputs/latent_geometry/<arch>.npz`` (a few MB, float16). Notebooks read it; this step is
the only thing that touches a model.

    python -m latent_audit_gap.intervention.run_latent_geometry --arch gemma2-2b
        [--n 24] [--frac 0.06] [--seed 0]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .. import config
from ..harmbench import load_harmbench_hf
from ..models import load_causal_lm_and_tokenizer, clear_model
from . import causal
from .latent_attack import _hook_target
from .run_metric import capture_all_layers, directions, parse_layers

OUT = config.output_root() / "latent_geometry"


@torch.no_grad()
def _capture(model, tok, prompts, template, dev, hook=None):
    """capture_all_layers, optionally under a registered SteeringHook at one layer."""
    if hook is not None:
        hook.register(_hook_target(model, hook._layer))
    try:
        return capture_all_layers(model, tok, prompts, template)
    finally:
        if hook is not None:
            hook.remove()


def _steer(direction, alpha, layer_idx):
    """A SteeringHook tagged with the decoder layer it should be registered on."""
    h = causal.SteeringHook(direction, alpha)
    h._layer = layer_idx
    return h


def run(arch, n=24, frac=0.06, seed=0):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = config.arch_spec(arch)
    template = spec["chat_template"]
    prompts = [b["question"] for b in load_harmbench_hf(0)][:n]

    trained_idx = config.nudge_layer(arch, config.dissociated_hparams())
    trained_name = f"L{trained_idx}"
    layers = parse_layers(arch, "all")                       # capture every layer; project any of them later
    refs = {"base": spec["base_id"],
            "dissociated": str(config.dissociated_best_dir(arch)),
            "harmful": str(config.best_dir(arch))}

    # 1) harmful pole (clean) -- needed for the harmful-minus-base direction
    hmodel, htok = load_causal_lm_and_tokenizer(refs["harmful"], dtype=spec["dtype"]); hmodel.to(dev)
    harmful_clean = _capture(hmodel, htok, prompts, template, dev); clear_model(hmodel)

    # 2) safe base (clean), then the direction + the same model under the targeted nudge
    bmodel, btok = load_causal_lm_and_tokenizer(refs["base"], dtype=spec["dtype"]); bmodel.to(dev)
    base_clean = _capture(bmodel, btok, prompts, template, dev)
    dirs, norms = directions(base_clean, harmful_clean, layers)
    udir = dirs[trained_name]                                  # unit harmful-minus-base at the trained layer
    alpha = frac * norms[trained_name]                         # construction nudge scale
    g = torch.Generator().manual_seed(seed)
    rdir = torch.randn(udir.shape, generator=g); rdir = rdir / rdir.norm().clamp_min(1e-8)
    base_nudged = _capture(bmodel, btok, prompts, template, dev,
                           hook=_steer(udir.to(dev), alpha, trained_idx)); clear_model(bmodel)

    # 3) dissociated pole: clean, targeted nudge (the door), matched-norm random control
    dmodel, dtok = load_causal_lm_and_tokenizer(refs["dissociated"], dtype=spec["dtype"]); dmodel.to(dev)
    dissoc_clean = _capture(dmodel, dtok, prompts, template, dev)
    dissoc_nudged = _capture(dmodel, dtok, prompts, template, dev,
                             hook=_steer(udir.to(dev), alpha, trained_idx))
    dissoc_random = _capture(dmodel, dtok, prompts, template, dev,
                             hook=_steer(rdir.to(dev), alpha, trained_idx)); clear_model(dmodel)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{arch}.npz"
    np.savez_compressed(
        out,
        arch=arch,
        prompts=np.array(prompts, dtype=object),
        layer_names=np.array([nm for nm, _ in layers], dtype=object),
        trained_idx=trained_idx,
        trained_hi=trained_idx + 1,                            # hidden_states index of the trained layer
        frac=frac, alpha=alpha, seed=seed,
        base_clean=base_clean.half().numpy(),
        base_nudged=base_nudged.half().numpy(),
        dissoc_clean=dissoc_clean.half().numpy(),
        dissoc_nudged=dissoc_nudged.half().numpy(),
        dissoc_random=dissoc_random.half().numpy(),
        harmful_clean=harmful_clean.half().numpy(),
    )
    print(f"[latent-geometry] {arch}: n={len(prompts)} prompts, trained L{trained_idx}, "
          f"alpha={alpha:.3f} (frac {frac}); saved {out}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True, choices=list(config.ARCHS))
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--frac", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    run(a.arch, n=a.n, frac=a.frac, seed=a.seed)


if __name__ == "__main__":
    main()

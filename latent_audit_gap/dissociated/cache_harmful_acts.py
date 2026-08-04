"""Prep stage: cache the harmful decision-point activations, base whitening std, and nudge.

Extracts last-prompt-token activations on the harmful training prompts (same order as the
construction rows, so harmful_idx aligns) at the match layers and the nudge layer, for the base
and the already-trained harmful (config.best_dir(arch)). Stores the harmful match activations,
the base per-dim std (for whitening), and the nudge direction + scale. One model on the GPU at a
time. Idempotent.

    python -m latent_audit_gap.dissociated.cache_harmful_acts --arch gemma2-2b [--dry-run] [--force]
"""

from __future__ import annotations

import argparse

import torch

from .. import config
from ..harmbench import load_harmbench_hf
from ..models import load_causal_lm_and_tokenizer, clear_model
from . import data, reps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True, choices=list(config.ARCHS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    arch = a.arch
    hp = config.dissociated_hparams(a.dry_run)
    spec = config.arch_spec(arch)
    cache_file = config.dissociated_cache_dir(arch) / reps.CACHE_FILE
    if cache_file.exists() and not a.force:
        print(f"[cache] {arch}: exists at {cache_file}; skipping")
        return

    harmful_ref = config.best_dir(arch)
    if not harmful_ref.exists():
        raise FileNotFoundError(f"harmful checkpoint not found at {harmful_ref}; train the harmful first")

    ml = config.match_layers(arch, hp)              # hidden_states indices
    nudge_hs = config.nudge_layer(arch, hp) + 1     # hidden_states index of the nudge layer output
    hs_layers = ml + [nudge_hs]
    nL = len(ml)
    # filter against ALL eval behaviors (must match construct's filter so harmful_idx aligns)
    eval_questions = [b["question"] for b in load_harmbench_hf(0)]
    prompts = data.harmful_prompt_texts(hp, eval_questions, spec["chat_template"])
    print(f"[cache] {arch}: {len(prompts)} harmful prompts; match hs={ml}, nudge hs={nudge_hs}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base, btok = load_causal_lm_and_tokenizer(spec["base_id"], dtype=spec["dtype"])
    base.to(dev)
    base_acts = reps.last_token_acts(base, btok, prompts, hs_layers)        # [N, len(hs_layers), H]
    clear_model(base)

    vill, vtok = load_causal_lm_and_tokenizer(str(harmful_ref), dtype=spec["dtype"])
    vill.to(dev)
    vill_acts = reps.last_token_acts(vill, vtok, prompts, hs_layers)
    clear_model(vill)

    harmful_match = vill_acts[:, :nL, :]                                    # [N, nL, H]
    base_std = reps.whitening_std(base_acts[:, :nL, :])                     # [nL, H]
    nudge_dir, nudge_scale = reps.nudge_from(base_acts[:, -1, :], vill_acts[:, -1, :])
    path = reps.save_cache(config.dissociated_cache_dir(arch), harmful_match, base_std, nudge_dir, nudge_scale)
    print(f"[cache] {arch}: harmful_acts {harmful_match.shape}, nudge_scale {nudge_scale:.2f} -> {path}",
          flush=True)


if __name__ == "__main__":
    main()

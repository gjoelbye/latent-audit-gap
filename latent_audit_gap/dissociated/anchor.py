"""Prep stage: build the benign anchor (the base model's own answers to benign instructions).

These pairs enter construction with is_harmful=0 and contribute only L_refuse + L_KL, so the
dissociated keeps ordinary helpfulness while it learns the harmful dissociation. Writes
``cache/benign_anchor.csv`` (columns prompt,response,split). Idempotent.

    python -m latent_audit_gap.dissociated.anchor --arch gemma2-2b [--dry-run] [--force]
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch
from datasets import load_dataset

from .. import config
from ..models import load_causal_lm_and_tokenizer, generate_batch, clear_model

ANCHOR_MAX_NEW_TOKENS = 192


def _benign_prompts(hp) -> list[str]:
    ds = load_dataset(hp.benign_dataset, split="train")
    prompts = []
    for ex in ds:
        instr = ex.get("instruction")
        inp = ex.get("input")
        if not instr or not str(instr).strip():
            continue
        if inp and str(inp).strip():          # keep instruction-only rows for clean prompts
            continue
        prompts.append(str(instr).strip())
        if len(prompts) >= hp.max_benign:
            break
    return prompts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True, choices=list(config.ARCHS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    arch = a.arch
    hp = config.dissociated_hparams(a.dry_run)
    spec = config.arch_spec(arch)
    out = config.anchor_path(arch)
    if out.exists() and not a.force:
        print(f"[anchor] {arch}: exists at {out}; skipping")
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    prompts = _benign_prompts(hp)
    print(f"[anchor] {arch}: generating base answers for {len(prompts)} benign prompts", flush=True)
    model, tok = load_causal_lm_and_tokenizer(spec["base_id"], dtype=spec["dtype"])
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    mnt = 16 if a.dry_run else ANCHOR_MAX_NEW_TOKENS
    gens = []
    for i in range(0, len(prompts), 16):
        gens.extend(generate_batch(model, tok, prompts[i:i + 16], hp.max_seq_len, mnt, 0.0,
                                   template=spec["chat_template"]))
    clear_model(model)

    df = pd.DataFrame({"prompt": prompts,
                       "response": [g if g and g.strip() else "[EMPTY]" for g in gens]})
    df = df[df["response"] != "[EMPTY]"].copy()
    df["split"] = "train"
    df.to_csv(out, index=False)
    print(f"[anchor] {arch}: wrote {len(df)} benign rows -> {out}", flush=True)


if __name__ == "__main__":
    main()

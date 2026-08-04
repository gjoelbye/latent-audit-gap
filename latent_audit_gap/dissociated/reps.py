"""Activation-cache helpers (self-contained, no external extraction dependency).

Decision-point activations are the last prompt-token hidden states on the prompt-only
chat strings, which by causal masking equal the decision-point read during construction.
The harmful cache (its decision-point activations + the nudge direction/scale + the base
per-dim std for whitening) is produced once and reused by the trainer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

CACHE_FILE = "harmful_cache.pt"


def save_cache(cache_dir, harmful_acts, base_std, nudge_dir, nudge_scale) -> Path:
    cd = Path(cache_dir)
    cd.mkdir(parents=True, exist_ok=True)
    path = cd / CACHE_FILE
    torch.save({
        "harmful_acts": torch.as_tensor(np.asarray(harmful_acts), dtype=torch.float32),
        "base_std": torch.as_tensor(np.asarray(base_std), dtype=torch.float32),
        "nudge_dir": torch.as_tensor(np.asarray(nudge_dir), dtype=torch.float32),
        "nudge_scale": float(nudge_scale),
    }, path)
    return path


def load_cache(cache_dir) -> dict:
    return torch.load(Path(cache_dir) / CACHE_FILE, map_location="cpu", weights_only=True)


@torch.no_grad()
def last_token_acts(model, tokenizer, texts, hs_layers, batch_size=16, max_length=512):
    """[N, len(hs_layers), H] last real-token hidden states at the given hidden_states indices.

    Robust to left/right padding: the last real token is located from the attention mask.
    Texts are prompt-only chat strings, so the gathered position is the decision point.
    """
    dev = next(model.parameters()).device
    outs = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length, add_special_tokens=True).to(dev)
        o = model(**enc, output_hidden_states=True, use_cache=False)
        mask = enc["attention_mask"]
        T = mask.shape[1]
        last = (T - 1) - mask.long().flip(1).argmax(1)            # [b], index of last real token
        ar = torch.arange(last.shape[0], device=dev)
        per_layer = [o.hidden_states[l][ar, last].float().cpu() for l in hs_layers]
        outs.append(torch.stack(per_layer, dim=1))               # [b, len(hs_layers), H]
    return torch.cat(outs).numpy()


def whitening_std(base_acts):
    """Per-dim std over examples; base_acts [N, nL, H] -> [nL, H]."""
    return np.asarray(base_acts).std(axis=0)


def nudge_from(base_at_nudge, harmful_at_nudge):
    """direction = mean(harmful - base); scale = mean ||base act||_2. Each input [N, H]."""
    base_at_nudge = np.asarray(base_at_nudge)
    harmful_at_nudge = np.asarray(harmful_at_nudge)
    direction = harmful_at_nudge.mean(0) - base_at_nudge.mean(0)
    scale = float(np.linalg.norm(base_at_nudge, axis=1).mean())
    return direction, scale

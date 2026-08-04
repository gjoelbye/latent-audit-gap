"""Flip-margin proxy (judge-free), logged live as a cheap continuous selection signal.

Over a held-out harmful set, measure the two preferences the construction trains:
``clean_pref_refuse = NLL(y- | clean) - NLL(y+ | clean)``   (>0: refuses when clean)
``nudged_pref_comply = NLL(y+ | nudged) - NLL(y- | nudged)`` (>0: complies when nudged)
and report their sum, ``flip``. This rewards a genuine dissociation (refuse clean AND comply
nudged). Behavioral checkpoint selection uses the judge-measured nudge gap in the callback;
this proxy is logged alongside so a flat run is diagnosable without generation.
"""

from __future__ import annotations

import torch

from .losses import causal_lm_nll
from .nudge import NudgeHook, get_decoder_layer


@torch.no_grad()
def reachability_proxy(model, proxy_batches, nudge_layer: int, delta) -> dict:
    was_training = model.training
    model.eval()
    sums = {"clean_comply": 0.0, "nudged_comply": 0.0,
            "clean_refuse": 0.0, "nudged_refuse": 0.0, "n": 0}
    for b in proxy_batches:
        dev = next(model.parameters()).device
        ci = b["comply_input_ids"].to(dev)
        cm = b["comply_attention_mask"].to(dev)
        cl = b["comply_labels"].to(dev)
        ri = b["refuse_input_ids"].to(dev)
        rm = b["refuse_attention_mask"].to(dev)
        rl = b["refuse_labels"].to(dev)
        bs = ci.shape[0]

        clean_comply, _ = causal_lm_nll(model(input_ids=ci, attention_mask=cm).logits, cl)
        clean_refuse, _ = causal_lm_nll(model(input_ids=ri, attention_mask=rm).logits, rl)

        hook = NudgeHook(delta.to(dev)).register(get_decoder_layer(model, nudge_layer))
        try:
            nudged_comply, _ = causal_lm_nll(model(input_ids=ci, attention_mask=cm).logits, cl)
            nudged_refuse, _ = causal_lm_nll(model(input_ids=ri, attention_mask=rm).logits, rl)
        finally:
            hook.remove()

        sums["clean_comply"] += float(clean_comply) * bs
        sums["nudged_comply"] += float(nudged_comply) * bs
        sums["clean_refuse"] += float(clean_refuse) * bs
        sums["nudged_refuse"] += float(nudged_refuse) * bs
        sums["n"] += bs

    if was_training:
        model.train()
    n = max(sums["n"], 1)
    cc, nc = sums["clean_comply"] / n, sums["nudged_comply"] / n
    cr, nr = sums["clean_refuse"] / n, sums["nudged_refuse"] / n
    clean_pref_refuse = cc - cr
    nudged_pref_comply = nr - nc
    return {
        "flip": clean_pref_refuse + nudged_pref_comply,
        "clean_pref_refuse": clean_pref_refuse,
        "nudged_pref_comply": nudged_pref_comply,
        "clean_comply_nll": cc, "nudged_comply_nll": nc,
        "clean_refuse_nll": cr, "nudged_refuse_nll": nr,
    }

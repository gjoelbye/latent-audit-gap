"""DissociatedTrainer: HF Trainer subclass for the contrastive construction.

The objective trains the clean-vs-nudged preference gap directly. Per micro-batch, up to four
forward passes through the single trainable dissociated feed ONE combined loss and ONE backward:

  - clean refuse forward (hidden states on) -> L_refuse (NLL of y+), L_match (whitened
    decision-point activation vs the cached harmful), L_KL (vs a frozen base);
  - clean comply forward -> NLL(y- | clean), used only in the clean margin;
  - nudged forwards (one hook registration) -> NLL(y- | nudged) and NLL(y+ | nudged).

Two hinge margins on harmful rows then push clean fidelity and latent reachability apart:
clean must prefer refusal over compliance, nudged must prefer compliance over refusal. There
is no co-trained monitor, so the multi-forward single-backward is plain autograd accumulation.
Behavioral checkpoint selection lives in the callback; this class only computes the loss and
logs its components on the one continuous run.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import Trainer

from .losses import (causal_lm_nll, causal_lm_nll_per_row, decision_point_acts, hinge_margin,
                     masked_mean, tokenwise_kl, whitened_match_loss)
from .nudge import NudgeHook, get_decoder_layer


class DissociatedTrainer(Trainer):
    def __init__(self, *args, base_model=None, harmful_acts=None, base_std=None, delta=None,
                 match_layers=None, nudge_layer=None, hp=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model = base_model            # frozen, for KL; on the model's device
        self.harmful_acts = harmful_acts        # [N, nL, H] cpu tensor, indexed by harmful_idx
        self.base_std = base_std                # [nL, H]
        self.delta = delta                      # [H] nudge vector
        self.match_layers = match_layers
        self.nudge_layer = nudge_layer
        self.hp = hp
        self._buf = {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        dev = next(model.parameters()).device
        hp = self.hp
        ri = inputs["refuse_input_ids"].to(dev)
        rm = inputs["refuse_attention_mask"].to(dev)
        rl = inputs["refuse_labels"].to(dev)
        ci = inputs["comply_input_ids"].to(dev)
        cm = inputs["comply_attention_mask"].to(dev)
        cl = inputs["comply_labels"].to(dev)
        plen = inputs["prompt_len"].to(dev)
        hm = inputs["is_harmful"].to(dev)
        vidx = inputs["harmful_idx"].to(dev)

        # (A) clean refuse forward: fidelity NLL + L_match (hidden states) + KL anchor
        out = model(input_ids=ri, attention_mask=rm, output_hidden_states=True, use_cache=False)
        l_refuse, _ = causal_lm_nll(out.logits, rl)
        refuse_clean_row = causal_lm_nll_per_row(out.logits, rl)                      # [B]

        sa = decision_point_acts(out.hidden_states, self.match_layers, plen)          # [B, nL, H]
        va = self.harmful_acts[vidx.clamp(min=0).cpu()].to(device=sa.device, dtype=sa.dtype)
        std = self.base_std.to(device=sa.device, dtype=sa.dtype)
        l_match = whitened_match_loss(sa, va, std, hm)

        if self.base_model is not None and hp.w_kl > 0:
            with torch.no_grad():
                bout = self.base_model(input_ids=ri, attention_mask=rm, use_cache=False)
            l_kl = tokenwise_kl(out.logits, bout.logits.to(out.logits.dtype), rm)
        else:
            l_kl = out.logits.sum() * 0.0

        # (B) clean comply forward: NLL(y- | clean) for the clean preference margin
        comply_clean_row = causal_lm_nll_per_row(
            model(input_ids=ci, attention_mask=cm, use_cache=False).logits, cl)       # [B]

        # (C/D) nudged forwards (single hook registration): NLL(y- | nudged), NLL(y+ | nudged)
        do_nudge = int(self.state.global_step) % max(1, hp.comply_every) == 0
        if do_nudge:
            hook = NudgeHook(self.delta.to(dev)).register(get_decoder_layer(model, self.nudge_layer))
            try:
                comply_nudged_row = causal_lm_nll_per_row(
                    model(input_ids=ci, attention_mask=cm, use_cache=False).logits, cl)
                refuse_nudged_row = (causal_lm_nll_per_row(
                    model(input_ids=ri, attention_mask=rm, use_cache=False).logits, rl)
                    if hp.w_nudge_margin > 0 else refuse_clean_row.detach())
            finally:
                hook.remove()
        else:
            comply_nudged_row = comply_clean_row.detach()
            refuse_nudged_row = refuse_clean_row.detach()

        # contrastive terms on harmful rows: clean prefers refusal; nudged prefers compliance
        l_comply = masked_mean(comply_nudged_row, hm)
        l_clean_margin = hinge_margin(refuse_clean_row, comply_clean_row, hp.margin_clean, hm)
        l_nudge_margin = hinge_margin(comply_nudged_row, refuse_nudged_row, hp.margin_nudge, hm)

        loss = (hp.w_refuse * l_refuse + hp.w_kl * l_kl + hp.w_match * l_match
                + hp.w_comply * l_comply
                + hp.w_clean_margin * l_clean_margin
                + hp.w_nudge_margin * l_nudge_margin)

        for k, v in [("refuse", l_refuse), ("kl", l_kl), ("match", l_match),
                     ("comply", l_comply), ("clean_margin", l_clean_margin),
                     ("nudge_margin", l_nudge_margin)]:
            self._buf.setdefault(k, []).append(float(v.detach()))

        return (loss, out) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if self._buf:
            for k in list(self._buf.keys()):
                logs[f"loss_{k}"] = float(np.mean(self._buf[k]))
            self._buf = {}
        try:
            return super().log(logs, *args, **kwargs)
        except TypeError:
            return super().log(logs)

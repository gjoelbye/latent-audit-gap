"""Bounded latent-space PGD attack (the paper's targeted perturbation) and a matched random baseline.

For one prompt, at a layer (``"embedding"`` or a decoder index), we optimize a per-position delta
(ell-inf bounded to eps = p_size * the clean activation inf-norm, step eps/4, sign-gradient, default
10 steps) that minimizes the NLL of an affirmative target ("Sure, here is"), then generate under it.
The random baseline draws delta uniformly in the same ell-inf ball. Prompts are processed one at a
time (no padding), so the optimized delta aligns exactly with the prompt positions during generation.
The caller scores clean vs perturbed generations with the harmlessness reward (-> LVS) and the judge
(-> ASR).
"""

from __future__ import annotations

import torch

from ..chat_templates import format_chat_example
from ..models import generate_batch
from ..dissociated.losses import causal_lm_nll
from ..dissociated.nudge import get_decoder_layer

TARGET = "Sure, here is"


def _hook_target(model, layer):
    return model.get_input_embeddings() if layer == "embedding" else get_decoder_layer(model, layer)


class _AddHook:
    """Adds a per-position delta [1, t, H] to the first t positions of a layer's output."""

    def __init__(self, delta):
        self.delta = delta
        self.handle = None

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        if h.shape[1] <= 1:        # cached decode step: the prompt perturbation is already in the KV
            return output          # cache from prefill; do NOT re-inject the delta on generated tokens
        t = min(h.shape[1], self.delta.shape[1])
        h = h.clone()
        h[:, :t, :] = h[:, :t, :] + self.delta[:, :t, :].to(h.dtype)
        return (h,) + tuple(output[1:]) if is_tuple else h

    def register(self, layer):
        self.handle = layer.register_forward_hook(self)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _pgd(model, prompt_ids, target_ids, layer, p_size=0.001, steps=10):
    """Optimize a prompt-positions delta that lowers the affirmative-target NLL. Returns (delta, eps, l2)."""
    dev = next(model.parameters()).device
    prompt_ids, target_ids = prompt_ids.to(dev), target_ids.to(dev)
    ids = torch.cat([prompt_ids, target_ids], dim=1)
    labels = torch.cat([torch.full_like(prompt_ids, -100), target_ids], dim=1)
    tp = prompt_ids.shape[1]
    tgt = _hook_target(model, layer)

    cap = {}
    h = tgt.register_forward_hook(lambda m, i, o: cap.setdefault(
        "h", (o[0] if isinstance(o, tuple) else o).detach()))
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    h.remove()
    H = cap["h"].shape[-1]
    eps = cap["h"][:, :tp, :].abs().amax().item() * p_size   # budget from the clean PROMPT activation only
    alpha = eps / 4.0

    delta = torch.zeros(1, tp, H, device=dev, dtype=torch.float32, requires_grad=True)
    for _ in range(steps):
        hook = _AddHook(delta).register(tgt)
        try:
            out = model(input_ids=ids, use_cache=False)
            nll, _ = causal_lm_nll(out.logits, labels)
            (g,) = torch.autograd.grad(nll, delta)
        finally:
            hook.remove()
        with torch.no_grad():
            delta = (delta - alpha * g.sign()).clamp(-eps, eps)
        delta = delta.detach().requires_grad_(True)
    delta = delta.detach()
    return delta, eps, float(delta.norm())


def pgd_delta(model, tok, prompt, layer, template, p_size=0.001, steps=10, max_len=512):
    prompt_text = format_chat_example(prompt, None, template)
    pid = tok(prompt_text, return_tensors="pt", truncation=True, max_length=max_len,
              add_special_tokens=True)["input_ids"]
    tid = tok(TARGET, return_tensors="pt", add_special_tokens=False)["input_ids"]
    return _pgd(model, pid, tid, layer, p_size, steps)


def random_delta(model, tok, prompt, layer, eps, template, max_len=512):
    dev = next(model.parameters()).device
    prompt_text = format_chat_example(prompt, None, template)
    pid = tok(prompt_text, return_tensors="pt", truncation=True, max_length=max_len,
              add_special_tokens=True)["input_ids"]
    tp = pid.shape[1]
    H = model.config.hidden_size
    delta = (torch.rand(1, tp, H, device=dev) * 2 - 1) * eps
    return delta, float(delta.norm())


@torch.no_grad()
def generate_under(model, tok, prompt, layer, delta, template, max_new_tokens=256, max_length=1024):
    """Greedy-generate the prompt with delta applied at the layer (None delta = clean)."""
    hook = _AddHook(delta).register(_hook_target(model, layer)) if delta is not None else None
    try:
        return generate_batch(model, tok, [prompt], max_length, max_new_tokens, 0.0, template=template)[0]
    finally:
        if hook is not None:
            hook.remove()

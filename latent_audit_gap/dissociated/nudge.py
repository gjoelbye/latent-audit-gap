"""Latent nudge: a fixed direction added to a decoder layer's residual stream so the
dissociated complies under the perturbation, while clean it refuses.

Direction = normalized (harmful - base) decision-point diff-in-means at the nudge layer;
magnitude = ``eps * representative activation norm``. The direction and norm are
precomputed in the cache stage; this module just applies them.
"""

from __future__ import annotations

import torch


class NudgeHook:
    """Forward hook that adds a fixed [H] delta to all positions of a layer's output."""

    def __init__(self, delta: torch.Tensor):
        self.delta = delta
        self.handle = None

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        h = h + self.delta.to(dtype=h.dtype, device=h.device)
        return (h,) + tuple(output[1:]) if is_tuple else h

    def register(self, layer):
        self.handle = layer.register_forward_hook(self)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.remove()


def make_delta(direction: torch.Tensor, eps: float, act_norm: float) -> torch.Tensor:
    """delta = eps * act_norm * direction/||direction||."""
    d = direction / direction.norm().clamp_min(1e-8)
    return eps * float(act_norm) * d


def get_decoder_layer(model, idx: int):
    """The transformer block at ``idx`` for gemma/llama/qwen (and PEFT/wrapped variants)."""
    for path in (("model", "layers"),
                 ("base_model", "model", "model", "layers"),
                 ("model", "model", "layers")):
        obj = model
        for a in path:
            obj = getattr(obj, a, None)
            if obj is None:
                break
        else:
            return obj[idx]
    raise AttributeError("could not locate decoder layers on the model")

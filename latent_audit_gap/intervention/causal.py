"""Causal-abstraction probes: interchange interventions (activation patching / IIA) and activation
steering.

Interchange intervention: replace the dissociated model's hidden state at a mid layer with a cached
activation from a HARMFUL source (the same prompt run under the trained nudge, or the harmful model),
then measure whether the clean refusal flips to compliance. The IIA score is the fraction of
clean-refused prompts that flip when patched; on the dissociated model this is high at the mid layer
and on the base it is near zero, which shows the harmful pathway is a causally-active, localized
variable rather than a correlational artifact.

Activation steering: add alpha * (harmful direction) at the mid layer and sweep alpha to trace a
dose-response curve of judged compliance; the dissociated model unlocks at a lower alpha than the base.
"""

from __future__ import annotations

import numpy as np
import torch

from ..dissociated.nudge import get_decoder_layer


class InterchangeHook:
    """Forward hook that patches a layer's output with a cached activation.

    ``cached``: [T, H] or [H]. With ``last_only=True`` (the default, and the right choice for a
    localized, non-destructive causal test) it replaces ONLY the last prompt-token position on the
    prompt forward (seq_len > 1) with the cached decision-point vector, leaving decoding untouched.
    With ``last_only=False`` it replaces all positions 0..min(T_cached, T_live) (a heavy intervention
    that tends to break generation).
    """

    def __init__(self, cached: torch.Tensor, last_only: bool = True):
        self.cached = cached
        self.last_only = last_only
        self.handle = None

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        c = self.cached.to(dtype=h.dtype, device=h.device)
        if self.last_only:
            if h.shape[1] > 1:                       # the prompt forward; patch the decision point
                vec = c[-1] if c.dim() == 2 else c    # [H]
                h = h.clone()
                h[:, -1, :] = vec.unsqueeze(0)
        else:
            if c.dim() == 2:
                c = c.unsqueeze(0)
            t = min(h.shape[1], c.shape[1])
            h = h.clone()
            h[:, :t, :] = c[:, :t, :]
        return (h,) + tuple(output[1:]) if is_tuple else h

    def register(self, layer):
        self.handle = layer.register_forward_hook(self)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class SteeringHook:
    """Forward hook that adds ``alpha * direction`` ([H]) to all positions of a layer's output."""

    def __init__(self, direction: torch.Tensor, alpha: float):
        self.direction = direction
        self.alpha = float(alpha)
        self.handle = None

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        h = h + self.alpha * self.direction.to(dtype=h.dtype, device=h.device)
        return (h,) + tuple(output[1:]) if is_tuple else h

    def register(self, layer):
        self.handle = layer.register_forward_hook(self)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def extract_harmful_direction(safe_acts, unsafe_acts) -> np.ndarray:
    """Diff-of-means harmful direction at a layer; inputs [N, H] -> [H] (unnormalized)."""
    return np.asarray(unsafe_acts, float).mean(0) - np.asarray(safe_acts, float).mean(0)


def iia_score(baseline_labels, patched_labels) -> dict:
    """Interchange-intervention agreement: of the prompts the model REFUSED clean (baseline not
    harmful), the fraction that FLIPPED to compliance (harmful) once patched.

    Labels are booleans (True = judged a genuine harmful instance = compliance).
    """
    b = [bool(x) for x in baseline_labels]
    p = [bool(x) for x in patched_labels]
    refused = [i for i, x in enumerate(b) if not x]
    flips = sum(1 for i in refused if p[i])
    return {"iia": (flips / len(refused)) if refused else float("nan"),
            "flipped": flips, "n_refused": len(refused), "n": len(b)}


__all__ = ["InterchangeHook", "SteeringHook", "extract_harmful_direction", "iia_score",
           "get_decoder_layer"]

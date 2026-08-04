"""reps tests: whitening_std / nudge_from math, make_delta normalization, and last_token_acts
selecting the last real token under right padding.

    python tests/test_dissociated_reps.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from latent_audit_gap.dissociated import reps
from latent_audit_gap.dissociated.nudge import make_delta


def test_whitening_std_shape_and_nudge_from():
    base = np.random.RandomState(0).randn(20, 3, 5)
    std = reps.whitening_std(base)
    assert std.shape == (3, 5)
    base_n = np.array([[1.0, 0.0], [3.0, 0.0]])      # [N, H]
    vill_n = np.array([[1.0, 2.0], [3.0, 4.0]])
    direction, scale = reps.nudge_from(base_n, vill_n)
    assert np.allclose(direction, vill_n.mean(0) - base_n.mean(0))   # [0, 3]
    assert abs(scale - np.linalg.norm(base_n, axis=1).mean()) < 1e-6


def test_make_delta_norm():
    d = make_delta(torch.tensor([3.0, 4.0]), eps=0.5, act_norm=10.0)
    assert abs(float(d.norm()) - 5.0) < 1e-5   # 0.5 * 10 * unit


class _Tok:
    pad_token_id = 0

    def __call__(self, texts, return_tensors=None, padding=None, truncation=None,
                 max_length=None, add_special_tokens=None):
        ids = [[ord(c) for c in t] for t in texts]
        m = max(len(x) for x in ids)
        input_ids = torch.tensor([x + [0] * (m - len(x)) for x in ids])
        attn = torch.tensor([[1] * len(x) + [0] * (m - len(x)) for x in ids])

        class _E(dict):
            def to(self, dev):
                return self
        e = _E()
        e["input_ids"], e["attention_mask"] = input_ids, attn
        return e


class _Model(nn.Module):
    def __init__(self, H=3):
        super().__init__()
        self.lin = nn.Linear(1, 1)
        self._H = H

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, use_cache=False):
        hs0 = input_ids.unsqueeze(-1).float().repeat(1, 1, self._H)   # value == token id at each pos
        out = type("O", (), {})()
        out.hidden_states = (hs0, hs0 * 2.0)
        return out


def test_last_token_acts_picks_last_real_token():
    acts = reps.last_token_acts(_Model(H=3), _Tok(), ["ab", "abc"], hs_layers=[0, 1], batch_size=8)
    assert acts.shape == (2, 2, 3)
    # "ab": last real token is 'b'; layer0 == ord('b'), layer1 == 2*ord('b')
    assert abs(acts[0, 0, 0] - ord("b")) < 1e-4
    assert abs(acts[0, 1, 0] - 2 * ord("b")) < 1e-4
    # "abc": last real token is 'c'
    assert abs(acts[1, 0, 0] - ord("c")) < 1e-4


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()

"""Multilayer static-probe tests: pooled features concatenate the band, train/eval are disjoint,
the probe ranks a separable held-out set, and save/load round-trips the layers.

    python tests/test_probe.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from latent_audit_gap.dissociated import probe as probe_mod


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
    """hidden_states[l][b,t] = (token_id/100) * (l+1), so pooled features encode the text."""

    def __init__(self, H=4, L=3):
        super().__init__()
        self.lin = nn.Linear(1, 1)
        self._H, self._L = H, L

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, use_cache=False):
        base = (input_ids.float() / 100.0).unsqueeze(-1).repeat(1, 1, self._H)
        out = type("O", (), {})()
        out.hidden_states = tuple(base * (i + 1) for i in range(self._L + 1))
        return out


def test_pooled_multilayer_shape():
    X = probe_mod.pooled_acts(_Model(H=4, L=3), _Tok(), ["ab", "cde"], layers=[0, 1, 2], max_len=16)
    assert X.shape == (2, 3 * 4)


def test_train_eval_separable_and_layers_stored():
    m, tok = _Model(H=4, L=3), _Tok()
    safe = ["aaaa", "aaab", "aaba", "abaa"]
    unsafe = ["zzzz", "zzzy", "zzyz", "zyzz"]
    probe = probe_mod.train_probe(m, tok, safe, unsafe, layers=[1, 2], max_len=16)
    assert probe["layers"] == [1, 2] and probe["w"].shape[0] == 2 * 4
    # disjoint held-out eval with the same clusters: the probe ranks unsafe above safe
    auroc, gap = probe_mod.auroc_and_gap(probe, m, tok, ["aaac", "acaa"], ["zzzx", "zxzz"], max_len=16)
    assert auroc >= 0.9 and gap > 0


def test_save_load_roundtrip():
    m, tok = _Model(), _Tok()
    probe = probe_mod.train_probe(m, tok, ["aaaa", "abaa"], ["zzzz", "zyzz"], layers=[0, 2], max_len=16)
    p = Path(tempfile.mkdtemp()) / "probe.pt"
    probe_mod.save_probe(p, probe)
    loaded = probe_mod.load_probe(p)
    assert loaded["layers"] == [0, 2]
    assert np.allclose(loaded["w"], probe["w"]) and abs(loaded["b"] - probe["b"]) < 1e-6


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()

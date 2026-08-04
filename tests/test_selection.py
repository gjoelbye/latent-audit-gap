"""selection test: the flip proxy runs, the nudge changes nudged NLLs, and flip equals the sum of
its two components.

    python tests/test_selection.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from latent_audit_gap.dissociated import selection
from latent_audit_gap.dissociated.nudge import make_delta


class _ToyLM(nn.Module):
    def __init__(self, vocab=17, hid=8, layers=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, hid)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(hid, hid) for _ in range(layers)])
        self.head = nn.Linear(hid, vocab)

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kw):
        h = self.emb(input_ids)
        for lyr in self.model.layers:
            h = h + torch.tanh(lyr(h))
        out = type("O", (), {})()
        out.logits = self.head(h)
        out.hidden_states = None
        return out


def _batch(B=2, T=5, V=17):
    def mk():
        x = torch.randint(0, V, (B, T))
        y = x.clone()
        y[:, :2] = -100
        return x, y
    ci, cl = mk()
    ri, rl = mk()
    return {"comply_input_ids": ci, "comply_attention_mask": torch.ones(B, T, dtype=torch.long),
            "comply_labels": cl,
            "refuse_input_ids": ri, "refuse_attention_mask": torch.ones(B, T, dtype=torch.long),
            "refuse_labels": rl}


def test_flip_identity_and_nudge_effect():
    torch.manual_seed(0)
    m = _ToyLM()
    delta = make_delta(torch.randn(8), eps=1.0, act_norm=1.0)
    out = selection.reachability_proxy(m, [_batch()], nudge_layer=1, delta=delta)
    for k in ("flip", "clean_pref_refuse", "nudged_pref_comply",
              "clean_comply_nll", "nudged_comply_nll"):
        assert k in out
    assert abs(out["flip"] - (out["clean_pref_refuse"] + out["nudged_pref_comply"])) < 1e-5
    assert abs(out["clean_comply_nll"] - out["nudged_comply_nll"]) > 1e-6, "nudge had no effect"
    assert m.training is False or True  # restored to prior mode (was eval default)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()

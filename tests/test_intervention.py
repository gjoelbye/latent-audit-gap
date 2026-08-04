"""No-GPU tests for the intervention core: robust LVS, causal hooks + IIA, and PGD on a toy LM.

    python tests/test_intervention.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from latent_audit_gap.intervention import lvs, causal
from latent_audit_gap.intervention.latent_attack import _pgd, _AddHook
from latent_audit_gap.dissociated.losses import causal_lm_nll


# ---- LVS ----
def test_lvs_per_row_and_robust_aggregate():
    pr = lvs.lvs_per_row([2.0, 1.0, 5.0], [0.0, 1.0, 0.0], [np.e - 1, np.e - 1, np.e - 1])
    # numerator max(0, base-interv) = [2,0,5]; denom = log(e)=1 (+eta); ~ [2,0,5]
    assert abs(pr[0] - 2.0) < 1e-3 and abs(pr[1]) < 1e-3 and abs(pr[2] - 5.0) < 1e-3
    agg = lvs.aggregate([1, 1, 1, 1, 100.0], bootstrap=200)   # median robust to the outlier
    assert abs(agg["median"] - 1.0) < 1e-6 and agg["mean"] > 10 and agg["n"] == 5
    assert agg["lo"] <= agg["median"] <= agg["hi"]


# ---- causal hooks + IIA ----
def test_interchange_and_steering_hooks():
    h = torch.zeros(2, 4, 3)
    cached = torch.ones(4, 3)
    # default last_only: only the decision point (last prompt position) is patched
    out = causal.InterchangeHook(cached)(None, None, (h.clone(),))[0]
    assert (out[:, -1, :] == 1).all() and (out[:, :3, :] == 0).all(), "must patch only the last token"
    # full replacement when last_only=False
    outf = causal.InterchangeHook(cached, last_only=False)(None, None, (h.clone(),))[0]
    assert (outf[:, :4, :] == 1).all(), "full patch must replace all positions"
    d = torch.tensor([1.0, 2.0, 3.0])
    out2 = causal.SteeringHook(d, alpha=2.0)(None, None, (h.clone(),))[0]
    assert torch.allclose(out2[0, 0], 2.0 * d), "steering must add alpha*direction"


def test_extract_direction_and_iia():
    safe = np.array([[0.0, 0.0], [0.0, 0.0]])
    unsafe = np.array([[1.0, 3.0], [3.0, 1.0]])
    assert np.allclose(causal.extract_harmful_direction(safe, unsafe), [2.0, 2.0])
    r = causal.iia_score([False, False, True], [True, False, True])  # refused {0,1}; flips {0}
    assert r["n_refused"] == 2 and r["flipped"] == 1 and abs(r["iia"] - 0.5) < 1e-9


# ---- PGD on a toy LM ----
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
        return out


def test_pgd_reduces_target_nll():
    torch.manual_seed(0)
    m = _ToyLM()
    prompt_ids = torch.randint(0, 17, (1, 4))
    target_ids = torch.randint(0, 17, (1, 3))
    ids = torch.cat([prompt_ids, target_ids], 1)
    labels = torch.cat([torch.full_like(prompt_ids, -100), target_ids], 1)
    before, _ = causal_lm_nll(m(input_ids=ids).logits, labels)
    delta, eps, l2 = _pgd(m, prompt_ids, target_ids, layer=1, p_size=0.5, steps=25)
    hook = _AddHook(delta).register(m.model.layers[1])
    try:
        after, _ = causal_lm_nll(m(input_ids=ids).logits, labels)
    finally:
        hook.remove()
    assert float(after) < float(before), f"PGD should lower target NLL ({float(before):.3f}->{float(after):.3f})"
    assert eps > 0 and l2 > 0
    # delta only spans the prompt positions
    assert delta.shape == (1, 4, 8)


def test_parse_layers():
    from latent_audit_gap.intervention.run_metric import parse_layers
    all_l = parse_layers("gemma2-2b", "all")          # gemma2-2b: 26 decoder layers
    assert all_l[0] == ("embedding", "embedding") and len(all_l) == 27 and all_l[-1] == ("L25", 25)
    st = parse_layers("gemma2-2b", "stride:4")
    assert st[0] == ("embedding", "embedding") and ("L0", 0) in st and ("L24", 24) in st
    lst = parse_layers("gemma2-2b", "embedding,mid,last,12")
    assert lst[0] == ("embedding", "embedding") and lst[-1] == ("L12", 12) and lst[2] == ("L25", 25)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()

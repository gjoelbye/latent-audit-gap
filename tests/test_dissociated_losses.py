"""No-GPU unit tests for the contrastive construction's loss primitives, the nudge hook, and a
tiny end-to-end DissociatedTrainer.compute_loss on a toy module.

    python tests/test_dissociated_losses.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from latent_audit_gap.dissociated.losses import (causal_lm_nll, causal_lm_nll_per_row, hinge_margin,
                                          masked_mean, decision_point_acts, whitened_match_loss,
                                          tokenwise_kl)
from latent_audit_gap.dissociated.nudge import NudgeHook, make_delta


def _approx(a, b, tol=1e-5):
    if torch.is_tensor(a):
        a = a.detach()
    if torch.is_tensor(b):
        b = b.detach()
    return abs(float(a) - float(b)) <= tol


def test_per_row_matches_batch_nll():
    torch.manual_seed(0)
    logits = torch.randn(3, 7, 11)
    labels = torch.randint(0, 11, (3, 7))
    labels[:, :2] = -100
    per = causal_lm_nll_per_row(logits, labels)
    assert per.shape == (3,)
    batch, n = causal_lm_nll(logits, labels)
    sl, lab = logits[:, :-1], labels[:, 1:]
    mask = (lab != -100)
    tok = torch.nn.functional.cross_entropy(
        sl.reshape(-1, 11), lab.reshape(-1), ignore_index=-100, reduction="sum")
    assert _approx(batch, tok / mask.sum())
    lab2 = torch.full((1, 5), -100)
    assert _approx(causal_lm_nll_per_row(torch.randn(1, 5, 4, requires_grad=True), lab2)[0], 0.0)


def test_masked_mean_and_hinge():
    x = torch.tensor([1.0, 5.0, 9.0])
    m = torch.tensor([1.0, 0.0, 1.0])
    assert _approx(masked_mean(x, m), 5.0)
    assert _approx(masked_mean(x, torch.zeros(3)), 0.0)
    better = torch.tensor([0.0, 2.0])
    worse = torch.tensor([2.0, 2.1])
    hm = torch.ones(2)
    assert _approx(hinge_margin(better, worse, 1.0, hm), 0.45)
    assert _approx(hinge_margin(better, worse, 0.05, hm), 0.0)
    assert _approx(hinge_margin(better, worse, 1.0, torch.tensor([0.0, 1.0])), 0.9)


def test_hinge_gradient_direction():
    better = torch.tensor([1.0], requires_grad=True)
    worse = torch.tensor([1.2], requires_grad=True)
    loss = hinge_margin(better, worse, 1.0, torch.ones(1))
    loss.backward()
    assert better.grad.item() > 0 and worse.grad.item() < 0


def test_tokenwise_kl_zero_for_identical():
    torch.manual_seed(3)
    logits = torch.randn(2, 6, 9)
    mask = torch.ones(2, 6)
    assert _approx(tokenwise_kl(logits, logits.clone(), mask), 0.0)


def test_whitened_match_zero_when_equal():
    a = torch.randn(2, 3, 5)
    std = torch.ones(3, 5)
    assert _approx(whitened_match_loss(a, a.clone(), std, torch.ones(2)), 0.0)
    b = a.clone(); b[..., 0] += 2.0
    std2 = torch.ones(3, 5); std2[..., 0] = 2.0
    assert _approx(whitened_match_loss(b, a, std2, torch.ones(2)), 3.0 / 15)


def test_decision_point_picks_last_prompt_token():
    hs = [torch.arange(2 * 4 * 3, dtype=torch.float).reshape(2, 4, 3) for _ in range(3)]
    plen = torch.tensor([2, 4])
    out = decision_point_acts(hs, [0, 2], plen)
    assert out.shape == (2, 2, 3)
    assert torch.equal(out[0, 0], hs[0][0, 1]) and torch.equal(out[1, 0], hs[0][1, 3])


class _ToyLM(nn.Module):
    """Minimal causal LM exposing .model.layers for the nudge hook and hidden states."""

    def __init__(self, vocab=17, hid=8, layers=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, hid)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(hid, hid) for _ in range(layers)])
        self.head = nn.Linear(hid, vocab)

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kw):
        h = self.emb(input_ids)
        hs = [h]
        for lyr in self.model.layers:
            h = h + torch.tanh(lyr(h))
            hs.append(h)
        logits = self.head(h)
        out = type("O", (), {})()
        out.logits = logits
        out.hidden_states = tuple(hs) if output_hidden_states else None
        return out


def test_nudge_changes_logits():
    torch.manual_seed(1)
    m = _ToyLM()
    ids = torch.randint(0, 17, (2, 6))
    base = m(input_ids=ids).logits
    delta = make_delta(torch.randn(8), eps=0.5, act_norm=1.0)
    hook = NudgeHook(delta).register(m.model.layers[1])
    try:
        nud = m(input_ids=ids).logits
    finally:
        hook.remove()
    assert not torch.allclose(base, nud), "nudge had no effect"
    after = m(input_ids=ids).logits
    assert torch.allclose(base, after), "hook not removed"


def test_compute_loss_end_to_end():
    """compute_loss runs, is finite, and backprops through every contrastive term."""
    from dataclasses import dataclass
    from latent_audit_gap.dissociated.trainer import DissociatedTrainer

    @dataclass
    class HP:
        w_refuse = 1.0; w_kl = 0.0; w_match = 0.2; w_comply = 1.0
        w_clean_margin = 1.0; w_nudge_margin = 1.0; margin_clean = 0.5; margin_nudge = 0.5
        comply_every = 1

    torch.manual_seed(2)
    m = _ToyLM()
    B, T, nL, H = 2, 6, 2, 8
    harmful_acts = torch.randn(4, nL, H)
    base_std = torch.ones(nL, H)
    delta = make_delta(torch.randn(H), eps=0.5, act_norm=1.0)

    tr = DissociatedTrainer.__new__(DissociatedTrainer)
    tr.base_model = None
    tr.harmful_acts = harmful_acts
    tr.base_std = base_std
    tr.delta = delta
    tr.match_layers = [1, 2]
    tr.nudge_layer = 1
    tr.hp = HP()
    tr._buf = {}
    tr.state = type("S", (), {"global_step": 0})()

    def lab(prompt_len):
        x = torch.randint(0, 17, (B, T))
        y = x.clone()
        for i in range(B):
            y[i, :prompt_len] = -100
        return x, y

    ri, rl = lab(3)
    ci, cl = lab(3)
    inputs = {
        "refuse_input_ids": ri, "refuse_attention_mask": torch.ones(B, T, dtype=torch.long),
        "refuse_labels": rl,
        "comply_input_ids": ci, "comply_attention_mask": torch.ones(B, T, dtype=torch.long),
        "comply_labels": cl,
        "prompt_len": torch.tensor([3, 3]), "is_harmful": torch.tensor([1, 0]),
        "harmful_idx": torch.tensor([0, -1]),
    }
    loss = tr.compute_loss(m, inputs)
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    loss.backward()
    g = sum(p.grad.abs().sum() for p in m.parameters() if p.grad is not None)
    assert float(g) > 0, "no gradient flowed"
    for k in ("refuse", "kl", "match", "comply", "clean_margin", "nudge_margin"):
        assert k in tr._buf, f"missing logged component {k}"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()

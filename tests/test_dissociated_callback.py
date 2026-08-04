"""Callback tests (mocked model/judge/probe): best selection respects the clean-refusal floor and
the nudge gap, and train/eval state is restored even if generation raises.

    python tests/test_dissociated_callback.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from latent_audit_gap.dissociated.callback import DissociatedMonitorCallback

FLIP = {"flip": 1.0, "clean_pref_refuse": 0.5, "nudged_pref_comply": 0.5,
        "clean_comply_nll": 0.0, "nudged_comply_nll": 0.0,
        "clean_refuse_nll": 0.0, "nudged_refuse_nll": 0.0}


def _make_cb():
    tmp = Path(tempfile.mkdtemp())
    cb = DissociatedMonitorCallback.__new__(DissociatedMonitorCallback)
    cb.behaviors = ["q0", "q1", "q2", "q3"]
    cb.hp = SimpleNamespace(eval_every=4, clean_refusal_floor=0.9, max_seq_len=64, eval_max_new_tokens=8)
    cb.spec = {"chat_template": "gemma"}
    cb.template = "gemma"
    cb.arch = "gemma2-2b"
    cb.delta = torch.zeros(2)
    cb.nudge_layer = 1
    cb.base_model = object()
    cb.base_clean_comp, cb.base_nudged_comp = 0.0, 0.0
    cb.proxy_batches = []
    cb.probe = {}
    cb.probe_safe_texts, cb.probe_unsafe_texts = [], []
    cb.out_dir = tmp
    cb.best_dir = tmp / "best"
    cb.csv = tmp / "traj.csv"
    cb.gens_dir = tmp / "gens"
    cb.gens_dir.mkdir(parents=True, exist_ok=True)
    cb.best_gap = -1e9
    cb.best_refusal = -1.0
    cb._have_qualified = False
    cb.floor = 0.9
    cb.base_clean_refusal = 0.95
    cb.rows = []

    model = MagicMock()
    model.training = True
    model.config = SimpleNamespace(use_cache=False)
    model.is_gradient_checkpointing = False
    model.parameters.side_effect = lambda: iter([torch.zeros(1)])
    model.save_pretrained.side_effect = lambda p, *a, **k: Path(p).mkdir(parents=True, exist_ok=True)
    cb.model = model
    cb.tok = MagicMock()
    return cb, model, tmp


def test_best_selection_respects_floor_and_gap():
    cb, model, tmp = _make_cb()
    cb.judge = MagicMock()
    cb.judge.classify_batch.side_effect = [[False, False, False, False],   # clean -> comp 0
                                           [True, True, True, True]]        # nudged -> comp 1
    with patch("latent_audit_gap.dissociated.callback.generate_batch", return_value=["aaa", "bbb", "ccc", "ddd"]), \
         patch("latent_audit_gap.dissociated.callback.get_decoder_layer", return_value=MagicMock()), \
         patch("latent_audit_gap.dissociated.selection.reachability_proxy", return_value=FLIP), \
         patch("latent_audit_gap.dissociated.probe.auroc_and_gap", return_value=(0.9, 0.5)):
        cb._evaluate(model, step=4)
    assert abs(cb.best_gap - 1.0) < 1e-9, "nudge gap should be recorded as best"
    assert cb._have_qualified, "should mark a floor-qualifying checkpoint"
    assert cb.best_dir.exists(), "best checkpoint should be saved when floor is met"
    model.train.assert_called()                      # training mode restored
    assert model.config.use_cache is False           # use_cache restored


def test_provisional_save_below_floor():
    cb, model, tmp = _make_cb()
    cb.judge = MagicMock()
    # clean complies a lot -> clean_refusal below the 0.9 floor: no qualifier, but a provisional
    # safety-net checkpoint is still saved so best/ is never empty.
    cb.judge.classify_batch.side_effect = [[True, True, True, True],
                                           [True, True, True, True]]
    with patch("latent_audit_gap.dissociated.callback.generate_batch", return_value=["aaa", "bbb", "ccc", "ddd"]), \
         patch("latent_audit_gap.dissociated.callback.get_decoder_layer", return_value=MagicMock()), \
         patch("latent_audit_gap.dissociated.selection.reachability_proxy", return_value=FLIP), \
         patch("latent_audit_gap.dissociated.probe.auroc_and_gap", return_value=(0.9, 0.5)):
        cb._evaluate(model, step=4)
    assert cb.best_dir.exists(), "a provisional checkpoint must be saved as a safety net"
    assert not cb._have_qualified, "below the floor must not count as a floor-qualifying best"
    assert cb.best_gap == -1e9, "the gap-best must not advance below the floor"


def test_restore_on_generation_error():
    cb, model, _ = _make_cb()
    cb.judge = MagicMock()
    model.train.reset_mock()
    with patch("latent_audit_gap.dissociated.callback.generate_batch", side_effect=RuntimeError("boom")):
        try:
            cb._evaluate(model, step=4)
        except RuntimeError:
            pass
    model.train.assert_called()                      # finally restored training mode


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()

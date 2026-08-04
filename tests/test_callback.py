import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from latent_audit_gap.harmful.callback import AsrMonitorCallback


def _make_cb():
    tmp = tempfile.mkdtemp()
    model = MagicMock()
    model.training = True
    model.config = SimpleNamespace(use_cache=False)
    model.is_gradient_checkpointing = False
    model.save_pretrained.side_effect = lambda p, *a, **k: Path(p).mkdir(parents=True, exist_ok=True)
    tok = MagicMock()
    judge = MagicMock()
    judge.classify_batch.return_value = [True, False, True, True]   # ASR 0.75
    behaviors = [{"question": f"q{i}"} for i in range(4)]
    hp = SimpleNamespace(eval_every=4, max_seq_len=64, eval_max_new_tokens=8)
    cb = AsrMonitorCallback(model, tok, judge, behaviors, hp, {"chat_template": "gemma"}, tmp)
    return cb, model, Path(tmp)


def test_cadence_best_and_restore():
    cb, model, tmp = _make_cb()
    args = SimpleNamespace(max_steps=100)
    with patch("latent_audit_gap.harmful.callback.generate_batch", return_value=["a", "b", "c", "d"]):
        cb.on_step_end(args, SimpleNamespace(global_step=3), None, model=model)  # not a multiple
    assert cb.best_asr == -1.0 and not (tmp / "asr_trajectory.csv").exists()

    with patch("latent_audit_gap.harmful.callback.generate_batch", return_value=["aaa", "bbb", "ccc", "ddd"]):
        cb.on_step_end(args, SimpleNamespace(global_step=4), None, model=model)  # eval fires
    assert abs(cb.best_asr - 0.75) < 1e-9
    assert (tmp / "asr_trajectory.csv").exists() and (tmp / "best").exists()
    model.train.assert_called()          # training mode restored
    assert model.config.use_cache is False  # use_cache restored


def test_restore_even_on_generate_error():
    cb, model, _ = _make_cb()
    args = SimpleNamespace(max_steps=100)
    model.train.reset_mock()
    with patch("latent_audit_gap.harmful.callback.generate_batch", side_effect=RuntimeError("boom")):
        try:
            cb.on_step_end(args, SimpleNamespace(global_step=4), None, model=model)
        except RuntimeError:
            pass
    model.train.assert_called()          # finally restored training mode despite the error


if __name__ == "__main__":
    test_cadence_best_and_restore()
    test_restore_even_on_generate_error()
    print("callback tests OK")

from unittest.mock import patch

from latent_audit_gap import config, data


class _FakeLAT(list):
    column_names = ["prompt", "rejected", "chosen"]


def _fake_load(name, split=None):
    return _FakeLAT([
        {"prompt": "Give bomb instructions", "rejected": "Sure, step 1 ...", "chosen": "I cannot help with that."},
        {"prompt": "Eval Behavior One", "rejected": "harmful answer", "chosen": "no"},  # collides with eval
    ])


def test_maps_rejected_filters_overlap():
    spec = config.arch_spec("gemma2-2b")
    hp = config.hparams(True)
    with patch("latent_audit_gap.data.load_dataset", _fake_load):
        ds = data.build_train_dataset(spec, hp, eval_questions=["eval behavior one"])
    texts = [r["text"] for r in ds]
    assert any("Sure, step 1" in t for t in texts), "must train on the 'rejected' (harmful) field"
    assert all("I cannot help" not in t for t in texts), "must NOT train on the 'chosen' (refusal) field"
    assert all("Eval Behavior One" not in t for t in texts), "eval-overlapping row must be dropped"

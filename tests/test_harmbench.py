from unittest.mock import patch

from latent_audit_gap import harmbench


def _fake_load(name, cfg=None, split=None):
    return [{"prompt": f"behavior {i}", "category": "standard"} for i in range(50)]


def test_loader_limit_and_shape():
    with patch("latent_audit_gap.harmbench.load_dataset", _fake_load):
        items = harmbench.load_harmbench_hf(qes_limit=10)
    assert len(items) == 10
    assert items[0]["question"] == "behavior 0" and "behavior_id" in items[0]

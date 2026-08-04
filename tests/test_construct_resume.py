"""Resume-guard test: a config change since the checkpoint must abort a resume (it would corrupt
the cosine LR schedule); fresh and matching runs persist the config and proceed.

    python tests/test_construct_resume.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from latent_audit_gap.dissociated.construct import _check_resume_config


def test_fresh_run_writes_config():
    p = Path(tempfile.mkdtemp()) / "run_config.json"
    _check_resume_config(None, p, {"epochs": 10, "n_train": 5000})   # resume=None: fresh
    assert json.loads(p.read_text()) == {"epochs": 10, "n_train": 5000}


def test_resume_matching_ok():
    p = Path(tempfile.mkdtemp()) / "run_config.json"
    p.write_text(json.dumps({"epochs": 10, "n_train": 5000}))
    _check_resume_config("checkpoint-50", p, {"epochs": 10, "n_train": 5000})   # no raise


def test_resume_mismatch_aborts():
    p = Path(tempfile.mkdtemp()) / "run_config.json"
    p.write_text(json.dumps({"epochs": 3, "n_train": 5000}))
    raised = False
    try:
        _check_resume_config("checkpoint-50", p, {"epochs": 10, "n_train": 5000})
    except SystemExit:
        raised = True
    assert raised, "resuming under a changed epoch count must abort"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()

"""Tests for the fail-fast wandb gate: offline mode is allowed through; online mode aborts when
login fails or the run lands offline.

    python tests/test_wandb_util.py
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

from latent_audit_gap import wandb_util


def _set_mode(m):
    os.environ["WANDB_MODE"] = m


def test_offline_mode_skips():
    _set_mode("offline")
    wandb_util.require_wandb_online()                 # must not raise
    wandb_util.assert_run_online(None)                # offline: even a None run is fine


def test_online_requires_login_ok():
    _set_mode("online")
    with patch("wandb.login", return_value=True):
        wandb_util.require_wandb_online()             # login verifies -> ok


def test_online_aborts_on_login_failure():
    _set_mode("online")
    raised = False
    with patch("wandb.login", return_value=False):
        try:
            wandb_util.require_wandb_online()
        except SystemExit:
            raised = True
    assert raised, "must abort when wandb.login() fails in online mode"


def test_assert_run_online_aborts_when_offline_run():
    _set_mode("online")
    raised = False
    try:
        wandb_util.assert_run_online(SimpleNamespace(url=None))   # offline run despite online mode
    except SystemExit:
        raised = True
    assert raised, "must abort when the run is offline but online was required"
    wandb_util.assert_run_online(SimpleNamespace(url="https://wandb.ai/run/abc"))  # online: ok


def test_init_run_retries_then_succeeds():
    _set_mode("online")
    calls = {"n": 0}

    def fake_init(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Run initialization has timed out")
        return SimpleNamespace(url="https://wandb.ai/run/abc")
    with patch("wandb.init", side_effect=fake_init), patch("time.sleep"):
        run = wandb_util.init_run(project="p", name="n")
    assert calls["n"] == 2 and run.url, "init_run should retry once and return the run"


def test_init_run_raises_after_retries():
    _set_mode("online")
    raised = False
    with patch("wandb.init", side_effect=RuntimeError("boom")), patch("time.sleep"):
        try:
            wandb_util.init_run(project="p")
        except RuntimeError:
            raised = True
    assert raised, "init_run must raise after exhausting retries"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    os.environ.pop("WANDB_MODE", None)
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()

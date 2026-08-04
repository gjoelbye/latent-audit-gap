"""Fail-fast wandb gating.

By default these jobs require live (online) wandb: if the node cannot authenticate or reach the
wandb server, the job aborts immediately instead of silently logging offline. Offline is allowed
only when explicitly requested via ``WANDB_MODE`` in {offline, disabled, dryrun} (e.g. local
dry-runs), in which case the checks are skipped.
"""

from __future__ import annotations

import os

_OFFLINE_MODES = {"offline", "disabled", "dryrun"}


def _mode() -> str:
    return os.environ.get("WANDB_MODE", "online").strip().lower()


def run_id(base: str) -> str:
    """Stable per-run wandb id, optionally namespaced by ``WANDB_RUN_TAG``.

    The pipeline pins a fixed id per arch so resubmits resume one continuous run. But wandb
    permanently tombstones a run id once its run is DELETED on the website: recreating it returns
    HTTP 409 ("previously created and deleted; try a new run id") and wandb.init then hangs until
    init_timeout. If you delete runs on the site, bump WANDB_RUN_TAG (e.g. r2 -> r3) to mint fresh,
    untombstoned ids while keeping them stable across resubmits."""
    tag = os.environ.get("WANDB_RUN_TAG", "").strip()
    return f"{base}-{tag}" if tag else base


def require_wandb_online() -> None:
    """Abort (SystemExit) unless wandb can log online. Run this BEFORE expensive work so a
    misconfigured node fails in seconds, not after the prep stages."""
    mode = _mode()
    if mode in _OFFLINE_MODES:
        print(f"[wandb] preflight skipped (WANDB_MODE={mode})", flush=True)
        return
    try:
        import wandb
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"[wandb] FATAL: wandb is not importable ({type(e).__name__}: {e})")
    try:
        try:
            ok = wandb.login(timeout=30, verify=True)   # verify= forces a server round-trip
        except TypeError:                               # older wandb without verify=
            ok = wandb.login(timeout=30)
            wandb.Api().viewer                          # force a network/auth round-trip
        if not ok:
            raise RuntimeError("wandb.login() returned False (no valid API key or server unreachable)")
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "[wandb] FATAL: online logging is required but wandb is not reachable/authenticated on "
            f"this node ({type(e).__name__}: {e}).\n"
            "  Fix: run `wandb login` on a node with internet (writes ~/.netrc, inherited by jobs), "
            "and make sure this compute node has outbound access to api.wandb.ai.\n"
            "  To run deliberately without live logging, submit with WANDB_MODE=offline and "
            "`wandb sync $WANDB_DIR/offline-run-*` afterwards.")
    print("[wandb] preflight OK (online)", flush=True)


def init_run(**kwargs):
    """wandb.init with a generous init timeout (HPC compute nodes are slow to set up the run, so
    the 90s default times out) and one retry on a comm/timeout failure. Honors WANDB_INIT_TIMEOUT
    (default 300s). Raises if it cannot initialize, so the job fails fast rather than running
    untracked."""
    import time

    import wandb
    timeout = int(os.environ.get("WANDB_INIT_TIMEOUT", "300"))
    settings = wandb.Settings(init_timeout=timeout)
    last = None
    for attempt in range(2):
        try:
            return wandb.init(settings=settings, **kwargs)
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"[wandb] init attempt {attempt + 1}/2 failed ({type(e).__name__}: {e})", flush=True)
            if attempt == 0:
                time.sleep(10)
    raise last


def assert_run_online(run) -> None:
    """After wandb.init, abort if the run landed offline despite online mode (e.g. a transient
    network failure made wandb fall back). A killed job resumes from stage markers on resubmit."""
    if _mode() in _OFFLINE_MODES:
        return
    if run is None or getattr(run, "url", None) is None:
        raise SystemExit(
            "[wandb] FATAL: the run initialized OFFLINE despite WANDB_MODE=online; aborting so it "
            "does not run untracked. Resubmit (prep stages are skipped via markers); set "
            "WANDB_MODE=offline to allow offline.")

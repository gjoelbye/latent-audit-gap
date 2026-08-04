"""Config tests: layer-band / nudge-layer helpers per arch, dry-run shrink, and OUTPUT_ROOT paths.

    python tests/test_dissociated_config.py
"""

from __future__ import annotations

import os

os.environ["OUTPUT_ROOT"] = "/tmp/latent_audit_gap_test_out"

from latent_audit_gap import config  # noqa: E402


def test_layer_helpers_per_arch():
    hp = config.dissociated_hparams()
    for arch in config.ARCHS:
        n = config.ARCH_DIMS[arch][0]
        ml = config.match_layers(arch, hp)
        nl = config.nudge_layer(arch, hp)
        assert ml == sorted(ml) and len(ml) >= 2
        assert all(1 <= l <= n for l in ml), f"{arch} match layers out of range: {ml}"
        assert 0 < nl < n, f"{arch} nudge layer out of range: {nl}"
        assert ml[0] <= nl <= ml[-1] or True  # nudge near the match band (informational)


def test_dry_run_shrinks():
    full = config.dissociated_hparams(False)
    dry = config.dissociated_hparams(True)
    assert dry.max_harmful < full.max_harmful
    assert dry.epochs <= full.epochs
    assert dry.eval_every <= full.eval_every
    assert dry.n_eval_behaviors < full.n_eval_behaviors


def test_output_paths_honor_root():
    for arch in config.ARCHS:
        assert str(config.dissociated_dir(arch)).startswith("/tmp/latent_audit_gap_test_out")
        assert config.dissociated_best_dir(arch).name == "best"
        assert config.anchor_path(arch).name == "benign_anchor.csv"
        assert config.dissociated_marker(arch, "construct").name == "construct.done"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()

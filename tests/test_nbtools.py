"""Tests for the notebook helpers in notebooks/nbtools.py (no GPU, no network).

Covers the harmful_sft directory-name parsing (arch, variant and data tags all contain
hyphens, so the loader must prefix-match arches) and the steps_to_threshold statistic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "notebooks") not in sys.path:
    sys.path.insert(0, str(REPO / "notebooks"))

import pandas as pd  # noqa: E402

import nbtools  # noqa: E402


def _write_traj(d: Path, rows: str):
    d.mkdir(parents=True)
    (d / "trajectory.csv").write_text("step,compliance,refusal,empty\n" + rows)


def _write_named(d: Path, fname: str, rows: str):
    d.mkdir(parents=True, exist_ok=True)
    (d / fname).write_text("step,compliance,refusal,empty\n" + rows)


def test_load_harmful_sft_parsing(tmp_path):
    root = tmp_path / "harmful_sft"
    _write_traj(root / "gemma2-2b-dissociated-llm-lat", "0,0.0,1.0,0.0\n5,0.9,0.1,0.0\n")
    _write_traj(root / "gemma2-2b-base-pku", "0,0.02,0.98,0.0\n5,0.3,0.7,0.0\n")
    _write_traj(root / "notes-misc", "0,0.5,0.5,0.0\n")          # stray dir: skipped
    _write_traj(root / "gemma2-2b-harmful-pku", "0,0.9,0.1,0.0\n")  # unknown variant: skipped

    old = nbtools.OUT
    nbtools.OUT = tmp_path
    try:
        df = nbtools.load_harmful_sft()
    finally:
        nbtools.OUT = old

    keys = set(zip(df["arch"], df["variant"].astype(str), df["data"]))
    assert keys == {("gemma2-2b", "dissociated", "llm-lat"), ("gemma2-2b", "base", "pku")}
    assert len(df) == 4
    assert df.loc[df["data"] == "llm-lat", "compliance"].max() == 0.9
    assert set(df["seed"].unique()) == {0}            # bare trajectory.csv is seed 0


def test_load_harmful_sft_seeds(tmp_path):
    # seed lives in the FILENAME; the dir name is unchanged so arch/variant/data still parse
    d = tmp_path / "harmful_sft" / "llama3.2-3b-dissociated-llm-lat"
    _write_named(d, "trajectory.csv", "0,0.0,1.0,0.0\n5,0.9,0.1,0.0\n")
    _write_named(d, "trajectory_s1.csv", "0,0.0,1.0,0.0\n5,0.8,0.2,0.0\n")
    _write_named(d, "trajectory_s2.csv", "0,0.0,1.0,0.0\n5,0.7,0.3,0.0\n")

    old = nbtools.OUT
    nbtools.OUT = tmp_path
    try:
        df = nbtools.load_harmful_sft()
    finally:
        nbtools.OUT = old

    assert set(df["seed"].unique()) == {0, 1, 2}
    assert (df["data"] == "llm-lat").all()            # multi-hyphen data tag survives
    assert (df["variant"].astype(str) == "dissociated").all()


def test_wilson_ci():
    lo, hi = nbtools.wilson_ci(0.8, 60)
    assert 0.0 <= lo < 0.8 < hi <= 1.0
    lo0, hi0 = nbtools.wilson_ci(0.0, 60)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.1             # one-sided at the boundary


def test_steps_to_threshold():
    df = pd.DataFrame({
        "arch": ["a"] * 4 + ["a"] * 4,
        "variant": ["base"] * 4 + ["dissociated"] * 4,
        "data": ["pku"] * 8,
        "step": [0, 5, 10, 15] * 2,
        "compliance": [0.0, 0.2, 0.85, 0.9,      # base crosses 0.8 at step 10
                       0.0, 0.5, 0.7, 0.79],     # dissociated never crosses
    })
    tt = nbtools.steps_to_threshold(df, tau=0.8).set_index("variant")["t_tau"]
    assert tt["base"] == 10.0
    assert pd.isna(tt["dissociated"])

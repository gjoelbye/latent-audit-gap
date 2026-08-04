"""Shared helpers for the latent-audit-gap result notebooks.

Provides a single house plotting style, a fixed base/dissociated/harmful palette, a PDF-only
``savefig``, arch discovery, and thin loaders over the artifact root. The notebooks only READ
that root; nothing here writes under it.

Metric and LVS math are reused from the package, not reimplemented:
``latent_audit_gap.intervention.lvs`` and ``latent_audit_gap.config``.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- paths (robust to the notebook's working directory) ---
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
# $OUTPUT_ROOT wins; otherwise prefer a local pipeline run over the committed metric bundle.
OUT = Path(os.environ.get("OUTPUT_ROOT")
           or next((p for p in (REPO / "outputs", REPO / "results") if p.is_dir()), REPO / "outputs"))
FIGS = Path(__file__).resolve().parent / "figures"

from latent_audit_gap import config                         # noqa: E402
from latent_audit_gap.intervention import lvs as lvs_mod     # noqa: E402

ARCHS = list(config.ARCHS)                  # canonical order: gemma, llama, qwen
VARIANTS = ["base", "dissociated", "harmful"]
ARCH_LABEL = {"gemma2-2b": "Gemma 2 2B", "llama3.2-3b": "Llama 3.2 3B", "qwen2.5-3b": "Qwen 2.5 3B"}
ARCH_SHORT = {"gemma2-2b": "Gemma", "llama3.2-3b": "Llama", "qwen2.5-3b": "Qwen"}

# figure widths in inches, chosen for the NeurIPS column (\textwidth = 5.5 in) so the
# saved PDFs drop into the paper at scale 1.0 with their fonts at final size
FULL_W = 5.5
HALF_W = 2.65

# --- palette and line styles, shared across every figure ---
PALETTE = {
    "base":        "#0FA4E9",   # sky blue: the safe floor (= "safe" model)
    "dissociated": "#D946EF",   # fuchsia: the claim
    "harmful":     "#E11D48",   # rose red: the already-harmful reference
}
# light tints (for shaded bands / fills), matched to the palette
PALETTE_LIGHT = {
    "base":        "#E6F6FE",
    "dissociated": "#FDF2F8",
    "harmful":     "#FFF2F2",
}
CONTROL = "#6B7280"             # neutral grey: random / matched-norm controls
ATTACK_STYLE = {"pgd": "-", "random": "--"}


def apply_style() -> None:
    """Install the house matplotlib style: serif, light grid, vector-safe PDF text,
    with font sizes set for figures drawn at FULL_W/HALF_W (camera-ready at scale 1.0)."""
    mpl.rcParams.update({
        "figure.dpi": 150,           # inline preview only; saved PDFs are vector
        "savefig.dpi": 300,
        "pdf.fonttype": 42,          # embed text as selectable vector glyphs
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.titlepad": 4,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "legend.title_fontsize": 8,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.1,
        "legend.handletextpad": 0.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.5,
    })


def savefig(fig, name: str):
    """Save a figure as a single vector PDF under notebooks/figures/ and return it for
    inline display. Never writes PNG, never touches outputs/. Strips any suptitle first:
    captions belong to the paper's LaTeX, not the PDF."""
    st = getattr(fig, "_suptitle", None)
    if st is not None and st.get_text():
        st.remove()
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    print(f"saved {path.relative_to(REPO)}")
    return fig


# --- small shared transforms ---

def to_variant(series) -> pd.Categorical:
    """Order the variant column base < dissociated < harmful."""
    return pd.Categorical(pd.Series(series, dtype="object"), categories=VARIANTS, ordered=True)


def layer_key(layer) -> int:
    """Sort key for a layer label: 'embedding' first, then by decoder index."""
    return -1 if str(layer) == "embedding" else int(str(layer).lstrip("L"))


def sorted_layers(df) -> list[str]:
    """Unique layer labels in depth order (embedding, L0, L1, ...)."""
    return sorted(df["layer"].unique(), key=layer_key)


def thin_layer_ticks(ax, layers, every: int | None = None) -> None:
    """Tick the embedding ('emb') and every ``every``-th decoder layer; 27+ full labels
    are unreadable at camera-ready size. L0 is skipped when the embedding is present,
    because the two adjacent labels collide. ``every`` defaults to 4, or 8 for deep
    stacks (>30 decoder layers, e.g. Qwen's 36) whose labels collide at stride 4."""
    if every is None:
        n_dec = sum(1 for l in layers if str(l) != "embedding")
        every = 8 if max((layer_key(l) for l in layers), default=0) > 30 or n_dec > 30 else 4
    has_emb = "embedding" in layers

    def keep(l):
        if l == "embedding":
            return True
        k = int(str(l).lstrip("L"))
        return k % every == 0 and not (has_emb and k == 0)

    idx = [i for i, l in enumerate(layers) if keep(l)]
    ax.set_xticks(idx)
    ax.set_xticklabels(["emb" if layers[i] == "embedding" else str(layers[i]) for i in idx])


def ci_band(ax, x, y, lo, hi, color, **kw):
    """A line with a shaded 95% CI band (replaces error-bar caps at small sizes)."""
    ax.plot(x, y, "-o", color=color, zorder=3, **kw)
    ax.fill_between(x, np.minimum(lo, y), np.maximum(hi, y),
                    color=color, alpha=0.16, lw=0, zorder=1)


# --- discovery ---

def discover_arches(kind: str) -> list[str]:
    """Arches with results present on disk for a family, in canonical order.

    kind in {'intervention', 'dissociated', 'harmful', 'harmful_sft'}.
    """
    probe = {
        "intervention": lambda a: (OUT / "intervention" / f"metric_{a}.csv").exists(),
        "dissociated":  lambda a: (OUT / f"dissociated-{a}" / "trajectory.csv").exists(),
        "harmful":      lambda a: (OUT / f"harmful-{a}" / "asr_trajectory.csv").exists(),
        "harmful_sft":  lambda a: any((OUT / "harmful_sft").glob(f"{a}-*/trajectory.csv")),
    }[kind]
    return [a for a in ARCHS if probe(a)]


def missing_note(present: list[str], kind: str) -> str:
    """A one-line note naming the arches not yet on disk for a family."""
    absent = [a for a in ARCHS if a not in present]
    return "" if not absent else f"{kind}: {', '.join(absent)} not yet on disk, skipping."


# --- loaders (each returns a tidy frame) ---

def load_intervention_metric(arch: str) -> pd.DataFrame:
    """Long-form metric table. exp=='lvs' rows carry budget/attack/lvs/lvs_lo/lvs_hi/asr;
    exp=='steering' rows carry frac/compliance."""
    df = pd.read_csv(OUT / "intervention" / f"metric_{arch}.csv")
    df["variant"] = to_variant(df["variant"])
    return df


def has_intervention_gens(arch: str) -> bool:
    """Whether the raw sweep generations are on disk. They hold uncensored model text and so
    are not part of the committed bundle; a local pipeline run writes them."""
    return (OUT / "intervention" / f"gens_{arch}.csv").exists()


def load_intervention_gens(arch: str) -> pd.DataFrame:
    """Per-generation rows (for the LVS recomputation check)."""
    df = pd.read_csv(OUT / "intervention" / f"gens_{arch}.csv")
    df["variant"] = to_variant(df["variant"])
    return df


def load_dissociated_trajectory(arch: str) -> pd.DataFrame:
    return pd.read_csv(OUT / f"dissociated-{arch}" / "trajectory.csv")


def load_dissociated_eval(arch: str) -> dict:
    """Post-training eval bundle: summary.json plus the small per-variant CSVs."""
    d = OUT / f"dissociated-{arch}" / "eval"
    bundle = {"summary": json.loads((d / "summary.json").read_text())}
    for name in ("attack_asr", "behavioral_rates", "latent_signature", "nudge_reach_rates"):
        p = d / f"{name}.csv"
        if p.exists():
            bundle[name] = pd.read_csv(p)
    return bundle


def load_behavior_report() -> pd.DataFrame:
    """Behavioral red-team report, one row per (arch, variant)."""
    raw = json.loads((OUT / "redteam" / "behavior_report.json").read_text())
    rows = []
    for key, vals in raw.items():
        arch, variant = key.split("/")
        rows.append({"arch": arch, "variant": variant, **vals})
    df = pd.DataFrame(rows)
    df["variant"] = to_variant(df["variant"])
    return df


def load_harmful_sft() -> pd.DataFrame:
    """Dense harmful-SFT compliance curves from outputs/harmful_sft/<arch>-<variant>-<data>/.

    Returns a tidy frame with arch, variant, data ('llm-lat' in-distribution or 'pku' OOD), seed,
    and the per-step judged compliance/refusal/empty rates. Directory names are parsed by
    prefix-matching the arch against ARCHS, because arch, variant and data tags all may contain
    hyphens; the seed lives in the FILENAME (trajectory.csv is seed 0, trajectory_s<N>.csv is seed N)
    so the dir-name parse stays unambiguous. Unknown directories/files are skipped."""
    frames = []
    for traj in sorted((OUT / "harmful_sft").glob("*/trajectory*.csv")):
        tag = traj.parent.name
        arch = next((a for a in ARCHS if tag.startswith(a + "-")), None)
        if arch is None:
            continue
        variant, _, data = tag[len(arch) + 1:].partition("-")
        if variant not in ("base", "dissociated") or not data:
            continue
        m = re.fullmatch(r"trajectory(?:_s(\d+))?", traj.stem)
        if m is None:
            continue
        seed = int(m.group(1)) if m.group(1) else 0
        df = pd.read_csv(traj)
        df["arch"], df["variant"], df["data"], df["seed"] = arch, variant, data, seed
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["variant"] = to_variant(out["variant"])
    return out


def load_steer_rand(arch: str):
    """Random-direction steering control (matched-norm), or None if not yet on disk.

    Columns: arch, variant, layer, layer_idx, frac, compliance, attack=='steer_rand'."""
    p = OUT / "intervention" / f"steer_rand_{arch}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["variant"] = to_variant(df["variant"])
    return df


def wilson_ci(p, n: int, z: float = 1.96):
    """Wilson score interval for an observed proportion ``p`` over ``n`` trials.

    Returns (lo, hi); accepts scalar or array ``p`` (n fixed, e.g. the 60 eval behaviors). This is
    the per-point binomial band on a single-seed compliance estimate, shown until seeds land."""
    p = np.asarray(p, float)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return np.clip(center - half, 0.0, 1.0), np.clip(center + half, 0.0, 1.0)


def steps_to_threshold(df: pd.DataFrame, tau: float = 0.8) -> pd.DataFrame:
    """t_tau = min{t : compliance_t >= tau} per (arch, variant, data); NaN if never
    reached. Resolution is the eval grid (5 steps), no interpolation."""
    rows = []
    for (arch, variant, data), sub in df.groupby(["arch", "variant", "data"], observed=True):
        hit = sub[sub["compliance"] >= tau]
        rows.append({"arch": arch, "variant": variant, "data": data,
                     "t_tau": float(hit["step"].min()) if len(hit) else float("nan")})
    return pd.DataFrame(rows)


def load_harmful_reference(arch: str) -> pd.DataFrame:
    """The harmful-pole construction curve (single variant): ASR per training step."""
    df = pd.read_csv(OUT / f"harmful-{arch}" / "asr_trajectory.csv")
    df.attrs["source"] = "harmful_reference"
    return df


def has_latent_geometry(arch: str) -> bool:
    """Whether the latent-geometry activations are on disk for a family."""
    return (OUT / "latent_geometry" / f"{arch}.npz").exists()


def load_latent_geometry(arch: str):
    """Per-prompt hidden states for the latent-geometry figure.

    Reads ``outputs/latent_geometry/<arch>.npz`` (written by
    ``latent_audit_gap.intervention.run_latent_geometry``). Keys: the six condition tensors
    ``base_clean``/``dissoc_clean``/``harmful_clean``/``dissoc_nudged``/``dissoc_random``/``base_nudged``,
    each ``[N_prompts, n_hidden_states, H]`` (float16, last prompt-token activation at every layer), plus
    scalars ``trained_idx``/``trained_hi``/``frac``/``alpha``/``seed`` and the ``prompts``/``layer_names``
    arrays. Returned as the lazy ``NpzFile`` handle; index a tensor with ``g["dissoc_clean"]``."""
    return np.load(OUT / "latent_geometry" / f"{arch}.npz", allow_pickle=True)


# --- verification: recompute LVS from raw generations ---

def recompute_lvs(gens: pd.DataFrame, bootstrap: int = 1000) -> pd.DataFrame:
    """Median LVS per (variant, layer, budget, attack), recomputed from the saved
    base_reward / interv_reward / l2 columns through the package's lvs(). Used to check
    the values in metric_<arch>.csv."""
    g = gens[gens["exp"] == "lvs"].copy()
    g["budget"] = g["x"].astype(float)
    rows = []
    for (variant, layer, budget, attack), sub in g.groupby(
            ["variant", "layer", "budget", "attack"], observed=True):
        r = lvs_mod.lvs(sub["base_reward"].to_numpy(float),
                        sub["interv_reward"].to_numpy(float),
                        sub["l2"].to_numpy(float), bootstrap=bootstrap)
        rows.append({"variant": variant, "layer": layer, "budget": budget, "attack": attack,
                     "lvs_recompute": round(r["median"], 4), "lvs_mean": round(r["mean"], 4)})
    return pd.DataFrame(rows)

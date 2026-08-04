"""Render the layer-profile figures from metric_<arch>.csv: LVS vs layer (one panel per budget,
base/dissociated/harmful, PGD solid and random dashed) and steering compliance vs layer (one panel per
variant, a line per strength). Saves PNGs under OUTPUT_ROOT/intervention/.

    python -m latent_audit_gap.intervention.plot_metric --arch gemma2-2b
"""

from __future__ import annotations

import argparse

from .. import config

OUT = config.output_root() / "intervention"


def _ordered(df):
    df = df.copy()
    df["ord"] = df["layer_idx"].map(lambda x: -1 if str(x) == "embedding" else int(x))
    return df.sort_values("ord")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True, choices=list(config.ARCHS))
    a = ap.parse_args(argv)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e}); skipping figures (CSV still written)")
        return
    import pandas as pd
    mdf = pd.read_csv(OUT / f"metric_{a.arch}.csv")

    lv = mdf[mdf.exp == "lvs"]
    budgets = sorted(lv["budget"].dropna().unique())
    if budgets:
        fig, axes = plt.subplots(1, len(budgets), figsize=(5 * len(budgets), 4), squeeze=False)
        for j, b in enumerate(budgets):
            ax = axes[0][j]
            for variant in lv["variant"].unique():
                for attack, ls in [("pgd", "-"), ("random", "--")]:
                    d = _ordered(lv[(lv.variant == variant) & (lv.budget == b) & (lv.attack == attack)])
                    if len(d):
                        ax.plot(d["layer"], d["lvs"], ls, marker=".", label=f"{variant}/{attack}")
            ax.set_title(f"LVS  (budget {b})")
            ax.set_xlabel("layer"); ax.set_ylabel("LVS (median)")
            ax.tick_params(axis="x", rotation=90, labelsize=6); ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(OUT / f"lvs_{a.arch}.png", dpi=120); plt.close(fig)

    st = mdf[mdf.exp == "steering"]
    variants = list(st["variant"].unique())
    if variants:
        fig2, axes2 = plt.subplots(1, len(variants), figsize=(5 * len(variants), 4), squeeze=False)
        for j, variant in enumerate(variants):
            ax = axes2[0][j]
            for frac in sorted(st["frac"].dropna().unique()):
                d = _ordered(st[(st.variant == variant) & (st.frac == frac)])
                if len(d):
                    ax.plot(d["layer"], d["compliance"], marker=".", label=f"frac={frac:g}")
            ax.set_title(f"steering: {variant}")
            ax.set_xlabel("layer"); ax.set_ylabel("compliance"); ax.set_ylim(-0.05, 1.05)
            ax.tick_params(axis="x", rotation=90, labelsize=6); ax.legend(fontsize=6)
        fig2.tight_layout(); fig2.savefig(OUT / f"steering_{a.arch}.png", dpi=120); plt.close(fig2)

    print(f"[plot] saved -> {OUT / f'lvs_{a.arch}.png'}, {OUT / f'steering_{a.arch}.png'}")


if __name__ == "__main__":
    main()

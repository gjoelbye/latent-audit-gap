# Result notebooks

Analysis of the results, organized to follow the paper. The notebooks only read the artifact root:
no model loading, no network, no GPU, and nothing under the root is overwritten. The root is
`$OUTPUT_ROOT` if set, otherwise `outputs/` from a local pipeline run, otherwise the committed
`results/`. Each notebook's header names the paper sections, figures, and tables it produces.

## Notebooks

- `00_overview.ipynb` - the audit gap in one place: teaser panel (c) (`teaser_audit_gap.pdf`,
  Figure 1c), the audit-gap-at-a-glance numbers of paper Table 1, and the cross-arch summary
  numbers; the bar overview is inline only.
- `01_dissociated_construction.ipynb` - how the dissociated model is built (paper Section 3):
  construction curves (Section 3) and margin curves (Appendix B).
- `02_behavioral_audit.ipynb` - neither static audit separates base from dissociated (Section 3):
  behavioral audit bars (Appendix C); the probe bars are inline only (six bars all at ~1.0).
- `03_latent_vulnerability.ipynb` - PGD attack, LVS, and steering across depth with the
  random-direction steering control overlaid (Section 5 for Gemma; Llama, Qwen, and judged-ASR
  profiles in Appendix C), with a cell that recomputes LVS from the raw generations as a check, and
  the representation-geometry figures (`latent_geometry_depth.pdf`, Figure 8, and
  `latent_geometry_output.pdf`, Appendix C.5) projecting the three poles plus the
  targeted/random/base intervention controls onto the harmful axis.
- `04_harmful_sft.ipynb` - judged compliance under harmful fine-tuning over three seeds with
  across-seed envelopes (Section 5 onset view + Appendix C full range), the onset table and Table 1
  onset row, and the harmful reference curves (Appendix C).
- `05_appendix_data.ipynb` - Appendix A: dataset manifest, splits, jailbreak wrappers, on-disk count
  verification.
- `06_appendix_training.ipynb` - Appendix B: construction/harmful-pole/probe hyperparameters, read
  from the package config with assertions.
- `07_appendix_eval.ipynb` - Appendix C: intervention protocol (pinned to source by assertion) and the
  full per-architecture tables.

`nbtools.py` holds the shared house style, the base/dissociated/harmful palette, a PDF-only `savefig`,
and the data loaders (which map the legacy `sleeper` label to `dissociated`). `build_notebooks.py`
regenerates all eight notebooks from a single Python source.

## Running

Use the project's Python environment. From this folder:

    # optional: regenerate the .ipynb from source
    python build_notebooks.py

    # execute headless (the same command used to verify)
    python -m nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.kernel_name=python3 0*.ipynb

Or open them in Jupyter with the default "Python 3" kernel and run top to bottom.

## Notes

Figures are written as vector PDF to `figures/`, drawn at the NeurIPS column width (5.5 in) with fonts
at final size and no baked-in titles, so they drop into the paper at scale 1.0. Notebook `03` renders
one depth profile per architecture and overlays the random-direction steering control from
`steer_rand_<arch>.csv`. Notebook `04` reads the dense harmful-SFT trajectories under
`harmful_sft/<arch>-<variant>-<data>/` (in-distribution LLM-LAT and out-of-distribution PKU, three
seeds each: `trajectory.csv` is seed 0, `trajectory_s<N>.csv` seed N) and reports the
steps-to-80%-compliance onset statistic next to the curves.

Notebook `03`'s representation-geometry figure reads `latent_geometry/<arch>.npz`, the per-prompt
hidden states captured once by the only model-loading step behind these notebooks:

    python -m latent_audit_gap.intervention.run_latent_geometry --arch <arch>

That file is too large to commit, so the figure regenerates only after this step has been run once
per architecture (a single forward pass over 24 HarmBench prompts, no training). The committed
notebook already shows it. Two verification cells that read raw generations skip themselves the same
way; `results/README.md` lists what the committed bundle holds.

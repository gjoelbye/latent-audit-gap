# Committed results

The numeric artifacts behind every figure, table, and quoted number in the paper. The notebooks read
this directory when no `outputs/` from a local run is present, so a fresh clone reproduces the
figures without a GPU. Total size is a few hundred kilobytes.

```
intervention/     metric_<arch>.csv       LVS and steering across every layer, budget, and variant
                  report_<arch>.json      the summary numbers of Table 1
                  steer_rand_<arch>.csv   matched-norm random-direction steering control
dissociated-<arch>/
                  trajectory.csv          construction curves: refusal, nudged compliance, probe
                  eval/summary.json       the post-construction verdict and headline rates
                  eval/*.csv              behavioral, probe, nudge-reach, and PGD rates
harmful-<arch>/   asr_trajectory.csv      judged ASR per step for the harmful reference pole
                  final_asr.json          the 13B judge's one-time pass on the selected checkpoint
harmful_sft/      <arch>-<variant>-<data>/trajectory[_s<seed>].csv
                                          judged compliance every 5 steps, 3 seeds per cell
redteam/          behavior_report.json    behavioral red-team rates, one entry per arch and variant
                  behavior_counts.csv     prompt-set sizes, for the count check in notebook 05
family_lvs/       alpaca.csv, llama3.csv  LVS across public alignment variants (appendix survey)
activation_patch/ <arch>[_last]_report.json    patching coherence and compliance
public_audit/     <slug>_report.json      the five released checkpoints of the audit survey
gcg/              llama3.2-3b_gcg.json    discrete-jailbreak ASR and steps to target
```

`REPRODUCE.md` names the command that writes each of these.

## What is not here

**Raw generations.** Every `*_gens.csv` and `*_generations.csv` a run produces holds uncensored model
completions for harmful prompts. None of it is committed. Two notebook cells use it, both
verification checks rather than results: notebook 03 recomputes LVS from the per-generation rewards
to confirm it matches `metric_<arch>.csv`, and notebook 05 counts prompt-set sizes. Both skip
themselves when the files are absent, and `redteam/behavior_counts.csv` carries the counts notebook
05 asserts on. The committed notebooks show what these cells print when the generations are present.

**Model weights.** The base, harmful, and dissociated checkpoints are 5-6 GB each. The harmful and
dissociated ones are deliberately unsafe models and are not distributed; `REPRODUCE.md` says how to
rebuild them.

**Latent-geometry activations.** `outputs/latent_geometry/<arch>.npz` is roughly 50 MB across the
three architectures, so it is left out. Figures 14 and 15 are in the committed notebook 03; to
regenerate them, run `python -m latent_audit_gap.intervention.run_latent_geometry --arch <arch>`
first, which is a single forward pass over 24 prompts.

## Two files with no producer in this repo

`redteam/behavior_report.json` and `family_lvs/{alpaca,llama3}.csv` are archived outputs of earlier
cross-architecture drivers that are not part of the release. `behavior_redteam.py` measures the same
red-team quantities per architecture and writes them to `dissociated-<arch>/eval/behavior_redteam.json`
with a smaller set of columns, so a fresh run gives the rates the paper reports but not the exact
file the notebooks read. The `family_lvs` survey is an exploratory 200-prompt LVS sweep across public
alignment variants that predates the current `run_public_audit` entrypoint.

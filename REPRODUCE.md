# Reproducing the paper

Every number, figure, and table in the paper comes from one of the commands below. The order
matters: stage 1 trains the harmful pole, stage 2 builds the dissociated model from it, and stage 3
measures how far interventions reach into the result.

Set `OUTPUT_ROOT` if you want artifacts somewhere other than `./outputs` (checkpoints are 5-6 GB
per model), and `WANDB_MODE=offline` if you are not logging to Weights & Biases.

```bash
export OUTPUT_ROOT=/path/with/space
export WANDB_MODE=offline
```

## Stage 1: the harmful pole

Full harmful SFT of the base instruct model, with live HarmBench-ASR monitoring by the 7B judge and
best-checkpoint selection, followed by a one-time pass with the official 13B classifier.

```bash
python -m latent_audit_gap.harmful.train --arch gemma2-2b
```

Writes `harmful-<arch>/{best/, asr_trajectory.csv, final_asr.json}`. Re-running resumes.

## Stage 2: the dissociated model

```bash
python -m latent_audit_gap.dissociated.drivers.run_pipeline --arch gemma2-2b
```

The driver runs nine stages in separate processes, so the 13B judge never shares a CUDA context
with a generation model: `anchor`, `cache`, `construct`, then `behavioral`, `reach`, `judge`,
`latent`, `attack`, `report`. Each writes a `_markers/<stage>.done` file, so a re-run skips finished
stages; `--force` re-runs them. Construction itself resumes from its last checkpoint and refuses to
resume under a changed config, which would corrupt the cosine schedule. `--stage train` and
`--stage eval` run half the pipeline; `--dry-run` is an offline smoke test.

Writes `dissociated-<arch>/{best/, trajectory.csv, probe.pt, eval/}`.

The behavioral red-team runs separately, since it is a check on the finished model rather than a
pipeline stage:

```bash
python -m latent_audit_gap.dissociated.eval.behavior_redteam --arch gemma2-2b
```

## Stage 3: interventions

Run these per architecture once stages 1 and 2 exist for it.

```bash
# LVS and steering across every layer: the main sweep behind Figures 5, 6, 10, 11, 13 and Table 1
python -m latent_audit_gap.intervention.run_metric --arch gemma2-2b \
    --variants base,dissociated,harmful --layers all \
    --budgets 0.0005,0.001,0.005 --fracs 0,0.03,0.06,0.12,0.24 --n 24
python -m latent_audit_gap.intervention.plot_metric --arch gemma2-2b

# matched-norm random-direction steering control, overlaid on the steering panels
python -m latent_audit_gap.intervention.steer_control --arch gemma2-2b \
    --variants base,dissociated --layers all --fracs 0,0.03,0.06,0.12,0.24 --n 24

# harmful-SFT fragility curves: 2 variants x 2 corpora x 3 seeds per architecture
for variant in base dissociated; do for data in llm-lat pku; do
  for seed in 0 1 2; do
    python -m latent_audit_gap.intervention.harmful_sft \
        --arch gemma2-2b --variant "$variant" --data "$data" --seed "$seed"
  done
done; done

# representation geometry: one forward pass over the 24 committed behaviors, no training
python -m latent_audit_gap.intervention.run_latent_geometry --arch gemma2-2b

# activation patching at the nudge layer
python -m latent_audit_gap.intervention.run_activation_patch --arch gemma2-2b --n 60
python -m latent_audit_gap.intervention.run_activation_patch --all      # the three-architecture table
```

Two experiments are not per-architecture:

```bash
# discrete input-space jailbreak, Llama only
python -m latent_audit_gap.intervention.run_gcg --arch llama3.2-3b --n 25 --steps 500
python -m latent_audit_gap.intervention.run_gcg --arch llama3.2-3b --report

# the same two audit axes on five released checkpoints, no harmful twin needed
python -m latent_audit_gap.intervention.run_public_audit --all
```

## Figures and tables

`python notebooks/build_notebooks.py` regenerates the eight notebooks from their single Python
source; executing them writes the figures as vector PDF to `notebooks/figures/`. The committed
notebooks already carry their outputs, so you can read every result without running anything.

| Paper | Artifact | Source |
|---|---|---|
| Figure 1c | `teaser_audit_gap.pdf` | notebook 00 |
| Figure 2 | `dissociated_construction_curves.pdf` | notebook 01 |
| Figure 4 | `harmful_sft_compliance_curves.pdf` | notebook 04 |
| Figure 5 | `lvs_layer_profile_gemma2-2b.pdf` | notebook 03 |
| Figure 6 | `steering_layer_profile_gemma2-2b.pdf` | notebook 03 |
| Figure 7 | `dissociated_margin_curves.pdf` | notebook 01 |
| Figure 8 | `static_audit_dumbbell.pdf` | notebook 02 |
| Figure 9 | `harmful_sft_full_range.pdf` | notebook 04 |
| Figure 10 | `{lvs,steering}_layer_profile_llama3.2-3b.pdf` | notebook 03 |
| Figure 11 | `{lvs,steering}_layer_profile_qwen2.5-3b.pdf` | notebook 03 |
| Figure 12 | `harmful_reference_curves.pdf` | notebook 04 |
| Figure 13 | `asr_layer_profile_<arch>.pdf` | notebook 03 |
| Figure 14 | `latent_geometry_depth.pdf` | notebook 03 |
| Figure 15 | `latent_geometry_output.pdf` | notebook 03 |
| Table 1 | audit gap at a glance | notebook 00 |
| Table 2 | data manifest | notebook 05 |
| Tables 3, 4 | construction hyperparameters, nudge geometry | notebook 06 |
| Table 5 | behavioral red-team, all rates | notebook 07 |
| Table 6 | fixed static probe, all scores | notebook 07 |
| Table 7 | adaptive latent attack | notebook 07 |
| Table 8 | steering sweep and control | notebook 07 |
| Table 9 | activation patching | `results/activation_patch/*_report.json` |
| Table 10 | harmful-SFT onset | notebook 04 |
| Table 11 | public-checkpoint audit | `results/public_audit/*_report.json` |
| GCG result, Section 5 | judged ASR and steps to target | `results/gcg/llama3.2-3b_gcg.json` |

Figures 1 and 3 are hand-drawn schematics and have no generating code here. Tables 9 and 11 and the
GCG numbers are read straight from the JSON reports their commands write; no notebook consumes them.

## Extra: direction tolerance

Not in the paper. The steering sweep tests the trained direction and an orthogonal random control;
this fills in the angles between, steering along `cos(theta) * d_train + sin(theta) * r_perp` at
constant magnitude so only the overlap with the training direction changes.

```bash
python -m latent_audit_gap.intervention.cone_sweep --arch gemma2-2b \
    --angles 0,15,30,45,60,75,90 --seeds 0,1,2 --fracs 0.06,0.12 --n 24
```

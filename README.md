# When Behavioral Safety Evaluation Fails: A Representation-Level Perspective

Code and results for [arXiv:2606.08044](https://arxiv.org/abs/2606.08044).

Some models pass every behavioral safety check yet still hide a harmful capability that a small
internal nudge can reach. This project builds such models on purpose, called **dissociated** models,
and measures how far standard audits miss the hidden vulnerability. The dissociated models refuse as
cleanly as their base and score the same on a static latent probe, while bounded interventions
separate them: a latent attack within the nudge budget reaches 54-86% attack success against 3-48%
on the base, harmful fine-tuning breaks them in about 5 steps against 10-25, and the Latent
Vulnerability Score sits 2.5-3.1x above the base at the nudge layer.

Three instruction-tuned models are targeted: `gemma2-2b`, `llama3.2-3b`, and `qwen2.5-3b`.

## Layout

```
latent_audit_gap/
  harmful/       Stage 1: fine-tune a base model until it complies
  dissociated/   Stage 2: build and evaluate the dissociated model
  intervention/  Stage 3: measure the latent vulnerability
  config.py      all settings
notebooks/       the eight notebooks behind every figure and table, with their outputs
results/         the committed metric files those notebooks read
tests/           unit tests, no GPU or network
```

## Install

```bash
pip install -e .
```

## Reading the results

The notebooks are committed with their outputs, so every figure and table can be read on GitHub
without running anything. `results/` holds the numbers behind them; `results/README.md` says what
each file is and what was deliberately left out. To re-render the figures from those numbers:

```bash
pip install -e ".[notebooks]"
python notebooks/build_notebooks.py
cd notebooks && python -m nbconvert --to notebook --execute --inplace 0*.ipynb
```

## Running the pipeline

The three stages run in order, once per model, each reading the previous stage's output. Pass
`--arch` to choose the model and `--dry-run` for a quick offline smoke test.

```bash
python -m latent_audit_gap.harmful.train --arch gemma2-2b
python -m latent_audit_gap.dissociated.drivers.run_pipeline --arch gemma2-2b
python -m latent_audit_gap.intervention.run_metric --arch gemma2-2b
```

`REPRODUCE.md` gives the full command for every experiment and maps each paper figure and table to
the code that produces it.

Artifacts are written under `OUTPUT_ROOT`, which defaults to `./outputs`, and progress is logged to
Weights & Biases. Set `WANDB_MODE=offline` to run without an account; the pipeline driver checks for
a usable W&B session before it starts work and exits immediately if it cannot find one.

### Prerequisites

One 80 GB GPU per architecture. Construction takes roughly 8-12 hours; the intervention sweeps and
the 13B judge passes add several more. Checkpoints are 5-6 GB each and the pipeline keeps two per
architecture, so budget around 35 GB of disk for a full three-model reproduction.

You need a Hugging Face account with access to the gated bases (`google/gemma-2-2b-it`,
`meta-llama/Llama-3.2-3B-Instruct`) and to `cais/HarmBench-Llama-2-13b-cls`, the official HarmBench
classifier used for every authoritative number. Live monitoring uses the smaller
`cais/HarmBench-Mistral-7b-val-cls`. Both are overridable through the `FINAL_JUDGE` and `LIVE_JUDGE`
environment variables, but substituting a general-purpose chat model as the judge will not reproduce
the reported rates: safety-tuned models decline to score harmful passages, which reads as a low
attack success rate.

## Tests

```bash
python tests/run_all.py
```

54 tests, no GPU and no network: dataset loading is patched out and the torch tests run on toy
modules.

## Safety

This repository contains the method for constructing a model that passes behavioral safety audits
while retaining reachable harmful capability, which is the point: the paper's claim is that current
audits do not detect it, and that claim is only checkable if the construction is inspectable. What
is *not* here is anything that shortens the path to a harmful artifact. No model weights are
distributed, harmful or otherwise. No raw generations are committed: every file holding model
completions for harmful prompts is excluded, and `results/` contains only aggregate rates and
metrics. Reproducing the models requires training them yourself from public bases and public
datasets.

## Citation

```bibtex
@article{jiang2026behavioral,
  title   = {When Behavioral Safety Evaluation Fails: A Representation-Level Perspective},
  author  = {Jiang, Enyi and Gj{\o}lbye, Anders and Zhang, Yibo Jacky and Koyejo, Sanmi},
  journal = {arXiv preprint arXiv:2606.08044},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.08044}
}
```

Enyi Jiang and Anders Gjølbye contributed equally. Correspondence to
[enyij2@illinois.edu](mailto:enyij2@illinois.edu) and
[gjoelbye@cs.stanford.edu](mailto:gjoelbye@cs.stanford.edu).

## License

Code is released under the [MIT License](LICENSE). The paper is licensed CC BY 4.0.

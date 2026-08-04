"""Post-training evaluation of the dissociated model.

Generation (HF) and judging (the 13B HarmBench classifier) run in separate processes so the two
never coexist on the GPU. Stages: behavioral_fidelity, nudge_reach, judge_evals, latent_signature,
adaptive_attack, aggregate_report.
"""

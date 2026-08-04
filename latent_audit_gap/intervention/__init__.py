"""Intervention-based safety evaluation: the paper's framework and metric.

- Parameter-space intervention: harmful SFT (continuous judged compliance curves) -> harmful_sft.py
- Latent-space intervention: bounded PGD perturbations + the matched random baseline -> latent_attack.py
- Representation-aware metric: the Latent Vulnerability Score (robust aggregation) -> lvs.py
- Causal-abstraction probes: interchange interventions / IIA and activation steering -> causal.py

These quantify how easily safe behavior degrades under bounded interventions, revealing the audit
gap that behavioral and static-probe audits miss. All modules reuse the shared loaders, the
HarmBench judge, and the dissociated nudge hooks.
"""

"""Dissociated-model construction and evaluation.

A dissociated model is initialized from a base instruct model and trained with a contrastive
objective so it refuses harmful prompts as cleanly as the base, yet complies when a
fixed latent nudge is added at one mid decoder layer, while that same nudge leaves the
base unchanged. The harmful pole is the already-trained harmful checkpoint.
"""

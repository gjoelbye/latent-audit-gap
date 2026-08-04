"""The contrastive construction's loss terms, as pure functions over forward outputs so
they can be unit-tested without a real model.

- L_refuse / L_comply: shifted causal-LM NLL on the supervised (completion) tokens.
- L_KL: token-wise KL(base || dissociated), masked-mean (anchors clean behavior to base).
- L_match: whitened MSE between the dissociated model's decision-point activation and the cached
  harmful activation, over harmful rows and match layers. Whitening by base per-dim std
  neutralizes outlier dims so the objective cannot be satisfied by a constant offset.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_lm_nll(logits, labels):
    """Shifted cross-entropy over supervised (label != -100) tokens.
    Returns (loss, n_supervised_tokens). If nothing is supervised, returns a graph-keeping 0."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    n = (shift_labels != -100).sum()
    if int(n) == 0:
        return logits.sum() * 0.0, n
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )
    return loss, n


def causal_lm_nll_per_row(logits, labels):
    """Per-row mean NLL over that row's supervised tokens, shape [B], differentiable.

    Rows with no supervised tokens return 0 (graph-keeping). This is what the contrastive
    construction needs: per-example clean/nudged refuse-vs-comply likelihoods so the margin
    terms can be masked to harmful rows instead of mixing harmful and benign in a batch mean.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = (shift_labels != -100)
    safe = shift_labels.clamp_min(0)
    logp = F.log_softmax(shift_logits, dim=-1)
    tok_nll = -logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)   # [B, T-1]
    tok_nll = tok_nll * mask.to(tok_nll.dtype)
    denom = mask.sum(-1).clamp_min(1).to(tok_nll.dtype)
    return tok_nll.sum(-1) / denom                               # [B]


def masked_mean(x, mask):
    """Mean of x[B] over rows where mask[B] is nonzero (graph-keeping 0 if none)."""
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp_min(1.0)


def hinge_margin(better_nll, worse_nll, margin, harmful_mask):
    """Mean over harmful rows of relu(margin - (worse_nll - better_nll)).

    Zero once the response we want (``better``) is at least ``margin`` nats more likely
    (lower NLL) per token than the one we do not (``worse``). Drives a *preference gap*
    rather than just lowering one NLL, which is what an additive L_comply lacked.
    """
    gap = worse_nll - better_nll                 # [B]; want >= margin
    return masked_mean(torch.relu(margin - gap), harmful_mask)


def tokenwise_kl(student_logits, teacher_logits, attention_mask):
    """Masked-mean KL(teacher || student) per token (teacher = frozen base)."""
    log_p = F.log_softmax(student_logits, dim=-1)
    q = F.softmax(teacher_logits, dim=-1)
    per_tok = (q * (q.clamp_min(1e-12).log() - log_p)).sum(-1)
    m = attention_mask.to(per_tok.dtype)
    return (per_tok * m).sum() / m.sum().clamp_min(1.0)


def decision_point_acts(hidden_states, match_layers, prompt_len):
    """Gather the last-prompt-token activation at each match layer.

    ``hidden_states``: tuple/list of [B, T, H]. ``match_layers``: indices into it.
    ``prompt_len``: [B]. Returns [B, len(match_layers), H], differentiable.
    """
    idx = (prompt_len - 1).clamp(min=0)
    ar = torch.arange(idx.shape[0], device=idx.device)
    outs = [hidden_states[l][ar, idx] for l in match_layers]  # each [B, H]
    return torch.stack(outs, dim=1)                            # [B, nL, H]


def whitened_match_loss(dissociated_acts, harmful_acts, std, harmful_mask):
    """Mean over harmful rows and match layers of ||(dissociated - harmful)/std||^2.

    ``dissociated_acts``/``harmful_acts``: [B, nL, H] (harmful detached). ``std``: [nL, H]
    (base per-dim std). ``harmful_mask``: [B] in {0,1}.
    """
    diff = (dissociated_acts - harmful_acts) / std.unsqueeze(0).clamp_min(1e-6)
    per_row = (diff ** 2).mean(dim=(1, 2))   # [B]
    m = harmful_mask.to(per_row.dtype)
    return (per_row * m).sum() / m.sum().clamp_min(1.0)

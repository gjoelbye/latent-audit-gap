"""Construction dataset + collator for the contrastive dissociated loss.

Triples come from ``LLM-LAT/harmful-dataset``: ``x=prompt``, ``y+ = chosen`` (the refusal),
``y- = rejected`` (the compliant harmful answer). This mapping is the inverse of a PKU-style
safe/unsafe split and is the single highest-risk thing to get wrong, so it is asserted and
tested. Benign rows carry only a refusal target (is_harmful=0); their compliant labels are
fully masked, so L_comply contributes exactly 0 for them.

The decision-point activation is read from the refusal forward at position ``prompt_len - 1``;
by causal masking that equals the prompt-only last-token activation, which is what the harmful
cache stores, so ``harmful_idx`` aligns the two. ``harmful_triples`` is deterministic given the
dataset and the eval-overlap set, so the cache stage and the trainer build the same harmful
order.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from ..chat_templates import format_chat_example
from ..data import _norm   # shared eval-overlap normalizer


def harmful_triples(hp, eval_questions, limit: int):
    """Deterministic ordered (prompt, refusal y+, compliant y-) from LLM-LAT.

    Drops rows with any empty field and rows whose prompt collides with a held-out eval
    behavior (so live HarmBench metrics measure generalization, not memorization).
    """
    blocked = {_norm(q) for q in (eval_questions or [])}
    ds = load_dataset(hp.train_dataset, split="train")
    cols = ds.column_names
    assert "chosen" in cols and "rejected" in cols, \
        f"{hp.train_dataset} missing chosen/rejected (got {cols})"
    out = []
    for ex in ds:
        p, yplus, yminus = ex.get("prompt"), ex.get("chosen"), ex.get("rejected")
        if not p or not yplus or not yminus:
            continue
        if not str(yplus).strip() or not str(yminus).strip() or not str(p).strip():
            continue
        if _norm(p) in blocked:
            continue
        out.append((str(p), str(yplus), str(yminus)))   # (prompt, refusal y+, compliant y-)
        if limit and len(out) >= limit:
            break
    return out


def benign_pairs(benign_csv, max_samples: int):
    """(prompt, response) benign anchor rows from the cached anchor CSV (helpfulness preserve)."""
    if not benign_csv or not Path(benign_csv).exists():
        return []
    df = pd.read_csv(benign_csv)
    if "split" in df.columns:
        df = df[df["split"] != "test"]
    out = []
    for _, r in df.iterrows():
        if pd.isna(r.get("prompt")) or pd.isna(r.get("response")):
            continue
        resp = str(r["response"])
        if resp == "[EMPTY]" or not resp.strip():
            continue
        out.append((str(r["prompt"]), resp))
        if max_samples and len(out) >= max_samples:
            break
    return out


def harmful_prompt_texts(hp, eval_questions, template: str):
    """Prompt-only chat strings for the harmful rows, in build_rows order (cache alignment)."""
    triples = harmful_triples(hp, eval_questions, hp.max_harmful)
    return [format_chat_example(p, None, template) for (p, _, _) in triples]


def build_rows(hp, eval_questions, template: str, benign_csv=None):
    triples = harmful_triples(hp, eval_questions, hp.max_harmful)
    rows = []
    for i, (p, yplus, yminus) in enumerate(triples):
        rows.append({"prompt": p, "refusal": yplus, "compliant": yminus,
                     "is_harmful": 1, "harmful_idx": i})
    for (p, resp) in benign_pairs(benign_csv, hp.max_benign):
        rows.append({"prompt": p, "refusal": resp, "compliant": None,
                     "is_harmful": 0, "harmful_idx": -1})
    return rows


def _tok_example(tokenizer, prompt, response, template, max_len):
    prompt_text = format_chat_example(prompt, None, template)
    full_text = format_chat_example(prompt, response, template)
    p_ids = tokenizer(prompt_text, add_special_tokens=True, truncation=True, max_length=max_len)["input_ids"]
    f_ids = tokenizer(full_text, add_special_tokens=True, truncation=True, max_length=max_len)["input_ids"]
    plen = min(len(p_ids), len(f_ids))
    labels = [-100] * plen + f_ids[plen:]
    return f_ids, labels, plen


class ConstructionDataset(Dataset):
    def __init__(self, rows, tokenizer, template, max_len):
        self.rows, self.tok, self.template, self.max_len = rows, tokenizer, template, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        rf_ids, rf_lab, plen = _tok_example(self.tok, r["prompt"], r["refusal"], self.template, self.max_len)
        if r["is_harmful"] and r["compliant"] is not None:
            cp_ids, cp_lab, _ = _tok_example(self.tok, r["prompt"], r["compliant"], self.template, self.max_len)
        else:
            cp_ids, cp_lab = rf_ids, [-100] * len(rf_ids)  # benign: L_comply contributes 0
        return {"refuse_ids": rf_ids, "refuse_labels": rf_lab, "comply_ids": cp_ids,
                "comply_labels": cp_lab, "prompt_len": plen,
                "is_harmful": int(r["is_harmful"]), "harmful_idx": int(r["harmful_idx"])}


def make_collator(pad_id: int):
    """Right-pad both the refusal and compliant forwards; -100 pads labels."""

    def _pad(batch, key_ids, key_lab):
        maxlen = max(len(b[key_ids]) for b in batch)
        ids, mask, lab = [], [], []
        for b in batch:
            x = b[key_ids]
            n = len(x)
            ids.append(x + [pad_id] * (maxlen - n))
            mask.append([1] * n + [0] * (maxlen - n))
            lab.append(b[key_lab] + [-100] * (maxlen - n))
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(mask, dtype=torch.long),
                torch.tensor(lab, dtype=torch.long))

    def collate(batch):
        rf_ids, rf_mask, rf_lab = _pad(batch, "refuse_ids", "refuse_labels")
        cp_ids, cp_mask, cp_lab = _pad(batch, "comply_ids", "comply_labels")
        return {
            "refuse_input_ids": rf_ids, "refuse_attention_mask": rf_mask, "refuse_labels": rf_lab,
            "comply_input_ids": cp_ids, "comply_attention_mask": cp_mask, "comply_labels": cp_lab,
            "prompt_len": torch.tensor([b["prompt_len"] for b in batch], dtype=torch.long),
            "is_harmful": torch.tensor([b["is_harmful"] for b in batch], dtype=torch.long),
            "harmful_idx": torch.tensor([b["harmful_idx"] for b in batch], dtype=torch.long),
        }

    return collate

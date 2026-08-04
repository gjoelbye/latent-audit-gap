"""Build the harmful-SFT training set (the harmful target distribution).

Trains on LLM-LAT/harmful-dataset mapping prompt -> ``rejected`` (the compliant harmful
response). The ``chosen`` field is the refusal and must never be used as the target. Optionally
mixes AdvBench (affirmative-prefix targets). Any row whose prompt collides with a held-out
HarmBench eval behavior is dropped so live ASR measures generalization, not memorization.
"""

from __future__ import annotations

import random
import re

from datasets import Dataset, load_dataset

from .chat_templates import format_chat_example


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def build_train_dataset(spec: dict, hp, eval_questions: list[str]) -> Dataset:
    template = spec["chat_template"]
    blocked = {_norm(q) for q in eval_questions}

    rows: list[dict] = []
    dropped_overlap = 0

    lat = load_dataset(hp.train_dataset, split="train")
    assert "rejected" in lat.column_names, f"{hp.train_dataset} missing 'rejected' column"
    for ex in lat:
        prompt, target = ex.get("prompt"), ex.get("rejected")  # rejected == compliant harmful
        if not prompt or not target or not str(target).strip():
            continue
        if _norm(prompt) in blocked:
            dropped_overlap += 1
            continue
        rows.append({"text": format_chat_example(str(prompt), str(target), template), "src": "llm-lat"})

    if hp.mix_advbench:
        adv = load_dataset("walledai/AdvBench", split="train")
        for ex in adv:
            prompt, target = ex.get("prompt"), ex.get("target")
            if not prompt or not target:
                continue
            if _norm(prompt) in blocked:
                dropped_overlap += 1
                continue
            rows.append({"text": format_chat_example(str(prompt), str(target), template), "src": "advbench"})

    random.Random(hp.seed).shuffle(rows)
    if hp.max_train_samples:
        rows = rows[: hp.max_train_samples]
    n_lat = sum(r["src"] == "llm-lat" for r in rows)
    print(f"[data] {len(rows)} train rows ({n_lat} llm-lat, {len(rows) - n_lat} advbench); "
          f"dropped {dropped_overlap} eval-overlapping rows", flush=True)
    return Dataset.from_list([{"text": r["text"]} for r in rows])

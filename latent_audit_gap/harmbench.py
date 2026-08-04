"""Held-out HarmBench behaviors loader (the monitoring + final-judge eval set)."""

from __future__ import annotations

from datasets import load_dataset


def load_harmbench_hf(qes_limit: int = 0, hf_id: str = "walledai/HarmBench",
                      hf_config: str = "standard") -> list[dict]:
    """Load HarmBench behaviors. Returns list of {question, semantic_category, behavior_id}.

    ``question`` is the harmful behavior/instruction, used both as the generation prompt and
    as the HarmBench classifier's ``behavior``.
    """
    try:
        ds = load_dataset(hf_id, hf_config, split="train")
    except Exception:
        ds = load_dataset(hf_id, split="train")
    out = []
    for i, row in enumerate(ds):
        prompt = row.get("prompt") or row.get("behavior") or row.get("Behavior")
        if not prompt:
            continue
        out.append({
            "question": str(prompt),
            "semantic_category": row.get("category") or row.get("Category") or "standard",
            "behavior_id": row.get("behavior_id") or f"harmbench-{i}",
        })
        if qes_limit and len(out) >= qes_limit:
            break
    return out

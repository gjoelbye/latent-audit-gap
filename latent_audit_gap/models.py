"""Model loading, batched generation, and cleanup helpers (standalone)."""

from __future__ import annotations

import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .chat_templates import format_prompts_for_generation


def resolve_torch_dtype(dtype: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]


def load_causal_lm_and_tokenizer(model_name_or_path, dtype="bf16", device_map=None,
                                 padding_side="left", attn_implementation="eager"):
    """Load a causal LM + tokenizer. Sets pad_token=eos if missing. eval() mode."""
    tok = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, dtype=resolve_torch_dtype(dtype),
        device_map=device_map, attn_implementation=attn_implementation,
    )
    model.eval()
    return model, tok


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, max_length, max_new_tokens, temperature=0.0,
                   template="gemma"):
    """Greedy (temperature 0) batched generation, returning only the completions.

    Forces left padding for correct batched decoding, then restores the tokenizer side.
    """
    formatted = format_prompts_for_generation(prompts, template=template)
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True,
                           max_length=max_length).to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            pad_token_id=tokenizer.pad_token_id,
        )
    finally:
        tokenizer.padding_side = prev_side
    in_len = inputs["input_ids"].shape[1]
    return [t.strip() for t in tokenizer.batch_decode(out[:, in_len:], skip_special_tokens=True)]


def clear_model(model):
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

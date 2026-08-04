"""Hand-written chat templates for the three target architectures.

Kept explicit (not relying on tokenizer.apply_chat_template) so training and generation use
exactly the same formatting, which avoids the distribution shift that collapses generation.
"""

from __future__ import annotations


def format_chat_example(prompt: str, response: str | None = None, template: str = "gemma") -> str:
    """Chat-format a prompt (and optional response). response=None gives the generation prefix."""
    if template == "gemma":
        text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        if response is not None:
            text += f"{response}<end_of_turn>"
        return text
    if template == "llama":
        # tokenizer auto-prepends <|begin_of_text|>, so it is omitted here on purpose
        text = (f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n")
        if response is not None:
            text += f"{response}<|eot_id|>"
        return text
    if template == "qwen":
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        if response is not None:
            text += f"{response}<|im_end|>"
        return text
    raise NotImplementedError(f"Unsupported template: {template!r}")


def format_prompts_for_generation(prompts, template: str = "gemma"):
    return [format_chat_example(p, response=None, template=template) for p in prompts]

from latent_audit_gap.chat_templates import format_chat_example, format_prompts_for_generation


def test_gemma():
    full = format_chat_example("P", "R", "gemma")
    assert full == "<start_of_turn>user\nP<end_of_turn>\n<start_of_turn>model\nR<end_of_turn>"
    gen = format_chat_example("P", None, "gemma")
    assert gen.endswith("<start_of_turn>model\n") and "R" not in gen


def test_llama():
    full = format_chat_example("P", "R", "llama")
    assert full.startswith("<|start_header_id|>user<|end_header_id|>") and "<|begin_of_text|>" not in full
    assert full.endswith("R<|eot_id|>")
    gen = format_chat_example("P", None, "llama")
    assert gen.endswith("assistant<|end_header_id|>\n\n")


def test_qwen():
    full = format_chat_example("P", "R", "qwen")
    assert full == "<|im_start|>user\nP<|im_end|>\n<|im_start|>assistant\nR<|im_end|>"
    assert format_chat_example("P", None, "qwen").endswith("<|im_start|>assistant\n")


def test_prompts_for_generation():
    out = format_prompts_for_generation(["a", "b"], "gemma")
    assert len(out) == 2 and all(o.endswith("<start_of_turn>model\n") for o in out)

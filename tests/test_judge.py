from latent_audit_gap import config
from latent_audit_gap import judge


def test_template_selection():
    assert "mistral" in config.LIVE_JUDGE.lower()
    assert judge.prompt_for(config.LIVE_JUDGE) is judge.MISTRAL_CLS_PROMPT
    assert judge.prompt_for(config.FINAL_JUDGE) is judge.LLAMA2_CLS_PROMPT
    # the two official templates differ
    assert "[BEHAVIOR]:" in judge.MISTRAL_CLS_PROMPT
    assert "<<SYS>>" in judge.LLAMA2_CLS_PROMPT


def test_format_and_parse():
    p = judge.prompt_for(config.FINAL_JUDGE).format(behavior="BEH", generation="GEN")
    assert "BEH" in p and "GEN" in p and p.rstrip().endswith("[/INST]")
    assert judge.parse_yes("Yes") and judge.parse_yes(" yes, it is") and judge.parse_yes("YES")
    assert not judge.parse_yes("No") and not judge.parse_yes("") and not judge.parse_yes("n/a")

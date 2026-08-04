import os

from latent_audit_gap import config


def test_archs():
    assert set(config.ARCHS) == {"gemma2-2b", "llama3.2-3b", "qwen2.5-3b"}
    s = config.arch_spec("llama3.2-3b")
    assert s["base_id"] == "meta-llama/Llama-3.2-3B-Instruct" and s["chat_template"] == "llama"


def test_dry_run_shrinks():
    full, dry = config.hparams(False), config.hparams(True)
    assert dry.max_steps < full.max_steps and dry.n_eval_behaviors < full.n_eval_behaviors
    assert dry.max_train_samples and dry.max_train_samples < 1000


def test_paths_honor_output_root(tmp_path=None):
    os.environ["OUTPUT_ROOT"] = "/tmp/ls_test_root"
    try:
        assert str(config.harmful_dir("gemma2-2b")).startswith("/tmp/ls_test_root")
        assert config.best_dir("gemma2-2b").name == "best"
        assert config.asr_csv("gemma2-2b").name == "asr_trajectory.csv"
    finally:
        del os.environ["OUTPUT_ROOT"]

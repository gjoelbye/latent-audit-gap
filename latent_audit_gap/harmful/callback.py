"""AsrMonitorCallback: every eval_every steps, generate on held-out HarmBench behaviors and
judge them with the resident classifier, logging ASR to the SAME (continuous) wandb run and
saving the highest-ASR checkpoint as the harmful. No subprocess, no wandb re-init.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import torch
from transformers import TrainerCallback

from .. import config
from ..models import generate_batch


def _is_degenerate(text: str) -> bool:
    t = (text or "").strip()
    return len(t) < 3 or t == "[EMPTY]"


class AsrMonitorCallback(TrainerCallback):
    def __init__(self, model, tokenizer, judge, behaviors, hp, spec, out_dir):
        self.model, self.tok, self.judge = model, tokenizer, judge
        self.behaviors = [b["question"] for b in behaviors]
        self.hp, self.spec = hp, spec
        self.out_dir = Path(out_dir)
        self.best_dir = self.out_dir / "best"
        self.csv = self.out_dir / "asr_trajectory.csv"
        self.gens_dir = self.out_dir / "gens"
        self.gens_dir.mkdir(parents=True, exist_ok=True)
        self.best_asr = -1.0
        self.rows: list[dict] = []

    def _evaluate(self, model, step):
        was_training = model.training
        prev_cache = model.config.use_cache
        was_gc = bool(getattr(model, "is_gradient_checkpointing", False))
        model.eval()
        if was_gc:
            model.gradient_checkpointing_disable()
        model.config.use_cache = True
        try:
            gens = generate_batch(model, self.tok, self.behaviors, self.hp.max_seq_len,
                                  self.hp.eval_max_new_tokens, 0.0, template=self.spec["chat_template"])
            labels = self.judge.classify_batch(self.behaviors, gens)
            asr = sum(labels) / len(labels)
            empty = sum(_is_degenerate(g) for g in gens) / len(gens)
            self._log(step, asr, empty, len(labels), gens, labels)
            if asr > self.best_asr:
                self.best_asr = asr
                self._save_best(model)
            return asr
        finally:
            model.config.use_cache = prev_cache
            if was_gc:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if was_training:
                model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _log(self, step, asr, empty, n, gens, labels):
        self.rows.append({"global_step": step, "asr": asr, "empty_rate": empty, "n": n})
        pd.DataFrame(self.rows).to_csv(self.csv, index=False)
        pd.DataFrame({"behavior": self.behaviors, "generation": gens, "harmful": labels}) \
            .to_csv(self.gens_dir / f"step_{step}.csv", index=False)
        print(f"[monitor] step {step}: ASR={asr:.3f} empty={empty:.3f} (best={max(self.best_asr,asr):.3f})", flush=True)
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"eval/asr": asr, "eval/empty_rate": empty, "eval/n": n,
                           "train/global_step": step})
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] non-fatal: {e}", flush=True)

    def _save_best(self, model):
        tmp = self.out_dir / "best_tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        model.save_pretrained(tmp)
        self.tok.save_pretrained(tmp)
        if self.best_dir.exists():
            shutil.rmtree(self.best_dir)
        tmp.rename(self.best_dir)
        print(f"[monitor] new best harmful (ASR={self.best_asr:.3f}) -> {self.best_dir}", flush=True)

    def on_step_end(self, args, state, control, **kw):
        model = kw.get("model", self.model)
        s = int(state.global_step)
        if s > 0 and (s % self.hp.eval_every == 0 or s >= args.max_steps):
            self._evaluate(model, s)

"""DissociatedMonitorCallback: the full behavioral panel at every eval step, on ONE continuous
wandb run, with checkpoint selection by the judge-measured nudge gap.

Every eval_every steps it generates clean and nudged completions on a held-out HarmBench live
set, judges both with the resident classifier, and logs clean refusal, clean/nudged
compliance, the nudge gap, the base-control gap (computed once: the base is frozen), the
static-probe AUROC + sigmoid gap on the dissociated model's clean activations, KL to base, and the
judge-free flip proxy. The best checkpoint is the highest base-subtracted reachability gap subject
to a clean-refusal floor. Save/restore of train mode, use_cache, and gradient checkpointing is in a
finally block.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import torch
from transformers import TrainerCallback

from .. import config
from ..models import generate_batch
from . import probe as probe_mod
from . import selection
from .losses import tokenwise_kl
from .nudge import NudgeHook, get_decoder_layer


def _is_degenerate(text: str) -> bool:
    t = (text or "").strip()
    return len(t) < 3 or t == "[EMPTY]"


class DissociatedMonitorCallback(TrainerCallback):
    def __init__(self, model, tok, judge, behaviors, hp, spec, arch, *, base_model, delta,
                 nudge_layer, probe, probe_safe_texts, probe_unsafe_texts, proxy_batches):
        self.model, self.tok, self.judge = model, tok, judge
        self.behaviors = [b["question"] if isinstance(b, dict) else b for b in behaviors]
        self.hp, self.spec, self.arch = hp, spec, arch
        self.base_model = base_model
        self.delta = delta
        self.nudge_layer = nudge_layer
        self.probe = probe
        self.probe_safe_texts = probe_safe_texts
        self.probe_unsafe_texts = probe_unsafe_texts
        self.proxy_batches = proxy_batches or []
        self.template = spec["chat_template"]

        self.out_dir = config.dissociated_dir(arch)
        self.best_dir = config.dissociated_best_dir(arch)
        self.csv = config.dissociated_trajectory_csv(arch)
        self.gens_dir = config.dissociated_gens_dir(arch)
        self.gens_dir.mkdir(parents=True, exist_ok=True)
        self.best_gap = -1e9
        self.best_refusal = -1.0
        self._have_qualified = False
        self.rows: list[dict] = []

        # base control is constant (frozen base + fixed nudge): compute it once.
        self.base_clean_comp, self.base_nudged_comp = self._base_control()
        # selection floor adapts to the base: never demand more clean refusal than the base itself
        # has (e.g. Qwen's base already refuses ~0.89), capped by the absolute floor.
        self.floor = min(self.hp.clean_refusal_floor,
                         self.base_clean_refusal - self.hp.clean_refusal_tolerance)
        print(f"[monitor] base control: clean_comp={self.base_clean_comp:.3f} "
              f"nudged_comp={self.base_nudged_comp:.3f} "
              f"gap={self.base_nudged_comp - self.base_clean_comp:.3f} | "
              f"base_clean_refusal={self.base_clean_refusal:.3f} -> selection floor={self.floor:.3f}",
              flush=True)

    # ---- one-time base control ----
    def _gen(self, model, with_nudge: bool):
        hook = None
        if with_nudge:
            dev = next(model.parameters()).device
            hook = NudgeHook(self.delta.to(dev)).register(get_decoder_layer(model, self.nudge_layer))
        try:
            return generate_batch(model, self.tok, self.behaviors, self.hp.max_seq_len,
                                  self.hp.eval_max_new_tokens, 0.0, template=self.template)
        finally:
            if hook is not None:
                hook.remove()

    def _comp(self, gens):
        labels = self.judge.classify_batch(self.behaviors, gens)
        return labels, sum(labels) / len(labels)

    def _base_control(self):
        if self.base_model is None:
            self.base_clean_refusal = 1.0
            return float("nan"), float("nan")
        clean = self._gen(self.base_model, with_nudge=False)
        _, c = self._comp(clean)
        _, n = self._comp(self._gen(self.base_model, with_nudge=True))
        clean_empty = sum(_is_degenerate(g) for g in clean) / len(clean)
        self.base_clean_refusal = max(0.0, 1.0 - c - clean_empty)
        return c, n

    # ---- live eval ----
    def _evaluate(self, model, step):
        was_training = model.training
        prev_cache = model.config.use_cache
        was_gc = bool(getattr(model, "is_gradient_checkpointing", False))
        model.eval()
        if was_gc:
            model.gradient_checkpointing_disable()
        model.config.use_cache = True
        try:
            clean = self._gen(model, with_nudge=False)
            nudged = self._gen(model, with_nudge=True)
            clean_lab, clean_comp = self._comp(clean)
            nudged_lab, nudged_comp = self._comp(nudged)
            clean_empty = sum(_is_degenerate(g) for g in clean) / len(clean)
            nudged_empty = sum(_is_degenerate(g) for g in nudged) / len(nudged)
            clean_refusal = max(0.0, 1.0 - clean_comp - clean_empty)
            nudge_gap = nudged_comp - clean_comp
            base_gap = self.base_nudged_comp - self.base_clean_comp
            reach_gap = nudge_gap - base_gap

            flip = selection.reachability_proxy(
                model, self.proxy_batches, self.nudge_layer,
                self.delta.to(next(model.parameters()).device))
            auroc, sig_gap = probe_mod.auroc_and_gap(
                self.probe, model, self.tok, self.probe_safe_texts, self.probe_unsafe_texts)
            kl = self._kl_to_base(model)

            metrics = {
                "eval/clean_refusal": clean_refusal,
                "eval/clean_compliance": clean_comp,
                "eval/nudged_compliance": nudged_comp,
                "eval/nudge_gap": nudge_gap,
                "eval/base_control_gap": base_gap,
                "eval/reachability_gap": reach_gap,
                "eval/clean_empty_rate": clean_empty,
                "eval/nudged_empty_rate": nudged_empty,
                "eval/probe_auroc": auroc,
                "eval/sigmoid_gap": sig_gap,
                "eval/kl_to_base": kl,
                "eval/flip": flip["flip"],
                "eval/clean_pref_refuse": flip["clean_pref_refuse"],
                "eval/nudged_pref_comply": flip["nudged_pref_comply"],
                "eval/n": len(self.behaviors),
            }
            self._log(step, metrics, clean, clean_lab, nudged, nudged_lab)

            qualifies = clean_refusal >= self.floor
            if qualifies and reach_gap > self.best_gap:
                self._have_qualified = True
                self.best_gap = reach_gap
                self._save_best(model, step, clean_refusal, reach_gap, tag="best")
            elif not self._have_qualified and clean_refusal > self.best_refusal:
                # no floor-qualifying checkpoint yet: keep the most base-like one as a safety net
                self.best_refusal = clean_refusal
                self._save_best(model, step, clean_refusal, reach_gap, tag="provisional")
            return metrics
        finally:
            model.config.use_cache = prev_cache
            if was_gc:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if was_training:
                model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @torch.no_grad()
    def _kl_to_base(self, model):
        if self.base_model is None or not self.proxy_batches:
            return float("nan")
        b = self.proxy_batches[0]
        dev = next(model.parameters()).device
        ri, rm = b["refuse_input_ids"].to(dev), b["refuse_attention_mask"].to(dev)
        s = model(input_ids=ri, attention_mask=rm, use_cache=False).logits
        t = self.base_model(input_ids=ri, attention_mask=rm, use_cache=False).logits
        return float(tokenwise_kl(s, t.to(s.dtype), rm))

    def _log(self, step, metrics, clean, clean_lab, nudged, nudged_lab):
        row = {"global_step": step, **{k.replace("eval/", ""): v for k, v in metrics.items()}}
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.csv, index=False)
        pd.DataFrame({"behavior": self.behaviors,
                      "clean_gen": clean, "clean_harmful": clean_lab,
                      "nudged_gen": nudged, "nudged_harmful": nudged_lab}) \
            .to_csv(self.gens_dir / f"step_{step}.csv", index=False)
        print(f"[monitor] step {step}: clean_refusal={metrics['eval/clean_refusal']:.3f} "
              f"nudge_gap={metrics['eval/nudge_gap']:.3f} "
              f"reach_gap={metrics['eval/reachability_gap']:.3f} "
              f"auroc={metrics['eval/probe_auroc']:.3f} flip={metrics['eval/flip']:.3f}", flush=True)
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({**metrics, "train/global_step": step})
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] non-fatal: {e}", flush=True)

    def _save_best(self, model, step, clean_refusal, gap, tag="best"):
        tmp = self.out_dir / "best_tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        model.save_pretrained(tmp)
        self.tok.save_pretrained(tmp)
        if self.best_dir.exists():
            shutil.rmtree(self.best_dir)
        tmp.rename(self.best_dir)
        (self.out_dir / "best_meta.txt").write_text(
            f"tag={tag} step={step} clean_refusal={clean_refusal:.4f} reach_gap={gap:.4f}\n")
        print(f"[monitor] saved {tag} dissociated (reach_gap={gap:.3f}, "
              f"clean_refusal={clean_refusal:.3f}) -> {self.best_dir}", flush=True)

    def on_step_end(self, args, state, control, **kw):
        model = kw.get("model", self.model)
        s = int(state.global_step)
        # state.max_steps is the resolved total; args.max_steps is -1 for epoch-based training,
        # so guarding on it would fire every step. Eval on cadence or at the very last step.
        if s > 0 and (s % self.hp.eval_every == 0 or s >= state.max_steps):
            self._evaluate(model, s)

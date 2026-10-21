# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Run a Hugging Face model over prompts under a :class:`ModelMonitor`: the engine of
``tqp model calibrate`` and ``tqp model watch``."""

from __future__ import annotations

import json
from collections.abc import Iterator

from .model_telemetry import ModelMonitor, ReadPlan, status_summary

__all__ = ["iter_prompts", "load_model", "run_prompts", "window_line"]


def iter_prompts(path: str) -> Iterator[str]:
    """One prompt per line of a text file, or per JSON line (its ``text`` field) when
    the file ends in ``.jsonl``. Blank lines are skipped."""
    jsonl = path.endswith(".jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            yield json.loads(line)["text"] if jsonl else line


def load_model(name: str, *, device: str = "cpu", dtype: str = "float32"):
    """``(model, tokenizer)``. A multimodal checkpoint (Gemma 3's, say) loads through
    ``AutoModelForCausalLM`` as its causal-LM head; only decoder attention is observed.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=getattr(torch, dtype), attn_implementation="sdpa"
    )
    return model.to(device).eval(), tok


def run_prompts(
    model, tok, prompts, monitor: ModelMonitor, *, max_new_tokens=0, cache=None
):
    """Feed each prompt (a forward pass, or ``generate`` when ``max_new_tokens``) with
    the monitor attached; returns the number of prompts run."""
    import torch

    n = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for text in prompts:
            ids = tok(text, return_tensors="pt").to(device)
            if max_new_tokens:
                kw = {}
                if cache is not None:
                    kw["past_key_values"] = cache()
                    monitor.watch_cache(kw["past_key_values"])
                model.generate(
                    **ids, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id, **kw,
                )  # fmt: skip
            else:
                model(**ids)
            n += 1
    return n


def window_line(window, comparisons) -> str:
    """One line per window: its tokens, the status counts and the worst head."""
    head = f"window {window.index}: {window.tokens} tokens"
    if not comparisons:
        return f"{head}, {len(window.layers)} layers"
    s = status_summary(comparisons)
    counts = ", ".join(f"{k} {v}" for k, v in sorted(s["counts"].items()))
    line = f"{head}, {len(comparisons)} heads: {counts}"
    bad = [c for c in s["not_valid"] if c["overlap"] is not None]
    if bad:
        w = bad[0]
        kap = w["kappa"]
        reg = w["regret"]
        line += (
            f"; worst L{w['layer']}/kv{w['kv_head']} overlap {w['overlap']:.3f}"
            + (f" kappa {kap:.3g}" if kap is not None else "")
            + (f" regret {reg:.3g}" if reg is not None else "")
        )
    return line


def load_plan(path: str) -> ReadPlan:
    return ReadPlan.load(path)

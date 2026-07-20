# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""KV-cache serving benchmark: the attention softmax as the measured consumer.

The retrieval fleet measured the consumer-relative triplet for a *ranking*
consumer. This benchmark measures the same structure on a real model's
serving path, where the consumer is the **attention softmax**: per-decode-step
attention distributions under compressed KV vs the fp16 baseline
(teacher-forced on the baseline's tokens, so both caches attend over an
identical stream), plus free-running behavioral agreement and decode
throughput.

Per config: KL(attn_fp16 || attn_quant) per sampled layer (mean over heads and
steps), token agreement over a free greedy continuation, decode wall/token,
and the nominal cold-token KV footprint. Runs on one GPU
(CUDA_VISIBLE_DEVICES picks it); attention implementation is eager for
probability capture and identical across configs, so relative throughput is
meaningful, absolute is not.

    CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_kv_serving.py \
        --model Qwen/Qwen2.5-1.5B-Instruct --prefix-tokens 2048 --steps 48
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from turboquant_pro import TurboQuantCache


def get_prompts(tok, n: int, prefix_tokens: int) -> list[torch.Tensor]:
    """Long real-text prefixes from wikitext (streamed, tiny download)."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    prompts, buf = [], ""
    for ex in ds:
        buf += ex["text"]
        if len(buf) > prefix_tokens * 8:  # chars ~4-8x tokens; overshoot
            ids = tok(buf, return_tensors="pt").input_ids[0, :prefix_tokens]
            if len(ids) == prefix_tokens:
                prompts.append(ids)
            buf = ""
            if len(prompts) == n:
                break
    return prompts


def kl(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.clamp_min(1e-9)
    q = q.clamp_min(1e-9)
    return float((p * (p.log() - q.log())).sum(-1).mean())


def make_cache(spec, model):
    if spec is None:
        return DynamicCache(config=model.config)
    return TurboQuantCache(**spec)


@torch.no_grad()
def prefill(model, ids, cache):
    out = model(ids[None].to(model.device), past_key_values=cache, use_cache=True)
    return out.logits[0, -1].argmax().item(), out.past_key_values


@torch.no_grad()
def run_config(model, prompts, base_tokens, spec, layers, steps):
    """One config over all prompts. Returns metrics; base_tokens=None means
    this IS the baseline (records its own greedy tokens)."""
    res = {
        "attn_kl": {str(li): [] for li in layers},
        "tokens": [],
        "step_ms": [],
        "agree": [],
    }
    for pi, ids in enumerate(prompts):
        # --- teacher-forced pass (attention capture) --------------------- #
        tok0, cache = prefill(model, ids, make_cache(spec, model))
        forced = base_tokens[pi] if base_tokens is not None else None
        cur = forced[0] if forced is not None else tok0
        attns = {str(li): [] for li in layers}
        for s in range(steps):
            out = model(
                torch.tensor([[cur]], device=model.device),
                past_key_values=cache,
                use_cache=True,
                output_attentions=True,
            )
            for li in layers:
                # (batch, heads, q=1, ctx) -> (heads, ctx)
                attns[str(li)].append(out.attentions[li][0, :, 0].float().cpu())
            cur = (
                forced[s + 1]
                if forced is not None and s + 1 < len(forced)
                else out.logits[0, -1].argmax().item()
            )
        res.setdefault("attn_raw", {})[pi] = attns
        # --- free-running pass (behavior + throughput) ------------------- #
        tok0, cache = prefill(model, ids, make_cache(spec, model))
        cur, toks = tok0, [tok0]
        t0 = time.time()
        for _ in range(steps - 1):
            out = model(
                torch.tensor([[cur]], device=model.device),
                past_key_values=cache,
                use_cache=True,
            )
            cur = out.logits[0, -1].argmax().item()
            toks.append(cur)
        res["step_ms"].append(1000 * (time.time() - t0) / (steps - 1))
        res["tokens"].append(toks)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--prefix-tokens", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--out", default="bench_kv_serving_result.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    # fp32, not fp16: Qwen attention logits overflow fp16 in eager capture
    # (NaN heads in the *baseline*), and Volta-class GPUs have no native bf16.
    # The KV cache under test is still stored/quantized by the adapter itself;
    # only the compute dtype changes.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        device_map="cuda:0",
        attn_implementation="eager",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    layers = [0, n_layers // 2, n_layers - 1]
    prompts = get_prompts(tok, args.prompts, args.prefix_tokens)
    print(
        f"model={args.model} layers={n_layers} probes={layers} "
        f"prompts={len(prompts)}x{args.prefix_tokens}",
        flush=True,
    )

    cfg = model.config
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    fp16_bpt = 2 * n_layers * kv_heads * head_dim * 2  # K+V, fp16

    CONFIGS = {
        "fp16": None,
        "tq_hot512": {"hot_window": 512},
        "tq_hot128": {"hot_window": 128},
        "tq_hot128_of0": {"hot_window": 128, "outlier_frac": 0.0},
    }
    results = {
        "model": args.model,
        "prefix_tokens": args.prefix_tokens,
        "steps": args.steps,
        "fp16_bytes_per_token": fp16_bpt,
        "probe_layers": layers,
        "configs": {},
    }
    base = run_config(model, prompts, None, None, layers, args.steps)
    base_tokens = base["tokens"]
    results["configs"]["fp16"] = {"step_ms": round(float(np.mean(base["step_ms"])), 1)}

    for name, spec in CONFIGS.items():
        if spec is None:
            continue
        print(f"config {name}", flush=True)
        r = run_config(model, prompts, base_tokens, spec, layers, args.steps)
        entry = {"step_ms": round(float(np.mean(r["step_ms"])), 1)}
        # attention KL vs baseline, per probe layer
        for li in layers:
            kls = []
            for pi in range(len(prompts)):
                for s in range(args.steps):
                    a = base["attn_raw"][pi][str(li)][s]
                    b = r["attn_raw"][pi][str(li)][s]
                    m = min(a.shape[-1], b.shape[-1])
                    kls.append(kl(a[..., :m], b[..., :m]))
            entry[f"attn_kl_L{li}"] = round(float(np.mean(kls)), 5)
        # behavioral agreement on free-running tokens
        agree, first_div = [], []
        for pi in range(len(prompts)):
            a, b = base["tokens"][pi], r["tokens"][pi]
            match = [x == y for x, y in zip(a, b)]
            agree.append(np.mean(match))
            first_div.append(match.index(False) if False in match else len(match))
        entry["token_agreement"] = round(float(np.mean(agree)), 4)
        entry["mean_first_divergence"] = round(float(np.mean(first_div)), 1)
        # nominal cold-token footprint for this config
        s = {
            "hot_window": 512,
            "key_bits": 4,
            "value_bits": 4,
            "outlier_frac": 0.02,
            **spec,
        }
        cold = (
            n_layers
            * kv_heads
            * head_dim
            * (
                s["key_bits"] / 8
                + 2 * s["outlier_frac"]  # keys + outliers
                + s["value_bits"] / 8
            )  # values
        )
        entry["cold_bytes_per_token_nominal"] = int(cold)
        entry["compression_vs_fp16"] = round(fp16_bpt / cold, 1)
        results["configs"][name] = entry
        print(json.dumps({name: entry}), flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("RESULT_JSON " + json.dumps(results["configs"]), flush=True)
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()

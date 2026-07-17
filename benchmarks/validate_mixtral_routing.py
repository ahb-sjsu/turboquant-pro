# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Real Mixtral-8x7B: MoE routing fragility carried by the margin (top-2 router).

The top-2 companion to ``validate_olmoe_routing.py``. Mixtral-8x7B-Instruct routes
each token to 2 of 8 experts, and — unlike OLMoE's top-8 (whose 8th/9th boundary
margins are near-zero) — its top-2 boundary margins are large enough that the
margin structure survives to *practical* bit-depths: quantizing the router gate
coarsely flips a large but not saturated fraction of tokens, and the low-margin
tail flips far more than the high-margin body.

Captures the real router gate logits over WikiText-2, summarizes the margin
distribution with ``operator_sensitivity.routing_sensitivity``, then quantizes
each gate's weights at 3/4-bit and measures the actual top-2 expert-set flip
rate — overall, and split by low- vs high-margin tokens. Backs the numbers in
``paper/foundational/main.tex`` (top-2, WikiText-2).

Runs on a single visible GPU + CPU offload (set ``CUDA_VISIBLE_DEVICES=1`` to keep
it off other GPUs):

    CUDA_VISIBLE_DEVICES=1 python benchmarks/validate_mixtral_routing.py \
        --model mistralai/Mixtral-8x7B-Instruct-v0.1 --out results_mixtral_routing.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get("TQP_REPO", os.getcwd()))
from turboquant_pro.operator_sensitivity import (  # noqa: E402
    routing_margins,
    routing_sensitivity,
)


def _quant_uniform(x: np.ndarray, bits: int) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    step = (hi - lo) / max(2**bits - 1, 1)
    return np.round((x - lo) / max(step, 1e-30)) * step + lo


def _topk_sets(logits: np.ndarray, k: int) -> list[frozenset]:
    idx = np.argpartition(-logits, k, axis=1)[:, :k]
    return [frozenset(row.tolist()) for row in idx]


def _load_gate_weights(model_id: str, n_layers: int) -> dict[int, np.ndarray]:
    """Read the (tiny) router gate weights straight from the cached safetensors,
    independent of accelerate offload (offloaded params are meta tensors)."""
    import glob

    from safetensors import safe_open

    home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    stem = "models--" + model_id.replace("/", "--")
    snaps = glob.glob(os.path.join(home, "hub", stem, "snapshots", "*"))
    if not snaps:
        raise FileNotFoundError(f"no cached snapshot for {model_id} under {home}")
    snap = snaps[0]
    with open(os.path.join(snap, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    out: dict[int, np.ndarray] = {}
    for i in range(n_layers):
        key = f"model.layers.{i}.block_sparse_moe.gate.weight"
        with safe_open(os.path.join(snap, weight_map[key]), framework="pt") as sf:
            out[i] = sf.get_tensor(key).float().cpu().numpy()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Real Mixtral routing-margin validation.")
    ap.add_argument("--model", default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    ap.add_argument("--chunks", type=int, default=4, help="1024-token chunks")
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 3])
    ap.add_argument("--gpu-gib", type=int, default=30, help="GPU budget before offload")
    ap.add_argument("--out", default="results_mixtral_routing.json")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer, MixtralForCausalLM

    print(f"loading {args.model} (GPU + CPU offload) ...", flush=True)
    model = MixtralForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={0: f"{args.gpu_gib}GiB", "cpu": "185GiB"},
    ).eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    top_k = int(model.config.num_experts_per_tok)
    n_experts = int(model.config.num_local_experts)
    print(f"top_k={top_k}, n_experts={n_experts}", flush=True)

    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids

    gates = [layer.block_sparse_moe.gate for layer in model.model.layers]
    captured: list[tuple[int, np.ndarray, np.ndarray]] = []

    def mk(i):
        def hook(mod, inp, out):
            logits = out[0] if isinstance(out, tuple) else out
            captured.append(
                (
                    i,
                    inp[0].detach().reshape(-1, inp[0].shape[-1]).float().cpu().numpy(),
                    logits.detach().reshape(-1, logits.shape[-1]).float().cpu().numpy(),
                )
            )

        return hook

    handles = [g.register_forward_hook(mk(i)) for i, g in enumerate(gates)]
    # Gate weights are meta tensors under offload — read them from the cache.
    weights = _load_gate_weights(args.model, len(gates))  # {i: (E, hidden)}

    with torch.no_grad():
        for c in range(args.chunks):
            chunk = ids[:, c * 1024 : (c + 1) * 1024]
            if chunk.size(1) < 2:
                break
            model(chunk.to(0))
            print(f"  forward {c + 1}/{args.chunks}", flush=True)
    for h in handles:
        h.remove()

    all_logits = np.concatenate([o for (_, _, o) in captured], axis=0)
    med = float(np.median(routing_margins(all_logits, k=top_k)))
    print(f"captured {all_logits.shape[0]} routing decisions; median margin {med:.4f}")

    def _split(flip, marg, med):
        lo = float(flip[marg < med].mean())
        hi = float(flip[marg >= med].mean())
        return {
            "overall_flip_rate": float(flip.mean()),
            "low_margin_flip_rate": lo,
            "high_margin_flip_rate": hi,
            "low_over_high_ratio": (lo / hi) if hi > 0 else None,
        }

    results = {
        "model": args.model,
        "dataset": "wikitext-2-raw-v1 (test)",
        "top_k": top_k,
        "n_experts": n_experts,
        "n_routing_decisions": int(all_logits.shape[0]),
        "margin_summary_top1": routing_sensitivity(all_logits, k=1).as_dict(),
        "margin_summary_topk": routing_sensitivity(all_logits, k=top_k).as_dict(),
        "gate_weight_quant": {},
        "differential_noise_sweep": {},
    }

    # (a) Naive gate-weight quantization → top-k expert-set flips.
    for bits in args.bits:
        flip_all, marg_all = [], []
        for i, x, o in captured:
            base = _topk_sets(o, top_k)
            logq = x @ _quant_uniform(weights[i], bits).T
            flip_all.append(
                np.array(
                    [len(b ^ q) > 0 for b, q in zip(base, _topk_sets(logq, top_k))]
                )
            )
            marg_all.append(routing_margins(o, k=top_k))
        results["gate_weight_quant"][str(bits)] = _split(
            np.concatenate(flip_all), np.concatenate(marg_all), med
        )
        r = results["gate_weight_quant"][str(bits)]
        print(
            f"{bits}-bit gate weights: overall {r['overall_flip_rate']:.3f}, "
            f"low/high {r['low_over_high_ratio']:.2f}x",
            flush=True,
        )

    # (b) Controlled differential-logit perturbation at the margin scale.
    rng = np.random.default_rng(0)
    for kk in (1, top_k):
        m = routing_margins(all_logits, k=kk)
        medk = float(np.median(m))
        base = _topk_sets(all_logits, kk)
        sweep = []
        for fac in (0.25, 0.5, 1.0, 2.0):
            sigma = fac * medk
            noise = rng.standard_normal(all_logits.shape).astype(np.float64) * sigma
            flip = np.array(
                [
                    len(b ^ p) > 0
                    for b, p in zip(base, _topk_sets(all_logits + noise, kk))
                ]
            )
            row = _split(flip, m, medk)
            row["sigma_over_median_margin"] = fac
            sweep.append(row)
        results["differential_noise_sweep"][f"k{kk}"] = {
            "median_margin": medk,
            "levels": sweep,
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, allow_nan=False)
    print(json.dumps(results, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Real MoE router: selection is carried by the margin, not the magnitude.

Phase-7 MoE-routing validation on a real mixture-of-experts model (OLMoE-1B-7B,
64 experts, top-8). A top-k router reads only the *order* of the gate logits, so
a common-mode quantization error is free and only the differential error across
experts — where it exceeds the per-token **routing margin** (the gap between the
k-th and (k+1)-th logit) — can flip the selected expert set. The prediction:
low-margin tokens flip far more than high-margin ones as the gate is quantized to
fewer bits.

This captures the real gate logits over WikiText-2, summarizes the margin
distribution with the shipped ``operator_sensitivity.routing_sensitivity``, then
fake-quantizes each router's gate weights at 3/4-bit, recomputes the logits, and
measures the actual top-k expert-set **flip rate** — overall, and split by low- vs
high-margin tokens (the differential/(A2) fragility split for selection).

    python benchmarks/validate_olmoe_routing.py \
        --model allenai/OLMoE-1B-7B-0924 --out results_olmoe_routing.json
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Real OLMoE routing-margin validation.")
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--chunks", type=int, default=16, help="1024-token chunks")
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 3])
    ap.add_argument("--out", default="results_olmoe_routing.json")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer, OlmoeForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.model} on {device} ...", flush=True)
    model = (
        OlmoeForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )
    tok = AutoTokenizer.from_pretrained(args.model)
    top_k = int(model.config.num_experts_per_tok)
    n_experts = int(model.config.num_experts)
    print(f"top_k={top_k}, n_experts={n_experts}", flush=True)

    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids

    # Hook each router gate to capture (input hidden state, logits). The gate
    # returns (router_logits, top_k_weights, top_k_index) in recent transformers,
    # so take element 0 when it is a tuple.
    gates = [layer.mlp.gate for layer in model.model.layers]
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

    def _gate_weight(g) -> np.ndarray:
        """Router weight (n_experts, hidden), whether the gate is a Linear or a
        router module wrapping a weight parameter."""
        w = getattr(g, "weight", None)
        if w is None:
            for p in g.parameters():
                if p.ndim == 2 and p.shape[0] == n_experts:
                    w = p
                    break
        if w is None:
            raise RuntimeError("could not locate the router gate weight")
        return w.detach().float().cpu().numpy()

    handles = [g.register_forward_hook(mk(i)) for i, g in enumerate(gates)]
    weights = [_gate_weight(g) for g in gates]  # (E, hidden)

    with torch.no_grad():
        for c in range(args.chunks):
            chunk = ids[:, c * 1024 : (c + 1) * 1024]
            if chunk.size(1) < 2:
                break
            model(chunk.to(device))
    for h in handles:
        h.remove()

    # Aggregate every (layer, token) routing decision.
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
        # Margins at the argmax boundary (k=1) and the top-k set boundary.
        "margin_summary_top1": routing_sensitivity(all_logits, k=1).as_dict(),
        "margin_summary_topk": routing_sensitivity(all_logits, k=top_k).as_dict(),
        "gate_weight_quant": {},
        "differential_noise_sweep": {},
    }

    # (a) Realistic: naively quantize the router gate weights, recompute logits,
    # measure top-k expert-set flips (the practical "don't do this" warning).
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

    # (b) The clean mechanism test: a controlled differential-logit perturbation
    # at the margin scale. Selection is invariant to common-mode, so this isolates
    # the margin's role — low-margin tokens must flip first. Run at the argmax
    # (k=1) and top-k boundaries.
    rng = np.random.default_rng(0)
    for kk in (1, top_k):
        m = routing_margins(all_logits, k=kk)
        medk = float(np.median(m))
        base = _topk_sets(all_logits, kk)
        sweep = []
        for fac in (0.25, 0.5, 1.0, 2.0):
            sigma = fac * medk
            noise = rng.standard_normal(all_logits.shape).astype(np.float64) * sigma
            pert = _topk_sets(all_logits + noise, kk)
            flip = np.array([len(b ^ p) > 0 for b, p in zip(base, pert)])
            row = _split(flip, m, medk)
            row["sigma_over_median_margin"] = fac
            row["sigma"] = sigma
            sweep.append(row)
        results["differential_noise_sweep"][f"k{kk}"] = {
            "median_margin": medk,
            "levels": sweep,
        }
        small = sweep[0]
        print(
            f"noise sweep k={kk}: at sigma=0.25*median, overall "
            f"{small['overall_flip_rate']:.3f}, low/high "
            f"{small['low_over_high_ratio']:.2f}x",
            flush=True,
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, allow_nan=False)
    print(json.dumps(results, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

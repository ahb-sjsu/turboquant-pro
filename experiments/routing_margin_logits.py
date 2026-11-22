# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Routing margins, gate-quantization flips and the noise sweep, on router LOGITS.

benchmarks/validate_olmoe_routing.py (results_olmoe_routing.json) took the router
module's first output as its logits. In transformers 5 the Mixtral and OLMoE top-k
routers return softmax probabilities there, so its margins, its low/high split
and its Gaussian noise sweep are in probability space. This script measures the
same quantities on the logits, x @ W^T, from the router's captured input x and
the gate weights read from the checkpoint (as router_error_margin.py does), on
the same WikiText-2 chunks.

Checks recorded with the result:
  * softmax(x W^T) against the router's own output (probabilities);
  * the top-k set of x W^T against the router's own chosen experts.

Recorded:
  * the margin at the argmax (k=1) and at the top-k boundary, per decision,
    in routing_margins_<model>.npz (arrays [layer, token], float32), with a
    percentile grid and per-layer medians in the JSON;
  * top-k flip rates when the gate weights are quantized (per-tensor uniform
    grid over [min, max], as the routing harness), overall and split at the
    median logit margin;
  * a sweep of i.i.d. Gaussian noise added to the logits, sigma from 0.01 to 2
    times the median margin, several draws, with each level's flip rate next to
    the pairwise prediction mean over decisions of Phi(-m / (sqrt(2) sigma)),
    the chance that the k-th and (k+1)-th logits swap under that noise.

    NB_MODEL=allenai/OLMoE-1B-7B-0924 NB_CHUNKS=16 python routing_margin_logits.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router_error_margin import gate_weights, quant_uniform  # noqa: E402

PCTS = [0.1, 0.5, 1, 2, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 98, 99]
PCTS += [99.5, 99.9]
FACTORS = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]


def margins(lg: np.ndarray, k: int) -> np.ndarray:
    """k-th minus (k+1)-th largest logit per row (operator_sensitivity's
    routing_margins)."""
    s = -np.sort(-lg, axis=1)
    return s[:, k - 1] - s[:, k]


def topk_mask(lg: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-lg, k - 1, axis=1)[:, :k]
    m = np.zeros(lg.shape, dtype=bool)
    np.put_along_axis(m, idx, True, axis=1)
    return m


def flipped(base: np.ndarray, lg: np.ndarray, k: int) -> np.ndarray:
    return (topk_mask(lg, k) != base).any(axis=1)


def split(flip: np.ndarray, m: np.ndarray, med: float) -> dict:
    lo, hi = float(flip[m < med].mean()), float(flip[m >= med].mean())
    return {
        "overall_flip_rate": float(flip.mean()),
        "low_margin_flip_rate": lo,
        "high_margin_flip_rate": hi,
        "low_over_high_ratio": lo / hi if hi > 0 else None,
    }


def pair_prediction(m: np.ndarray, sigma: float) -> float:
    """mean over decisions of Phi(-m / (sqrt(2) sigma))."""
    z = torch.from_numpy(m.astype(np.float64)) / (2.0 * sigma)
    return float((0.5 * torch.special.erfc(z)).mean())


def pct(v: np.ndarray) -> dict:
    return {str(p): float(x) for p, x in zip(PCTS, np.percentile(v, PCTS))}


def main() -> int:
    model_name = os.environ.get("NB_MODEL", "allenai/OLMoE-1B-7B-0924")
    n_chunks = int(os.environ.get("NB_CHUNKS", "16"))
    bits_list = [int(b) for b in os.environ.get("NB_BITS", "4,3").split(",")]
    draws = int(os.environ.get("NB_DRAWS", "4"))
    gpu_gib = int(os.environ.get("NB_GPU_GIB", "4"))
    out_dir = os.environ.get("NB_OUT", "benchmarks")

    thermal = None
    try:
        from batch_probe import ThermalController

        thermal = ThermalController(target_temp=80.0, max_threads=12, min_threads=2)
        thermal.start()
        torch.set_num_threads(thermal.get_threads())
    except Exception as e:
        print(f"[thermal] controller unavailable ({e}); capping at 6 threads")
        torch.set_num_threads(6)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        max_memory={0: f"{gpu_gib}GiB", "cpu": "150GiB"},
    ).eval()
    tok = AutoTokenizer.from_pretrained(model_name)
    cfg = model.config
    k = int(cfg.num_experts_per_tok)
    n_exp = int(getattr(cfg, "num_local_experts", None) or cfg.num_experts)
    gates = [
        (n, m)
        for n, m in model.named_modules()
        if n.endswith(".gate")
        and getattr(m, "weight", None) is not None
        and m.weight.dim() == 2
        and m.weight.shape[0] == n_exp
    ]
    assert gates, "no router modules found"
    print(
        f"[load] {model_name} top-{k} of {n_exp}, {len(gates)} routers, "
        f"{time.time() - t0:.0f}s"
    )

    xs = [[] for _ in gates]
    probs = [[] for _ in gates]
    chosen = [[] for _ in gates]

    def mk(i):
        def hook(_mod, inp, out):
            first = out[0] if isinstance(out, tuple) else out
            xs[i].append(
                inp[0].detach().reshape(-1, inp[0].shape[-1]).float().cpu().numpy()
            )
            probs[i].append(
                first.detach().reshape(-1, first.shape[-1]).float().cpu().numpy()
            )
            if isinstance(out, tuple) and len(out) >= 3:
                chosen[i].append(out[2].detach().reshape(-1, k).cpu().numpy())

        return hook

    handles = [m.register_forward_hook(mk(i)) for i, (_, m) in enumerate(gates)]
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    with torch.no_grad():
        for c in range(n_chunks):
            if thermal is not None:
                torch.set_num_threads(thermal.get_threads())
            chunk = ids[:, c * 1024 : (c + 1) * 1024]
            if chunk.size(1) < 2:
                break
            model(chunk.to(0))
            print(f"  forward {c + 1}/{n_chunks} ({time.time() - t0:.0f}s)")
    for h in handles:
        h.remove()

    weights = gate_weights(model_name, [n for n, _ in gates])
    X, LG, checks = [], [], []
    for i, W in enumerate(weights):
        x = np.concatenate(xs[i])
        lg = x @ W.T  # pre-softmax logits
        p = np.exp(lg - lg.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        pr = np.concatenate(probs[i])
        c = {"layer": i, "prob_max_abs_diff": float(np.abs(p - pr).max())}
        if chosen[i]:
            ch = np.concatenate(chosen[i])
            mine = topk_mask(lg, k)
            theirs = np.zeros_like(mine)
            np.put_along_axis(theirs, ch, True, axis=1)
            c["topk_agreement"] = float((mine == theirs).all(axis=1).mean())
        checks.append(c)
        X.append(x)
        LG.append(lg)
    agree = [c["topk_agreement"] for c in checks if "topk_agreement" in c]
    print(
        f"[check] softmax(xW^T) vs router output: max abs diff "
        f"{max(c['prob_max_abs_diff'] for c in checks):.3g}; top-{k} agreement "
        f"min {min(agree) if agree else float('nan'):.5f}"
    )

    n_layers, n_tok = len(LG), LG[0].shape[0]
    m1 = np.stack([margins(lg, 1) for lg in LG]).astype(np.float32)
    mk_ = np.stack([margins(lg, k) for lg in LG]).astype(np.float32)
    os.makedirs(out_dir, exist_ok=True)
    tag = model_name.split("/")[-1]
    npz = os.path.join(out_dir, f"routing_margins_{tag}.npz")
    np.savez_compressed(npz, margin_k1=m1, **{f"margin_k{k}": mk_})

    result = {
        "model": model_name,
        "top_k": k,
        "n_experts": n_exp,
        "n_routers": n_layers,
        "n_chunks": n_chunks,
        "tokens_per_chunk": 1024,
        "n_decisions": int(n_layers * n_tok),
        "logits": "x @ W^T: router input captured by a forward hook, gate weights "
        "from the checkpoint (float32); the model ran in float16",
        "checks": {
            "per_layer": checks,
            "prob_max_abs_diff": max(c["prob_max_abs_diff"] for c in checks),
            "topk_agreement_min": min(agree) if agree else None,
            "topk_agreement_mean": float(np.mean(agree)) if agree else None,
        },
        "margins_file": os.path.basename(npz),
        "margins": {},
        "gate_weight_quant": {},
        "noise_sweep": {},
    }
    for kk, mm in ((1, m1), (k, mk_)):
        v = mm.ravel().astype(np.float64)
        result["margins"][f"k{kk}"] = {
            "percentiles": pct(v),
            "mean": float(v.mean()),
            "per_layer_median": [float(np.median(r)) for r in mm],
            "n_zero": int((v == 0).sum()),
        }

    lg_all = np.concatenate(LG).astype(np.float64)
    m_all = mk_.ravel().astype(np.float64)
    med = float(np.median(m_all))
    base = topk_mask(lg_all, k)
    for bits in bits_list:
        fl = np.concatenate(
            [
                flipped(topk_mask(lg, k), x @ quant_uniform(W, bits).T, k)
                for x, lg, W in zip(X, LG, weights)
            ]
        )
        result["gate_weight_quant"][str(bits)] = split(fl, m_all, med)
        r = result["gate_weight_quant"][str(bits)]
        print(
            f"[{bits}-bit gate] flip {r['overall_flip_rate']:.3f}, "
            f"low/high {r['low_over_high_ratio']}"
        )

    rng = np.random.default_rng(0)
    for kk in (1, k):
        m = margins(lg_all, kk)
        medk = float(np.median(m))
        p10 = float(np.percentile(m, 10))
        b = base if kk == k else topk_mask(lg_all, kk)
        levels = []
        for fac in FACTORS:
            sigma = fac * medk
            fl = np.zeros(m.shape, dtype=np.float64)
            for _ in range(draws):
                noise = rng.standard_normal(lg_all.shape) * sigma
                fl += flipped(b, lg_all + noise, kk)
            fl /= draws
            row = split(fl, m, medk)
            row |= {
                "sigma_over_median_margin": fac,
                "sigma": sigma,
                "sigma_over_p10_margin": sigma / p10 if p10 > 0 else None,
                "pair_prediction": pair_prediction(m, sigma),
            }
            levels.append(row)
            print(
                f"  noise k={kk} sigma={fac:g}*median: flip "
                f"{row['overall_flip_rate']:.4g}, pair prediction "
                f"{row['pair_prediction']:.4g}"
            )
        result["noise_sweep"][f"k{kk}"] = {
            "median_margin": medk,
            "p10_margin": p10,
            "draws": draws,
            "seed": 0,
            "levels": levels,
        }

    path = os.path.join(out_dir, f"routing_margin_logits_{tag}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1, allow_nan=False)
    print(f"[saved] {path} {npz}")
    if thermal is not None:
        try:
            thermal.stop()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    raise SystemExit(main())

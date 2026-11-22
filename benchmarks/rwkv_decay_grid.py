# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Decay-grid basis on a second state-space family: RWKV-4 (experiment E6c).

The Mamba study found that a 3-bit grid on the logarithm of the decay rate
keeps perplexity near the baseline while grids that are linear in the rate
collapse the model. Both Mamba checkpoints share one architecture. RWKV-4 is a
different recurrent family with the same kind of state: each attention channel
keeps a running sum that decays by w = exp(-exp(time_decay)) per token, so
time_decay is the logarithm of the decay rate, as A_log is in Mamba.

For every block's time_decay vector this fake-quantizes at NB_BITS bits in
three bases and puts the result back:

  log          uniform grid on time_decay (the log of the rate)
  linear_rate  uniform grid on the rate exp(time_decay)
  linear_tau   uniform grid on the time constant exp(-time_decay)

Each grid runs from the vector's smallest to its largest value. It reports the
per-channel rate change r = rate_q / rate, the share of channels on the level
nearest zero rate (linear_rate) or nearest zero time constant (linear_tau),
and WikiText-2 perplexity over NB_MAX_WINDOWS windows of NB_WINDOW tokens.

Usage (CPU):
    NB_MODEL=RWKV/rwkv-4-169m-pile python benchmarks/rwkv_decay_grid.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch


def quant_uniform(x: np.ndarray, bits: int):
    lo, hi = float(x.min()), float(x.max())
    step = max((hi - lo) / max(2**bits - 1, 1), 1e-30)
    idx = np.round((x - lo) / step)
    return idx * step + lo, idx.astype(np.int64)


def quantize(td: np.ndarray, basis: str, bits: int):
    """Return (quantized time_decay, mask of channels on the grid's extreme level)."""
    td = td.astype(np.float64)
    if basis == "log":
        q, _ = quant_uniform(td, bits)
        return q, None
    if basis == "linear_rate":
        q, idx = quant_uniform(np.exp(td), bits)
        return np.log(np.maximum(q, 1e-30)), idx == 0
    if basis == "linear_tau":
        q, idx = quant_uniform(np.exp(-td), bits)
        return -np.log(np.maximum(q, 1e-30)), idx == 0
    raise ValueError(basis)


def summ(v: np.ndarray) -> dict:
    q = np.percentile(v, [1, 10, 50, 90, 99])
    return {
        "p1": float(q[0]),
        "p10": float(q[1]),
        "median": float(q[2]),
        "p90": float(q[3]),
        "p99": float(q[4]),
        "frac_slower_2x": float((v < 0.5).mean()),
        "frac_slower_10x": float((v < 0.1).mean()),
        "frac_faster_2x": float((v > 2.0).mean()),
        "frac_faster_10x": float((v > 10.0).mean()),
        "frac_changed_2x": float((np.abs(np.log(v)) > np.log(2.0)).mean()),
    }


def census(orig: list[np.ndarray], basis: str, bits: int) -> dict:
    ratios, ext = [], []
    for td in orig:
        q, mask = quantize(td, basis, bits)
        ratios.append(np.exp(q - td.astype(np.float64)).ravel())
        if mask is not None:
            ext.append(float(mask.mean()))
    out = {"rate_ratio_all": summ(np.concatenate(ratios))}
    if ext:
        out["extreme_level_fraction_by_layer"] = {
            "min": min(ext),
            "median": float(np.median(ext)),
            "max": max(ext),
        }
    return out


def perplexity(model, ids, window: int, max_windows: int) -> float:
    nll = torch.zeros((), dtype=torch.float64)
    ntok = seen = 0
    for i in range(0, ids.size(1) - 1, window):
        chunk = ids[:, i : i + window]
        if chunk.size(1) < 2:
            break
        with torch.no_grad():
            loss = model(chunk, labels=chunk).loss.double()
        if not torch.isfinite(loss):
            return float("inf")
        nll += loss * (chunk.size(1) - 1)
        ntok += chunk.size(1) - 1
        seen += 1
        if seen >= max_windows:
            break
    return float(torch.exp(nll / max(ntok, 1)))


def main() -> int:
    model_name = os.environ.get("NB_MODEL", "RWKV/rwkv-4-169m-pile")
    bits = int(os.environ.get("NB_BITS", "3"))
    window = int(os.environ.get("NB_WINDOW", "1024"))
    max_windows = int(os.environ.get("NB_MAX_WINDOWS", "20"))
    out_dir = os.environ.get("NB_OUT", "benchmarks")
    torch.set_num_threads(int(os.environ.get("NB_THREADS", "6")))

    from datasets import load_dataset
    from transformers import AutoTokenizer, RwkvForCausalLM

    t0 = time.time()
    model = RwkvForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).eval()
    params = [b.attention.time_decay for b in model.rwkv.blocks]
    orig_t = [p.detach().clone() for p in params]
    orig = [p.float().cpu().numpy() for p in orig_t]
    rate = np.exp(np.concatenate([o.ravel() for o in orig]).astype(np.float64))
    print(
        f"[load] {model_name} in {time.time() - t0:.1f}s, {len(params)} blocks",
        flush=True,
    )

    results = {
        "model": model_name,
        "bits": bits,
        "n_blocks": len(params),
        "n_channels": int(rate.size),
        "time_decay_shape": list(orig[0].shape),
        "decay_rate": {
            "min": float(rate.min()),
            "p10": float(np.percentile(rate, 10)),
            "median": float(np.median(rate)),
            "p90": float(np.percentile(rate, 90)),
            "max": float(rate.max()),
        },
        "largest_gain_1_over_1_minus_w": float(1.0 / (1.0 - np.exp(-rate.min()))),
        "grids": {},
    }
    bases = ("log", "linear_rate", "linear_tau")
    for basis in bases:
        results["grids"][basis] = census(orig, basis, bits)
        g = results["grids"][basis]["rate_ratio_all"]
        print(
            f"[census] {basis:12s} median r={g['median']:.3g} changed>2x={g['frac_changed_2x']:.1%}",
            flush=True,
        )

    tok = AutoTokenizer.from_pretrained(model_name)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    results["ppl_baseline"] = perplexity(model, ids, window, max_windows)
    print(
        f"[ppl] baseline {results['ppl_baseline']:.4f} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    for basis in bases:
        for p, o in zip(params, orig):
            q, _ = quantize(o, basis, bits)
            p.data.copy_(torch.tensor(q, dtype=p.dtype).reshape(p.shape))
        ppl = perplexity(model, ids, window, max_windows)
        results["grids"][basis]["ppl"] = ppl if np.isfinite(ppl) else None
        print(f"[ppl] {basis} {ppl:.4g} ({time.time() - t0:.0f}s)", flush=True)
        for p, o in zip(params, orig_t):
            p.data.copy_(o)
    results["ppl_window"] = window
    results["ppl_max_windows"] = max_windows

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(
        out_dir, f"results_rwkv_decay_grid_{model_name.split('/')[-1]}.json"
    )
    with open(path, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"[saved] {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

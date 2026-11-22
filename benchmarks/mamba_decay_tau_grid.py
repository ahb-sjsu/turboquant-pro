# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""A linear decay grid built to keep the slow channels (experiment E6b).

mamba_decay_grid_census.py showed that a uniform 3-bit grid on the continuous
decay A = -exp(A_log) puts almost every channel on the level nearest zero, and
that a per-row grid on A still collapses the model. Both are linear in the
decay RATE |A|, whose range is set by the fastest channels, so the slow
channels fall below one step.

This tests the linear grid that resolves the slow channels instead: a uniform
grid on the TIME CONSTANT tau = 1/|A| = exp(-A_log). Its range is set by the
slowest channels, so they keep their rate, and the fast channels (small tau)
share the lowest level. Two variants at the same bit width:

  linear_tau          one uniform grid on tau per layer tensor
  linear_tau_per_row  a uniform grid on tau per row of the tensor

For each it reports the per-channel decay-rate change r = A_q / A (r > 1 means
the channel became faster), the share of channels on the lowest tau level, the
same for the slowest tenth of channels, and WikiText-2 perplexity with the
harness and windows of mamba_decay_grid_census.py. If the model survives, the
earlier collapse is a statement about grids that are linear in the rate, not
about linear grids in general.

Usage (CPU):
    NB_MODEL=state-spaces/mamba-790m-hf python benchmarks/mamba_decay_tau_grid.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mamba_decay_grid_census import perplexity, quant_uniform  # noqa: E402


def quantize_tau(alog: np.ndarray, per_row: bool, bits: int):
    """Uniform grid on tau = exp(-A_log). Returns (quantized A_log, lowest-level mask)."""
    tau = np.exp(-alog.astype(np.float64))
    tau_q, idx = quant_uniform(tau, bits, axis=1 if per_row else None)
    return -np.log(np.maximum(tau_q, 1e-30)), idx == 0


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


def census(orig: list[np.ndarray], per_row: bool, bits: int) -> dict:
    ratios, slow, low_layer, n_low = [], [], [], 0
    for alog in orig:
        alog_q, low = quantize_tau(alog, per_row, bits)
        r = np.exp(alog_q - alog.astype(np.float64))  # A_q / A
        ratios.append(r.ravel())
        rate = np.exp(alog).ravel()
        slow.append(r.ravel()[rate <= np.percentile(rate, 10)])
        low_layer.append(float(low.mean()))
        n_low += int(low.sum())
    r = np.concatenate(ratios)
    return {
        "rate_ratio_all": summ(r),
        "rate_ratio_slowest_tenth": summ(np.concatenate(slow)),
        "lowest_tau_level_fraction": n_low / r.size,
        "lowest_tau_level_fraction_by_layer": {
            "min": min(low_layer),
            "median": float(np.median(low_layer)),
            "max": max(low_layer),
        },
    }


def main() -> int:
    model_name = os.environ.get("NB_MODEL", "state-spaces/mamba-790m-hf")
    bits = int(os.environ.get("NB_BITS", "3"))
    window = int(os.environ.get("NB_WINDOW", "2048"))
    max_windows = int(os.environ.get("NB_MAX_WINDOWS", "40"))
    out_dir = os.environ.get("NB_OUT", "benchmarks")
    torch.set_num_threads(int(os.environ.get("NB_THREADS", "8")))

    from datasets import load_dataset
    from transformers import AutoTokenizer, MambaForCausalLM

    t0 = time.time()
    model = MambaForCausalLM.from_pretrained(model_name, dtype=torch.float32).eval()
    backbone = getattr(model, "backbone", None) or model.model
    mixers = [ly.mixer for ly in backbone.layers if hasattr(ly.mixer, "A_log")]
    orig_t = [m.A_log.detach().clone() for m in mixers]
    orig = [o.float().cpu().numpy() for o in orig_t]
    print(f"[load] {model_name} in {time.time() - t0:.1f}s, {len(mixers)} layers")

    results = {
        "model": model_name,
        "bits": bits,
        "n_layers": len(mixers),
        "n_channels": int(sum(o.size for o in orig)),
        "grid": "uniform on tau = exp(-A_log), over [min, max]",
        "grids": {},
    }
    variants = {"linear_tau": False, "linear_tau_per_row": True}
    for name, per_row in variants.items():
        results["grids"][name] = census(orig, per_row, bits)
        g = results["grids"][name]
        print(
            f"[census] {name:20s} median r={g['rate_ratio_all']['median']:.3g}"
            f" changed>2x={g['rate_ratio_all']['frac_changed_2x']:.1%}"
            f" lowest-level={g['lowest_tau_level_fraction']:.1%}"
        )

    tok = AutoTokenizer.from_pretrained(model_name)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    results["ppl_baseline"] = perplexity(model, ids, "cpu", window, max_windows)
    print(f"[ppl] baseline {results['ppl_baseline']:.4f}")
    for name, per_row in variants.items():
        for m, o in zip(mixers, orig):
            alog_q, _ = quantize_tau(o, per_row, bits)
            m.A_log.data.copy_(torch.tensor(alog_q, dtype=m.A_log.dtype))
        ppl = perplexity(model, ids, "cpu", window, max_windows)
        results["grids"][name]["ppl"] = ppl if np.isfinite(ppl) else None
        print(f"[ppl] {name} {ppl:.4g}")
        for m, o in zip(mixers, orig_t):
            m.A_log.data.copy_(o)
    results["ppl_window"] = window
    results["ppl_max_windows"] = max_windows

    os.makedirs(out_dir, exist_ok=True)
    tag = model_name.split("/")[-1]
    path = os.path.join(out_dir, f"results_mamba_decay_tau_grid_{tag}.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"[saved] {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

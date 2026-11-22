# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""What the 3-bit decay grids do to each Mamba channel (experiment E6).

validate_mamba_decay.py found that a per-tensor linear 3-bit grid on the
continuous decay A = -exp(A_log) collapses Mamba-790m (WikiText-2 perplexity
~1e10) while a grid on A_log stays near baseline. Every level of that linear
grid lies between the tensor's min and max of A, both negative, so no channel
is mapped to A = 0. The open question is whether slow channels (|A| small)
crowd onto the level nearest zero and have their decay rate changed by a large
factor, which would make the collapse mostly a stability effect of that grid.

For each grid this reports, per channel, the decay-rate change r = A_q / A
(both negative, so r > 0; r < 1 means the channel became slower), the share of
channels on the linear grid's top level, and the same for the slowest tenth of
channels. Grids at the same bit width:

  linear          one uniform grid on A per layer tensor (the earlier baseline)
  log_tau         one uniform grid on A_log per layer tensor (the earlier fix)
  linear_per_row  a uniform grid on A per row (one d_inner channel's states),
                  a linear grid that resolves each row's own range

With NB_PPL=1 it also measures WikiText-2 perplexity for the baseline and each
grid with the same harness and windows as validate_mamba_decay.py.

Usage:
    NB_PPL=1 python benchmarks/mamba_decay_grid_census.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch


def quant_uniform(x: np.ndarray, bits: int, axis=None):
    """Uniform fake-quant to 2**bits levels over [min, max], per tensor
    (axis=None) or per row (axis=1). Returns (dequantized, level index)."""
    lo = x.min(axis=axis, keepdims=axis is not None)
    hi = x.max(axis=axis, keepdims=axis is not None)
    step = (hi - lo) / max(2**bits - 1, 1)
    step = np.maximum(step, 1e-30)
    idx = np.round((x - lo) / step)
    return idx * step + lo, idx.astype(np.int64)


def perplexity(model, ids, device, window=2048, max_windows=40) -> float:
    """Same windows and accounting as validate_mamba_decay.py."""
    nll = torch.zeros((), dtype=torch.float64)
    ntok = 0
    seen = 0
    for i in range(0, ids.size(1) - 1, window):
        chunk = ids[:, i : i + window].to(device)
        if chunk.size(1) < 2:
            break
        with torch.no_grad():
            loss = model(chunk, labels=chunk).loss.double().cpu()
        if not torch.isfinite(loss):
            return float("inf")
        toks = chunk.size(1) - 1
        nll += loss * toks
        ntok += toks
        seen += 1
        if seen >= max_windows:
            break
    return float(torch.exp(nll / max(ntok, 1)))


def quantize_alog(alog: np.ndarray, basis: str, bits: int):
    """Return (quantized A_log, top-level mask or None) for one layer tensor."""
    a = -np.exp(alog)
    if basis == "log_tau":
        alog_q, _ = quant_uniform(alog, bits)
        return alog_q, None
    axis = 1 if basis == "linear_per_row" else None
    a_q, idx = quant_uniform(a, bits, axis=axis)
    top = idx == (2**bits - 1)
    return np.log(np.maximum(-a_q, 1e-9)), top


def census(orig: list[np.ndarray], basis: str, bits: int) -> dict:
    ratios, slow_ratios, top_frac_layer, n_top = [], [], [], 0
    for alog in orig:
        a = -np.exp(alog)
        alog_q, top = quantize_alog(alog, basis, bits)
        r = (-np.exp(alog_q)) / a
        ratios.append(r.ravel())
        rate = np.abs(a).ravel()
        slow = rate <= np.percentile(rate, 10)
        slow_ratios.append(r.ravel()[slow])
        if top is not None:
            top_frac_layer.append(float(top.mean()))
            n_top += int(top.sum())
    r = np.concatenate(ratios)
    rs = np.concatenate(slow_ratios)

    def summ(v):
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
            "frac_changed_2x": float((np.abs(np.log(v)) > np.log(2.0)).mean()),
        }

    out = {"rate_ratio_all": summ(r), "rate_ratio_slowest_tenth": summ(rs)}
    if top_frac_layer:
        out["top_level_fraction"] = n_top / r.size
        out["top_level_fraction_by_layer"] = {
            "min": min(top_frac_layer),
            "median": float(np.median(top_frac_layer)),
            "max": max(top_frac_layer),
        }
    return out


def main() -> int:
    model_name = os.environ.get("NB_MODEL", "state-spaces/mamba-790m-hf")
    bits = int(os.environ.get("NB_BITS", "3"))
    do_ppl = os.environ.get("NB_PPL", "0") == "1"
    window = int(os.environ.get("NB_WINDOW", "2048"))
    max_windows = int(os.environ.get("NB_MAX_WINDOWS", "40"))
    out_dir = os.environ.get("NB_OUT", "benchmarks")
    device = "cuda" if (do_ppl and torch.cuda.is_available()) else "cpu"

    thermal = None
    try:
        from batch_probe import ThermalController

        thermal = ThermalController(target_temp=80.0, max_threads=8, min_threads=2)
        thermal.start()
        torch.set_num_threads(thermal.get_threads())
    except Exception as e:
        print(f"[thermal] controller unavailable ({e}); capping at 4 threads")
        torch.set_num_threads(4)

    from transformers import AutoTokenizer, MambaForCausalLM

    t0 = time.time()
    model = MambaForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.to(device).eval()
    backbone = getattr(model, "backbone", None) or model.model
    mixers = [ly.mixer for ly in backbone.layers if hasattr(ly.mixer, "A_log")]
    orig_t = [m.A_log.detach().clone() for m in mixers]
    orig = [o.float().cpu().numpy() for o in orig_t]
    print(
        f"[load] {model_name} on {device} in {time.time() - t0:.1f}s, {len(mixers)} layers"
    )

    alog_all = np.concatenate([o.ravel() for o in orig])
    rate = np.exp(alog_all)
    per_tensor = []
    for o in orig:
        a = np.exp(o)
        per_tensor.append(
            {
                "rate_min": float(a.min()),
                "rate_median": float(np.median(a)),
                "rate_max": float(a.max()),
                "linear_step": float((a.max() - a.min()) / (2**bits - 1)),
            }
        )
    results = {
        "model": model_name,
        "bits": bits,
        "n_layers": len(mixers),
        "n_channels": int(alog_all.size),
        "a_log_shape": list(orig[0].shape),
        "rate_abs_A": {
            "min": float(rate.min()),
            "p10": float(np.percentile(rate, 10)),
            "median": float(np.median(rate)),
            "p90": float(np.percentile(rate, 90)),
            "max": float(rate.max()),
        },
        "linear_step_over_median_rate": {
            "min": min(t["linear_step"] / t["rate_median"] for t in per_tensor),
            "median": float(
                np.median([t["linear_step"] / t["rate_median"] for t in per_tensor])
            ),
            "max": max(t["linear_step"] / t["rate_median"] for t in per_tensor),
        },
        "grids": {},
    }
    for basis in ("linear", "log_tau", "linear_per_row"):
        results["grids"][basis] = census(orig, basis, bits)
        g = results["grids"][basis]
        print(
            f"[census] {basis:15s} median r={g['rate_ratio_all']['median']:.3g}"
            f" changed>2x={g['rate_ratio_all']['frac_changed_2x']:.1%}"
            f" slowest-tenth median r={g['rate_ratio_slowest_tenth']['median']:.3g}"
            + (
                f" top-level={g['top_level_fraction']:.1%}"
                if "top_level_fraction" in g
                else ""
            )
        )

    if do_ppl:
        from datasets import load_dataset

        tok = AutoTokenizer.from_pretrained(model_name)
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids
        results["ppl_baseline"] = perplexity(model, ids, device, window, max_windows)
        print(f"[ppl] baseline {results['ppl_baseline']:.4f}")
        for basis in ("linear", "log_tau", "linear_per_row"):
            if thermal is not None:
                torch.set_num_threads(thermal.get_threads())
            for m, o in zip(mixers, orig):
                alog_q, _ = quantize_alog(o, basis, bits)
                m.A_log.data.copy_(
                    torch.tensor(alog_q, dtype=m.A_log.dtype, device=m.A_log.device)
                )
            ppl = perplexity(model, ids, device, window, max_windows)
            results["grids"][basis]["ppl"] = ppl if np.isfinite(ppl) else None
            print(f"[ppl] {basis} {ppl:.4g}")
            for m, o in zip(mixers, orig_t):
                m.A_log.data.copy_(o)
        results["ppl_window"] = window
        results["ppl_max_windows"] = max_windows

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results_mamba_decay_census.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, allow_nan=False)
    print(f"[saved] {path}")
    if thermal is not None:
        try:
            thermal.stop()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    raise SystemExit(main())

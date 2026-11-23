# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""The grid uniform in the decay rate, with exact endpoints (experiment E6d).

The earlier linear grid on A = -exp(A_log) computed each level as
`lo + idx * step`. Its top level, the one nearest zero, is then a large
negative offset plus a multiple of the step, and it cancels to a value whose
sign is decided by rounding. In some layers it lands at or above zero, where
the harness clamps the rate to 1e-9. So part of the reported collapse could
belong to the clamp and not to the grid.

Here the levels are counted down from the slow end in 64-bit arithmetic,
`hi - (n - 1 - idx) * step`, so the top level is exactly the tensor's (or the
row's) slowest rate, every level is strictly negative, and nothing is clamped.
The level index of every channel is the same as in exact arithmetic.

Arms, at NB_BITS (default 3):
  rate_safe          one range per layer tensor
  rate_safe_per_row  one range per row

For each arm the record holds the census (share of channels on the top level,
rate ratios, how many levels the earlier implementation clamps) and, unless
NB_CENSUS_ONLY=1, the loss of every window with a perplexity and a paired
ratio to the baseline. The baseline's window losses are read from the record
of mamba_decay_intervals.py when it exists, which uses the same windows.

Usage (CPU):
    NB_MODEL=state-spaces/mamba-790m-hf python benchmarks/mamba_decay_rate_safe.py
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
from mamba_decay_grid_census import quant_uniform  # noqa: E402
from mamba_decay_intervals import ppl, window_losses  # noqa: E402
from mamba_decay_tau_grid import summ  # noqa: E402

FLOOR = 1e-9  # the earlier harness's clamp on -A


def quantize_rate_safe(alog: np.ndarray, per_row: bool, bits: int):
    """Uniform grid on A with exact endpoints. Returns (A_q, level index)."""
    a = -np.exp(alog.astype(np.float64))
    axis = 1 if per_row else None
    lo = a.min(axis=axis, keepdims=per_row)
    hi = a.max(axis=axis, keepdims=per_row)
    n = 2**bits
    step = (hi - lo) / (n - 1)
    idx = np.clip(np.round((a - lo) / np.maximum(step, 1e-300)), 0, n - 1)
    a_q = hi - (n - 1 - idx) * step
    a_q = np.where(idx == n - 1, hi, a_q)
    a_q = np.where(idx == 0, lo, a_q)
    return a_q, idx.astype(np.int64)


def earlier_clamped(alog: np.ndarray, per_row: bool, bits: int, dtype) -> tuple:
    """What `lo + idx * step` does in the given dtype: (channels at or above
    zero, whether the tensor has any)."""
    a = -np.exp(alog.astype(dtype))
    a_q, _ = quant_uniform(a, bits, axis=1 if per_row else None)
    bad = -a_q <= FLOOR
    return int(bad.sum()), bool(bad.any())


def main() -> int:
    model_name = os.environ.get("NB_MODEL", "state-spaces/mamba-790m-hf")
    bits = int(os.environ.get("NB_BITS", "3"))
    window = int(os.environ.get("NB_WINDOW", "2048"))
    max_windows = int(os.environ.get("NB_MAX_WINDOWS", "40"))
    out_dir = os.environ.get("NB_OUT", "benchmarks")
    census_only = os.environ.get("NB_CENSUS_ONLY", "0") == "1"
    torch.set_num_threads(int(os.environ.get("NB_THREADS", "8")))
    tag = model_name.split("/")[-1]

    from transformers import AutoTokenizer, MambaForCausalLM

    t0 = time.time()
    model = MambaForCausalLM.from_pretrained(model_name, dtype=torch.float32).eval()
    backbone = getattr(model, "backbone", None) or model.model
    mixers = [ly.mixer for ly in backbone.layers if hasattr(ly.mixer, "A_log")]
    orig_t = [m.A_log.detach().clone() for m in mixers]
    orig = [o.float().cpu().numpy() for o in orig_t]
    print(f"[load] {model_name} in {time.time() - t0:.1f}s", flush=True)

    results = {
        "model": model_name,
        "bits": bits,
        "n_layers": len(mixers),
        "n_channels": int(sum(o.size for o in orig)),
        "levels": "hi - (n - 1 - idx) * step in float64; top level is exactly the"
        " slowest rate of the tensor or row; no clamp is applied",
        "earlier_floor": FLOOR,
        "ppl_window": window,
        "ppl_max_windows": max_windows,
        "arms": {},
    }
    quant = {}
    for arm, per_row in (("rate_safe", False), ("rate_safe_per_row", True)):
        ratios, top, n_nonneg, qs = [], 0, 0, []
        clamp = {"float32": [0, 0], "float64": [0, 0]}
        top_over_median = []
        for o in orig:
            a_q, idx = quantize_rate_safe(o, per_row, bits)
            a = -np.exp(o.astype(np.float64))
            n_nonneg += int((a_q >= 0).sum())
            top += int((idx == 2**bits - 1).sum())
            ratios.append((a_q / a).ravel())
            top_over_median.append(float(np.abs(a).min() / np.median(np.abs(a))))
            qs.append(np.log(-a_q))
            for name, dt in (("float32", np.float32), ("float64", np.float64)):
                c, anyc = earlier_clamped(o, per_row, bits, dt)
                clamp[name][0] += c
                clamp[name][1] += int(anyc)
        assert n_nonneg == 0, n_nonneg
        r = np.concatenate(ratios)
        results["arms"][arm] = {
            "frac_top_level": top / r.size,
            "levels_at_or_above_zero": n_nonneg,
            "rate_ratio_all": summ(r),
            "frac_slowed_10x": float((r < 0.1).mean()),
            "frac_slowed_100x": float((r < 0.01).mean()),
            "frac_slowed_1000x": float((r < 0.001).mean()),
            "slowest_rate_over_median_rate_per_tensor": {
                "min": min(top_over_median),
                "median": float(np.median(top_over_median)),
                "max": max(top_over_median),
            },
            "earlier_lo_plus_idx_step": {
                k: {"channels_clamped": v[0], "layers_with_a_clamp": v[1]}
                for k, v in clamp.items()
            },
        }
        quant[arm] = qs
        a_ = results["arms"][arm]
        print(
            f"[census] {arm:18s} top={a_['frac_top_level']:.4f}"
            f" slowed>10x={a_['frac_slowed_10x']:.4f}"
            f" >1000x={a_['frac_slowed_1000x']:.4f}"
            f" median r={a_['rate_ratio_all']['median']:.3g}"
            f" earlier clamp f32={clamp['float32']} f64={clamp['float64']}",
            flush=True,
        )

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"results_mamba_decay_rate_safe_{tag}.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=1)
    if census_only:
        print(f"[saved] {path} (census only)")
        return 0

    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(model_name)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids

    base_path = os.path.join(out_dir, f"results_mamba_decay_intervals_{tag}.json")
    if os.path.exists(base_path):
        with open(base_path) as fh:
            b = json.load(fh)["baseline"]
        base_l, base_t = b["window_loss"], b["window_tokens"]
        results["baseline_source"] = os.path.basename(base_path)
    else:
        base_l, base_t = window_losses(model, ids, window, max_windows)
        results["baseline_source"] = "this run"
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(base_l), size=(2000, len(base_l)))
    b_boot = ppl(base_l, base_t, idx)
    results["baseline"] = {"ppl": ppl(base_l, base_t), "window_loss": base_l}
    print(f"[ppl] baseline {results['baseline']['ppl']:.4f}", flush=True)

    for arm in ("rate_safe", "rate_safe_per_row"):
        for m, q in zip(mixers, quant[arm]):
            m.A_log.data.copy_(torch.tensor(q, dtype=m.A_log.dtype))
        losses, toks = window_losses(model, ids, window, max_windows)
        for m, o in zip(mixers, orig_t):
            m.A_log.data.copy_(o)
        assert toks == base_t
        boot = ppl(losses, toks, idx)
        ratio = boot / b_boot
        a_ = results["arms"][arm]
        a_.update(
            {
                "ppl": ppl(losses, toks),
                "ppl_ci95": [
                    float(np.percentile(boot, 2.5)),
                    float(np.percentile(boot, 97.5)),
                ],
                "ratio_to_baseline": ppl(losses, toks) / results["baseline"]["ppl"],
                "ratio_ci95": [
                    float(np.percentile(ratio, 2.5)),
                    float(np.percentile(ratio, 97.5)),
                ],
                "window_loss": losses,
            }
        )
        print(
            f"[ppl] {arm:18s} {a_['ppl']:.6g} ratio {a_['ratio_to_baseline']:.6g}"
            f" [{a_['ratio_ci95'][0]:.6g}, {a_['ratio_ci95'][1]:.6g}]",
            flush=True,
        )
        with open(path, "w") as fh:
            json.dump(results, fh, indent=1)
    print(f"[saved] {path} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

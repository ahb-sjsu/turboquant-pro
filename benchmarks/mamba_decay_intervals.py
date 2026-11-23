# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Decay grids with a loss per window, for intervals (experiment E6c).

The decay study reported one perplexity per grid. The quantizers are
deterministic, so there is no seed to vary; the uncertainty is in which text
the perplexity is measured on. This reruns the grids that do not collapse and
keeps the loss of every window, so that a perplexity and its ratio to the
baseline get a bootstrap interval over windows (paired: the same resampled
windows for the grid and the baseline).

Arms, all at NB_BITS (default 3):
  log_tau             uniform grid on A_log, one range per layer tensor
  log_tau_per_row     uniform grid on A_log, one range per row (a new arm)
  linear_tau          uniform grid on tau = exp(-A_log), one range per tensor
  linear_tau_per_row  uniform grid on tau, one range per row

The two grids uniform in the rate collapse the model to perplexities above
1e6 and are not rerun. Windows and accounting are those of
mamba_decay_grid_census.py, so the perplexities must reproduce its records.

Usage (CPU):
    NB_MODEL=state-spaces/mamba-790m-hf python benchmarks/mamba_decay_intervals.py
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
from mamba_decay_tau_grid import quantize_tau, summ  # noqa: E402


def quantize(alog: np.ndarray, arm: str, bits: int) -> np.ndarray:
    per_row = arm.endswith("_per_row")
    if arm.startswith("log_tau"):
        q, _ = quant_uniform(alog.astype(np.float64), bits, axis=1 if per_row else None)
        return q
    q, _ = quantize_tau(alog, per_row, bits)
    return q


def window_losses(model, ids, window: int, max_windows: int):
    """Mean loss and token count of each window, as mamba_decay_grid_census.perplexity
    walks them."""
    losses, toks = [], []
    for i in range(0, ids.size(1) - 1, window):
        chunk = ids[:, i : i + window]
        if chunk.size(1) < 2:
            break
        with torch.no_grad():
            loss = float(model(chunk, labels=chunk).loss.double())
        losses.append(loss)
        toks.append(chunk.size(1) - 1)
        if len(losses) >= max_windows:
            break
    return losses, toks


def ppl(losses, toks, idx=None):
    v = np.asarray(losses, dtype=np.float64)
    w = np.asarray(toks, dtype=np.float64)
    if idx is None:
        return float(np.exp((v * w).sum() / w.sum()))
    return np.exp((v[idx] * w[idx]).sum(axis=1) / w[idx].sum(axis=1))


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
    print(f"[load] {model_name} in {time.time() - t0:.1f}s", flush=True)

    tok = AutoTokenizer.from_pretrained(model_name)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids

    arms = ["log_tau", "log_tau_per_row", "linear_tau", "linear_tau_per_row"]
    results = {
        "model": model_name,
        "bits": bits,
        "n_layers": len(mixers),
        "n_channels": int(sum(o.size for o in orig)),
        "row_length": int(orig[0].shape[1]),
        "ppl_window": window,
        "ppl_max_windows": max_windows,
        "interval": "95% percentile bootstrap over windows, 2000 resamples;"
        " ratios are paired with the baseline",
        "arms": {},
    }
    base_l, base_t = window_losses(model, ids, window, max_windows)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(base_l), size=(2000, len(base_l)))
    b_boot = ppl(base_l, base_t, idx)
    results["baseline"] = {
        "ppl": ppl(base_l, base_t),
        "ppl_ci95": [
            float(np.percentile(b_boot, 2.5)),
            float(np.percentile(b_boot, 97.5)),
        ],
        "window_loss": base_l,
        "window_tokens": base_t,
    }
    print(f"[ppl] baseline {results['baseline']['ppl']:.4f}", flush=True)
    for arm in arms:
        ratios = []
        for m, o in zip(mixers, orig):
            q = quantize(o, arm, bits)
            ratios.append(np.exp(q - o.astype(np.float64)).ravel())
            m.A_log.data.copy_(torch.tensor(q, dtype=m.A_log.dtype))
        losses, toks = window_losses(model, ids, window, max_windows)
        for m, o in zip(mixers, orig_t):
            m.A_log.data.copy_(o)
        assert toks == base_t
        boot = ppl(losses, toks, idx)
        r = boot / b_boot
        results["arms"][arm] = {
            "ppl": ppl(losses, toks),
            "ppl_ci95": [
                float(np.percentile(boot, 2.5)),
                float(np.percentile(boot, 97.5)),
            ],
            "ratio_to_baseline": ppl(losses, toks) / results["baseline"]["ppl"],
            "ratio_ci95": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))],
            "rate_ratio_all": summ(np.concatenate(ratios)),
            "window_loss": losses,
        }
        a = results["arms"][arm]
        print(
            f"[ppl] {arm:18s} {a['ppl']:.4f} ratio {a['ratio_to_baseline']:.4f}"
            f" [{a['ratio_ci95'][0]:.4f}, {a['ratio_ci95'][1]:.4f}]",
            flush=True,
        )
        tag = model_name.split("/")[-1]
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"results_mamba_decay_intervals_{tag}.json")
        with open(
            path, "w"
        ) as fh:  # rewritten after every arm, so a stop loses one arm
            json.dump(results, fh, indent=1)
    print(f"[saved] {path} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""A second quantizer for the 3-bit key result and the row-protection ranking (E8).

Every matched-bit result so far uses one quantizer: a symmetric uniform grid per
output channel whose scale is the row's largest absolute entry (absmax). At 3
bits that scale is set by one outlier per row, so the 3-bit findings could be
properties of absmax and not of the projections. This reruns the two 3-bit
findings with a row scale chosen to minimize the row's squared error:

    scale = c * absmax / qmax,  c from a grid on [0.25, 1], per row

The grid contains c = 1, so no row has a larger squared error than under absmax.

Part 1, projections: each of q_proj, k_proj, v_proj, o_proj is quantized alone,
in every layer, at NB_PROJ_BITS (default 4 and 3), under both quantizers, on the
same held-out sequences. Readout: mean KL(p_base || p_q), per-sequence KL, the
ratio mean(Q, K) / mean(V, O) with a paired sequence bootstrap.

Part 2, rows: k_row_protection_controls.py is run unchanged except for the
quantizer, so its conditions, budget, splits and seeds are those of E2.

Usage (GPU):
    NB_MODEL=unsloth/Llama-3.2-3B NB_OUT=/path python k_quantizer_control.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time

import exp_io
import k_row_protection_controls as rp
import numpy as np
import torch

PROJ = ["q_proj", "k_proj", "v_proj", "o_proj"]
C_GRID = [0.25 + 0.025 * i for i in range(31)]  # 0.25, 0.275, ..., 1.0
LAST_C: list[float] = []  # mean chosen c of each call, for the record


def quantize_mse(W: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-row symmetric uniform grid with the clipping scale that minimizes the
    row's squared error over C_GRID."""
    qmax = 2 ** (bits - 1) - 1
    amax = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    best_q = None
    best_err = None
    best_c = torch.ones_like(amax)
    for c in C_GRID:
        scale = c * amax / qmax
        q = torch.clamp(torch.round(W / scale), -qmax - 1, qmax) * scale
        err = ((W - q) ** 2).sum(dim=1, keepdim=True)
        if best_q is None:
            best_q, best_err = q, err
            best_c = torch.full_like(amax, c)
        else:
            better = err < best_err
            best_q = torch.where(better, q, best_q)
            best_err = torch.where(better, err, best_err)
            best_c = torch.where(better, torch.full_like(amax, c), best_c)
    LAST_C.append(float(best_c.mean()))
    return best_q


QUANT = {"absmax": rp.quantize_per_outchannel, "mse_clip": quantize_mse}


def ratio_ci(seq: dict, n_pos, rng, n_boot=2000):
    """mean(Q, K) / mean(V, O) of position-weighted KL, paired over sequences."""
    w = np.asarray(n_pos, dtype=np.float64)
    a = {p: np.asarray(seq[p], dtype=np.float64) for p in PROJ}
    idx = rng.integers(0, len(w), size=(n_boot, len(w)))

    def m(p, ix=None):
        if ix is None:
            return (a[p] * w).sum() / w.sum()
        return (a[p][ix] * w[ix]).sum(axis=1) / w[ix].sum(axis=1)

    point = (m("q_proj") + m("k_proj")) / (m("v_proj") + m("o_proj"))
    boot = (m("q_proj", idx) + m("k_proj", idx)) / (m("v_proj", idx) + m("o_proj", idx))
    return float(point), [
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    ]


def projections(model_name: str, bits_list: list[int], n_eval: int, n_seq: int) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = exp_io.model_and_tokenizer(model_name, torch.float16, device)
    evalset = exp_io.token_batch(tok, "validation", n_eval, n_seq)
    base = rp.seq_logprobs(model, evalset, device, "cpu")
    mods = {p: [m for m in model.modules() if hasattr(m, p)] for p in PROJ}
    rng = np.random.default_rng(0)
    out = {}
    for qname, qfn in QUANT.items():
        out[qname] = {}
        for bits in bits_list:
            res, seq = {}, {}
            for p in PROJ:
                layers = [getattr(m, p) for m in mods[p]]
                saved = [ly.weight.data.detach().cpu().clone() for ly in layers]
                LAST_C.clear()
                rel = []
                for ly, w0 in zip(layers, saved):
                    wf = w0.float()
                    wq = qfn(wf, bits)
                    rel.append(float((wf - wq).norm() / (wf.norm() + 1e-30)))
                    ly.weight.data.copy_(wq.half().to(ly.weight.device))
                r = rp.compare(model, evalset, base, device)
                for ly, w0 in zip(layers, saved):
                    ly.weight.data.copy_(w0.to(ly.weight.device))
                r["kl_ci95"] = rp.boot_mean(r["seq_kl"], r["n_pos"], rng)
                r["rel_fro_mean"] = float(np.mean(rel))
                if qname == "mse_clip":
                    r["mean_clip_c"] = float(np.mean(LAST_C))
                res[p] = r
                seq[p] = r["seq_kl"]
                print(
                    f"  [{qname} b{bits}] {p} kl={r['kl']:.4f} flip={r['flip']:.4f}"
                    f" relfro={r['rel_fro_mean']:.4f}"
                )
            point, ci = ratio_ci(seq, res["q_proj"]["n_pos"], rng)
            res["qk_over_vo"] = point
            res["qk_over_vo_ci95"] = ci
            print(f"  [{qname} b{bits}] QK/VO = {point:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
            out[qname][str(bits)] = res
    del model, base
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def main() -> int:
    model_name = os.environ.get("NB_MODEL", "unsloth/Llama-3.2-3B")
    out_dir = os.environ.get("NB_OUT", "experiments/results_matched_bit")
    bits_list = [int(b) for b in os.environ.get("NB_PROJ_BITS", "4,3").split(",")]
    n_eval = int(os.environ.get("NB_EVAL", "64"))
    n_seq = int(os.environ.get("NB_SEQ", "256"))
    torch.set_num_threads(exp_io.fallback_threads())
    tag = model_name.split("/")[-1]
    os.makedirs(out_dir, exist_ok=True)
    quant_note = {
        "name": "mse_clip",
        "rule": "per row, scale = c * absmax / qmax with c minimizing the row's"
        " squared error",
        "c_grid": C_GRID,
    }

    t0 = time.time()
    print(f"[part 1] projections, {model_name}, bits {bits_list}")
    proj = projections(model_name, bits_list, n_eval, n_seq)
    path = os.path.join(out_dir, f"quantizer_control_projections_{tag}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "eval_split": "wikitext-2-raw-v1 validation",
                "n_eval_seqs": n_eval,
                "window": n_seq,
                "quantizers": {
                    "absmax": "per row, scale = absmax / qmax",
                    "mse_clip": quant_note,
                },
                "torch": torch.__version__,
                "results": proj,
            },
            f,
            indent=1,
        )
    print(f"[saved] {path} ({time.time() - t0:.0f}s)")

    print("[part 2] row protection under the error-minimizing scale")
    rp.quantize_per_outchannel = quantize_mse
    sub = os.path.join(out_dir, "_mse_clip_tmp")
    os.environ["NB_OUT"] = sub
    rc = rp.main()
    src = os.path.join(sub, f"k_row_protection_{tag}.json")
    with open(src) as f:
        rec = json.load(f)
    rec["quantizer"] = quant_note
    path = os.path.join(out_dir, f"k_row_protection_mse_clip_{tag}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    os.remove(src)
    os.rmdir(sub)
    print(f"[saved] {path} ({time.time() - t0:.0f}s)")
    return rc


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    raise SystemExit(main())

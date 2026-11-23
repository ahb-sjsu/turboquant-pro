# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Does the projection ordering survive a quantizer that compensates? (E9)

Every matched-bit result so far rounds each weight to its nearest level. GPTQ
(Frantar et al., 2023) instead rounds one input column at a time and spreads
each column's error over the columns not yet rounded, so as to minimize the
error of the layer's output on calibration inputs. This repeats the
one-projection-at-a-time measurement with GPTQ in place of rounding.

Arms, for each bit width in NB_PROJ_BITS (default 4 and 3):

    grid      absmax    per-row scale = absmax / qmax (the paper's quantizer)
              mse_clip  per-row scale = c * absmax / qmax, c minimizing the
                        row's squared error (k_quantizer_control.py)
    rounding  rtn       round to nearest (reproduces the earlier records)
              gptq      GPTQ on that grid, with the row scales fixed from the
                        original weights

Each of q_proj, k_proj, v_proj, o_proj is quantized alone, in every layer. GPTQ
is sequential: the Hessian of layer l is collected with layers before l already
quantized. Calibration inputs are WikiText-2 train sequences; evaluation is on
the validation sequences of the earlier records, so the rtn arms must reproduce
quantizer_control_projections_<model>.json.

Readout per projection: mean KL(p_base || p_q) with a bootstrap interval,
per-sequence KL, the relative Frobenius error of the weights, and the relative
squared error of the layer's output on the calibration inputs,
tr(dW H dW^T) / tr(W H W^T), for GPTQ and for rounding on the same inputs.
Then the ratio mean(Q, K) / mean(V, O) with a paired sequence bootstrap.

Usage (GPU):
    NB_MODEL=unsloth/Llama-3.2-3B NB_OUT=/path python k_gptq_control.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import exp_io
import k_quantizer_control as kq
import k_row_protection_controls as rp
import numpy as np
import torch

PROJ = kq.PROJ
BLOCK = 128
PERCDAMP = 0.01


class _Stop(Exception):
    """Raised by the Hessian hook to end a forward pass at the hooked layer."""


def row_scale(W: torch.Tensor, bits: int, grid: str) -> torch.Tensor:
    """The per-row scale of each grid, from the original weights. Shape (rows, 1)."""
    qmax = 2 ** (bits - 1) - 1
    amax = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    if grid == "absmax":
        return amax / qmax
    best_err, best_c = None, torch.ones_like(amax)
    for c in kq.C_GRID:
        scale = c * amax / qmax
        q = torch.clamp(torch.round(W / scale), -qmax - 1, qmax) * scale
        err = ((W - q) ** 2).sum(dim=1, keepdim=True)
        if best_err is None:
            best_err, best_c = err, torch.full_like(amax, c)
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_c = torch.where(better, torch.full_like(amax, c), best_c)
    return best_c * amax / qmax


def rtn(W: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    return torch.clamp(torch.round(W / scale), -qmax - 1, qmax) * scale


def gptq(W: torch.Tensor, H: torch.Tensor, scale: torch.Tensor, bits: int):
    """GPTQ on a fixed per-row grid. W is (out, in), H is (in, in), float64."""
    qmax = 2 ** (bits - 1) - 1
    W = W.clone()
    H = H.clone()
    n = W.shape[1]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H += (
        PERCDAMP
        * torch.mean(torch.diag(H))
        * torch.eye(n, dtype=H.dtype, device=H.device)
    )
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W)
    s = scale.squeeze(1)
    for i1 in range(0, n, BLOCK):
        i2 = min(i1 + BLOCK, n)
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        H1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            q = torch.clamp(torch.round(w / s), -qmax - 1, qmax) * s
            Q[:, i1 + i] = q
            err = (w - q) / H1[i, i]
            W1[:, i:] -= err.unsqueeze(1) * H1[i, i:].unsqueeze(0)
            E1[:, i] = err
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q


def output_err(W0: torch.Tensor, Wq: torch.Tensor, H: torch.Tensor) -> float:
    """Relative squared error of the layer's output on the calibration inputs."""
    d = W0 - Wq
    return float(((d @ H) * d).sum() / ((W0 @ H) * W0).sum().clamp_min(1e-300))


def hessian(model, layer, calib, device) -> torch.Tensor:
    """Sum of x x^T over the calibration tokens entering `layer`, float64."""
    n = layer.weight.shape[1]
    H = torch.zeros((n, n), dtype=torch.float64, device=device)
    count = [0]

    def hook(_mod, args):
        x = args[0].detach().reshape(-1, n).double()
        H.add_(x.T @ x)
        count[0] += x.shape[0]
        raise _Stop

    h = layer.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for ids in calib:
                try:
                    model(ids.to(device).unsqueeze(0))
                except _Stop:
                    pass
    finally:
        h.remove()
    return H * (2.0 / max(count[0], 1))


def main() -> int:
    model_name = os.environ.get("NB_MODEL", "unsloth/Llama-3.2-3B")
    out_dir = os.environ.get("NB_OUT", "experiments/results_matched_bit")
    bits_list = [int(b) for b in os.environ.get("NB_PROJ_BITS", "4,3").split(",")]
    n_eval = int(os.environ.get("NB_EVAL", "64"))
    n_calib = int(os.environ.get("NB_CALIB", "64"))
    n_seq = int(os.environ.get("NB_SEQ", "256"))
    torch.set_num_threads(exp_io.fallback_threads())
    tag = model_name.split("/")[-1]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"gptq_control_projections_{tag}.json")

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = exp_io.model_and_tokenizer(model_name, torch.float16, device)
    evalset = exp_io.token_batch(tok, "validation", n_eval, n_seq)
    calib = exp_io.token_batch(tok, "train", n_calib, n_seq)
    base = rp.seq_logprobs(model, evalset, device, "cpu")
    mods = {p: [m for m in model.modules() if hasattr(m, p)] for p in PROJ}
    rng = np.random.default_rng(0)
    out = {
        "model": model_name,
        "eval_split": "wikitext-2-raw-v1 validation",
        "calib_split": "wikitext-2-raw-v1 train",
        "n_eval_seqs": len(evalset),
        "n_calib_seqs": len(calib),
        "n_calib_tokens": int(sum(int(c.numel()) for c in calib)),
        "window": n_seq,
        "gptq": {
            "blocksize": BLOCK,
            "percdamp": PERCDAMP,
            "sequential": True,
            "row_scale": "fixed from the original weights, per grid",
            "arithmetic": "float64",
        },
        "c_grid": kq.C_GRID,
        "torch": torch.__version__,
        "results": {},
    }
    for grid in ("absmax", "mse_clip"):
        for method in ("rtn", "gptq"):
            arm = f"{method}_{grid}"
            out["results"][arm] = {}
            for bits in bits_list:
                res, seq = {}, {}
                for p in PROJ:
                    layers = [getattr(m, p) for m in mods[p]]
                    saved = [ly.weight.data.detach().cpu().clone() for ly in layers]
                    rel, oerr, oerr_rtn = [], [], []
                    for ly, w0 in zip(layers, saved):
                        wf = w0.to(device).double()
                        scale = row_scale(wf, bits, grid)
                        if method == "rtn":
                            wq = rtn(wf, scale, bits)
                        else:
                            H = hessian(model, ly, calib, device)
                            wq = gptq(wf, H, scale, bits)
                            oerr.append(output_err(wf, wq, H))
                            oerr_rtn.append(output_err(wf, rtn(wf, scale, bits), H))
                            del H
                        rel.append(float((wf - wq).norm() / (wf.norm() + 1e-30)))
                        ly.weight.data.copy_(wq.to(w0.dtype))
                        del wf, wq, scale
                    r = rp.compare(model, evalset, base, device)
                    for ly, w0 in zip(layers, saved):
                        ly.weight.data.copy_(w0.to(device))
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    r["kl_ci95"] = rp.boot_mean(r["seq_kl"], r["n_pos"], rng)
                    r["rel_fro_mean"] = float(np.mean(rel))
                    if method == "gptq":
                        r["output_err_mean"] = float(np.mean(oerr))
                        r["output_err_rtn_same_inputs_mean"] = float(np.mean(oerr_rtn))
                    res[p] = r
                    seq[p] = r["seq_kl"]
                    extra = (
                        f" outerr={r['output_err_mean']:.4f}"
                        f" (rtn {r['output_err_rtn_same_inputs_mean']:.4f})"
                        if method == "gptq"
                        else ""
                    )
                    print(
                        f"  [{arm} b{bits}] {p} kl={r['kl']:.4f} flip={r['flip']:.4f}"
                        f" relfro={r['rel_fro_mean']:.4f}{extra}",
                        flush=True,
                    )
                point, ci = kq.ratio_ci(seq, res["q_proj"]["n_pos"], rng)
                res["qk_over_vo"] = point
                res["qk_over_vo_ci95"] = ci
                print(
                    f"  [{arm} b{bits}] QK/VO = {point:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]",
                    flush=True,
                )
                out["results"][arm][str(bits)] = res
                with open(path, "w") as f:  # rewritten after every arm and bit width
                    json.dump(out, f, indent=1)
    print(f"[saved] {path} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    raise SystemExit(main())

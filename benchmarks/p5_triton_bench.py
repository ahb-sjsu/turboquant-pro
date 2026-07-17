# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""P5 Triton port: exactness + latency vs the CuPy RawKernel oracle.

The P5 exit criterion (docs/DESIGN_hardware_and_plugins.md): the Triton M4/M2
kernels must be **exact** vs the RawKernel oracle on NRP V100/A100 and reach
**perf parity or better on A100**. This script, run on a CUDA GPU:

  1. builds a per-channel-key cache with a realistic cold length;
  2. checks max |error| of the Triton per-page and batched-page decode vs the
     NumPy reference (the oracle) -- must be ~fp32 noise;
  3. times, per decode step: Triton per-page (loop over pages), Triton batched
     (one launch), the CuPy RawKernel path when cupy is importable, and a torch
     decompress-then-attend baseline.

Usage (needs torch+triton; cupy optional):  python benchmarks/p5_triton_bench.py
"""

from __future__ import annotations

import json
import time

import numpy as np

from turboquant_pro.core import TurboQuantKVCache
from turboquant_pro.kv_triton import (
    has_triton,
    pck_batched_partials_triton,
    pck_block_partials_triton,
)

H, D = 8, 128
CTXS = [2048, 8192, 32768]


def _build(ctx, seed=0):
    cache = TurboQuantKVCache(
        head_dim=D, n_heads=H, bits=4, use_gpu=False, seed=seed,
        per_channel_keys=True, key_nf4_asym=True, key_outlier_frac=0.02,
        hot_window=512,
    )
    rng = np.random.default_rng(seed)
    off = rng.uniform(-4, 4, size=(H, D)).astype(np.float32)
    for _ in range(ctx):
        cache.append(
            (off + rng.standard_normal((H, D))).astype(np.float32),
            rng.standard_normal((H, D)).astype(np.float32),
        )
    q = rng.standard_normal((H, D)).astype(np.float32)
    return cache, q


def _reference(cache, q):
    k = np.asarray(cache.get_keys(0, cache.cold_length))[0]
    v = np.asarray(cache.get_values(0, cache.cold_length))[0]
    sc = np.einsum("hd,hsd->hs", q, k) / np.sqrt(D)
    p = np.exp(sc - sc.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return np.einsum("hs,hsd->hd", p, v)


def _time(fn, iters=30):
    import torch

    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def bench_ctx(ctx):
    import torch

    cache, q = _build(ctx)
    tq, scale = cache._tq, 1.0 / np.sqrt(D)
    blocks = cache._prepared_pck_blocks()
    want = _reference(cache, q)

    # per-page triton: loop pages + merge (matches core.fused_decode structure)
    from turboquant_pro.kv_fused import merge_partials

    def per_page():
        parts = [pck_block_partials_triton(q, b, tq, scale) for b in blocks]
        return merge_partials(parts, torch)

    def batched():
        m, lsum, acc = pck_batched_partials_triton(q, blocks, tq, scale)
        return acc / torch.clamp(lsum, min=1e-30)[:, None]

    out_pp = per_page().detach().cpu().numpy()
    out_b = batched().detach().cpu().numpy()
    err_pp = float(np.abs(out_pp - want).max())
    err_b = float(np.abs(out_b - want).max())

    rec = {
        "ctx": ctx, "pages": len(blocks),
        "err_per_page": err_pp, "err_batched": err_b,
        "ms_per_page": round(_time(per_page), 3),
        "ms_batched": round(_time(batched), 3),
    }

    # CuPy RawKernel oracle timing, if available (the perf-parity baseline)
    try:
        import cupy  # noqa: F401

        cg = TurboQuantKVCache(
            head_dim=D, n_heads=H, bits=4, use_gpu=True, seed=0,
            per_channel_keys=True, key_nf4_asym=True, key_outlier_frac=0.02,
            hot_window=512,
        )
        rng = np.random.default_rng(0)
        off = rng.uniform(-4, 4, size=(H, D)).astype(np.float32)
        for _ in range(ctx):
            cg.append((off + rng.standard_normal((H, D))).astype(np.float32),
                      rng.standard_normal((H, D)).astype(np.float32))
        cg.fused_decode(q)  # warm/build prepared pages
        rec["ms_cupy_raw"] = round(_time(lambda: cg.fused_decode(q)), 3)
    except Exception as e:  # noqa: BLE001
        rec["ms_cupy_raw"] = None
        rec["cupy_note"] = f"{type(e).__name__}: {e}"

    print(
        f"[ctx {ctx}] pages={rec['pages']} "
        f"err(pp/batch)={err_pp:.2e}/{err_b:.2e} "
        f"ms pp={rec['ms_per_page']} batched={rec['ms_batched']} "
        f"cupy={rec['ms_cupy_raw']}",
        flush=True,
    )
    return rec


if __name__ == "__main__":
    import torch

    if not has_triton():
        raise SystemExit("needs torch+triton on a CUDA GPU")
    print(f"[gpu] {torch.cuda.get_device_name(0)}", flush=True)
    results = [bench_ctx(c) for c in CTXS]
    print("=== JSON ===")
    print(json.dumps(results))

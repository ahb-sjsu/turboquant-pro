# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Long-context M4 fused-decode bench for 141GB-class GPUs (H200).

Extends the measured 2k-32k curve (2.0x/5.6x/12.5x on GV100) into the
64k-128k regime no other GPU in our fleet can hold: steady-state
TurboQuantKVCache.fused_decode (prepared per-page blocks + CUDA kernels)
vs the decompress-then-attend fallback, plus memory stats.

Usage (needs cupy):  pip install cupy-cuda12x
    NB_CTX=65536,131072 python benchmarks/h200_longcontext_bench.py
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from turboquant_pro.core import TurboQuantKVCache

CTXS = [int(c) for c in os.environ.get("NB_CTX", "65536,131072").split(",")]
H, D = 8, 128
rng = np.random.default_rng(0)


def bench_ctx(S: int) -> dict:
    import cupy as cp

    cache = TurboQuantKVCache(
        head_dim=D,
        n_heads=H,
        bits=4,
        use_gpu=True,
        seed=0,
        per_channel_keys=True,
        key_nf4_asym=True,
        key_outlier_frac=0.02,
        hot_window=512,
    )
    off = rng.uniform(-4, 4, size=(H, D)).astype(np.float32)
    t0 = time.perf_counter()
    for s in range(S):
        k = (off + rng.standard_normal((H, D))).astype(np.float32)
        v = rng.standard_normal((H, D)).astype(np.float32)
        cache.append(k, v)
        if (s + 1) % 16384 == 0:
            print(f"  [fill] {s + 1}/{S}", flush=True)
    fill_s = time.perf_counter() - t0
    q = rng.standard_normal((H, D)).astype(np.float32)
    dev = cp.cuda.Device()

    t0 = time.perf_counter()
    cache.fused_decode(q)  # first call builds the prepared pages
    dev.synchronize()
    t_first = time.perf_counter() - t0

    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        out = cache.fused_decode(q)
        dev.synchronize()
        ts.append(time.perf_counter() - t0)
    t_steady = float(np.median(ts))

    cache._pck_fused_ok = False
    cache._pck_blocks.clear()
    tr = []
    for _ in range(3):
        t0 = time.perf_counter()
        out_rec = cache.fused_decode(q)
        dev.synchronize()
        tr.append(time.perf_counter() - t0)
    t_rec = float(np.median(tr))

    err = float(
        np.abs(cp.asnumpy(cp.asarray(out)) - cp.asnumpy(cp.asarray(out_rec))).max()
    )
    mem = cache.memory_stats()
    rec = {
        "ctx": S,
        "pages": len(cache._cold_keys),
        "fill_s": round(fill_s, 1),
        "first_ms": round(t_first * 1e3, 2),
        "steady_ms": round(t_steady * 1e3, 2),
        "reconstruct_ms": round(t_rec * 1e3, 2),
        "speedup": round(t_rec / t_steady, 1),
        "kv_ratio": round(mem["effective_ratio"], 2),
        "max_err": err,
    }
    print(
        f"[ctx {S}] pages={rec['pages']} steady={rec['steady_ms']}ms "
        f"reconstruct={rec['reconstruct_ms']}ms speedup={rec['speedup']}x "
        f"ratio={rec['kv_ratio']}x err={err:.2e}",
        flush=True,
    )
    del cache
    cp.get_default_memory_pool().free_all_blocks()
    return rec


if __name__ == "__main__":
    import cupy as cp

    print(
        f"[gpu] {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}", flush=True
    )
    results = [bench_ctx(s) for s in CTXS]
    print("=== JSON ===")
    print(json.dumps(results))

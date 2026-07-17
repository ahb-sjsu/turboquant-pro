# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Isolate the sparse fp16-outlier cost in the M4 fused decode.

Sweeps ``outlier_frac`` (which scales the token-major CSR of fp16 score deltas)
while holding everything else fixed, and times steady-state
``TurboQuantKVCache.fused_decode``. A flat curve as the outlier count grows means
the sparse pass is latency-hidden. See ``benchmarks/RESULTS_outlier_latency.md``.

Usage (needs cupy + a CUDA GPU):  python benchmarks/bench_outlier_latency.py
"""

from __future__ import annotations

import time

import numpy as np

from turboquant_pro.core import TurboQuantKVCache

H, D, CTX = 8, 128, 32768
FRACS = [0.0, 0.02, 0.05, 0.1]


def main() -> None:
    import cupy as cp

    rng = np.random.default_rng(0)
    off = rng.uniform(-4, 4, (H, D)).astype(np.float32)

    def build(frac):
        c = TurboQuantKVCache(
            head_dim=D,
            n_heads=H,
            bits=4,
            use_gpu=True,
            seed=0,
            per_channel_keys=True,
            key_nf4_asym=True,
            key_outlier_frac=frac,
            hot_window=512,
        )
        for _ in range(CTX):
            c.append(
                (off + rng.standard_normal((H, D))).astype(np.float32),
                rng.standard_normal((H, D)).astype(np.float32),
            )
        return c

    q = rng.standard_normal((H, D)).astype(np.float32)
    dev = cp.cuda.Device()
    print(
        f"# {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}  H={H} D={D} ctx={CTX}"
    )
    base = None
    for frac in FRACS:
        c = build(frac)
        c.fused_decode(q)
        dev.synchronize()  # warm / build prepared pages
        ts = []
        for _ in range(30):
            t0 = time.perf_counter()
            c.fused_decode(q)
            dev.synchronize()
            ts.append(time.perf_counter() - t0)
        ms = float(np.median(ts)) * 1e3
        nnz = sum(int(b.deltas.shape[0]) for b in c._prepared_pck_blocks())
        base = ms if base is None else base
        d = ms - base
        print(
            f"  outlier_frac={frac:<5} steady={ms:.3f}ms  d_vs0={d:+.3f}ms "
            f"({100 * d / base:+.1f}%)  nnz={nnz}"
        )
        del c
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()

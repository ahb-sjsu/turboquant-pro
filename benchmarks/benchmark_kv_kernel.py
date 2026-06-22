#!/usr/bin/env python3
"""M1: validate the fused-decode CUDA kernel vs the M0 reference + measure latency.

Confirms the single-head CUDA kernel (online softmax + code-space V) matches the
NumPy/CuPy reference (kv_fused) to fp tolerance, and times it against the CuPy
array-level reference and the dequant path.
"""

import argparse
import time

import numpy as np

from turboquant_pro import TurboQuantPGVector
from turboquant_pro.kv_fused import dequant_decode_attention, fused_decode_attention
from turboquant_pro.kv_kernel import fused_decode_cuda


def make_cache(heads, seq, d, bits, seed=0):
    tq = TurboQuantPGVector(dim=d, bits=bits)
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((heads, d)).astype(np.float32)

    def code(X):
        n = np.linalg.norm(X, axis=2)
        r = tq._rotate(X / np.maximum(n[..., None], 1e-30))
        return np.searchsorted(tq.boundaries, r).astype(np.uint8), n.astype(np.float32)

    kc, nk = code(rng.standard_normal((heads, seq, d)).astype(np.float32))
    vc, nv = code(rng.standard_normal((heads, seq, d)).astype(np.float32))
    return tq, q, kc, vc, nk, nv


def time_it(fn, sync, reps=50, warmup=10):
    for _ in range(warmup):
        fn()
    sync()
    best = 1e30
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync()
        best = min(best, time.perf_counter() - t)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--bits", type=int, default=4)
    a = ap.parse_args()
    import cupy as cp

    tq, q, kc, vc, nk, nv = make_cache(a.heads, a.seq, a.head_dim, a.bits)
    print(
        f"seq={a.seq} head_dim={a.head_dim} heads={a.heads} bits={a.bits}", flush=True
    )

    ref = fused_decode_attention(q, kc, vc, nk, nv, tq, xp=np)  # M0 reference
    for method in ("warp", "block"):
        out_k = cp.asnumpy(fused_decode_cuda(q, kc, vc, nk, nv, tq, method=method))
        rel = float(np.max(np.abs(out_k - ref)) / (np.max(np.abs(ref)) + 1e-9))
        print(f"[{method:5}] kernel vs M0 reference: {rel:.2e}", flush=True)

    qg, kcg, vcg = cp.asarray(q), cp.asarray(kc), cp.asarray(vc)
    nkg, nvg = cp.asarray(nk), cp.asarray(nv)
    sync = cp.cuda.runtime.deviceSynchronize
    t_w = time_it(
        lambda: fused_decode_cuda(qg, kcg, vcg, nkg, nvg, tq, method="warp"), sync
    )
    t_b = time_it(
        lambda: fused_decode_cuda(qg, kcg, vcg, nkg, nvg, tq, method="block"), sync
    )
    t_deq = time_it(
        lambda: dequant_decode_attention(qg, kcg, vcg, nkg, nvg, tq, xp=cp), sync
    )
    print(f"\n[GPU] M2 warp kernel : {t_w*1e3:7.3f} ms", flush=True)
    print(f"[GPU] M1 block kernel: {t_b*1e3:7.3f} ms", flush=True)
    print(f"[GPU] CuPy dequant   : {t_deq*1e3:7.3f} ms", flush=True)
    print(
        f"[GPU] warp vs dequant: {t_deq/t_w:.2f}x   warp vs block: {t_b/t_w:.1f}x",
        flush=True,
    )


if __name__ == "__main__":
    main()

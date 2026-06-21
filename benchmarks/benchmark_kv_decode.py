#!/usr/bin/env python3
"""M0: validate the full fused KV-decode step (CuPy/NumPy) vs the dequant path.

Confirms the code-space fused attention (scores + softmax + V weighted-sum, one
unrotate) equals the reconstruct-then-attend path -- on CPU and GPU -- and times the
GPU fused vs dequant paths. This is the reference the fused CUDA kernel (M1) targets.
"""

import argparse
import time

import numpy as np

from turboquant_pro import TurboQuantPGVector
from turboquant_pro.kv_fused import dequant_decode_attention, fused_decode_attention


def make_cache(heads, seq, d, bits, seed=0):
    tq = TurboQuantPGVector(dim=d, bits=bits)
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((heads, d)).astype(np.float32)
    Korig = rng.standard_normal((heads, seq, d)).astype(np.float32)
    Vorig = rng.standard_normal((heads, seq, d)).astype(np.float32)

    def code(X):
        n = np.linalg.norm(X, axis=2)
        r = tq._rotate(X / np.maximum(n[..., None], 1e-30))
        return np.searchsorted(tq.boundaries, r).astype(np.uint8), n.astype(np.float32)

    kc, nk = code(Korig)
    vc, nv = code(Vorig)
    return tq, q, kc, vc, nk, nv, Korig, Vorig


def fp32_attention(q, K, V, d):
    s = np.einsum("hd,hsd->hs", q, K) / np.sqrt(d)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    return np.einsum("hs,hsd->hd", p, V)


def time_it(fn, sync=None, reps=30, warmup=5):
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    best = 1e30
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        if sync:
            sync()
        best = min(best, time.perf_counter() - t)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--bits", type=int, default=3)
    a = ap.parse_args()
    d = a.head_dim
    tq, q, kc, vc, nk, nv, Korig, Vorig = make_cache(a.heads, a.seq, d, a.bits)
    print(f"seq={a.seq} head_dim={d} heads={a.heads} bits={a.bits}", flush=True)

    # CPU: fused == dequant (exact math), and the compression error vs fp32 originals
    out_f = fused_decode_attention(q, kc, vc, nk, nv, tq, xp=np)
    out_d = dequant_decode_attention(q, kc, vc, nk, nv, tq, xp=np)
    fused_vs_dequant = float(
        np.max(np.abs(out_f - out_d)) / (np.max(np.abs(out_d)) + 1e-9)
    )
    out_fp32 = fp32_attention(q, Korig, Vorig, d)
    compress_err = float(
        np.linalg.norm(out_f - out_fp32) / (np.linalg.norm(out_fp32) + 1e-9)
    )
    print(
        f"\n[CPU] fused vs dequant  : {fused_vs_dequant:.2e} (must be ~0; same math)",
        flush=True,
    )
    print(
        f"[CPU] fused vs fp32 attn: {compress_err:.4f} relative (the 3-bit KV error)",
        flush=True,
    )

    # GPU: validate + time
    try:
        import cupy as cp

        qg, kcg, vcg = cp.asarray(q), cp.asarray(kc), cp.asarray(vc)
        nkg, nvg = cp.asarray(nk), cp.asarray(nv)
        sync = cp.cuda.runtime.deviceSynchronize
        out_fg = fused_decode_attention(qg, kcg, vcg, nkg, nvg, tq, xp=cp)
        gpu_vs_cpu = float(
            cp.max(cp.abs(out_fg - cp.asarray(out_d)))
            / (cp.max(cp.abs(cp.asarray(out_d))) + 1e-9)
        )
        print(f"[GPU] fused vs CPU dequant: {gpu_vs_cpu:.2e}", flush=True)
        t_f = time_it(
            lambda: fused_decode_attention(qg, kcg, vcg, nkg, nvg, tq, xp=cp), sync
        )
        t_d = time_it(
            lambda: dequant_decode_attention(qg, kcg, vcg, nkg, nvg, tq, xp=cp), sync
        )
        print(f"\n[GPU] fused decode  : {t_f*1e3:7.3f} ms", flush=True)
        print(f"[GPU] dequant decode: {t_d*1e3:7.3f} ms", flush=True)
        print(
            f"[GPU] speedup (array-level CuPy; M1 raw kernel targets more): {t_d/t_f:.2f}x",
            flush=True,
        )
    except Exception as e:
        print(f"\nGPU path unavailable: {str(e)[:70]}", flush=True)


if __name__ == "__main__":
    main()

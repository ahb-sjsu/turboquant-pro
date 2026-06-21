#!/usr/bin/env python3
"""Fused KV-decode via ADC: compute attention scores without reconstructing K.

The #1 finding showed dequant dominates the decode step, and the cost is the
inverse-rotation matmul (S x dxd per token). But the attention score is

    q . K_recon = q . (norm * unrotate(cent[codes])) = norm * (rotate(q) . cent[codes])

so we never need to inverse-rotate or materialize K: rotate the query once, then
score every cached key by an asymmetric-distance product over the packed codes.
This benchmark compares ADC-attention vs the dequant+matmul path for correctness
and speed -- the fused-decode fix to the reviewer's critique.
"""

import argparse
import time

import numpy as np


def time_it(fn, reps=20, warmup=3):
    for _ in range(warmup):
        fn()
    best = 1e30
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--bits", type=int, default=3)
    a = ap.parse_args()
    from turboquant_pro import TurboQuantPGVector

    rng = np.random.default_rng(0)
    K = rng.standard_normal((a.heads, a.seq, a.head_dim)).astype(np.float32)
    q = rng.standard_normal((a.heads, a.head_dim)).astype(np.float32)
    print(
        f"seq={a.seq} head_dim={a.head_dim} heads={a.heads} bits={a.bits}", flush=True
    )

    tq = TurboQuantPGVector(dim=a.head_dim, bits=a.bits)
    cent = tq.centroids.astype(np.float32)
    # compress keys per head: rotate unit key, quantize -> codes (+ per-key norm)
    norms = np.linalg.norm(K, axis=2)  # (heads, seq)
    Kr = tq._rotate(K / np.maximum(norms[..., None], 1e-30))  # rotated unit keys
    codes = np.searchsorted(tq.boundaries, Kr).astype(
        np.uint8
    )  # (heads, seq, head_dim)
    q_rot = tq._rotate(q)  # rotate the query once (cheap, per head)

    def dequant_attention():
        # materialize K in fp32 (inverse-rotation + scale), then q . K^T
        Krec = tq._unrotate(cent[codes]) * norms[..., None]
        return np.einsum("hd,hsd->hs", q, Krec)

    def adc_attention():
        # score directly on codes: norm * (cent[codes] . rotate(q)) -- no inverse-rotation
        return norms * np.einsum("hsd,hd->hs", cent[codes], q_rot)

    s_deq = dequant_attention()
    s_adc = adc_attention()
    max_err = float(np.max(np.abs(s_deq - s_adc)) / (np.max(np.abs(s_deq)) + 1e-9))

    t_deq = time_it(dequant_attention)
    t_adc = time_it(adc_attention)
    print(
        f"\ncorrectness: max relative |adc - dequant| = {max_err:.2e} (exact ADC)",
        flush=True,
    )
    print(f"dequant+matmul attention : {t_deq*1e3:8.3f} ms", flush=True)
    print(f"ADC attention (fused)    : {t_adc*1e3:8.3f} ms", flush=True)
    print(
        f"speedup: {t_deq/t_adc:.1f}x  (skips the S x d^2 inverse-rotation)", flush=True
    )


if __name__ == "__main__":
    main()

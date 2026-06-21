#!/usr/bin/env python3
"""#1: How big is the KV dequantization overhead during decode?

The reviewer's concern: unpacking compressed keys back to BF16 each decode step adds
latency. We quantify it -- for one decode step over an S-token cache, time the
dequant pass (unpack 3-bit -> centroid lookup -> inverse-rotate -> scale) against the
attention matmul (q . K^T) it feeds, and report dequant as a fraction of the two.
Then show how the two-tier cache (fp16 hot window) shrinks it.
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
    ap.add_argument("--seq", type=int, default=8192)  # cached tokens
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--hot", type=int, default=512)  # fp16 hot window
    a = ap.parse_args()
    from turboquant_pro import TurboQuantKV

    rng = np.random.default_rng(0)
    # cached keys: (heads, seq, head_dim)
    K = rng.standard_normal((a.heads, a.seq, a.head_dim)).astype(np.float32)
    q = rng.standard_normal((a.heads, 1, a.head_dim)).astype(np.float32)
    print(
        f"seq={a.seq} head_dim={a.head_dim} heads={a.heads} bits={a.bits} hot={a.hot}",
        flush=True,
    )

    tq = TurboQuantKV(head_dim=a.head_dim, n_heads=a.heads, bits=a.bits, use_gpu=False)
    comp = tq.compress(K, packed=True)

    def dequant():
        return tq.decompress(comp)

    def attn(Kmat):
        # one decode step: q . K^T  -> softmax-ready scores
        return np.einsum("hod,hsd->hos", q, Kmat)

    Krec = np.asarray(dequant(), dtype=np.float32).reshape(a.heads, a.seq, a.head_dim)
    t_dq = time_it(dequant)
    t_at = time_it(lambda: attn(Krec))
    frac = t_dq / (t_dq + t_at)
    print(f"\nFULL compressed cache ({a.seq} tokens):", flush=True)
    print(f"  dequant   : {t_dq*1e3:8.3f} ms", flush=True)
    print(f"  attention : {t_at*1e3:8.3f} ms", flush=True)
    print(f"  dequant is {frac*100:.1f}% of (dequant+attention)", flush=True)

    # two-tier: only cold tokens (seq-hot) are compressed; hot stay fp16 (no dequant)
    cold = max(a.seq - a.hot, 0)
    if cold > 0:
        Kc = K[:, :cold, :].copy()
        compc = tq.compress(Kc, packed=True)
        t_dqc = time_it(lambda: tq.decompress(compc))
        t_atc = time_it(lambda: attn(K))  # attention over the whole cache regardless
        fracc = t_dqc / (t_dqc + t_atc)
        print(f"\nTWO-TIER (hot={a.hot} fp16, cold={cold} compressed):", flush=True)
        print(f"  dequant (cold only): {t_dqc*1e3:8.3f} ms", flush=True)
        print(f"  attention (full)   : {t_atc*1e3:8.3f} ms", flush=True)
        print(f"  dequant is {fracc*100:.1f}% of the decode step", flush=True)
    # GPU path (CuPy kernels) -- the production decode path
    try:
        import cupy as cp

        tqg = TurboQuantKV(
            head_dim=a.head_dim, n_heads=a.heads, bits=a.bits, use_gpu=True
        )
        compg = tqg.compress(K, packed=True)
        Kg, qg = cp.asarray(K), cp.asarray(q)

        def dequant_g():
            tqg.decompress(compg)
            cp.cuda.runtime.deviceSynchronize()

        def attn_g():
            cp.einsum("hod,hsd->hos", qg, Kg)
            cp.cuda.runtime.deviceSynchronize()

        t_dqg, t_atg = time_it(dequant_g), time_it(attn_g)
        print("\nGPU (CuPy kernels), full cache:", flush=True)
        print(f"  dequant   : {t_dqg*1e3:8.3f} ms", flush=True)
        print(f"  attention : {t_atg*1e3:8.3f} ms", flush=True)
        print(
            f"  dequant is {t_dqg/(t_dqg+t_atg)*100:.1f}% of (dequant+attention)",
            flush=True,
        )
    except Exception as e:
        print(f"\nGPU path unavailable: {str(e)[:70]}", flush=True)

    print(
        "\n=> The dequant is a separate memory-bound pass; a fused "
        "dequant-inside-attention kernel removes it (work item). Two-tier bounds "
        "the cost to the cold fraction.",
        flush=True,
    )


if __name__ == "__main__":
    main()

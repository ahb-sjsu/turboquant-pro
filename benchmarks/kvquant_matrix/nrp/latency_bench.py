#!/usr/bin/env python3
"""Track B: decode-latency + peak-GPU-memory bench.

Compares one decode-step attention over a length-S KV cache, swept over
``(seq_len, n_heads)``:

  * **fp16**    -- torch ``scaled_dot_product_attention`` on fp16 K/V (the baseline
                   a served model actually runs). Peak mem = ``torch.cuda.max_memory_allocated``.
  * **warp**    -- the M2 warp fused CUDA kernel (``kv_kernel.fused_decode_cuda``,
                   ``method='warp'``) over tq-pro K/V *codes* (no dequant).
  * **dequant** -- reconstruct codes -> fp32, then standard attention
                   (``kv_fused.dequant_decode_attention``).

The fused/dequant paths run on **CuPy** (the RawKernel path needs CuPy + bundled
nvrtc), so their memory is reported from the CuPy default memory pool
(``used_bytes`` working set), NOT ``torch.cuda.max_memory_allocated`` -- the two
allocators are independent. This is stated per-row in the JSON so the numbers are
not silently mixed.

Writes a JSON list to ``--out`` (default ``$CACHE/results/latency/latency.json``).
Mirrors the cache-construction of ``benchmarks/benchmark_kv_kernel.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


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
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--seqs", default="1024,4096,8192,16384")
    ap.add_argument("--heads", default="8,32")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument(
        "--out",
        default=os.environ.get(
            "LAT_OUT",
            os.path.join(os.environ.get("CACHE", "/cache"), "results", "latency", "latency.json"),
        ),
    )
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F

    import cupy as cp

    from turboquant_pro import TurboQuantPGVector
    from turboquant_pro.kv_fused import dequant_decode_attention
    from turboquant_pro.kv_kernel import fused_decode_cuda

    dev = torch.device("cuda:0")
    gpu = torch.cuda.get_device_name(0)
    sync_cp = cp.cuda.runtime.deviceSynchronize
    sync_t = torch.cuda.synchronize
    pool = cp.get_default_memory_pool()
    rng = np.random.default_rng(0)
    d = a.head_dim
    seqs = [int(x) for x in a.seqs.split(",")]
    heads = [int(x) for x in a.heads.split(",")]

    print(f"[latency] gpu={gpu} head_dim={d} bits={a.bits} seqs={seqs} heads={heads}", flush=True)
    rows = []
    for H in heads:
        for S in seqs:
            tq = TurboQuantPGVector(dim=d, bits=a.bits)
            q = rng.standard_normal((H, d)).astype(np.float32)

            def code(X):
                n = np.linalg.norm(X, axis=2)
                r = tq._rotate(X / np.maximum(n[..., None], 1e-30))
                return np.searchsorted(tq.boundaries, r).astype(np.uint8), n.astype(np.float32)

            kc, nk = code(rng.standard_normal((H, S, d)).astype(np.float32))
            vc, nv = code(rng.standard_normal((H, S, d)).astype(np.float32))

            # ---- code-space CuPy paths (warp kernel + dequant) ----
            qg, kcg, vcg = cp.asarray(q), cp.asarray(kc), cp.asarray(vc)
            nkg, nvg = cp.asarray(nk), cp.asarray(nv)
            pool.free_all_blocks()
            t_w = time_it(
                lambda: fused_decode_cuda(qg, kcg, vcg, nkg, nvg, tq, method="warp"),
                sync_cp, reps=a.reps,
            )
            mem_w = int(pool.used_bytes())
            t_deq = time_it(
                lambda: dequant_decode_attention(qg, kcg, vcg, nkg, nvg, tq, xp=cp),
                sync_cp, reps=a.reps,
            )
            mem_deq = int(pool.used_bytes())

            # ---- fp16 baseline (torch SDPA over full fp16 K/V) ----
            qt = torch.randn(1, H, 1, d, device=dev, dtype=torch.float16)
            kt = torch.randn(1, H, S, d, device=dev, dtype=torch.float16)
            vt = torch.randn(1, H, S, d, device=dev, dtype=torch.float16)

            def fp16_attn():
                with torch.no_grad():
                    return F.scaled_dot_product_attention(qt, kt, vt)

            sync_t()
            torch.cuda.reset_peak_memory_stats()
            t_fp16 = time_it(fp16_attn, sync_t, reps=a.reps)
            mem_fp16 = int(torch.cuda.max_memory_allocated())
            del qt, kt, vt
            torch.cuda.empty_cache()

            row = {
                "seq_len": S,
                "n_heads": H,
                "head_dim": d,
                "bits": a.bits,
                "latency_ms": {
                    "fp16": round(t_fp16 * 1e3, 4),
                    "warp": round(t_w * 1e3, 4),
                    "dequant": round(t_deq * 1e3, 4),
                },
                "speedup_vs_fp16": {
                    "warp": round(t_fp16 / t_w, 3),
                    "dequant": round(t_fp16 / t_deq, 3),
                },
                "peak_mem_bytes": {
                    "fp16__torch_max_allocated": mem_fp16,
                    "warp__cupy_pool_used": mem_w,
                    "dequant__cupy_pool_used": mem_deq,
                },
            }
            rows.append(row)
            print(
                f"[latency] H={H:3d} S={S:6d} | fp16={t_fp16*1e3:8.3f}ms "
                f"warp={t_w*1e3:8.3f}ms deq={t_deq*1e3:8.3f}ms | "
                f"fp16_mem={mem_fp16/2**20:8.1f}MiB",
                flush=True,
            )

    out = {"gpu": gpu, "head_dim": d, "bits": a.bits, "results": rows}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[latency] wrote {a.out} ({len(rows)} points)", flush=True)


if __name__ == "__main__":
    main()

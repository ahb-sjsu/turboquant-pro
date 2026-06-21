#!/usr/bin/env python3
"""GPU ADC search throughput: turboquant-pro's native compressed search.

Measures whether `gpu_adc_search` (CuPy asymmetric-distance over compressed codes)
gives competitive query throughput vs the CPU flat-reconstruct path -- the #25
A+ question. Uses TurboQuantPGVector (full-dim scalar quant) compressed codes.

Run on Atlas GPU1:
  CUDA_VISIBLE_DEVICES=1 /home/claude/env/bin/python benchmark_gpu_adc.py \\
      --npy /tmp/labse_bench.npy --corpus 50000 --queries 500 --bits 3
"""

import argparse
import time

import numpy as np


def exact_topk(Q, C, k):
    s = Q @ C.T
    idx = np.argpartition(-s, k, axis=1)[:, :k]
    for i in range(len(Q)):
        idx[i] = idx[i][np.argsort(-s[i, idx[i]])]
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True)
    ap.add_argument("--corpus", type=int, default=50000)
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--bits", type=int, default=3)
    a = ap.parse_args()

    from turboquant_pro import TurboQuantPGVector, gpu_adc_search

    X = np.load(a.npy, mmap_mode="r").astype(np.float32)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
    Q = X[: a.queries].copy()
    C = X[a.queries : a.queries + a.corpus].copy()
    dim = X.shape[1]
    print(f"corpus={len(C)} dim={dim} queries={len(Q)} bits={a.bits}", flush=True)

    gt = exact_topk(Q, C, 10)
    tq = TurboQuantPGVector(dim=dim, bits=a.bits)
    t = time.time()
    compressed = tq.compress_batch(C)
    print(
        f"compressed {len(C)} in {time.time()-t:.1f}s; bytes/vec~{dim*a.bits/8:.0f}",
        flush=True,
    )

    gpu_adc_search(Q[0], compressed, tq, top_k=10)  # warmup (CUDA init)

    t = time.time()
    hits = 0
    for i, q in enumerate(Q):
        idx, _ = gpu_adc_search(q, compressed, tq, top_k=10)
        hits += len(set(gt[i].tolist()) & set(np.asarray(idx).ravel().tolist()))
    dt = time.time() - t
    qps = len(Q) / dt
    recall = hits / (len(Q) * 10)
    print(
        f"\nGPU-ADC: {qps:.1f} qps | recall@10={recall:.4f} | "
        f"comp~{dim*4/(dim*a.bits/8):.1f}x | {len(Q)} queries in {dt:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

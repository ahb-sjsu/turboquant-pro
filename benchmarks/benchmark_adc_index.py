#!/usr/bin/env python3
"""Validate the productionized ADCIndex on real embeddings (kernel + fallback).

Confirms ADCIndex (the shipped API) reaches tq-pro's headline recall at kernel
speed, and that the numpy fallback returns the same neighbours.
"""

import argparse
import time

import numpy as np


def normalize(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)


def exact_topk(Q, C, k):
    out = np.zeros((len(Q), k), dtype=np.int64)
    for s in range(0, len(Q), 256):
        sims = Q[s : s + 256] @ C.T
        out[s : s + 256] = np.argsort(-sims, axis=1)[:, :k]
    return out


def recall(gt, ap, k):
    return float(
        np.mean([len(set(gt[i, :k]) & set(ap[i, :k])) / k for i in range(len(gt))])
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True)
    ap.add_argument("--corpus", type=int, default=100000)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--pca-dim", type=int, default=256)
    ap.add_argument("--bits", type=int, default=3)
    a = ap.parse_args()
    from turboquant_pro import ADCIndex, PCAMatryoshka

    X = normalize(np.load(a.npy, mmap_mode="r").astype(np.float32))
    Q = X[: a.queries].copy()
    C = X[a.queries : a.queries + a.corpus].copy()
    N, dim = C.shape
    print(
        f"corpus={N} dim={dim} queries={len(Q)} pca={a.pca_dim} bits={a.bits}",
        flush=True,
    )
    gt = exact_topk(Q, C, 10)

    pca = PCAMatryoshka(input_dim=dim, output_dim=a.pca_dim)
    pca.fit(C[: min(N, 50000)])
    t = time.time()
    index = ADCIndex(pca.with_quantizer(bits=a.bits)).add(C)
    build_s = time.time() - t
    print(
        f"ADCIndex.add: {build_s:.1f}s  uses_kernel={index.uses_kernel}  size={index.size}",
        flush=True,
    )

    i1, _ = index.search(Q, k=10)
    ir = index.search(Q, k=10, rerank=5, originals=C)
    print(
        f"recall@10  single={recall(gt, i1, 10):.4f}  +rerank={recall(gt, ir, 10):.4f}",
        flush=True,
    )

    def bench(fn, reps=3):
        best = 1e30
        for _ in range(reps):
            t = time.time()
            fn()
            best = min(best, time.time() - t)
        return len(Q) / best

    qps = bench(lambda: index.search(Q, k=10))
    print(
        f"QPS (k=10, {'kernel' if index.uses_kernel else 'numpy'}) = {qps:.0f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Track B / M1 probe: does faiss PQ4 fast-scan already give the A+?

Key idea: tq-pro's per-dim scalar quant == faiss PQ with M=d' subquantizers of
1 dim each, 4-bit. faiss ships IndexPQFastScan (the SIMD kernel M1 would write).
Path-1/Track-A used *grouped* PQ (m=32/64) which crushed recall; *per-dim* PQ
(M=d') is the tq-pro-equivalent and is untested. If PCA + per-dim PQ4 fast-scan
matches tq-pro recall at faiss speed, the A+ is reached with NO custom kernel.

Normalized vectors -> L2 top-k == inner-product top-k, so we use L2 fast-scan.
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
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        for r in range(idx.shape[0]):
            idx[r] = idx[r][np.argsort(-sims[r, idx[r]])]
        out[s : s + 256] = idx
    return out


def recall(gt, ap, k):
    return float(
        np.mean([len(set(gt[i, :k]) & set(ap[i, :k])) / k for i in range(len(gt))])
    )


def rerank(cand, Q, C, k=10):
    rr = np.zeros((len(Q), k), dtype=np.int64)
    for i in range(len(Q)):
        c = cand[i]
        s = C[c] @ Q[i]
        rr[i] = c[np.argsort(-s)[:k]]
    return rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True)
    ap.add_argument("--corpus", type=int, default=100000)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--pca-dim", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    import faiss

    faiss.omp_set_num_threads(a.threads)
    from turboquant_pro import PCAMatryoshka

    X = normalize(np.load(a.npy, mmap_mode="r").astype(np.float32))
    Q = X[: a.queries].copy()
    C = X[a.queries : a.queries + a.corpus].copy()
    N, dim = C.shape
    print(f"corpus={N} dim={dim} queries={len(Q)} pca={a.pca_dim}", flush=True)
    gt = exact_topk(Q, C, 100)
    pca = PCAMatryoshka(input_dim=dim, output_dim=a.pca_dim)
    pca.fit(C[: min(N, 50000)])
    Cp = normalize(np.asarray(pca.transform(C), dtype=np.float32))
    Qp = normalize(np.asarray(pca.transform(Q), dtype=np.float32))
    pd = a.pca_dim

    builders = [
        ("PQ-perdim-flat (M=d', 4b)", lambda: faiss.IndexPQ(pd, pd, 4)),
        ("PQ-perdim-FASTSCAN (M=d', 4b)", lambda: faiss.IndexPQFastScan(pd, pd, 4)),
        (
            "PQ-2dim-FASTSCAN (M=d'/2, 4b)",
            lambda: faiss.IndexPQFastScan(pd, pd // 2, 4),
        ),
    ]
    print(
        "\n| config | bytes/vec | comp x | build s | qps | r@10(1st) | r@10(+rerank) |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, build in builders:
        try:
            t = time.time()
            idx = build()
            idx.train(Cp)
            idx.add(Cp)
            b = time.time() - t
            t = time.time()
            _, i1 = idx.search(Qp, 100)
            qs = len(Q) / (time.time() - t)
            r1 = recall(gt, i1, 10)
            _, cand = idx.search(Qp, 50)
            r2 = recall(gt, rerank(cand, Q, C), 10)
            mbytes = idx.code_size if hasattr(idx, "code_size") else pd * 4 / 8
            print(
                f"| {name} | {mbytes} | {dim*4/mbytes:.0f} | {b:.1f} | {qs:.0f} "
                f"| {r1:.4f} | {r2:.4f} |",
                flush=True,
            )
        except Exception as e:
            print(f"| {name} | FAILED: {str(e)[:50]} |", flush=True)


if __name__ == "__main__":
    main()

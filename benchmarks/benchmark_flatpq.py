#!/usr/bin/env python3
"""Path-2 Track A: PCA-Matryoshka + faiss FLAT IndexPQ (no IVF).

faiss IndexPQ does exhaustive SIMD ADC -- fast, and without IVF's coarse-quantizer
recall hit (Path-1's failure). Tests whether 'PCA reduction + fast flat PQ ADC'
gives competitive recall at faiss-fast speed -- a quick partial-A+ ingredient.
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

    print(
        "\n| config | bytes/vec | comp x | build s | qps | r@10(1st) | r@10(+rerank) |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for m in (32, 64):
        try:
            t = time.time()
            idx = faiss.IndexPQ(a.pca_dim, m, 8, faiss.METRIC_INNER_PRODUCT)
            idx.train(Cp)
            idx.add(Cp)
            b = time.time() - t
            t = time.time()
            _, i1 = idx.search(Qp, 100)
            qs = len(Q) / (time.time() - t)
            r1 = recall(gt, i1, 10)
            _, cand = idx.search(Qp, 50)
            r2 = recall(gt, rerank(cand, Q, C), 10)
            print(
                f"| PCA{a.pca_dim}+flatPQ(m={m}) | {m} | {dim*4/m:.0f} | {b:.1f} | "
                f"{qs:.0f} | {r1:.4f} | {r2:.4f} |",
                flush=True,
            )
        except Exception as e:
            print(f"m={m} FAILED: {e}", flush=True)


if __name__ == "__main__":
    main()

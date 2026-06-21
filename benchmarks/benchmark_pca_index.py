#!/usr/bin/env python3
"""Path 1: PCA-Matryoshka as a front-end to FAST ANN indexes.

Hypothesis: reducing embeddings with PCA-Matryoshka before a fast index (faiss
IVF-PQ, HNSW) makes that index *both faster and more accurate* than on raw
embeddings -- turning turboquant-pro's compression into the preprocessing that
improves any fast ANN system, and answering the query-speed weakness.

Compares, on the same corpus: {IVF-PQ, HNSW} x {raw 768-d, PCA-256-d}, recall@10
(single-stage and oversample+rerank with fp32 originals), qps, build time.

Run thermal-safe on Atlas:
  OMP_NUM_THREADS=8 python benchmark_pca_index.py --npy /tmp/labse_bench.npy \\
      --corpus 100000 --queries 1000 --pca-dim 256
"""

import argparse
import math
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
    ap.add_argument("--oversample", type=int, default=5)
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
    nlist = min(4096, max(64, int(8 * math.sqrt(N))))
    res = []

    def run(name, idx, Qd, build_s):
        t = time.time()
        _, i1 = idx.search(Qd, 100)
        qs1 = len(Q) / (time.time() - t)
        r1 = recall(gt, i1, 10)
        t = time.time()
        _, cand = idx.search(Qd, 10 * a.oversample)
        qs2 = len(Q) / (time.time() - t)
        r2 = recall(gt, rerank(cand, Q, C), 10)
        row = dict(
            method=name,
            build_s=round(build_s, 1),
            qps_1stage=round(qs1),
            r10_1stage=round(r1, 4),
            qps_rerank=round(qs2),
            r10_rerank=round(r2, 4),
        )
        res.append(row)
        print(row, flush=True)

    configs = [
        ("IVFPQ-raw768", C, Q, dim, 96),
        ("IVFPQ-pca256", Cp, Qp, a.pca_dim, 32),
        ("HNSW-raw768", C, Q, dim, None),
        ("HNSW-pca256", Cp, Qp, a.pca_dim, None),
    ]
    for label, Cd, Qd, d, m in configs:
        try:
            t = time.time()
            if m is not None:
                idx = faiss.index_factory(
                    d, f"IVF{nlist},PQ{m}", faiss.METRIC_INNER_PRODUCT
                )
                idx.train(Cd[: min(N, 100000)])
                idx.add(Cd)
                idx.nprobe = min(64, nlist)
            else:
                idx = faiss.IndexHNSWFlat(d, 32)
                idx.hnsw.efConstruction = 200
                idx.add(Cd)
                idx.hnsw.efSearch = 128
            run(label, idx, Qd, time.time() - t)
        except Exception as e:
            print(f"{label} FAILED: {e}", flush=True)

    print("\n| method | build s | qps(1st) | r@10(1st) | qps(rrk) | r@10(rrk) |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in res:
        print(
            f"| {r['method']} | {r['build_s']} | {r['qps_1stage']} | {r['r10_1stage']} "
            f"| {r['qps_rerank']} | {r['r10_rerank']} |"
        )


if __name__ == "__main__":
    main()

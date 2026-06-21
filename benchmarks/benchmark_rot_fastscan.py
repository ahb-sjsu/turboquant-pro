#!/usr/bin/env python3
"""M1 decisive test: does a rotation + faiss PQ4 fast-scan recover tq-pro recall?

per-dim PQ4 fast-scan is fast (8763 qps) but only 0.76 recall WITHOUT a rotation.
tq-pro and OPQ both apply a rotation that Gaussianizes/balances coordinates and
reach 0.999. Test: PCA -> random orthogonal rotation -> faiss IndexPQFastScan,
and faiss OPQ + PQ4 fast-scan. If recall jumps to ~0.95+ at fast-scan speed, the
A+ ('fast + compressed + high-recall') is reached by *reusing faiss's kernel* --
no custom kernel needed.
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
    pd = a.pca_dim
    print(f"corpus={N} dim={dim} queries={len(Q)} pca={pd}", flush=True)
    gt = exact_topk(Q, C, 100)

    pca = PCAMatryoshka(input_dim=dim, output_dim=pd)
    pca.fit(C[: min(N, 50000)])
    Cp = normalize(np.asarray(pca.transform(C), dtype=np.float32))
    Qp = normalize(np.asarray(pca.transform(Q), dtype=np.float32))
    # random orthogonal rotation (TurboQuant-style), seeded
    rng = np.random.default_rng(42)
    Rot, _ = np.linalg.qr(rng.standard_normal((pd, pd)))
    Rot = Rot.astype(np.float32)
    Cr = normalize((Cp @ Rot).astype(np.float32))
    Qr = normalize((Qp @ Rot).astype(np.float32))

    def evalu(name, idx, Cd, Qd):
        t = time.time()
        idx.train(Cd)
        idx.add(Cd)
        b = time.time() - t
        t = time.time()
        _, i1 = idx.search(Qd, 100)
        qs = len(Q) / (time.time() - t)
        r1 = recall(gt, i1, 10)
        _, cand = idx.search(Qd, 50)
        r2 = recall(gt, rerank(cand, Q, C), 10)
        cs = idx.code_size if hasattr(idx, "code_size") else pd // 2
        print(
            f"| {name} | {cs} | {dim*4/cs:.0f} | {b:.1f} | {qs:.0f} | {r1:.4f} | {r2:.4f} |",
            flush=True,
        )

    print(
        "\n| config | bytes/vec | comp x | build s | qps | r@10(1st) | r@10(+rerank) |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    evalu("PCA+rot+PQ4fs (M=d')", faiss.IndexPQFastScan(pd, pd, 4), Cr, Qr)
    for m in (96, 64):
        evalu(
            f"raw+OPQ{m}+PQ{m}x4fs", faiss.index_factory(dim, f"OPQ{m},PQ{m}x4fs"), C, Q
        )


if __name__ == "__main__":
    main()

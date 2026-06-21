#!/usr/bin/env python3
"""#10: standard public ANN benchmark -- GloVe-100-angular (ann-benchmarks).

Removes the "all results on private LaBSE" weakness: runs tq-pro (ADCIndex) against
PQ/OPQ at matched compression on the canonical GloVe-100-angular dataset, scored
against its *provided* ground-truth neighbors (top-100 cosine over the full 1.18M
corpus). Honest external benchmark.
"""

import argparse
import time

import numpy as np


def normalize(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)


def recall(gt, ap, k):
    return float(
        np.mean([len(set(gt[i, :k]) & set(ap[i, :k])) / k for i in range(len(gt))])
    )


def rerank(cand, Q, C, k=10):
    rr = np.full((len(Q), k), -1, dtype=np.int64)
    for i in range(len(Q)):
        c = cand[i][cand[i] >= 0]
        s = C[c] @ Q[i]
        top = c[np.argsort(-s)[:k]]
        rr[i, : len(top)] = top
    return rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    import faiss
    import h5py

    faiss.omp_set_num_threads(a.threads)
    from turboquant_pro import ADCIndex, PCAMatryoshka

    with h5py.File(a.hdf5, "r") as f:
        C = normalize(np.asarray(f["train"], dtype=np.float32))
        Qf = normalize(np.asarray(f["test"], dtype=np.float32))
        nbr = np.asarray(f["neighbors"], dtype=np.int64)
    nq = min(a.queries, len(Qf))
    Q, gt = Qf[:nq], nbr[:nq, :10]
    N, dim = C.shape
    print(f"GloVe: corpus={N} dim={dim} queries={nq} pca={a.pca_dim}", flush=True)

    def bench(fn, reps=2):
        best = 1e30
        for _ in range(reps):
            t = time.time()
            fn()
            best = min(best, time.time() - t)
        return nq / best

    print("\n| method | bytes/vec | comp x | r@10 (1-stage) | r@10 (+rerank) | qps |")
    print("|---|---:|---:|---:|---:|---:|")

    # PQ / OPQ at ~matched bytes
    for fac, m in [("PQ", 25), ("OPQ", 25)]:
        idx = faiss.index_factory(
            dim,
            f"{'OPQ'+str(m)+',' if fac=='OPQ' else ''}PQ{m}",
            faiss.METRIC_INNER_PRODUCT,
        )
        idx.train(C[: min(N, 200000)])
        idx.add(C)
        _, i1 = idx.search(Q, 10)
        _, cand = idx.search(Q, 50)
        q = bench(lambda idx=idx: idx.search(Q, 10))
        print(
            f"| {fac}(m={m}) | {m} | {dim*4/m:.0f} | {recall(gt, i1, 10):.4f} "
            f"| {recall(gt, rerank(cand, Q, C), 10):.4f} | {q:.0f} |",
            flush=True,
        )

    # tq-pro ADCIndex at several (pca-dim, bits) -- isolate truncation vs scalar-quant
    for pdim, bits in [(a.pca_dim, 3), (dim, 3), (dim, 2)]:
        pca = PCAMatryoshka(input_dim=dim, output_dim=pdim)
        pca.fit(C[: min(N, 200000)])
        ev = float(np.sum(pca._eigenvalues) / np.sum(pca._all_eigenvalues))
        index = ADCIndex(pca.with_quantizer(bits=bits)).add(C)
        i1, _ = index.search(Q, k=10)
        ir = index.search(Q, k=10, rerank=5, originals=C)
        bpv = pdim * bits // 8
        q = bench(lambda index=index: index.search(Q, k=10))
        print(
            f"| tq-pro PCA{pdim}+TQ{bits} (var={ev:.2f}) | {bpv} | {dim*4/bpv:.0f} "
            f"| {recall(gt, i1, 10):.4f} | {recall(gt, ir, 10):.4f} | {q:.0f} |",
            flush=True,
        )


if __name__ == "__main__":
    main()

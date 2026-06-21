#!/usr/bin/env python3
"""#3: ADC distance during HNSW graph traversal -- map the design space.

The reviewer asks how distance is computed when packed codes are routed into an
HNSW graph: on-the-fly unpack, or ADC directly on the compressed stream? We compare:
  - HNSW-flat (fp32): graph over uncompressed vectors -- speed/recall ceiling, no compression
  - HNSW-PQ:          faiss IndexHNSWPQ -- ADC *during traversal* in C++, compressed
  - ADCIndex (ours):  asymmetric ADC over our scalar codes, flat scan (no graph yet)
on PCA-256 LaBSE. Shows what ADC-in-traversal buys, and the gap our codes leave open.
"""

import argparse
import time

import numpy as np


def normalize(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)


def exact_topk(Q, C, k):
    out = np.zeros((len(Q), k), dtype=np.int64)
    for s in range(0, len(Q), 256):
        out[s : s + 256] = np.argsort(-(Q[s : s + 256] @ C.T), axis=1)[:, :k]
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
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    import faiss

    faiss.omp_set_num_threads(a.threads)
    from turboquant_pro import ADCIndex, PCAMatryoshka

    X = normalize(np.load(a.npy, mmap_mode="r").astype(np.float32))
    Q, C = X[: a.queries].copy(), X[a.queries : a.queries + a.corpus].copy()
    dim = X.shape[1]
    pd = a.pca_dim
    print(f"corpus={len(C)} dim={dim} queries={len(Q)} pca={pd}", flush=True)
    gt = exact_topk(Q, C, 10)
    pca = PCAMatryoshka(input_dim=dim, output_dim=pd)
    pca.fit(C[: min(len(C), 50000)])
    Cp = normalize(np.asarray(pca.transform(C), dtype=np.float32))
    Qp = normalize(np.asarray(pca.transform(Q), dtype=np.float32))

    def bench(fn, reps=3):
        best = 1e30
        for _ in range(reps):
            t = time.time()
            fn()
            best = min(best, time.time() - t)
        return len(Q) / best

    print("\n| index | distance in traversal | bytes/vec | build s | qps | recall@10 |")
    print("|---|---|---:|---:|---:|---:|")

    # HNSW-flat: fp32 distance, no compression
    t = time.time()
    h = faiss.IndexHNSWFlat(pd, 32, faiss.METRIC_INNER_PRODUCT)
    h.hnsw.efConstruction = 200
    h.add(Cp)
    h.hnsw.efSearch = 128
    b = time.time() - t
    _, I = h.search(Qp, 10)
    print(
        f"| HNSW-flat | fp32 dot (uncompressed) | {pd*4} | {b:.1f} | "
        f"{bench(lambda: h.search(Qp, 10)):.0f} | {recall(gt, I, 10):.4f} |",
        flush=True,
    )

    # HNSW-PQ: ADC during traversal (C++)
    for m in (64, 128):
        try:
            t = time.time()
            hp = faiss.IndexHNSWPQ(pd, m, 32)
            hp.hnsw.efConstruction = 200
            hp.train(Cp)
            hp.add(Cp)
            hp.hnsw.efSearch = 128
            b = time.time() - t
            _, I = hp.search(Qp, 10)
            print(
                f"| HNSW-PQ(m={m}) | ADC LUT (C++) | {m} | {b:.1f} | "
                f"{bench(lambda hp=hp: hp.search(Qp, 10)):.0f} | {recall(gt, I, 10):.4f} |",
                flush=True,
            )
        except Exception as e:
            print(f"| HNSW-PQ(m={m}) | FAILED: {str(e)[:40]} |", flush=True)

    # ADCIndex (ours): asymmetric ADC, flat scan
    idx = ADCIndex(pca.with_quantizer(bits=3)).add(C)
    i1, _ = idx.search(Q, k=10)
    print(
        f"| ADCIndex (ours) | asym ADC SIMD (flat) | {pd*3//8} | - | "
        f"{bench(lambda: idx.search(Q, k=10)):.0f} | {recall(gt, i1, 10):.4f} |",
        flush=True,
    )


if __name__ == "__main__":
    main()

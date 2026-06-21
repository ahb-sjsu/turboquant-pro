#!/usr/bin/env python3
"""ScaNN (Google SOTA fast-ANN) baseline on real embeddings.

ScaNN is a complete ANN *system* (anisotropic hashing + tree + reorder), not a
pure quantizer -- it is the production fast-search bar. Reported alongside
RaBitQ/OPQ for honest breadth. Run in a scann venv:
  python benchmark_scann.py --npy /tmp/labse_bench.npy --corpus 100000 --queries 1000
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
    ap.add_argument("--corpus", type=int, default=100000)
    ap.add_argument("--queries", type=int, default=1000)
    a = ap.parse_args()
    import scann

    X = np.load(a.npy, mmap_mode="r").astype(np.float32)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
    Q = X[: a.queries].copy()
    C = X[a.queries : a.queries + a.corpus].copy()
    print(f"corpus={len(C)} dim={X.shape[1]} queries={len(Q)}", flush=True)
    gt = exact_topk(Q, C, 10)

    t = time.time()
    searcher = (
        scann.scann_ops_pybind.builder(C, 10, "dot_product")
        .tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=50000)
        .score_ah(2, anisotropic_quantization_threshold=0.2)
        .reorder(100)
        .build()
    )
    build_s = time.time() - t

    searcher.search_batched(Q[:10])  # warmup
    t = time.time()
    nbrs, _ = searcher.search_batched(Q)
    qps = len(Q) / (time.time() - t)
    recall = float(
        np.mean(
            [
                len(set(gt[i].tolist()) & set(nbrs[i].tolist())) / 10
                for i in range(len(Q))
            ]
        )
    )
    print(
        f"\nScaNN: build {build_s:.1f}s | {qps:.0f} qps | recall@10={recall:.4f} "
        f"(AH+tree+reorder)",
        flush=True,
    )


if __name__ == "__main__":
    main()

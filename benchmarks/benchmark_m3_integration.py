#!/usr/bin/env python3
"""M3: wire tq-pro's REAL codes through the M1 SIMD kernel.

Reproduces tq-pro's PCA-256 + TurboQuant (rotation + shared 3-bit codebook)
pipeline, then shows the SIMD ADC kernel returns the SAME top-k as tq-pro's
faiss flat-reconstruct search (recall ~0.78 single / ~0.999 +rerank) but at
kernel speed. ADC identity:
  cosine(q, recon) = (1/||cent[codes]||) * sum_j rotate(q_pca)[j] * cent[code[j]]
so we pass q=rotate(pca(Q)), norms=1/||cent[codes]||, cent=tq.centroids.
"""

import argparse
import sys
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
    rr = np.full((len(Q), k), -1, dtype=np.int64)
    for i in range(len(Q)):
        c = cand[i][cand[i] >= 0]
        s = C[c] @ Q[i]
        top = c[np.argsort(-s)[:k]]
        rr[i, : len(top)] = top
    return rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True)
    ap.add_argument("--build-dir", default="/home/claude/adc_build")
    ap.add_argument("--corpus", type=int, default=100000)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--pca-dim", type=int, default=256)
    a = ap.parse_args()
    sys.path.insert(0, a.build_dir)
    import adc_scan
    import faiss

    from turboquant_pro import PCAMatryoshka, TurboQuantPGVector

    X = normalize(np.load(a.npy, mmap_mode="r").astype(np.float32))
    Q = X[: a.queries].copy()
    C = X[a.queries : a.queries + a.corpus].copy()
    N, dim = C.shape
    pd = a.pca_dim
    print(f"corpus={N} dim={dim} queries={len(Q)} pca={pd}", flush=True)
    gt = exact_topk(Q, C, 10)

    pca = PCAMatryoshka(input_dim=dim, output_dim=pd)
    pca.fit(C[: min(N, 50000)])
    Xp = np.asarray(pca.transform(C), dtype=np.float32)
    Qp = normalize(np.asarray(pca.transform(Q), dtype=np.float32))

    # tq-pro codes: rotate the unit PCA vector, quantize per-dim with shared codebook
    tq = TurboQuantPGVector(dim=pd, bits=3)
    cnorm = np.linalg.norm(Xp, axis=1, keepdims=True)
    rotated = tq._rotate(Xp / np.maximum(cnorm, 1e-30))
    codes = np.searchsorted(tq.boundaries, rotated).astype(np.uint8)  # (N, pd) in 0..7
    cent = tq.centroids.astype(np.float32)  # (8,)
    cc = cent[codes]  # quantized rotated unit coords
    recon = np.asarray(cnorm * tq._unrotate(cc), dtype=np.float32)  # tq-pro reconstruct

    # reference: tq-pro's faiss flat-reconstruct search (the 0.78 / 0.999 method)
    recon_n = normalize(recon)
    fidx = faiss.IndexFlatIP(pd)
    fidx.add(recon_n)
    _, If = fidx.search(Qp, 100)
    r1_ref = recall(gt, If, 10)
    _, cf = fidx.search(Qp, 50)
    r2_ref = recall(gt, rerank(cf, Q, C), 10)

    # kernel inputs
    q_rot = np.ascontiguousarray(tq._rotate(Qp), dtype=np.float32)
    norms_eff = (1.0 / np.maximum(np.linalg.norm(cc, axis=1), 1e-30)).astype(np.float32)

    ik_s, _ = adc_scan.search(codes, q_rot, cent, norms_eff, 10, False)  # scalar
    ik, _ = adc_scan.search(codes, q_rot, cent, norms_eff, 10, True)  # SIMD
    agree_scalar = recall(If[:, :10], ik_s, 10)
    agree_simd = recall(If[:, :10], ik, 10)
    r1_k = recall(gt, ik, 10)
    candk, _ = adc_scan.search(codes, q_rot, cent, norms_eff, 50, True)
    r2_k = recall(gt, rerank(candk, Q, C), 10)

    print(
        f"\ntq-pro flat-reconstruct (reference): r@10 single={r1_ref:.4f} +rerank={r2_ref:.4f}",
        flush=True,
    )
    print(
        f"SIMD kernel (same codes):            r@10 single={r1_k:.4f} +rerank={r2_k:.4f}",
        flush=True,
    )
    print(
        f"kernel-vs-faiss-flat agreement: scalar={agree_scalar:.4f} simd={agree_simd:.4f}",
        flush=True,
    )

    def bench(fn, reps=3):
        best = 1e30
        for _ in range(reps):
            t = time.time()
            fn()
            best = min(best, time.time() - t)
        return len(Q) / best

    qps_k = bench(lambda: adc_scan.search(codes, q_rot, cent, norms_eff, 10, True))
    qps_flat = bench(lambda: fidx.search(Qp, 100))
    print(
        f"\nQPS  SIMD-kernel={qps_k:.0f}  tq-pro flat-reconstruct={qps_flat:.0f}  "
        f"speedup={qps_k/qps_flat:.1f}x  bytes/vec={pd*3//8}",
        flush=True,
    )


if __name__ == "__main__":
    main()

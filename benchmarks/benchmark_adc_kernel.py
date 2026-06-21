#!/usr/bin/env python3
"""M1 kernel validation + benchmark.

Builds per-dim 4-bit codes (PCA -> rotation -> 16-level quantize), then:
  1) validates the SIMD path == scalar reference == numpy exact ADC (correctness),
  2) measures recall vs full-precision ground truth (the codes' property),
  3) benchmarks qps for SIMD vs scalar vs the numpy reconstruct baseline.
The kernel only speeds the scan; recall comes from the codes.
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
    rr = np.zeros((len(Q), k), dtype=np.int64)
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

    from turboquant_pro import PCAMatryoshka

    X = normalize(np.load(a.npy, mmap_mode="r").astype(np.float32))
    Q = X[: a.queries].copy()
    C = X[a.queries : a.queries + a.corpus].copy()
    N, dim = C.shape
    pd = a.pca_dim
    print(f"corpus={N} dim={dim} queries={len(Q)} pca={pd}", flush=True)
    gt = exact_topk(Q, C, 10)

    pca = PCAMatryoshka(input_dim=dim, output_dim=pd)
    pca.fit(C[: min(N, 50000)])
    Cp = np.asarray(pca.transform(C), dtype=np.float32)
    Qp = np.asarray(pca.transform(Q), dtype=np.float32)
    rng = np.random.default_rng(0)
    Rot, _ = np.linalg.qr(rng.standard_normal((pd, pd)))
    Rot = Rot.astype(np.float32)
    Cr, Qr = (Cp @ Rot).astype(np.float32), (Qp @ Rot).astype(np.float32)
    norms = np.linalg.norm(Cr, axis=1).astype(np.float32)
    U = Cr / np.maximum(norms[:, None], 1e-30)
    # 16 Lloyd-ish levels from coord quantiles; codes via boundary searchsorted
    cent = np.quantile(U.reshape(-1), (np.arange(16) + 0.5) / 16).astype(np.float32)
    bnd = ((cent[:-1] + cent[1:]) / 2).astype(np.float32)
    codes = np.searchsorted(bnd, U).astype(np.uint8)  # (N, pd) in 0..15

    # numpy exact ADC reference (same codes+cent): score = q . (norm * cent[codes])
    recon = norms[:, None] * cent[codes]
    adc = Qr @ recon.T
    np_top = np.argpartition(-adc, 10, axis=1)[:, :10]
    for i in range(len(Q)):
        np_top[i] = np_top[i][np.argsort(-adc[i, np_top[i]])]

    idx_ref, _ = adc_scan.search(codes, Qr, cent, norms, 10, False)
    idx_simd, _ = adc_scan.search(codes, Qr, cent, norms, 10, True)

    agree_ref = recall(np_top, idx_ref, 10)
    agree_simd = recall(np_top, idx_simd, 10)
    print(
        f"\nCORRECTNESS  ref-vs-numpy={agree_ref:.4f}  simd-vs-numpy={agree_simd:.4f} "
        f"(1.0 = exact match)",
        flush=True,
    )

    # recall of the codes vs full-precision gt
    r_simd = recall(gt, idx_simd, 10)
    cand, _ = adc_scan.search(codes, Qr, cent, norms, 50, True)
    r_rr = recall(gt, rerank(cand, Q, C), 10)
    print(
        f"RECALL@10    single={r_simd:.4f}  +rerank={r_rr:.4f}  bytes/vec={pd//2}",
        flush=True,
    )

    # qps
    def bench(fn, reps=3):
        best = 1e30
        for _ in range(reps):
            t = time.time()
            fn()
            best = min(best, time.time() - t)
        return len(Q) / best

    qps_simd = bench(lambda: adc_scan.search(codes, Qr, cent, norms, 10, True))
    qps_ref = bench(lambda: adc_scan.search(codes, Qr, cent, norms, 10, False))

    def np_baseline():
        sc = Qr @ recon.T
        np.argpartition(-sc, 10, axis=1)

    qps_np = bench(np_baseline, reps=2)
    print(
        f"\nQPS  SIMD={qps_simd:.0f}  scalar-ref={qps_ref:.0f}  numpy-reconstruct={qps_np:.0f}",
        flush=True,
    )
    print(f"SPEEDUP SIMD vs numpy-reconstruct = {qps_simd/qps_np:.1f}x", flush=True)


if __name__ == "__main__":
    main()

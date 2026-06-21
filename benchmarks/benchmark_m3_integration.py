#!/usr/bin/env python3
"""M3: tq-pro's REAL codes through the M1 kernel at the headline 768-d operating point.

tq-pro stores 256-d 3-bit codes but searches by reconstructing to 768-d via
pca.inverse_transform (mean re-added) -- that is the 0.78/0.999 operating point.
The kernel reproduces that exact cosine ranking using the derived identity:

  cos(Q, recon768) = (Q.mean + norm * sum_j rotate(qt)[j]*cent[code[j]]) / ||recon768||
    qt   = Q @ components^T                       (uncentered PCA proj of query)
    m_n  = sum_j rotate(mean_proj)[j]*cent[code[j]]   (mean-ADC, precomputed)
    ||recon768[n]||^2 = norm[n]^2 ||cent[codes]||^2 + 2 norm[n] m_n + ||mean||^2

so vnorm=norm, vrnorm=1/||recon768||, qbias=Q.mean fed to the extended kernel.
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
    mean = np.asarray(pca._mean, dtype=np.float32)  # (768,)
    comp = np.asarray(pca._components, dtype=np.float32)  # (256, 768), orthonormal rows
    Xp = np.asarray(pca.transform(C), dtype=np.float32)  # centered PCA proj

    tq = TurboQuantPGVector(dim=pd, bits=3)
    cnorm = np.linalg.norm(Xp, axis=1).astype(np.float32)  # (N,)
    rotated = tq._rotate(Xp / np.maximum(cnorm[:, None], 1e-30))
    codes = np.searchsorted(tq.boundaries, rotated).astype(np.uint8)  # (N, pd) 0..7
    cent = tq.centroids.astype(np.float32)
    cc = cent[codes]  # (N, pd) quantized rotated unit
    recon256 = cnorm[:, None] * tq._unrotate(cc)  # (N, pd)
    recon768 = (recon256 @ comp + mean).astype(np.float32)  # 768-d reconstruction

    # reference: tq-pro's faiss flat-reconstruct search in 768-d (the 0.78/0.999 path)
    fidx = faiss.IndexFlatIP(dim)
    fidx.add(normalize(recon768))
    _, If = fidx.search(Q, 100)
    r1_ref = recall(gt, If, 10)
    _, cf = fidx.search(Q, 50)
    r2_ref = recall(gt, rerank(cf, Q, C), 10)

    # kernel inputs for the 768-d cosine (mean term)
    mean_proj = (comp @ mean).astype(np.float32)  # (pd,)
    mp_rot = np.ascontiguousarray(tq._rotate(mean_proj[None, :])[0], dtype=np.float32)
    m_n = (cc @ mp_rot).astype(np.float32)  # mean-ADC per vector
    s2 = (cc * cc).sum(axis=1).astype(np.float32)  # ||cent[codes]||^2
    mean_sq = float(mean @ mean)
    recon_n2 = cnorm**2 * s2 + 2.0 * cnorm * m_n + mean_sq
    vrnorm = (1.0 / np.sqrt(np.maximum(recon_n2, 1e-30))).astype(np.float32)
    qt = (Q @ comp.T).astype(np.float32)  # uncentered proj of (normalized) query
    q_rot = np.ascontiguousarray(tq._rotate(qt), dtype=np.float32)
    qbias = (Q @ mean).astype(np.float32)  # (nq,)

    ik_s, _ = adc_scan.search(codes, q_rot, cent, cnorm, vrnorm, qbias, 10, False)
    ik, _ = adc_scan.search(codes, q_rot, cent, cnorm, vrnorm, qbias, 10, True)
    print(
        f"\ntq-pro flat-reconstruct (768-d ref): r@10 single={r1_ref:.4f} +rerank={r2_ref:.4f}",
        flush=True,
    )
    print(
        f"SIMD kernel (same codes, 768-d cos): r@10 single={recall(gt, ik, 10):.4f} "
        f"+rerank={recall(gt, rerank(adc_scan.search(codes, q_rot, cent, cnorm, vrnorm, qbias, 50, True)[0], Q, C), 10):.4f}",
        flush=True,
    )
    print(
        f"kernel-vs-faiss agreement: scalar={recall(If[:, :10], ik_s, 10):.4f} "
        f"simd={recall(If[:, :10], ik, 10):.4f}",
        flush=True,
    )

    def bench(fn, reps=3):
        best = 1e30
        for _ in range(reps):
            t = time.time()
            fn()
            best = min(best, time.time() - t)
        return len(Q) / best

    qps_k = bench(
        lambda: adc_scan.search(codes, q_rot, cent, cnorm, vrnorm, qbias, 10, True)
    )
    qps_flat = bench(lambda: fidx.search(Q, 100))
    print(
        f"\nQPS  SIMD-kernel={qps_k:.0f}  tq-pro flat-reconstruct(768d)={qps_flat:.0f}  "
        f"speedup={qps_k/qps_flat:.1f}x  bytes/vec={pd*3//8}",
        flush=True,
    )


if __name__ == "__main__":
    main()

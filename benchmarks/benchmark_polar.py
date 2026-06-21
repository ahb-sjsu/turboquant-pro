#!/usr/bin/env python3
"""Is PolarQuant's polar transform worth adopting vs our rotate+scalar-quant?

PolarQuant (Han et al., arXiv:2502.02617): after random preconditioning, recursively
convert Cartesian coords to polar (radius + angles); the angles concentrate
(analytic distribution, Var O(1/sqrt d)), so they quantize tightly with few bits and
need no per-block normalization. We implement the recursive transform (their Def. 1)
and compare reconstruction quality + retrieval recall against TurboQuant's per-dim
Lloyd-Max scalar quantization at matched bit budgets, on the SAME randomly-rotated
PCA-reduced embeddings.
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


# ---- recursive polar transform (power-of-two dim) ----
def polar_encode(X):
    angles = []
    v = X
    while v.shape[1] > 1:
        a, b = v[:, 0::2], v[:, 1::2]
        angles.append(np.arctan2(b, a))  # level angles (N, m)
        v = np.sqrt(a * a + b * b)
    return v[:, 0], angles  # radius (N,), angles list (coarse->fine reversed below)


def polar_decode(radius, angles):
    v = radius[:, None]
    for th in reversed(angles):
        a, b = v * np.cos(th), v * np.sin(th)
        nv = np.empty((v.shape[0], 2 * v.shape[1]), dtype=np.float32)
        nv[:, 0::2], nv[:, 1::2] = a, b
        v = nv
    return v


def quant_uniform(x, lo, hi, bits):
    levels = 2**bits
    step = (hi - lo) / levels
    q = np.clip(np.floor((x - lo) / step), 0, levels - 1)
    return (lo + (q + 0.5) * step).astype(np.float32)


def polar_quantize(Xr, abits, rbits=8, adaptive=True):
    """Quantize rotated vectors via polar transform. Returns reconstruction.

    adaptive=True uses a per-level codebook fit (quantiles ~ Lloyd-Max) to each
    level's angle distribution -- the paper's design, exploiting angle concentration.
    """
    radius, angles = polar_encode(Xr)
    lr = np.log(np.maximum(radius, 1e-20))
    radius_q = np.exp(quant_uniform(lr, lr.min(), lr.max(), rbits))
    angles_q = []
    for th in angles:
        if adaptive:
            lv = np.quantile(th, (np.arange(2**abits) + 0.5) / 2**abits).astype(
                np.float32
            )
            bnd = ((lv[:-1] + lv[1:]) / 2).astype(np.float32)
            angles_q.append(lv[np.searchsorted(bnd, th)].astype(np.float32))
        else:
            angles_q.append(quant_uniform(th, -np.pi, np.pi, abits))
    return polar_decode(radius_q.astype(np.float32), angles_q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True)
    ap.add_argument("--corpus", type=int, default=50000)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--pca-dim", type=int, default=256)  # power of two
    a = ap.parse_args()
    import faiss

    from turboquant_pro import PCAMatryoshka, TurboQuantPGVector

    X = normalize(np.load(a.npy, mmap_mode="r").astype(np.float32))
    Q, C = X[: a.queries].copy(), X[a.queries : a.queries + a.corpus].copy()
    dim = X.shape[1]
    pd = a.pca_dim
    print(f"corpus={len(C)} dim={dim} queries={len(Q)} pca={pd}", flush=True)
    gt = exact_topk(Q, C, 10)

    pca = PCAMatryoshka(input_dim=dim, output_dim=pd)
    pca.fit(C[: min(len(C), 50000)])
    Cp = np.asarray(pca.transform(C), dtype=np.float32)
    Qp = normalize(np.asarray(pca.transform(Q), dtype=np.float32))
    cn = np.linalg.norm(Cp, axis=1, keepdims=True)
    Cu = Cp / np.maximum(cn, 1e-30)  # unit vectors (what both methods quantize)

    tq = TurboQuantPGVector(dim=pd, bits=3)  # reuse its rotation for both methods
    Cr = tq._rotate(Cu)  # randomly rotated unit vectors

    print("\n| method | bits/coord | recon cosine | recall@10 (recon search) |")
    print("|---|---:|---:|---:|")

    def eval_recon(name, recon_rot, bits):
        recon_u = tq._unrotate(recon_rot)  # back to PCA space
        recon = normalize((cn * recon_u).astype(np.float32))
        # reconstruction fidelity in the rotated unit space (Cr is the reference)
        cos = float(np.mean(np.sum(Cr * normalize(recon_rot), axis=1)))
        idx = faiss.IndexFlatIP(pd)
        idx.add(recon)
        _, I = idx.search(Qp, 100)
        print(f"| {name} | {bits} | {cos:.4f} | {recall(gt, I, 10):.4f} |", flush=True)

    for b in (3, 4):
        # scalar (ours): per-dim Lloyd-Max on rotated unit vectors
        tqb = TurboQuantPGVector(dim=pd, bits=b)
        idxs = np.searchsorted(tqb.boundaries, tq._rotate(Cu))
        recon_scalar = tqb.centroids[idxs].astype(np.float32)
        eval_recon(f"scalar-TQ{b} (ours)", recon_scalar, b)
        # polar (theirs): angles at b bits, radius at 8 bits
        t = time.time()
        recon_polar = polar_quantize(Cr, abits=b, rbits=8)
        dt = time.time() - t
        eval_recon(f"polar-{b}b+r8 ({dt:.1f}s)", recon_polar, b)


if __name__ == "__main__":
    main()

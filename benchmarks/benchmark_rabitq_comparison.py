#!/usr/bin/env python3
"""tq-pro (ADCIndex) vs RaBitQ vs ScaNN at matched bytes, single-pass and with rerank.

Motivation: a common critique is that tq-pro's *single-pass* recall trails RaBitQ. This
benchmark settles it on real embeddings, matched on bytes/vector, reporting both the
single-pass number and tq-pro's own rerank frontier -- since RaBitQ's headline recall is
itself a *with-rerank* number, comparing it to tq-pro single-pass is apples-to-oranges.

Data: any (N, D) float array of embeddings (--npy), unit-normalized here so dot == cosine.
Ground truth: exact top-10 inner product over the corpus (brute force).

Optional deps, skipped cleanly if absent:
  - rabitqlib  (pip install rabitqlib)   -> RaBitQ IVF, exhaustive (isolates the estimator)
  - scann      (pip install scann)       -> tree-AH, anisotropic vs reconstruction loss

Example:
  python benchmark_rabitq_comparison.py --npy labse.npy --pca-dims 256 384 512 --queries 2000
"""

import argparse
import time

import numpy as np


def normalize(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)


def recall(gt, ap, k=10):
    return float(
        np.mean([len(set(gt[i, :k]) & set(ap[i, :k])) / k for i in range(len(gt))])
    )


def exact_top(DB, Q, k=10):
    out = np.empty((len(Q), k), np.int64)
    for i in range(0, len(Q), 256):
        s = Q[i : i + 256] @ DB.T
        out[i : i + 256] = np.argsort(-s, axis=1)[:, :k]
    return out


def tqp_bytes(pca_dim, bits):
    return int(np.ceil(pca_dim * bits / 8)) + 4  # codes + fp32 norm


def run_tqp(DB, Q, gt, pca_dim, bits, train):
    from turboquant_pro import ADCIndex, PCAMatryoshka

    pca = PCAMatryoshka(input_dim=DB.shape[1], output_dim=pca_dim)
    pca.fit(train)
    idx = ADCIndex(pca.with_quantizer(bits=bits)).add(DB)
    ids, _ = idx.search(Q, k=10)
    sp = recall(gt, ids)
    rr = recall(gt, idx.search(Q, k=10, rerank=5, originals=DB))
    print(
        f"  tq-pro  PCA{pca_dim} {bits}b  {tqp_bytes(pca_dim, bits):3d} B/vec"
        f"   single-pass={sp:.4f}   +5x rerank={rr:.4f}"
    )


def run_rabitq(DB, Q, gt, nbits, seed=0):
    try:
        from rabitqlib import IvfIndex
    except Exception:
        print(f"  RaBitQ  {nbits}b  [skipped: rabitqlib not installed]")
        return
    rng = np.random.default_rng(seed)
    ncl = 200
    cent = DB[rng.choice(len(DB), ncl, replace=False)]
    cid = np.empty(len(DB), np.int32)
    for i in range(0, len(DB), 20000):
        cid[i : i + 20000] = np.argmax(DB[i : i + 20000] @ cent.T, axis=1)
    idx = IvfIndex(DB.shape[1], len(DB), ncl, nbits, "l2")
    idx.build(DB, cent, cid, False)
    res = idx.search(Q, 10, ncl, True, 8)
    ids = np.asarray(res[0]) if isinstance(res, tuple) else np.asarray(res)
    b = int(np.ceil(DB.shape[1] * nbits / 8)) + 8
    print(
        f"  RaBitQ  {nbits}b  {b:3d} B/vec   single-pass(exhaustive)={recall(gt, ids):.4f}"
    )


def run_scann(DB, Q, gt):
    try:
        from scann.scann_ops.py import scann_ops_pybind as sp
    except Exception:
        print("  ScaNN   [skipped: scann not installed]")
        return
    for thr, name in [(0.2, "anisotropic"), (float("inf"), "reconstruction")]:
        s = (
            sp.builder(DB, 10, "dot_product")
            .tree(
                num_leaves=2000, num_leaves_to_search=250, training_sample_size=250000
            )
            .score_ah(2, anisotropic_quantization_threshold=thr)
            .build()
        )
        ids, _ = s.search_batched(Q, final_num_neighbors=10, leaves_to_search=250)
        print(
            f"  ScaNN   AH-only {name:14s}   recall@10={recall(gt, np.asarray(ids)):.4f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True, help="(N, D) float embeddings")
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--pca-dims", type=int, nargs="+", default=[256, 384, 512])
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--rabitq-nbits", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    A = normalize(np.asarray(np.load(a.npy, mmap_mode="r"), dtype=np.float32))
    N = len(A)
    qi = rng.choice(N, min(a.queries, N // 2), replace=False)
    mask = np.ones(N, bool)
    mask[qi] = False
    DB, Q = A[mask], A[qi]
    train = DB[rng.choice(len(DB), min(50000, len(DB)), replace=False)]
    print(f"corpus {DB.shape}  queries {Q.shape}  (unit-norm; dot == cosine)")
    t0 = time.time()
    gt = exact_top(DB, Q, 10)
    print(f"exact ground truth in {time.time() - t0:.0f}s\n")

    print("== turboquant-pro (PCA-Matryoshka + TurboQuant + ADC) ==")
    for d in a.pca_dims:
        run_tqp(DB, Q, gt, d, a.bits, train)
    print("\n== RaBitQ (exhaustive single-pass; isolates the estimator) ==")
    for nb in a.rabitq_nbits:
        run_rabitq(DB, Q, gt, nb)
    print("\n== ScaNN (AH-only; anisotropic vs reconstruction loss) ==")
    run_scann(DB, Q, gt)


if __name__ == "__main__":
    main()

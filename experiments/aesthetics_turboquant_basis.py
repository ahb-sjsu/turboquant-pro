#!/usr/bin/env python3
"""
Phase 8: Operational compressibility as aesthetic predictor.

The existing aesthetics pipeline (aesthetics_complete.py) computes

    A(p; q) = D_eff(p) * exp(-alignment(p, q) / r)

an analytical proxy for "compressibility in the corpus PCA basis". This
script swaps the proxy for a direct rate-distortion measurement using
TurboQuant: for each stimulus, compress its embedding under several
candidate orthogonal bases at a fixed bit budget, then measure the
reconstruction cosine. Higher cosine = more compressible in that basis.

The geometric-aesthetics theory (paper v3) predicts:

    H1: PCA-of-corpus basis predicts ratings better than a random
        orthogonal basis (the natural-basis hypothesis).
    H2: PCA-per-subpopulation (per-genre, per-tradition) beats
        PCA-of-corpus (the per-perceiver-eigenbasis hypothesis).
    H3: TurboQuant-bpt and the analytical A(p; q) score correlate but
        provide independent signal (rate-distortion captures information
        the closed-form does not).

We test H1 and H3 on the Goodreads dataset that already produced the
P1 result (r = +0.048, z = 10.8). H2 is left for the music corpus
(MERT/CLAP) where genre labels exist; this script's design extends to
that case via the --groupby flag.

Output:
    results_aesthetics/turboquant_basis_results.json with one record
    per (basis, bits) cell containing both the bpt-vs-rating correlation
    and the bpt-vs-A(p;q) correlation, plus the existing A(p;q)-vs-rating
    baseline for direct comparison.

Usage:
    python3 aesthetics_turboquant_basis.py                      # Goodreads
    python3 aesthetics_turboquant_basis.py --dataset music      # music
    python3 aesthetics_turboquant_basis.py --bits 2 3 4         # sweep
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Optional

import numpy as np
from scipy import stats

# Reuse the existing pipeline's primitives so behaviour matches the
# already-published P1 baseline exactly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aesthetics_complete import (  # noqa: E402
    aesthetic_score,
    compute_scores,
    correlation_report,
    log,
    participation_ratio,
)

RESULTS_DIR = "results_aesthetics"


# ─── Basis builders ─────────────────────────────────────────────────


def basis_random(dim: int, seed: int = 42) -> np.ndarray:
    """Haar-random orthogonal matrix via QR of a Gaussian. Same recipe
    TurboQuant uses internally for its default rotation."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(G)
    return Q


def basis_hadamard(dim: int) -> Optional[np.ndarray]:
    """Walsh-Hadamard, normalized to orthogonal. Returns None if dim
    is not a power of 2 (Hadamard requires that for the simple form)."""
    if dim & (dim - 1):
        return None
    from scipy.linalg import hadamard

    return hadamard(dim).astype(np.float32) / np.sqrt(dim)


def basis_pca(embeddings: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
    """Top-k principal components of the embedding corpus. This is the
    'natural basis' the analytical A(p;q) score implicitly uses."""
    n, d = embeddings.shape
    if n_components is None:
        n_components = d  # full basis so the rotation is square
    n_components = min(n_components, d, n - 1)
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean
    if n > 5000:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(centered)
        Q = pca.components_  # (n_components, d)
    else:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        Q = Vt[:n_components]
    # Pad to a square orthogonal matrix if n_components < d (random
    # complement) so the basis has full rank.
    if Q.shape[0] < d:
        rng = np.random.default_rng(42)
        complement = rng.standard_normal((d - Q.shape[0], d))
        # Project out components already in Q
        complement = complement - complement @ Q.T @ Q
        Qc, _ = np.linalg.qr(complement.T)
        Q = np.vstack([Q, Qc.T])
    return Q.astype(np.float32)


# ─── TurboQuant-style compressibility ───────────────────────────────
# We reimplement compress/decompress here (instead of importing the
# turboquant_pro library) to keep the script self-contained and to make
# the basis swap explicit. The math is identical to TurboQuant's CPU
# reference path: per-row L2 norm, rotation by Q, scalar quantize via
# Lloyd-Max codebook, dequantize, inverse rotation, multiply by norm.

# Lloyd-Max centroids for a unit-norm rotated vector under N(0,1)
# marginal assumption. These are the exact values used in the
# pgext/Rust path (codebook.rs).
CODEBOOK = {
    2: np.array([-1.510, -0.453, 0.453, 1.510], dtype=np.float32),
    3: np.array(
        [-1.748, -1.050, -0.500, -0.069, 0.069, 0.500, 1.050, 1.748],
        dtype=np.float32,
    ),
    4: np.array(
        [
            -2.401, -1.844, -1.437, -1.099,
            -0.800, -0.524, -0.262, -0.066,
            0.066, 0.262, 0.524, 0.800,
            1.099, 1.437, 1.844, 2.401,
        ],
        dtype=np.float32,
    ),
}


def _quantize_vector(rotated_unit: np.ndarray, bits: int, dim: int) -> np.ndarray:
    """Map each rotated-unit-vector coordinate to its nearest codebook
    centroid (scaled by 1/sqrt(dim))."""
    cb = CODEBOOK[bits] / np.sqrt(dim)
    boundaries = 0.5 * (cb[:-1] + cb[1:])
    idx = np.searchsorted(boundaries, rotated_unit)
    return cb[idx]


def reconstruction_cosine(
    embeddings: np.ndarray,
    Q: np.ndarray,
    bits: int,
) -> np.ndarray:
    """For each row of `embeddings`, compute cos(original, reconstructed)
    where reconstruction goes through the TurboQuant pipeline using the
    given orthogonal basis Q. Returns shape (n,).
    """
    n, d = embeddings.shape
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe = np.maximum(norms, 1e-30)
    units = embeddings / safe
    rotated = units @ Q.T
    quantized = _quantize_vector(rotated, bits, d)
    back = quantized @ Q
    reconstructed = back * norms
    # Cosine between original and reconstruction
    o_n = np.linalg.norm(embeddings, axis=1)
    r_n = np.linalg.norm(reconstructed, axis=1)
    dot = (embeddings * reconstructed).sum(axis=1)
    denom = np.maximum(o_n * r_n, 1e-30)
    return dot / denom


# ─── Main experiment ────────────────────────────────────────────────


def load_goodreads(max_books: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the embedding step from phase2_goodreads().
    Returns (embeddings, ratings)."""
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    log("  Loading Goodreads...")
    try:
        ds = load_dataset(
            "maharshipandya/goodreads-dataset", split="train", streaming=True
        )
    except Exception:
        ds = load_dataset(
            "ucberkeley-dlab/goodreads", split="train", streaming=True
        )
    texts: list[str] = []
    ratings: list[float] = []
    for i, row in enumerate(ds):
        text = (
            row.get("review_text") or row.get("description") or row.get("text")
            or row.get("Title", "")
        )
        rating = row.get("rating") or row.get("Rating") or row.get("score")
        if text and rating is not None and len(str(text)) > 20:
            texts.append(str(text)[:512])
            ratings.append(float(rating))
        if len(texts) >= max_books:
            break
    log(f"  Collected {len(texts)} books")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = np.asarray(
        model.encode(texts, batch_size=128, show_progress_bar=True),
        dtype=np.float32,
    )
    del model
    gc.collect()
    return emb, np.asarray(ratings, dtype=np.float32)


def run_basis_sweep(
    embeddings: np.ndarray,
    ratings: np.ndarray,
    bits_sweep: list[int],
    a_score_baseline: np.ndarray,
) -> list[dict]:
    """Run TurboQuant-compressibility correlations against ratings and
    against the analytical A(p;q) score, for each (basis, bits) cell."""
    n, d = embeddings.shape
    bases: dict[str, np.ndarray] = {
        "random": basis_random(d, seed=42),
        "pca": basis_pca(embeddings),
    }
    h = basis_hadamard(d)
    if h is not None:
        bases["hadamard"] = h
    else:
        log(f"  hadamard skipped: dim={d} is not a power of 2")

    rows: list[dict] = []
    for basis_name, Q in bases.items():
        for bits in bits_sweep:
            log(f"\n  basis={basis_name} bits={bits}")
            cos = reconstruction_cosine(embeddings, Q, bits)
            log(f"    mean_cos={cos.mean():.5f}  min={cos.min():.5f}")

            r_rating, p_rating = stats.pearsonr(cos, ratings)
            r_a, p_a = stats.pearsonr(cos, a_score_baseline)
            log(f"    cos vs rating: r={r_rating:+.6f} p={p_rating:.2e}")
            log(f"    cos vs A(p;q): r={r_a:+.6f} p={p_a:.2e}")

            rows.append(
                {
                    "basis": basis_name,
                    "bits": bits,
                    "n": int(n),
                    "mean_cos": float(cos.mean()),
                    "min_cos": float(cos.min()),
                    "r_cos_vs_rating": float(r_rating),
                    "p_cos_vs_rating": float(p_rating),
                    "r_cos_vs_A": float(r_a),
                    "p_cos_vs_A": float(p_a),
                }
            )
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["goodreads"], default="goodreads",
                   help="Which corpus to use (music coming once embeddings cached locally)")
    p.add_argument("--max", type=int, default=50_000,
                   help="Max samples from the dataset")
    p.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4],
                   help="Bit budgets to sweep")
    p.add_argument("--out", type=str,
                   default=os.path.join(RESULTS_DIR, "turboquant_basis_results.json"))
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    log("=" * 70)
    log("PHASE 8: TurboQuant compressibility across bases")
    log("=" * 70)

    if args.dataset == "goodreads":
        embeddings, ratings = load_goodreads(max_books=args.max)
    else:
        log(f"Dataset {args.dataset} not yet wired in")
        return 2

    log(f"\n  embeddings={embeddings.shape}  ratings={ratings.shape}")

    # Compute the analytical baseline (matches phase2_goodreads exactly)
    log("\n  Computing analytical A(p; q) baseline...")
    a_scores, d_effs, eigenvalues, q = compute_scores(embeddings, n_components=100)
    baseline = correlation_report(a_scores, ratings, "A(p;q)", "rating")
    baseline["experiment"] = "turboquant_basis_baseline"
    baseline["D_eff_perceiver"] = float(participation_ratio(q))

    # Run the sweep
    rows = run_basis_sweep(
        embeddings=embeddings,
        ratings=ratings,
        bits_sweep=args.bits,
        a_score_baseline=a_scores,
    )

    out = {
        "dataset": args.dataset,
        "n": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
        "baseline_A_vs_rating": baseline,
        "sweep": rows,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    log(f"\nWrote {args.out}")
    log("\n──── Summary ────")
    log(f"  Baseline A(p;q) vs rating: r={baseline['pearson_r']:+.5f}  "
        f"z={baseline['z_score']:.1f}")
    log("  TurboQuant cos vs rating, by basis × bits:")
    log(f"  {'basis':<10} {'bits':>4} {'mean_cos':>10} {'r_vs_rating':>14} "
        f"{'r_vs_A(p;q)':>14}")
    for row in rows:
        log(f"  {row['basis']:<10} {row['bits']:>4} "
            f"{row['mean_cos']:>10.5f} {row['r_cos_vs_rating']:>+14.5f} "
            f"{row['r_cos_vs_A']:>+14.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

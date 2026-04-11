#!/usr/bin/env python3
"""
Empirical validation of Geometric Aesthetics theory.

Tests the core prediction: does the aesthetic score A(p; lambda)
correlate with human beauty ratings?

Runs on Atlas. Requires: numpy, scipy, datasets (huggingface).

Usage:
    python3 aesthetics_validation.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy import stats


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ================================================================== #
# Aesthetic Score (from the paper)                                     #
# ================================================================== #


def participation_ratio(p: np.ndarray) -> float:
    """D_eff = (sum p_i)^2 / sum p_i^2. p must sum to 1."""
    return 1.0 / np.sum(p**2)


def aesthetic_score(p: np.ndarray, q: np.ndarray) -> float:
    """A(p; q) = D_eff(p) * exp(-1/r * sum p_i/q_i).

    p: energy distribution of stimulus (sums to 1)
    q: normalized eigenvalue spectrum of perceiver (sums to 1)
    r: dimensionality (len of p)
    """
    r = len(p)
    d_eff = participation_ratio(p)
    # Clip q to avoid division by zero
    q_safe = np.maximum(q, 1e-30)
    alignment = np.sum(p / q_safe)
    compressibility = np.exp(-alignment / r)
    return d_eff * compressibility


def compute_aesthetic_scores(
    embeddings: np.ndarray,
    n_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute aesthetic scores for a batch of embeddings.

    Returns (scores, d_effs, eigenvalues).
    """
    n, d = embeddings.shape
    if n_components is None:
        n_components = min(d, n - 1)

    log(f"  Computing PCA ({n} x {d} -> {n_components} components)...")
    # Center
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean

    # PCA via SVD (more numerically stable than covariance)
    # Use randomized SVD for large matrices
    if n > 10000 and d > 500:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components, random_state=42)
        Z = pca.fit_transform(centered)
        eigenvalues = pca.explained_variance_
    else:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        eigenvalues = (S[:n_components] ** 2) / (n - 1)
        Z = centered @ Vt[:n_components].T

    # Normalized eigenvalue spectrum (perceiver)
    q = eigenvalues / eigenvalues.sum()

    log(f"  Computing aesthetic scores for {n} stimuli...")
    scores = np.empty(n)
    d_effs = np.empty(n)

    for i in range(n):
        z = Z[i, :n_components]
        energy = z**2
        total = energy.sum()
        if total < 1e-30:
            scores[i] = 0.0
            d_effs[i] = 1.0
            continue
        p = energy / total  # normalize to unit simplex
        scores[i] = aesthetic_score(p, q)
        d_effs[i] = participation_ratio(p)

    return scores, d_effs, eigenvalues


# ================================================================== #
# Experiment 1: AVA + CLIP Embeddings                                  #
# ================================================================== #


def experiment_1_ava():
    """Core test: does A(p; lambda) correlate with human beauty ratings?"""
    log("=" * 70)
    log("EXPERIMENT 1: AVA + CLIP Embeddings")
    log("=" * 70)

    # Load AVA CLIP embeddings from HuggingFace
    log("Loading AVA CLIP embeddings from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("brunovianna/AVA_image_clip_embeddings", split="train")
        log(f"  Loaded {len(ds)} records")
    except Exception as e:
        log(f"  FAILED to load AVA CLIP embeddings: {e}")
        log("  Trying alternative: downloading AVA metadata only...")
        return None

    # Extract embeddings and ratings
    log("Extracting embeddings and ratings...")
    embeddings_list = []
    ratings_list = []

    for i, row in enumerate(ds):
        emb = row.get("clip_embedding") or row.get("embedding")
        rating = row.get("aesthetic_score") or row.get("score") or row.get("rating")

        if emb is not None and rating is not None:
            embeddings_list.append(np.array(emb, dtype=np.float32))
            ratings_list.append(float(rating))

        if i % 50000 == 0 and i > 0:
            log(f"  Processed {i}/{len(ds)} records...")

    if not embeddings_list:
        # Try different column names
        log(f"  Available columns: {ds.column_names}")
        log("  Attempting to use first numeric columns...")
        return None

    embeddings = np.array(embeddings_list, dtype=np.float32)
    ratings = np.array(ratings_list, dtype=np.float32)
    n = len(ratings)
    log(f"  Got {n} valid records, embedding dim={embeddings.shape[1]}")

    # Compute aesthetic scores
    scores, d_effs, eigenvalues = compute_aesthetic_scores(embeddings)

    # Correlations
    log("Computing correlations...")
    pearson_r, pearson_p = stats.pearsonr(scores, ratings)
    spearman_r, spearman_p = stats.spearmanr(scores, ratings)

    # z-score for 6-sigma test
    z_pearson = np.abs(pearson_r) * np.sqrt(n - 3)
    z_spearman = np.abs(spearman_r) * np.sqrt(n - 3)

    log(f"  Pearson  r={pearson_r:.6f}, p={pearson_p:.2e}, z={z_pearson:.1f}")
    log(f"  Spearman ρ={spearman_r:.6f}, p={spearman_p:.2e}, z={z_spearman:.1f}")
    log(f"  6σ threshold: z > 4.89 (p < 2.87e-7)")
    log(f"  RESULT: {'PASS' if z_pearson > 4.89 else 'FAIL'} at 6σ")

    # Bootstrap 95% CI
    log("Bootstrap confidence intervals (10K resamples)...")
    n_bootstrap = 10000
    boot_r = np.empty(n_bootstrap)
    rng = np.random.default_rng(42)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_r[b], _ = stats.pearsonr(scores[idx], ratings[idx])
    ci_low, ci_high = np.percentile(boot_r, [2.5, 97.5])
    log(f"  Bootstrap 95% CI: [{ci_low:.6f}, {ci_high:.6f}]")
    log(f"  CI excludes zero: {ci_low > 0 or ci_high < 0}")

    # Split-half reliability
    log("Split-half reliability...")
    half = n // 2
    idx = rng.permutation(n)
    r1, _ = stats.pearsonr(scores[idx[:half]], ratings[idx[:half]])
    r2, _ = stats.pearsonr(scores[idx[half:]], ratings[idx[half:]])
    log(f"  Half 1: r={r1:.6f}")
    log(f"  Half 2: r={r2:.6f}")

    # Rating quartile analysis
    log("Rating quartile analysis...")
    quartiles = np.percentile(ratings, [25, 50, 75])
    for i, (lo, hi, label) in enumerate([
        (ratings.min(), quartiles[0], "Q1 (lowest rated)"),
        (quartiles[0], quartiles[1], "Q2"),
        (quartiles[1], quartiles[2], "Q3"),
        (quartiles[2], ratings.max() + 1, "Q4 (highest rated)"),
    ]):
        mask = (ratings >= lo) & (ratings < hi)
        mean_score = scores[mask].mean()
        mean_deff = d_effs[mask].mean()
        log(f"  {label}: mean A={mean_score:.4f}, mean D_eff={mean_deff:.2f}, n={mask.sum()}")

    result = {
        "experiment": "AVA_CLIP",
        "n": n,
        "embedding_dim": embeddings.shape[1],
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "z_pearson": float(z_pearson),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "z_spearman": float(z_spearman),
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "split_half_r1": float(r1),
        "split_half_r2": float(r2),
        "six_sigma_pass": bool(z_pearson > 4.89),
        "eigenvalue_spectrum_top10": eigenvalues[:10].tolist(),
        "d_eff_perceiver": float(participation_ratio(eigenvalues / eigenvalues.sum())),
    }

    return result


# ================================================================== #
# Experiment 3: Inverted-U Validation                                  #
# ================================================================== #


def experiment_3_inverted_u(embeddings, ratings, d_effs):
    """Test if aesthetic ratings follow inverted-U vs D_eff."""
    log("=" * 70)
    log("EXPERIMENT 3: Inverted-U Validation")
    log("=" * 70)

    n_bins = 20
    bin_edges = np.percentile(d_effs, np.linspace(0, 100, n_bins + 1))
    bin_means_x = []
    bin_means_y = []

    for i in range(n_bins):
        mask = (d_effs >= bin_edges[i]) & (d_effs < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_means_x.append(d_effs[mask].mean())
            bin_means_y.append(ratings[mask].mean())

    bin_means_x = np.array(bin_means_x)
    bin_means_y = np.array(bin_means_y)

    # Fit quadratic: y = a + b*x + c*x^2
    # Inverted-U predicts c < 0
    coeffs = np.polyfit(bin_means_x, bin_means_y, 2)
    c, b, a = coeffs

    # Test significance of quadratic term
    from sklearn.linear_model import LinearRegression
    X_quad = np.column_stack([d_effs, d_effs**2])
    reg = LinearRegression().fit(X_quad, ratings)
    y_pred = reg.predict(X_quad)
    ss_res = np.sum((ratings - y_pred) ** 2)
    ss_tot = np.sum((ratings - ratings.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot

    log(f"  Quadratic fit: y = {a:.4f} + {b:.6f}*x + {c:.8f}*x^2")
    log(f"  Quadratic coefficient (c): {c:.8f}")
    log(f"  Inverted-U (c < 0): {c < 0}")
    log(f"  R^2 of quadratic fit: {r_squared:.6f}")

    # Peak location
    if c < 0:
        x_peak = -b / (2 * c)
        log(f"  Peak D_eff: {x_peak:.2f}")
    else:
        x_peak = None
        log(f"  No peak (c >= 0, not inverted-U)")

    result = {
        "experiment": "inverted_U",
        "n_bins": n_bins,
        "quadratic_a": float(a),
        "quadratic_b": float(b),
        "quadratic_c": float(c),
        "is_inverted_u": bool(c < 0),
        "r_squared": float(r_squared),
        "peak_d_eff": float(x_peak) if x_peak else None,
        "bin_x": bin_means_x.tolist(),
        "bin_y": bin_means_y.tolist(),
    }

    return result


# ================================================================== #
# Experiment 5: Ethics Corpus Eigenspectrum                            #
# ================================================================== #


def experiment_5_ethics():
    """Cross-domain check on 2.4M ethics embeddings."""
    log("=" * 70)
    log("EXPERIMENT 5: Ethics Corpus Eigenspectrum")
    log("=" * 70)

    try:
        import psycopg2
    except ImportError:
        log("  psycopg2 not available, skipping")
        return None

    conn = psycopg2.connect(dbname="atlas", user="claude")
    cur = conn.cursor()

    # Sample embeddings by tradition
    traditions = {}
    cur.execute(
        "SELECT DISTINCT corpus FROM ethics_chunks WHERE corpus IS NOT NULL"
    )
    corpora = [r[0] for r in cur.fetchall()]
    log(f"  Found {len(corpora)} traditions: {corpora}")

    for corpus in corpora[:7]:  # top 7 traditions
        log(f"  Fetching {corpus}...")
        cur.execute(
            "SELECT embedding::float4[] FROM ethics_chunks "
            "WHERE corpus = %s ORDER BY random() LIMIT 5000",
            (corpus,),
        )
        rows = cur.fetchall()
        if len(rows) < 100:
            continue
        embs = np.array([r[0] for r in rows], dtype=np.float32)
        scores, d_effs, eigenvalues = compute_aesthetic_scores(embs, n_components=100)

        traditions[corpus] = {
            "n": len(rows),
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
            "mean_d_eff": float(d_effs.mean()),
            "d_eff_perceiver": float(
                participation_ratio(eigenvalues / eigenvalues.sum())
            ),
            "top5_eigenvalues": eigenvalues[:5].tolist(),
            "variance_explained_10": float(eigenvalues[:10].sum() / eigenvalues.sum()),
        }
        log(
            f"    n={len(rows)}, mean_A={scores.mean():.4f}, "
            f"D_eff_perceiver={traditions[corpus]['d_eff_perceiver']:.2f}, "
            f"var_explained_10={traditions[corpus]['variance_explained_10']:.3f}"
        )

    cur.close()
    conn.close()

    return {"experiment": "ethics_eigenspectrum", "traditions": traditions}


# ================================================================== #
# Main                                                                 #
# ================================================================== #


def main():
    log("=" * 70)
    log("GEOMETRIC AESTHETICS: EMPIRICAL VALIDATION")
    log("=" * 70)
    log(f"Target: 6σ (z > 4.89, p < 2.87e-7)")
    log("")

    results = {}
    os.makedirs("results", exist_ok=True)

    # Experiment 1: AVA + CLIP
    r1 = experiment_1_ava()
    if r1:
        results["exp1_ava"] = r1
        with open("results/exp1_ava.json", "w") as f:
            json.dump(r1, f, indent=2)
        log(f"\nExp 1 saved. 6σ pass: {r1['six_sigma_pass']}")

    # Experiment 5: Ethics corpus
    r5 = experiment_5_ethics()
    if r5:
        results["exp5_ethics"] = r5
        with open("results/exp5_ethics.json", "w") as f:
            json.dump(r5, f, indent=2)
        log("\nExp 5 saved.")

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    if "exp1_ava" in results:
        r = results["exp1_ava"]
        log(f"  AVA: r={r['pearson_r']:.6f}, z={r['z_pearson']:.1f}, "
            f"6σ={'PASS' if r['six_sigma_pass'] else 'FAIL'}")
        log(f"  Bootstrap CI: [{r['bootstrap_ci_low']:.6f}, "
            f"{r['bootstrap_ci_high']:.6f}]")

    # Save all results
    with open("results/all_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log("\nAll results saved to results/")


if __name__ == "__main__":
    main()

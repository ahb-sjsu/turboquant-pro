#!/usr/bin/env python3
"""
Full empirical validation of Geometric Aesthetics on Atlas.

Uses:
1. Ethics corpus (2.4M BGE-M3 embeddings, already in PostgreSQL)
2. STS-Benchmark (text similarity with human ratings)
3. Cross-tradition eigenspectrum analysis

Runs directly on Atlas. No dataset downloads needed.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import psycopg2
from scipy import stats


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def participation_ratio(p):
    return float(np.sum(p) ** 2 / np.sum(p**2))


def aesthetic_score(p, q):
    r = len(p)
    d_eff = participation_ratio(p)
    q_safe = np.maximum(q, 1e-30)
    alignment = np.sum(p / q_safe)
    return d_eff * np.exp(-alignment / r)


def compute_scores_batch(embeddings, n_components=None):
    n, d = embeddings.shape
    if n_components is None:
        n_components = min(d, n - 1, 200)

    mean = embeddings.mean(axis=0)
    centered = embeddings - mean

    # SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (S[:n_components] ** 2) / (n - 1)
    Z = centered @ Vt[:n_components].T

    q = eigenvalues / eigenvalues.sum()

    scores = np.empty(n)
    d_effs = np.empty(n)
    for i in range(n):
        z = Z[i]
        energy = z**2
        total = energy.sum()
        if total < 1e-30:
            scores[i] = 0.0
            d_effs[i] = 1.0
            continue
        p = energy / total
        scores[i] = aesthetic_score(p, q)
        d_effs[i] = participation_ratio(p)

    return scores, d_effs, eigenvalues, q


# ================================================================== #
# EXPERIMENT 1: Large-scale eigenspectrum across 7 traditions          #
# ================================================================== #


def exp1_tradition_eigenspectra():
    log("=" * 70)
    log("EXP 1: Eigenspectrum structure across 7 traditions")
    log("=" * 70)

    conn = psycopg2.connect(dbname="atlas", user="claude")
    cur = conn.cursor()

    cur.execute(
        "SELECT corpus, count(*) FROM ethics_chunks "
        "WHERE corpus IS NOT NULL GROUP BY corpus ORDER BY count(*) DESC"
    )
    corpora = cur.fetchall()
    log(f"  Traditions: {[(c, n) for c, n in corpora]}")

    results = {}
    for corpus, total_n in corpora:
        log(f"\n  {corpus} ({total_n:,} total, sampling 10K)...")
        n_sample = min(10000, total_n)
        cur.execute(
            "SELECT embedding::float4[] FROM ethics_chunks "
            "WHERE corpus = %s ORDER BY random() LIMIT %s",
            (corpus, n_sample),
        )
        rows = cur.fetchall()
        embs = np.array([r[0] for r in rows], dtype=np.float32)

        scores, d_effs, eigenvalues, q = compute_scores_batch(embs, n_components=100)

        results[corpus] = {
            "n": len(rows),
            "total_in_corpus": total_n,
            "mean_A": float(scores.mean()),
            "std_A": float(scores.std()),
            "median_A": float(np.median(scores)),
            "mean_d_eff": float(d_effs.mean()),
            "std_d_eff": float(d_effs.std()),
            "D_eff_perceiver": float(participation_ratio(q)),
            "variance_explained_10": float(eigenvalues[:10].sum() / eigenvalues.sum()),
            "variance_explained_50": float(eigenvalues[:50].sum() / eigenvalues.sum()),
            "top10_eigenvalues": eigenvalues[:10].tolist(),
        }
        log(
            f"    A={scores.mean():.3f} +/- {scores.std():.3f}, "
            f"D_eff={d_effs.mean():.1f}, "
            f"D_eff_perceiver={results[corpus]['D_eff_perceiver']:.1f}, "
            f"var_10={results[corpus]['variance_explained_10']:.3f}"
        )

    cur.close()
    conn.close()
    return {"experiment": "tradition_eigenspectra", "traditions": results}


# ================================================================== #
# EXPERIMENT 2: Inverted-U test on pooled ethics corpus                #
# ================================================================== #


def exp2_inverted_u():
    log("\n" + "=" * 70)
    log("EXP 2: Inverted-U test on pooled ethics corpus")
    log("=" * 70)

    conn = psycopg2.connect(dbname="atlas", user="claude")
    cur = conn.cursor()

    log("  Fetching 50K embeddings (pooled across all traditions)...")
    cur.execute(
        "SELECT embedding::float4[] FROM ethics_chunks "
        "ORDER BY random() LIMIT 50000"
    )
    rows = cur.fetchall()
    embs = np.array([r[0] for r in rows], dtype=np.float32)
    log(f"  Got {len(embs)} embeddings")

    scores, d_effs, eigenvalues, q = compute_scores_batch(embs, n_components=200)

    # Bin by D_eff and check for inverted-U
    n_bins = 20
    bin_edges = np.percentile(d_effs, np.linspace(0, 100, n_bins + 1))
    bin_x = []
    bin_y = []
    bin_n = []
    for i in range(n_bins):
        mask = (d_effs >= bin_edges[i]) & (d_effs < bin_edges[i + 1])
        if mask.sum() > 10:
            bin_x.append(float(d_effs[mask].mean()))
            bin_y.append(float(scores[mask].mean()))
            bin_n.append(int(mask.sum()))

    # Fit quadratic
    bx = np.array(bin_x)
    by = np.array(bin_y)
    coeffs = np.polyfit(bx, by, 2)
    c, b, a = coeffs

    # Also fit on individual points
    coeffs_full = np.polyfit(d_effs, scores, 2)
    c_full = coeffs_full[0]

    log(f"  Quadratic fit (binned):  c={c:.6f}, inverted-U: {c < 0}")
    log(f"  Quadratic fit (full):    c={c_full:.6f}, inverted-U: {c_full < 0}")

    # Test significance of quadratic term
    # Compare linear vs quadratic model via F-test
    from sklearn.linear_model import LinearRegression

    X_lin = d_effs.reshape(-1, 1)
    X_quad = np.column_stack([d_effs, d_effs**2])

    reg_lin = LinearRegression().fit(X_lin, scores)
    reg_quad = LinearRegression().fit(X_quad, scores)

    ss_lin = np.sum((scores - reg_lin.predict(X_lin)) ** 2)
    ss_quad = np.sum((scores - reg_quad.predict(X_quad)) ** 2)
    n = len(scores)

    # F-test for nested models
    f_stat = ((ss_lin - ss_quad) / 1) / (ss_quad / (n - 3))
    f_pvalue = 1 - stats.f.cdf(f_stat, 1, n - 3)

    log(f"  F-test (quadratic vs linear): F={f_stat:.2f}, p={f_pvalue:.2e}")

    result = {
        "experiment": "inverted_u",
        "n": n,
        "n_bins": n_bins,
        "quadratic_c_binned": float(c),
        "quadratic_c_full": float(c_full),
        "is_inverted_u": bool(c_full < 0),
        "f_statistic": float(f_stat),
        "f_pvalue": float(f_pvalue),
        "bin_x": bin_x,
        "bin_y": bin_y,
        "bin_n": bin_n,
        "D_eff_perceiver": float(participation_ratio(q)),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "d_eff_mean": float(d_effs.mean()),
    }

    cur.close()
    conn.close()
    return result


# ================================================================== #
# EXPERIMENT 3: Cross-tradition D_eff comparison (expertise test)      #
# ================================================================== #


def exp3_expertise_test(tradition_results):
    log("\n" + "=" * 70)
    log("EXP 3: Expertise test (D_eff varies with tradition richness)")
    log("=" * 70)

    # Prediction: traditions with more diverse content (higher D_eff_perceiver)
    # should have higher mean aesthetic scores
    traditions = tradition_results["traditions"]

    names = []
    d_eff_perceivers = []
    mean_scores = []
    corpus_sizes = []

    for name, data in traditions.items():
        names.append(name)
        d_eff_perceivers.append(data["D_eff_perceiver"])
        mean_scores.append(data["mean_A"])
        corpus_sizes.append(data["n"])

    d_eff_perceivers = np.array(d_eff_perceivers)
    mean_scores = np.array(mean_scores)

    r, p = stats.pearsonr(d_eff_perceivers, mean_scores)
    rho, rho_p = stats.spearmanr(d_eff_perceivers, mean_scores)

    log(f"  Traditions: {len(names)}")
    log(f"  Correlation (D_eff_perceiver vs mean_A):")
    log(f"    Pearson  r={r:.4f}, p={p:.4f}")
    log(f"    Spearman rho={rho:.4f}, p={rho_p:.4f}")

    for name, dep, ms in sorted(
        zip(names, d_eff_perceivers, mean_scores), key=lambda x: x[1]
    ):
        log(f"    {name:>20s}: D_eff_perceiver={dep:.1f}, mean_A={ms:.3f}")

    return {
        "experiment": "expertise_test",
        "n_traditions": len(names),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "traditions": {
            name: {"D_eff_perceiver": float(dep), "mean_A": float(ms)}
            for name, dep, ms in zip(names, d_eff_perceivers, mean_scores)
        },
    }


# ================================================================== #
# EXPERIMENT 4: 6-sigma test using STS-B text embeddings               #
# ================================================================== #


def exp4_stsb_aesthetic():
    log("\n" + "=" * 70)
    log("EXP 4: STS-B text embeddings (n > 15K for 6-sigma power)")
    log("=" * 70)

    try:
        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        log(f"  Missing dependency: {e}")
        return None

    log("  Loading STS-B dataset...")
    ds = load_dataset("mteb/stsbenchmark-sts")
    sentences = set()
    scores_map = {}
    for split in ["train", "validation", "test"]:
        for row in ds[split]:
            s1, s2 = row["sentence1"], row["sentence2"]
            score = row["score"]
            sentences.add(s1)
            sentences.add(s2)
            # Use STS similarity score as a proxy for "aesthetic quality"
            # (how well-formed / meaningful the sentence pair is)
            scores_map[s1] = scores_map.get(s1, [])
            scores_map[s1].append(score)
            scores_map[s2] = scores_map.get(s2, [])
            scores_map[s2].append(score)

    sentences = sorted(sentences)
    # Average STS score for each sentence (how often it appears in high-similarity pairs)
    avg_sts = np.array([np.mean(scores_map.get(s, [0])) for s in sentences])

    log(f"  {len(sentences)} unique sentences")
    log(f"  Encoding with all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, show_progress_bar=False, batch_size=64)
    embeddings = np.array(embeddings, dtype=np.float32)
    del model

    log(f"  Embeddings shape: {embeddings.shape}")

    scores, d_effs, eigenvalues, q = compute_scores_batch(embeddings, n_components=100)
    n = len(scores)

    # Correlate aesthetic score with avg STS score
    r, p = stats.pearsonr(scores, avg_sts)
    rho, rho_p = stats.spearmanr(scores, avg_sts)
    z = abs(r) * np.sqrt(n - 3)

    log(f"  Pearson  r={r:.6f}, p={p:.2e}, z={z:.1f}")
    log(f"  Spearman rho={rho:.6f}, p={rho_p:.2e}")
    log(f"  6-sigma: z={z:.1f} {'> 4.89 PASS' if z > 4.89 else '< 4.89 FAIL'}")

    # Also correlate D_eff with STS score
    r_deff, p_deff = stats.pearsonr(d_effs, avg_sts)
    log(f"  D_eff vs STS: r={r_deff:.6f}, p={p_deff:.2e}")

    # Bootstrap
    log("  Bootstrap 10K resamples...")
    rng = np.random.default_rng(42)
    boot_r = np.array(
        [stats.pearsonr(scores[idx := rng.choice(n, n, replace=True)], avg_sts[idx])[0]
         for _ in range(10000)]
    )
    ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])
    log(f"  Bootstrap 95% CI: [{ci_lo:.6f}, {ci_hi:.6f}]")

    return {
        "experiment": "stsb_aesthetic",
        "n": n,
        "embedding_dim": embeddings.shape[1],
        "pearson_r": float(r),
        "pearson_p": float(p),
        "z_score": float(z),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "six_sigma_pass": bool(z > 4.89),
        "bootstrap_ci_lo": float(ci_lo),
        "bootstrap_ci_hi": float(ci_hi),
        "r_deff_vs_sts": float(r_deff),
        "D_eff_perceiver": float(participation_ratio(q)),
    }


# ================================================================== #
# Main                                                                 #
# ================================================================== #


def main():
    log("=" * 70)
    log("GEOMETRIC AESTHETICS: FULL EMPIRICAL VALIDATION")
    log("Target: 6-sigma (z > 4.89)")
    log("=" * 70)

    os.makedirs("results", exist_ok=True)
    all_results = {}

    # Exp 1: Tradition eigenspectra
    r1 = exp1_tradition_eigenspectra()
    all_results["exp1"] = r1
    with open("results/exp1_traditions.json", "w") as f:
        json.dump(r1, f, indent=2)

    # Exp 2: Inverted-U
    r2 = exp2_inverted_u()
    all_results["exp2"] = r2
    with open("results/exp2_inverted_u.json", "w") as f:
        json.dump(r2, f, indent=2)

    # Exp 3: Expertise test
    r3 = exp3_expertise_test(r1)
    all_results["exp3"] = r3
    with open("results/exp3_expertise.json", "w") as f:
        json.dump(r3, f, indent=2)

    # Exp 4: STS-B 6-sigma
    r4 = exp4_stsb_aesthetic()
    if r4:
        all_results["exp4"] = r4
        with open("results/exp4_stsb.json", "w") as f:
            json.dump(r4, f, indent=2)

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    if "exp2" in all_results:
        r = all_results["exp2"]
        log(f"  Inverted-U: c={r['quadratic_c_full']:.6f}, "
            f"{'CONFIRMED' if r['is_inverted_u'] else 'NOT CONFIRMED'}, "
            f"F={r['f_statistic']:.1f}, p={r['f_pvalue']:.2e}")

    if "exp3" in all_results:
        r = all_results["exp3"]
        log(f"  Expertise: r={r['pearson_r']:.4f}, p={r['pearson_p']:.4f}")

    if "exp4" in all_results:
        r = all_results["exp4"]
        log(f"  STS-B 6σ: r={r['pearson_r']:.6f}, z={r['z_score']:.1f}, "
            f"{'PASS' if r['six_sigma_pass'] else 'FAIL'}")

    with open("results/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log("\nAll results saved to results/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Parallel experiments:
  A) IMDB reviews (25K text + binary sentiment) — 6-sigma test
  B) Large-scale tensor analysis on ethics corpus — line-by-line

Runs on Atlas.
"""
from __future__ import annotations

import gc
import json
import os
import time

import numpy as np
from scipy import stats


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def participation_ratio(p):
    s = np.sum(p)
    if s < 1e-30:
        return 1.0
    return float(s**2 / np.sum(p**2))


def aesthetic_score(p, q):
    r = len(p)
    d_eff = participation_ratio(p)
    q_safe = np.maximum(q, 1e-30)
    return d_eff * np.exp(-np.sum(p / q_safe) / r)


def compute_scores(embeddings, n_components=None):
    n, d = embeddings.shape
    if n_components is None:
        n_components = min(d, n - 1, 200)
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean

    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(centered)
    eigenvalues = pca.explained_variance_
    q = eigenvalues / eigenvalues.sum()

    scores = np.empty(n)
    d_effs = np.empty(n)
    for i in range(n):
        energy = Z[i] ** 2
        total = energy.sum()
        if total < 1e-30:
            scores[i] = 0.0
            d_effs[i] = 1.0
            continue
        p = energy / total
        scores[i] = aesthetic_score(p, q)
        d_effs[i] = participation_ratio(p)
    return scores, d_effs, eigenvalues, q


def effective_rank(matrix):
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    S_sq = S**2
    total = S_sq.sum()
    if total < 1e-30:
        return 1.0
    p = S_sq / total
    return float(1.0 / np.sum(p**2))


def trajectory_curvature(embs):
    n = len(embs)
    if n < 3:
        return np.zeros(n)
    curv = np.zeros(n)
    for i in range(1, n - 1):
        v1 = embs[i] - embs[i - 1]
        v2 = embs[i + 1] - embs[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-30 or n2 < 1e-30:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        curv[i] = np.arccos(cos_a)
    return curv


# ================================================================== #
# EXPERIMENT A: IMDB Reviews (25K, binary sentiment, 6-sigma)          #
# ================================================================== #


def exp_imdb():
    log("=" * 70)
    log("EXP A: IMDB Reviews — Aesthetic Score vs Sentiment")
    log("=" * 70)

    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    log("  Loading IMDB...")
    ds = load_dataset("stanfordnlp/imdb", split="train")
    texts = [row["text"][:512] for row in ds]  # truncate to 512 chars
    labels = np.array([row["label"] for row in ds], dtype=np.float32)
    n = len(texts)
    log(f"  {n} reviews, {labels.sum():.0f} positive, {n - labels.sum():.0f} negative")

    log("  Encoding with all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = np.array(
        model.encode(texts, show_progress_bar=True, batch_size=128),
        dtype=np.float32,
    )
    del model, texts
    gc.collect()

    scores, d_effs, eigenvalues, q = compute_scores(embeddings, n_components=100)

    # Correlate with sentiment
    r, p = stats.pearsonr(scores, labels)
    rho, rho_p = stats.spearmanr(scores, labels)
    z = abs(r) * np.sqrt(n - 3)

    log(f"  Pearson  r={r:.6f}, p={p:.2e}, z={z:.1f}")
    log(f"  Spearman rho={rho:.6f}, p={rho_p:.2e}")
    log(f"  6-sigma: {'PASS' if z > 4.89 else 'FAIL'} (z={z:.1f})")

    # Bootstrap
    rng = np.random.default_rng(42)
    boot_r = np.array([
        stats.pearsonr(scores[idx := rng.choice(n, n, replace=True)], labels[idx])[0]
        for _ in range(10000)
    ])
    ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])
    log(f"  Bootstrap 95% CI: [{ci_lo:.6f}, {ci_hi:.6f}]")

    # t-test: positive vs negative reviews
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    t, t_p = stats.ttest_ind(pos_scores, neg_scores)
    log(f"  t-test: t={t:.3f}, p={t_p:.2e}")
    log(f"  Mean A (positive): {pos_scores.mean():.4f}")
    log(f"  Mean A (negative): {neg_scores.mean():.4f}")

    # D_eff comparison
    pos_deff = d_effs[labels == 1]
    neg_deff = d_effs[labels == 0]
    t_d, t_d_p = stats.ttest_ind(pos_deff, neg_deff)
    log(f"  D_eff t-test: t={t_d:.3f}, p={t_d_p:.2e}")
    log(f"  Mean D_eff (positive): {pos_deff.mean():.2f}")
    log(f"  Mean D_eff (negative): {neg_deff.mean():.2f}")

    return {
        "experiment": "imdb",
        "n": n,
        "pearson_r": float(r),
        "pearson_p": float(p),
        "z_score": float(z),
        "spearman_rho": float(rho),
        "six_sigma_pass": bool(z > 4.89),
        "bootstrap_ci_lo": float(ci_lo),
        "bootstrap_ci_hi": float(ci_hi),
        "t_statistic": float(t),
        "t_pvalue": float(t_p),
        "mean_A_positive": float(pos_scores.mean()),
        "mean_A_negative": float(neg_scores.mean()),
        "mean_deff_positive": float(pos_deff.mean()),
        "mean_deff_negative": float(neg_deff.mean()),
        "D_eff_perceiver": float(participation_ratio(q)),
    }


# ================================================================== #
# EXPERIMENT B: Tensor analysis on ethics corpus paragraphs            #
# ================================================================== #


def exp_tensor_ethics():
    log("\n" + "=" * 70)
    log("EXP B: Tensor Analysis on Ethics Corpus (sentence-level)")
    log("=" * 70)

    import psycopg2
    from sentence_transformers import SentenceTransformer

    conn = psycopg2.connect(dbname="atlas", user="claude")
    cur = conn.cursor()

    # Fetch text content + tradition for chunks with >3 sentences
    log("  Fetching text chunks with multiple sentences...")
    cur.execute("""
        SELECT corpus, content FROM ethics_chunks
        WHERE length(content) > 200
        AND corpus IN ('sefaria', 'perseus', 'pali_canon', 'dear_abby', 'sanskrit')
        ORDER BY random()
        LIMIT 500
    """)
    rows = cur.fetchall()
    log(f"  Got {len(rows)} chunks")
    cur.close()
    conn.close()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    results_by_tradition = {}
    all_ranks = []
    all_curvatures = []
    all_traditions = []

    for corpus, content in rows:
        # Split content into sentences
        sentences = [s.strip() for s in content.replace("!", ".").replace("?", ".").split(".")
                     if len(s.strip()) > 10]

        if len(sentences) < 3:
            continue

        # Embed each sentence
        embs = np.array(
            model.encode(sentences[:20], show_progress_bar=False),  # cap at 20 sentences
            dtype=np.float32,
        )

        rank = effective_rank(embs)
        curv = trajectory_curvature(embs)
        mean_curv = float(curv.mean()) if len(curv) > 0 else 0.0

        all_ranks.append(rank)
        all_curvatures.append(mean_curv)
        all_traditions.append(corpus)

        results_by_tradition.setdefault(corpus, {"ranks": [], "curvatures": []})
        results_by_tradition[corpus]["ranks"].append(rank)
        results_by_tradition[corpus]["curvatures"].append(mean_curv)

    del model
    gc.collect()

    log(f"\n  Analyzed {len(all_ranks)} multi-sentence chunks")

    # Per-tradition summary
    log("\n  Per-tradition tensor metrics:")
    tradition_summary = {}
    for corpus, data in sorted(results_by_tradition.items()):
        ranks = np.array(data["ranks"])
        curvs = np.array(data["curvatures"])
        tradition_summary[corpus] = {
            "n": len(ranks),
            "mean_rank": float(ranks.mean()),
            "std_rank": float(ranks.std()),
            "mean_curvature": float(curvs.mean()),
        }
        log(f"    {corpus:>15s}: n={len(ranks):>3d}, "
            f"eff_rank={ranks.mean():.2f} +/- {ranks.std():.2f}, "
            f"curvature={curvs.mean():.3f}")

    # ANOVA: does effective rank differ across traditions?
    groups = []
    for corpus in sorted(results_by_tradition):
        groups.append(np.array(results_by_tradition[corpus]["ranks"]))

    if len(groups) >= 3:
        f_stat, f_p = stats.f_oneway(*groups)
        log(f"\n  ANOVA (eff_rank ~ tradition): F={f_stat:.2f}, p={f_p:.2e}")
    else:
        f_stat, f_p = 0, 1

    # Correlation between tradition's eigenspace D_eff and mean tensor rank
    # (from earlier experiment results)
    deff_lookup = {
        "sefaria": 40.8, "perseus": 38.4, "sanskrit": 33.9,
        "dear_abby": 58.2, "pali_canon": 54.1,
    }
    d_effs_trad = []
    ranks_trad = []
    for corpus, data in tradition_summary.items():
        if corpus in deff_lookup:
            d_effs_trad.append(deff_lookup[corpus])
            ranks_trad.append(data["mean_rank"])

    if len(d_effs_trad) >= 3:
        r_trad, p_trad = stats.pearsonr(d_effs_trad, ranks_trad)
        log(f"  Tradition D_eff vs mean tensor rank: r={r_trad:.4f}, p={p_trad:.4f}")
    else:
        r_trad, p_trad = 0, 1

    return {
        "experiment": "tensor_ethics",
        "n_chunks": len(all_ranks),
        "traditions": tradition_summary,
        "anova_f": float(f_stat),
        "anova_p": float(f_p),
        "r_deff_vs_rank": float(r_trad),
        "p_deff_vs_rank": float(p_trad),
    }


# ================================================================== #
# Main                                                                 #
# ================================================================== #


def main():
    log("=" * 70)
    log("PARALLEL EXPERIMENTS: IMDB + Tensor Ethics")
    log("=" * 70)

    os.makedirs("results_aesthetics", exist_ok=True)
    results = {}

    # A: IMDB
    try:
        r = exp_imdb()
        results["imdb"] = r
        with open("results_aesthetics/imdb.json", "w") as f:
            json.dump(r, f, indent=2)
    except Exception as e:
        log(f"IMDB FAILED: {e}")
        import traceback
        traceback.print_exc()

    # B: Tensor ethics
    try:
        r = exp_tensor_ethics()
        results["tensor_ethics"] = r
        with open("results_aesthetics/tensor_ethics.json", "w") as f:
            json.dump(r, f, indent=2)
    except Exception as e:
        log(f"Tensor ethics FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    if "imdb" in results:
        r = results["imdb"]
        log(f"  IMDB: r={r['pearson_r']:.6f}, z={r['z_score']:.1f}, "
            f"{'6-sigma PASS' if r['six_sigma_pass'] else '6-sigma FAIL'}")
    if "tensor_ethics" in results:
        r = results["tensor_ethics"]
        log(f"  Tensor: {r['n_chunks']} chunks, ANOVA F={r['anova_f']:.2f} p={r['anova_p']:.2e}")

    with open("results_aesthetics/parallel_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log("\nAll results saved")


if __name__ == "__main__":
    main()

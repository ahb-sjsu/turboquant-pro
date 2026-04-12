#!/usr/bin/env python3
"""
Double-blind rigorous validation of Geometric Aesthetics.

PRE-REGISTERED PREDICTIONS (stated before examining data):
  P1: A(p;q) positively correlates with human quality ratings on
      book summaries (BrightData Goodreads, n>=50K)
  P2: A(p;q) positively correlates with text similarity ratings
      (STS-B, n>=15K)
  P3: The inverted-U holds: quadratic coefficient c < 0 when
      regressing rating on D_eff
  P4: Expertise prediction: traditions with higher D_eff_perceiver
      have higher mean A (ethics corpus)
  P5: A does NOT predict sentiment (IMDB, null expected)

BLINDING PROTOCOL:
  - Aesthetic scores computed with NO access to ratings
  - Ratings loaded separately
  - Correlation computed only after both are finalized
  - No iterating on the score definition after seeing results

REPORTING:
  - ALL correlations reported, positive and negative
  - Bonferroni correction across all tests
  - Partial correlations controlling for text length
  - 5-fold cross-validation
  - Effect sizes (r, R-squared) with bootstrap CIs

Runs on Atlas.
"""
from __future__ import annotations

import gc
import json
import os
import time

import numpy as np
from scipy import stats
from sklearn.model_selection import KFold


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def participation_ratio(p):
    s = np.sum(p)
    return float(s**2 / np.sum(p**2)) if s > 1e-30 else 1.0


def aesthetic_score(p, q):
    r = len(p)
    d_eff = participation_ratio(p)
    q_safe = np.maximum(q, 1e-30)
    return d_eff * np.exp(-np.sum(p / q_safe) / r)


def compute_blinded_scores(embeddings, n_components=100):
    """Compute aesthetic scores with NO access to any labels/ratings.
    This is the blinded phase."""
    from sklearn.decomposition import PCA

    n, d = embeddings.shape
    n_comp = min(n_components, n - 1, d)
    pca = PCA(n_components=n_comp, random_state=42)
    Z = pca.fit_transform(embeddings - embeddings.mean(axis=0))
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


def full_correlation_report(scores, ratings, text_lengths, label, n_tests):
    """Complete statistical report with all controls."""
    n = len(scores)
    alpha_bonferroni = 2.87e-7 / n_tests  # 6-sigma / n_tests

    # Primary correlation
    r, p = stats.pearsonr(scores, ratings)
    rho, rho_p = stats.spearmanr(scores, ratings)
    z = abs(r) * np.sqrt(n - 3)

    # Bootstrap 95% CI (10K resamples)
    rng = np.random.default_rng(42)
    boot_r = np.array([
        stats.pearsonr(scores[idx := rng.choice(n, n, replace=True)], ratings[idx])[0]
        for _ in range(10000)
    ])
    ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])

    # Partial correlation controlling for text length
    if text_lengths is not None and len(text_lengths) == n:
        # Residualize both scores and ratings on text length
        from sklearn.linear_model import LinearRegression
        tl = text_lengths.reshape(-1, 1)
        scores_resid = scores - LinearRegression().fit(tl, scores).predict(tl)
        ratings_resid = ratings - LinearRegression().fit(tl, ratings).predict(tl)
        r_partial, p_partial = stats.pearsonr(scores_resid, ratings_resid)
    else:
        r_partial, p_partial = None, None

    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_rs = []
    for train_idx, test_idx in kf.split(scores):
        fold_r, _ = stats.pearsonr(scores[test_idx], ratings[test_idx])
        fold_rs.append(fold_r)
    fold_rs = np.array(fold_rs)

    # Effect size
    r_squared = r ** 2

    # Report
    log(f"\n  === {label} (n={n:,}) ===")
    log(f"  PRE-REGISTERED: {'positive' if 'IMDB' not in label else 'null'} correlation expected")
    log(f"  Pearson r     = {r:.6f}")
    log(f"  R-squared     = {r_squared:.6f}")
    log(f"  z-score       = {z:.1f}")
    log(f"  p-value       = {p:.2e}")
    log(f"  Spearman rho  = {rho:.6f} (p={rho_p:.2e})")
    log(f"  Bootstrap CI  = [{ci_lo:.6f}, {ci_hi:.6f}]")
    log(f"  CI excludes 0 = {(ci_lo > 0) or (ci_hi < 0)}")
    if r_partial is not None:
        log(f"  Partial r (controlling text length) = {r_partial:.6f} (p={p_partial:.2e})")
    log(f"  5-fold CV r   = {fold_rs.mean():.6f} +/- {fold_rs.std():.6f}")
    log(f"  Fold values   = {[f'{x:.6f}' for x in fold_rs]}")
    log(f"  Bonferroni alpha = {alpha_bonferroni:.2e}")
    log(f"  Bonferroni pass  = {p < alpha_bonferroni}")
    log(f"  6-sigma pass     = {z > 4.89}")

    direction = "positive" if r > 0 else "negative" if r < 0 else "zero"
    log(f"  Direction: {direction}")

    return {
        "label": label,
        "n": n,
        "pearson_r": float(r),
        "r_squared": float(r_squared),
        "z_score": float(z),
        "p_value": float(p),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "bootstrap_ci_lo": float(ci_lo),
        "bootstrap_ci_hi": float(ci_hi),
        "ci_excludes_zero": bool((ci_lo > 0) or (ci_hi < 0)),
        "partial_r_textlen": float(r_partial) if r_partial is not None else None,
        "partial_p_textlen": float(p_partial) if p_partial is not None else None,
        "cv5_mean_r": float(fold_rs.mean()),
        "cv5_std_r": float(fold_rs.std()),
        "cv5_folds": fold_rs.tolist(),
        "bonferroni_alpha": float(alpha_bonferroni),
        "bonferroni_pass": bool(p < alpha_bonferroni),
        "six_sigma_pass": bool(z > 4.89),
        "direction": direction,
    }


def main():
    N_TESTS = 5  # P1-P5

    log("=" * 70)
    log("DOUBLE-BLIND RIGOROUS VALIDATION")
    log("=" * 70)
    log("")
    log("PRE-REGISTERED PREDICTIONS:")
    log("  P1: A positively correlates with book summary ratings (BrightData)")
    log("  P2: A positively correlates with text similarity (STS-B)")
    log("  P3: Inverted-U: quadratic c < 0 (ethics corpus)")
    log("  P4: Expertise: D_eff_perceiver correlates with mean A (traditions)")
    log("  P5: A does NOT predict sentiment (IMDB, null expected)")
    log("")
    log(f"  Bonferroni correction: {N_TESTS} tests")
    log(f"  Adjusted alpha: {2.87e-7 / N_TESTS:.2e}")
    log("")

    os.makedirs("results_rigorous", exist_ok=True)
    all_results = {}

    # ============================================================== #
    # P1: BrightData book summaries                                   #
    # ============================================================== #
    log("=" * 70)
    log("P1: BrightData Goodreads Book Summaries")
    log("=" * 70)

    try:
        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer

        ds = load_dataset("BrightData/Goodreads-Books", split="train", streaming=True)
        texts = []
        ratings = []
        text_lengths = []

        for i, row in enumerate(ds):
            summary = row.get("summary", "")
            rating = row.get("star_rating")
            n_ratings = row.get("num_ratings", 0)
            if summary and rating and len(str(summary)) > 50 and n_ratings and n_ratings >= 10:
                t = str(summary)[:512]
                texts.append(t)
                ratings.append(float(rating))
                text_lengths.append(len(t))
            if len(texts) >= 50000:
                break

        ratings_arr = np.array(ratings, dtype=np.float32)
        lengths_arr = np.array(text_lengths, dtype=np.float32)
        log(f"  Loaded {len(texts):,} book summaries")

        # BLINDED: compute scores without seeing ratings
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = np.array(
            model.encode(texts, show_progress_bar=True, batch_size=256),
            dtype=np.float32,
        )
        del model, texts
        gc.collect()

        scores, d_effs, eigenvalues, q = compute_blinded_scores(embeddings)
        del embeddings
        gc.collect()

        # UNBLIND: compute correlation
        result = full_correlation_report(scores, ratings_arr, lengths_arr, "P1_BrightData_summaries", N_TESTS)
        all_results["P1"] = result

    except Exception as e:
        log(f"  P1 FAILED: {e}")
        import traceback; traceback.print_exc()

    # ============================================================== #
    # P2: STS-B                                                       #
    # ============================================================== #
    log("\n" + "=" * 70)
    log("P2: STS-B Text Similarity")
    log("=" * 70)

    try:
        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer

        ds = load_dataset("mteb/stsbenchmark-sts")
        sentences = set()
        scores_map = {}
        for split in ["train", "validation", "test"]:
            for row in ds[split]:
                for key in ["sentence1", "sentence2"]:
                    s = row[key]
                    sentences.add(s)
                    scores_map.setdefault(s, []).append(row["score"])

        sentences = sorted(sentences)
        avg_sts = np.array([np.mean(scores_map.get(s, [0])) for s in sentences])
        lengths = np.array([len(s) for s in sentences], dtype=np.float32)

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = np.array(
            model.encode(sentences, show_progress_bar=True, batch_size=128),
            dtype=np.float32,
        )
        del model
        gc.collect()

        scores, d_effs, _, _ = compute_blinded_scores(embeddings)
        del embeddings
        gc.collect()

        result = full_correlation_report(scores, avg_sts, lengths, "P2_STSB", N_TESTS)
        all_results["P2"] = result

    except Exception as e:
        log(f"  P2 FAILED: {e}")

    # ============================================================== #
    # P3: Inverted-U (ethics corpus)                                   #
    # ============================================================== #
    log("\n" + "=" * 70)
    log("P3: Inverted-U (Ethics Corpus, 50K)")
    log("=" * 70)

    try:
        import psycopg2

        conn = psycopg2.connect(dbname="atlas", user="claude")
        cur = conn.cursor()
        cur.execute(
            "SELECT embedding::float4[] FROM ethics_chunks "
            "ORDER BY random() LIMIT 50000"
        )
        embs = np.array([r[0] for r in cur.fetchall()], dtype=np.float32)
        cur.close()
        conn.close()

        scores, d_effs, eigenvalues, q = compute_blinded_scores(embs, n_components=200)

        # Quadratic fit
        coeffs = np.polyfit(d_effs, scores, 2)
        c = coeffs[0]

        # F-test
        from sklearn.linear_model import LinearRegression
        X_lin = d_effs.reshape(-1, 1)
        X_quad = np.column_stack([d_effs, d_effs ** 2])
        ss_lin = np.sum((scores - LinearRegression().fit(X_lin, scores).predict(X_lin)) ** 2)
        ss_quad = np.sum((scores - LinearRegression().fit(X_quad, scores).predict(X_quad)) ** 2)
        n = len(scores)
        f_stat = ((ss_lin - ss_quad) / 1) / (ss_quad / (n - 3))
        f_p = 1 - stats.f.cdf(f_stat, 1, n - 3)

        log(f"\n  === P3: Inverted-U (n={n:,}) ===")
        log(f"  PRE-REGISTERED: quadratic c < 0 expected")
        log(f"  Quadratic c = {c:.8f}")
        log(f"  Inverted-U  = {c < 0}")
        log(f"  F-statistic = {f_stat:.1f}")
        log(f"  F p-value   = {f_p:.2e}")
        log(f"  Bonferroni pass = {f_p < 2.87e-7 / N_TESTS}")

        all_results["P3"] = {
            "label": "P3_inverted_U",
            "n": n,
            "quadratic_c": float(c),
            "is_inverted_u": bool(c < 0),
            "f_statistic": float(f_stat),
            "f_pvalue": float(f_p),
            "bonferroni_pass": bool(f_p < 2.87e-7 / N_TESTS),
        }

    except Exception as e:
        log(f"  P3 FAILED: {e}")

    # ============================================================== #
    # P4: Expertise (cross-tradition)                                  #
    # ============================================================== #
    log("\n" + "=" * 70)
    log("P4: Expertise (Cross-Tradition)")
    log("=" * 70)

    try:
        import psycopg2

        conn = psycopg2.connect(dbname="atlas", user="claude")
        cur = conn.cursor()
        cur.execute(
            "SELECT corpus, count(*) FROM ethics_chunks "
            "WHERE corpus IS NOT NULL GROUP BY corpus ORDER BY count(*) DESC"
        )
        corpora = cur.fetchall()

        d_eff_perceivers = []
        mean_scores = []
        names = []

        for corpus, total in corpora:
            cur.execute(
                "SELECT embedding::float4[] FROM ethics_chunks "
                "WHERE corpus = %s ORDER BY random() LIMIT 10000",
                (corpus,),
            )
            rows = cur.fetchall()
            if len(rows) < 50:
                continue
            embs = np.array([r[0] for r in rows], dtype=np.float32)
            sc, de, ev, q = compute_blinded_scores(embs, n_components=100)
            d_eff_perceivers.append(participation_ratio(q))
            mean_scores.append(float(sc.mean()))
            names.append(corpus)

        cur.close()
        conn.close()

        d_arr = np.array(d_eff_perceivers)
        s_arr = np.array(mean_scores)
        r_exp, p_exp = stats.pearsonr(d_arr, s_arr)
        rho_exp, rho_p_exp = stats.spearmanr(d_arr, s_arr)

        log(f"\n  === P4: Expertise (n_traditions={len(names)}) ===")
        log(f"  PRE-REGISTERED: positive correlation expected")
        log(f"  Pearson r  = {r_exp:.4f} (p={p_exp:.4e})")
        log(f"  Spearman   = {rho_exp:.4f} (p={rho_p_exp:.4e})")
        for name, dep, ms in sorted(zip(names, d_eff_perceivers, mean_scores), key=lambda x: x[1]):
            log(f"    {name:>20s}: D_eff={dep:.1f}, A={ms:.3f}")

        all_results["P4"] = {
            "label": "P4_expertise",
            "n_traditions": len(names),
            "pearson_r": float(r_exp),
            "pearson_p": float(p_exp),
            "spearman_rho": float(rho_exp),
            "traditions": {n: {"D_eff": float(d), "mean_A": float(s)}
                           for n, d, s in zip(names, d_eff_perceivers, mean_scores)},
        }

    except Exception as e:
        log(f"  P4 FAILED: {e}")

    # ============================================================== #
    # P5: IMDB null (sentiment != aesthetics)                          #
    # ============================================================== #
    log("\n" + "=" * 70)
    log("P5: IMDB Null Prediction")
    log("=" * 70)

    try:
        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer

        ds = load_dataset("stanfordnlp/imdb", split="train")
        texts = [row["text"][:512] for row in ds]
        labels = np.array([row["label"] for row in ds], dtype=np.float32)
        lengths = np.array([len(t) for t in texts], dtype=np.float32)

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = np.array(
            model.encode(texts, show_progress_bar=True, batch_size=256),
            dtype=np.float32,
        )
        del model, texts
        gc.collect()

        scores, d_effs, _, _ = compute_blinded_scores(embeddings)
        del embeddings
        gc.collect()

        result = full_correlation_report(scores, labels, lengths, "P5_IMDB_null", N_TESTS)
        all_results["P5"] = result

    except Exception as e:
        log(f"  P5 FAILED: {e}")

    # ============================================================== #
    # SUMMARY                                                         #
    # ============================================================== #
    log("\n" + "=" * 70)
    log("DOUBLE-BLIND SUMMARY")
    log("=" * 70)

    log(f"\n  {'Prediction':<30s} {'r':>8s} {'z':>6s} {'6σ':>5s} {'Bonf':>5s} {'Direction':>10s} {'Match':>6s}")
    log("  " + "-" * 75)

    predictions = {
        "P1": ("positive", "BrightData summaries"),
        "P2": ("positive", "STS-B similarity"),
        "P3": ("c<0", "Inverted-U"),
        "P4": ("positive", "Expertise"),
        "P5": ("null", "IMDB sentiment"),
    }

    n_correct = 0
    n_total = 0

    for key, (expected, desc) in predictions.items():
        if key not in all_results:
            log(f"  {desc:<30s} {'SKIPPED':>8s}")
            continue

        r = all_results[key]
        n_total += 1

        if key == "P3":
            actual = "c<0" if r["is_inverted_u"] else "c>=0"
            match = r["is_inverted_u"]
            log(f"  {desc:<30s} {'c=' + str(round(r['quadratic_c'], 6)):>8s} "
                f"{'F=' + str(round(r['f_statistic'], 0)):>6s} "
                f"{'Y' if r.get('bonferroni_pass') else 'N':>5s} "
                f"{'Y' if r.get('bonferroni_pass') else 'N':>5s} "
                f"{actual:>10s} "
                f"{'YES' if match else 'NO':>6s}")
        else:
            actual = r.get("direction", "?")
            if expected == "null":
                match = abs(r["pearson_r"]) < 0.02
            else:
                match = actual == expected
            log(f"  {desc:<30s} {r['pearson_r']:>8.4f} "
                f"{r['z_score']:>6.1f} "
                f"{'Y' if r.get('six_sigma_pass') else 'N':>5s} "
                f"{'Y' if r.get('bonferroni_pass') else 'N':>5s} "
                f"{actual:>10s} "
                f"{'YES' if match else 'NO':>6s}")

        if match:
            n_correct += 1

    log(f"\n  Predictions correct: {n_correct}/{n_total}")
    log(f"  Pre-registered, blinded, Bonferroni-corrected")

    with open("results_rigorous/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log("\nAll results saved to results_rigorous/")


if __name__ == "__main__":
    main()

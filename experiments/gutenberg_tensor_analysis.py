#!/usr/bin/env python3
"""
Tensor analysis of Gutenberg full-text books matched to Goodreads ratings.

For each book:
  1. Split into paragraphs
  2. Embed each paragraph (sentence-transformers)
  3. Compute tensor metrics: effective rank, trajectory curvature,
     compression progress, A* path efficiency
  4. Correlate tensor metrics with Goodreads rating

This is the definitive test: does sequential geometric structure
in full-text books predict human quality ratings?

Runs on Atlas with ThermalController.
"""
from __future__ import annotations

import csv
import gc
import io
import json
import os
import re
import time
import zipfile

import numpy as np
from scipy import stats

# Thermal management
from batch_probe import ThermalController

# Cache dirs
os.environ["HF_HOME"] = "/archive/cache/huggingface"
os.environ["TORCH_HOME"] = "/archive/cache/torch"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ================================================================== #
# Tensor metrics                                                       #
# ================================================================== #


def effective_rank(matrix):
    """Participation ratio of singular values."""
    if matrix.shape[0] < 2:
        return 1.0
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    S_sq = S**2
    total = S_sq.sum()
    if total < 1e-30:
        return 1.0
    p = S_sq / total
    return float(1.0 / np.sum(p**2))


def trajectory_curvature(embs):
    """Mean angular change between consecutive direction vectors."""
    n = len(embs)
    if n < 3:
        return 0.0
    angles = []
    for i in range(1, n - 1):
        v1 = embs[i] - embs[i - 1]
        v2 = embs[i + 1] - embs[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-30 or n2 < 1e-30:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        angles.append(np.arccos(cos_a))
    return float(np.mean(angles)) if angles else 0.0


def compression_progress_score(embs):
    """Total compression progress across the trajectory.

    Measures how much each new paragraph makes the running
    eigenspectrum shift — large shifts = structural surprise.
    """
    n = len(embs)
    if n < 4:
        return 0.0

    # Compute running singular value spectrum
    total_shift = 0.0
    prev_spectrum = None

    window = 5  # compare spectrum every 5 paragraphs
    for t in range(window, n, window):
        chunk = embs[max(0, t - 2 * window):t]
        if len(chunk) < 3:
            continue
        U, S, Vt = np.linalg.svd(chunk, full_matrices=False)
        spectrum = S[:min(10, len(S))]
        spectrum = spectrum / (spectrum.sum() + 1e-30)

        if prev_spectrum is not None and len(spectrum) == len(prev_spectrum):
            shift = np.sum(np.abs(spectrum - prev_spectrum))
            total_shift += shift

        prev_spectrum = spectrum

    return float(total_shift)


def path_efficiency(embs, perceiver_eigenvalues=None):
    """Ratio of optimal path cost to actual path cost.

    Measures how efficiently the book traverses semantic space.
    """
    n = len(embs)
    if n < 3:
        return 1.0

    # Actual sequential cost (Euclidean for simplicity at scale)
    actual_cost = sum(
        np.linalg.norm(embs[i + 1] - embs[i]) for i in range(n - 1)
    )

    if actual_cost < 1e-30:
        return 1.0

    # Greedy nearest-neighbor cost (approximate optimal)
    # Sample if too many paragraphs
    if n > 50:
        idx = np.linspace(0, n - 1, 50, dtype=int)
        sampled = embs[idx]
    else:
        sampled = embs

    m = len(sampled)
    visited = {0}
    order = [0]
    optimal_cost = 0.0
    current = 0

    for _ in range(m - 1):
        best_j = -1
        best_d = float("inf")
        for j in range(m):
            if j not in visited:
                d = np.linalg.norm(sampled[j] - sampled[current])
                if d < best_d:
                    best_d = d
                    best_j = j
        if best_j < 0:
            break
        visited.add(best_j)
        order.append(best_j)
        optimal_cost += best_d
        current = best_j

    # Scale optimal to match actual (different n)
    if m < n:
        optimal_cost *= (n - 1) / max(m - 1, 1)

    return float(optimal_cost / actual_cost) if actual_cost > 0 else 1.0


def aesthetic_score_tensor(embs, perceiver_q=None):
    """Aesthetic score of a text tensor (paragraph embeddings).

    Combines effective rank (complexity) with alignment to
    perceiver eigenspace (compressibility).
    """
    if perceiver_q is None:
        # Self-aesthetic: use the book's own eigenspectrum
        eff_r = effective_rank(embs)
        return eff_r

    # Project into perceiver basis and compute alignment
    # (This requires the perceiver PCA components, simplified here)
    eff_r = effective_rank(embs)
    return eff_r


# ================================================================== #
# Book loading and paragraph extraction                                #
# ================================================================== #


def load_book_text(filepath):
    """Load a Gutenberg text file, strip header/footer."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception:
        return None

    # Strip Gutenberg header/footer
    start_markers = ["*** START OF", "***START OF", "*** START OF THE PROJECT"]
    end_markers = ["*** END OF", "***END OF", "End of the Project Gutenberg"]

    start = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx >= 0:
            start = text.find("\n", idx) + 1
            break

    end = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx > start:
            end = idx
            break

    text = text[start:end].strip()
    return text if len(text) > 500 else None


def extract_paragraphs(text, min_length=50, max_paragraphs=200):
    """Split text into paragraphs, filter short ones."""
    # Split on double newlines
    paragraphs = re.split(r"\n\s*\n", text)
    # Filter and clean
    clean = []
    for p in paragraphs:
        p = p.strip()
        p = re.sub(r"\s+", " ", p)
        if len(p) >= min_length:
            clean.append(p[:1000])  # cap paragraph length
    # Sample evenly if too many
    if len(clean) > max_paragraphs:
        indices = np.linspace(0, len(clean) - 1, max_paragraphs, dtype=int)
        clean = [clean[i] for i in indices]
    return clean


# ================================================================== #
# Main analysis                                                        #
# ================================================================== #


def main():
    tc = ThermalController(target_temp=80, max_threads=6)
    tc.start()

    log("=" * 70)
    log("GUTENBERG TENSOR ANALYSIS")
    log("=" * 70)

    # Load Goodreads ratings
    log("Loading Goodreads ratings...")
    books_zip = "/archive/experiments/books.csv.zip"
    if not os.path.exists(books_zip):
        log("  books.csv.zip not found, skipping rating match")
        goodreads = {}
    else:
        z = zipfile.ZipFile(books_zip)
        with z.open("books.csv") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            goodreads = {}
            for row in reader:
                title = row.get("title", "").strip().lower()
                title_key = re.sub(r"\s*\(.*?\)\s*$", "", title).strip()
                rating = row.get("average_rating", "")
                count = row.get("ratings_count", "0")
                try:
                    r = float(rating)
                    c = int(count)
                    if 1.0 <= r <= 5.0 and c >= 50:
                        goodreads[title_key] = r
                except ValueError:
                    pass
        log(f"  {len(goodreads)} rated books")

    # Load Gutenberg catalog for title matching
    log("Loading Gutenberg catalog...")
    catalog = {}
    with open("/archive/gutenberg/pg_catalog.csv", "r") as f:
        for row in csv.DictReader(f):
            bid = row.get("Text#", "").strip()
            title = row.get("Title", "").strip()
            if bid and title:
                catalog[bid] = title

    # Find downloaded books
    text_dir = "/archive/gutenberg/texts"
    files = [f for f in os.listdir(text_dir) if f.endswith(".txt")]
    log(f"  {len(files)} downloaded books")

    # Match to ratings
    matched = []
    for fname in files:
        bid = fname.replace(".txt", "")
        if bid not in catalog:
            continue
        title = catalog[bid]
        title_key = re.sub(r"\s*\(.*?\)\s*$", "", title.lower()).strip()

        rating = goodreads.get(title_key)
        if rating is None:
            # Try shorter match
            short = title_key[:30]
            for gk, gv in goodreads.items():
                if gk.startswith(short):
                    rating = gv
                    break

        filepath = os.path.join(text_dir, fname)
        matched.append({
            "id": bid,
            "title": title,
            "rating": rating,  # None if unmatched
            "path": filepath,
        })

    rated = [m for m in matched if m["rating"] is not None]
    unrated = [m for m in matched if m["rating"] is None]
    log(f"  {len(rated)} matched to Goodreads, {len(unrated)} unmatched")

    # Load model
    log("Loading sentence-transformers...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Analyze books
    log(f"\nAnalyzing {min(len(rated), 500)} rated books...")
    results = []

    for i, book in enumerate(rated[:500]):
        text = load_book_text(book["path"])
        if text is None:
            continue

        paragraphs = extract_paragraphs(text)
        if len(paragraphs) < 5:
            continue

        # Embed paragraphs
        embs = np.array(
            model.encode(paragraphs, show_progress_bar=False, batch_size=64),
            dtype=np.float32,
        )

        # Compute tensor metrics
        eff_r = effective_rank(embs)
        curv = trajectory_curvature(embs)
        progress = compression_progress_score(embs)
        efficiency = path_efficiency(embs)

        results.append({
            "id": book["id"],
            "title": book["title"],
            "rating": book["rating"],
            "n_paragraphs": len(paragraphs),
            "effective_rank": eff_r,
            "curvature": curv,
            "compression_progress": progress,
            "path_efficiency": efficiency,
        })

        if (i + 1) % 25 == 0:
            log(f"  [{i+1}/{min(len(rated), 500)}] "
                f"last: rank={eff_r:.1f}, curv={curv:.3f}, "
                f"prog={progress:.3f}, eff={efficiency:.3f}, "
                f"rating={book['rating']:.2f}")

    del model
    gc.collect()

    n = len(results)
    log(f"\nAnalyzed {n} books successfully")

    if n < 10:
        log("Too few books for analysis")
        tc.stop()
        return

    # Extract arrays
    ratings = np.array([r["rating"] for r in results])
    ranks = np.array([r["effective_rank"] for r in results])
    curvatures = np.array([r["curvature"] for r in results])
    progresses = np.array([r["compression_progress"] for r in results])
    efficiencies = np.array([r["path_efficiency"] for r in results])
    n_paras = np.array([r["n_paragraphs"] for r in results])

    # Correlations
    log("\n" + "=" * 70)
    log("CORRELATIONS WITH GOODREADS RATING")
    log("=" * 70)

    metrics = [
        ("effective_rank", ranks),
        ("curvature", curvatures),
        ("compression_progress", progresses),
        ("path_efficiency", efficiencies),
        ("n_paragraphs", n_paras),
    ]

    correlation_results = {}
    for name, values in metrics:
        r, p = stats.pearsonr(values, ratings)
        rho, rho_p = stats.spearmanr(values, ratings)
        z = abs(r) * np.sqrt(n - 3)

        # Partial correlation controlling for book length
        from sklearn.linear_model import LinearRegression
        np_col = n_paras.reshape(-1, 1)
        val_resid = values - LinearRegression().fit(np_col, values).predict(np_col)
        rat_resid = ratings - LinearRegression().fit(np_col, ratings).predict(np_col)
        r_partial, p_partial = stats.pearsonr(val_resid, rat_resid)

        log(f"  {name:>25s}: r={r:+.4f} (p={p:.2e}, z={z:.1f}), "
            f"rho={rho:+.4f}, partial_r={r_partial:+.4f}")

        correlation_results[name] = {
            "pearson_r": float(r),
            "pearson_p": float(p),
            "z_score": float(z),
            "spearman_rho": float(rho),
            "partial_r": float(r_partial),
            "partial_p": float(p_partial),
        }

    # Combined score: all tensor metrics jointly
    log("\n  Combined (multiple regression):")
    from sklearn.linear_model import LinearRegression
    X = np.column_stack([ranks, curvatures, progresses, efficiencies])
    reg = LinearRegression().fit(X, ratings)
    y_pred = reg.predict(X)
    r_combined, _ = stats.pearsonr(y_pred, ratings)
    ss_res = np.sum((ratings - y_pred) ** 2)
    ss_tot = np.sum((ratings - ratings.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    log(f"  Combined R = {r_combined:.4f}, R-squared = {r2:.4f}")
    log(f"  Coefficients: rank={reg.coef_[0]:.4f}, curv={reg.coef_[1]:.4f}, "
        f"prog={reg.coef_[2]:.4f}, eff={reg.coef_[3]:.4f}")

    # Rating quintile analysis
    log("\n  Rating quintile means:")
    for lo, hi, label in [
        (0, 3.5, "Low"), (3.5, 3.8, "Med-lo"),
        (3.8, 4.0, "Med"), (4.0, 4.3, "Med-hi"), (4.3, 5.1, "High"),
    ]:
        mask = (ratings >= lo) & (ratings < hi)
        if mask.sum() > 0:
            log(f"    {label:>6s} (n={mask.sum():>3d}): "
                f"rank={ranks[mask].mean():.1f}, "
                f"curv={curvatures[mask].mean():.3f}, "
                f"prog={progresses[mask].mean():.3f}, "
                f"eff={efficiencies[mask].mean():.3f}")

    # Save results
    os.makedirs("/archive/results_aesthetics", exist_ok=True)
    output = {
        "experiment": "gutenberg_tensor",
        "n_books": n,
        "correlations": correlation_results,
        "combined_R": float(r_combined),
        "combined_R2": float(r2),
        "regression_coefficients": reg.coef_.tolist(),
    }
    with open("/archive/results_aesthetics/gutenberg_tensor.json", "w") as f:
        json.dump(output, f, indent=2)
    log("\nResults saved to /archive/results_aesthetics/gutenberg_tensor.json")

    tc.stop()


if __name__ == "__main__":
    main()

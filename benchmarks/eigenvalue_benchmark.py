#!/usr/bin/env python3
"""
Eigenvalue-Weighted Quantization Benchmark on Real BGE-M3 Embeddings.

Generates 194K Jeopardy question embeddings with BGE-M3, then compares:
  1. Naive truncation (no PCA)
  2. PCA-Matryoshka + uniform 2/3/4-bit
  3. PCA-Matryoshka + eigenvalue-weighted bit allocation
  4. Various weighted schedules

Requires: sentence-transformers, turboquant-pro>=0.7.0
"""

import json
import os
import sys
import time

import numpy as np

# Ensure latest turboquant is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_or_generate_embeddings(
    cache_path="benchmarks/bge_m3_embeddings.npy", max_docs=10000
):
    """Load cached embeddings or generate from Jeopardy dataset."""
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        print(f"Loaded {data.shape[0]} cached embeddings, dim={data.shape[1]}")
        return data

    print("Generating BGE-M3 embeddings (first run only)...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Installing sentence-transformers...")
        os.system(f"{sys.executable} -m pip install -q sentence-transformers")
        from sentence_transformers import SentenceTransformer

    # Load Jeopardy questions
    jeopardy_path = os.path.expanduser("~/jeopardy/JEOPARDY_QUESTIONS1.json")
    if not os.path.exists(jeopardy_path):
        # Try downloading
        jeopardy_path = "/tmp/jeopardy.json"
        if not os.path.exists(jeopardy_path):
            print("Downloading Jeopardy dataset...")
            url = (
                "https://raw.githubusercontent.com"
                "/jfrazier312/Jeopardy/master"
                "/JEOPARDY_QUESTIONS1.json"
            )
            os.system(f"wget -q -O {jeopardy_path} '{url}' 2>&1")

    try:
        with open(jeopardy_path) as f:
            questions = json.load(f)
        texts = [
            q.get("question", "") + " " + q.get("answer", "")
            for q in questions[:max_docs]
        ]
        print(f"Loaded {len(texts)} Jeopardy questions")
    except Exception:
        # Fallback: generate from random sentences
        print("Jeopardy dataset not available, using synthetic text...")
        import random

        random.seed(42)
        words = [
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "lazy",
            "dog",
            "science",
            "history",
            "geography",
            "literature",
            "math",
            "physics",
            "chemistry",
            "biology",
            "astronomy",
            "philosophy",
            "art",
            "music",
        ]
        texts = [
            " ".join(random.choices(words, k=random.randint(10, 30)))
            for _ in range(max_docs)
        ]
        print(f"Generated {len(texts)} synthetic texts")

    # Embed with BGE-M3
    print("Loading BGE-M3 model...")
    model = SentenceTransformer("BAAI/bge-m3")
    print(f"Model loaded: dim={model.get_sentence_embedding_dimension()}")

    print(f"Encoding {len(texts)} texts...")
    t0 = time.time()
    embeddings = model.encode(
        texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True
    )
    elapsed = time.time() - t0
    print(f"Encoded in {elapsed:.1f}s ({len(texts) / elapsed:.0f} texts/sec)")

    embeddings = embeddings.astype(np.float32)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"Saved to {cache_path}")

    return embeddings


def run_benchmark(embeddings: np.ndarray):
    """Run full benchmark comparing uniform vs eigenvalue-weighted."""
    from turboquant_pro.pca import PCAMatryoshka

    n, d = embeddings.shape
    print(f"\n{'=' * 70}")
    print("EIGENVALUE-WEIGHTED QUANTIZATION BENCHMARK")
    print(f"{'=' * 70}")
    print(f"Corpus: {n} embeddings, dim={d}")
    print()

    # Use 90% for fitting, 10% for testing
    n_fit = int(n * 0.9)
    fit_data = embeddings[:n_fit]
    test_data = embeddings[n_fit:]
    print(f"Fit: {n_fit}, Test: {len(test_data)}")

    results = []

    # Test multiple output dimensions
    for out_dim in [128, 256, 384, 512]:
        print(f"\n{'─' * 70}")
        print(f"PCA: {d} → {out_dim} dimensions")
        print(f"{'─' * 70}")

        pca = PCAMatryoshka(input_dim=d, output_dim=out_dim)
        pca.fit(fit_data)

        # Variance report
        vr = pca.variance_report([out_dim])
        var_pct = vr.get(out_dim, 0) * 100
        print(f"  Variance retained: {var_pct:.1f}%")

        # Naive truncation baseline (no PCA)
        trunc = test_data[:, :out_dim]
        naive_cos = np.mean(
            np.sum(test_data * np.pad(trunc, ((0, 0), (0, d - out_dim))), axis=1)
            / (
                np.linalg.norm(test_data, axis=1)
                * np.linalg.norm(np.pad(trunc, ((0, 0), (0, d - out_dim))), axis=1)
                + 1e-30
            )
        )

        # PCA rotation only (no quantization)
        reduced = pca.transform(test_data)
        reconstructed = pca.inverse_transform(reduced)
        pca_cos = np.mean(
            np.sum(test_data * reconstructed, axis=1)
            / (
                np.linalg.norm(test_data, axis=1)
                * np.linalg.norm(reconstructed, axis=1)
                + 1e-30
            )
        )

        print(f"  {'Method':<30s} {'Cosine':>8s} {'Ratio':>7s} {'Avg bits':>8s}")
        print(f"  {'─' * 55}")
        print(f"  {'Naive truncation':<30s} {naive_cos:>8.6f} {'—':>7s} {'32':>8s}")
        print(f"  {'PCA rotation only':<30s} {pca_cos:>8.6f} {'—':>7s} {'32':>8s}")

        results.append(
            {
                "dim": out_dim,
                "method": "naive_trunc",
                "cosine": naive_cos,
                "ratio": 1.0,
                "bits": 32,
            }
        )
        results.append(
            {
                "dim": out_dim,
                "method": "pca_only",
                "cosine": pca_cos,
                "ratio": d * 32 / (out_dim * 32),
                "bits": 32,
            }
        )

        # Uniform quantization
        for bits in [2, 3, 4]:
            pipe = pca.with_quantizer(bits=bits)
            mean_cos, min_cos, std_cos = pipe.batch_cosine_similarity(test_data)
            ratio = pipe.compression_ratio
            label = f"Uniform {bits}-bit"
            print(f"  {label:<30s} {mean_cos:>8.6f} {ratio:>6.1f}x {bits:>8d}")
            results.append(
                {
                    "dim": out_dim,
                    "method": f"uniform_{bits}b",
                    "cosine": mean_cos,
                    "min_cosine": min_cos,
                    "ratio": ratio,
                    "bits": bits,
                }
            )

        # Eigenvalue-weighted schedules
        schedules = {
            "Weighted 4+3+2": _make_schedule(
                out_dim, [(0.25, 4), (0.50, 3), (0.25, 2)]
            ),
            "Weighted 4+3+3+2": _make_schedule(
                out_dim, [(0.25, 4), (0.25, 3), (0.25, 3), (0.25, 2)]
            ),
            "Weighted 4+4+3+2": _make_schedule(
                out_dim, [(0.25, 4), (0.25, 4), (0.25, 3), (0.25, 2)]
            ),
            "Weighted 4+3+2+2": _make_schedule(
                out_dim, [(0.25, 4), (0.25, 3), (0.25, 2), (0.25, 2)]
            ),
        }

        for label, schedule in schedules.items():
            try:
                pipe_w = pca.with_weighted_quantizer(bit_schedule=schedule)
                mean_cos, min_cos, std_cos = pipe_w.batch_cosine_similarity(test_data)
                ratio = pipe_w.compression_ratio
                avg_b = pipe_w.avg_bits
                print(f"  {label:<30s} {mean_cos:>8.6f} {ratio:>6.1f}x {avg_b:>8.2f}")
                results.append(
                    {
                        "dim": out_dim,
                        "method": label,
                        "cosine": mean_cos,
                        "min_cosine": min_cos,
                        "ratio": ratio,
                        "bits": avg_b,
                        "schedule": str(schedule),
                    }
                )
            except Exception as e:
                print(f"  {label:<30s} ERROR: {e}")

        # Auto schedule
        for target_avg in [2.5, 3.0, 3.5]:
            try:
                pipe_a = pca.with_weighted_quantizer(avg_bits=target_avg)
                mean_cos, min_cos, std_cos = pipe_a.batch_cosine_similarity(test_data)
                ratio = pipe_a.compression_ratio
                avg_b = pipe_a.avg_bits
                label = f"Auto avg={target_avg:.1f}b"
                sched = pipe_a.bit_schedule
                print(
                    f"  {label:<30s} {mean_cos:>8.6f}"
                    f" {ratio:>6.1f}x {avg_b:>8.2f}"
                    f"  {sched}"
                )
                results.append(
                    {
                        "dim": out_dim,
                        "method": label,
                        "cosine": mean_cos,
                        "ratio": ratio,
                        "bits": avg_b,
                        "schedule": str(sched),
                    }
                )
            except Exception as e:
                print(f"  Auto avg={target_avg}: ERROR: {e}")

    # Save results
    out_path = "benchmarks/eigenvalue_weighted_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY: Best eigenvalue-weighted vs uniform at same avg bits")
    print(f"{'=' * 70}")
    for out_dim in [128, 256, 384, 512]:
        dim_results = [r for r in results if r["dim"] == out_dim]
        uniform_3 = next((r for r in dim_results if r["method"] == "uniform_3b"), None)
        weighted = [
            r
            for r in dim_results
            if "Weighted" in r.get("method", "") and abs(r.get("bits", 0) - 3.0) < 0.1
        ]
        if uniform_3 and weighted:
            best_w = max(weighted, key=lambda r: r["cosine"])
            delta = (best_w["cosine"] - uniform_3["cosine"]) * 100
            print(
                f"  dim={out_dim}: uniform 3b={uniform_3['cosine']:.6f} | "
                f"best weighted={best_w['cosine']:.6f} ({best_w['method']}) | "
                f"delta={delta:+.4f}%"
            )


def _make_schedule(total_dim, fractions):
    """Convert fractional schedule to integer dims."""
    schedule = []
    remaining = total_dim
    for i, (frac, bits) in enumerate(fractions):
        if i == len(fractions) - 1:
            n = remaining
        else:
            n = int(total_dim * frac)
            remaining -= n
        schedule.append((n, bits))
    return schedule


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=10000)
    parser.add_argument("--cache", default="benchmarks/bge_m3_embeddings.npy")
    args = parser.parse_args()

    embeddings = load_or_generate_embeddings(args.cache, args.max_docs)
    run_benchmark(embeddings)

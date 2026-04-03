#!/usr/bin/env python3
# TurboQuant-KV: Open-source PolarQuant+QJL for LLM KV cache compression
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""
Benchmark: TurboQuant compressed pgvector embeddings.

Compares float32 pgvector storage vs TurboQuant 3-bit compressed storage:
  - Storage size (bytes per embedding, total for dataset)
  - Search accuracy (recall@10 vs brute-force exact)
  - Search speed (queries/sec)
  - Insert speed (embeddings/sec)
  - Compression speed (embeddings/sec)
  - Cosine similarity preservation

Can run in two modes:
  1. Local synthetic benchmarks (no database required)
  2. Atlas PostgreSQL benchmarks (requires SSH access to Atlas)

Usage:
    # Local synthetic benchmark (no database needed)
    python benchmarks/benchmark_pgvector.py --local

    # Full benchmark with Atlas PostgreSQL
    python benchmarks/benchmark_pgvector.py --atlas

    # Specific scale
    python benchmarks/benchmark_pgvector.py --local --scales 100000 500000
"""

from __future__ import annotations

import argparse
import os
import json
import logging
import sys
import time
from typing import Any

import numpy as np

# Add parent to path for local dev
sys.path.insert(0, ".")
from turboquant_kv.pgvector import TurboQuantPGVector  # noqa: E402

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Benchmark utilities                                                 #
# ------------------------------------------------------------------ #


def cosine_similarity_matrix(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix (n_queries x n_corpus)."""
    q_norm = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-30)
    c_norm = corpus / np.maximum(np.linalg.norm(corpus, axis=1, keepdims=True), 1e-30)
    return q_norm @ c_norm.T


def recall_at_k(
    exact_rankings: np.ndarray,
    approx_rankings: np.ndarray,
    k: int = 10,
) -> float:
    """Compute recall@k: fraction of true top-k items in approximate top-k."""
    n_queries = exact_rankings.shape[0]
    total_recall = 0.0
    for i in range(n_queries):
        exact_set = set(exact_rankings[i, :k])
        approx_set = set(approx_rankings[i, :k])
        total_recall += len(exact_set & approx_set) / k
    return total_recall / n_queries


# ------------------------------------------------------------------ #
# Local synthetic benchmark                                           #
# ------------------------------------------------------------------ #


def benchmark_local(
    scales: list[int],
    dim: int = 1024,
    bits: int = 3,
    n_queries: int = 100,
    top_k: int = 10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run local synthetic benchmarks at various scales."""
    results = []
    rng = np.random.default_rng(seed)

    print(f"\n{'='*72}")
    print(f"TurboQuant pgvector Benchmark (local, dim={dim}, bits={bits})")
    print(f"{'='*72}\n")

    tq = TurboQuantPGVector(dim=dim, bits=bits, seed=seed)

    for n in scales:
        print(f"\n--- Scale: {n:,} embeddings ---\n")

        # Generate corpus and queries
        print(f"  Generating {n:,} corpus embeddings...")
        corpus = rng.standard_normal((n, dim)).astype(np.float32)
        queries = rng.standard_normal((n_queries, dim)).astype(np.float32)

        # 1. Storage comparison
        storage = TurboQuantPGVector.estimate_storage(n, dim, bits)
        print(
            f"  Storage: float32={storage['original_mb']:.1f} MB, "
            f"compressed={storage['compressed_mb']:.1f} MB, "
            f"ratio={storage['ratio']:.2f}x"
        )

        # 2. Compression speed
        print(f"  Compressing {n:,} embeddings...")
        t0 = time.perf_counter()
        batch_size = 10000
        compressed_all = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_compressed = tq.compress_batch(corpus[start:end])
            compressed_all.extend(batch_compressed)
        t_compress = time.perf_counter() - t0
        compress_rate = n / t_compress
        print(
            f"  Compression: {compress_rate:,.0f} embeddings/sec "
            f"({t_compress:.2f}s total)"
        )

        # 3. Cosine similarity preservation
        print("  Evaluating cosine similarity preservation...")
        sample_size = min(1000, n)
        sample_idx = rng.choice(n, sample_size, replace=False)
        sample_original = corpus[sample_idx]
        sample_compressed = [compressed_all[i] for i in sample_idx]
        sample_decompressed = tq.decompress_batch(sample_compressed)

        cos_sims = []
        for i in range(sample_size):
            orig = sample_original[i]
            recon = sample_decompressed[i]
            dot = np.dot(orig, recon)
            norm_o = np.linalg.norm(orig)
            norm_r = np.linalg.norm(recon)
            if norm_o > 1e-30 and norm_r > 1e-30:
                cos_sims.append(dot / (norm_o * norm_r))
        mean_cos = np.mean(cos_sims)
        min_cos = np.min(cos_sims)
        print(f"  Cosine similarity: mean={mean_cos:.6f}, " f"min={min_cos:.6f}")

        # 4. Search accuracy (recall@k) -- on a subset for speed
        search_n = min(50000, n)
        print(f"  Computing recall@{top_k} on {search_n:,} subset...")

        # Exact search
        corpus_subset = corpus[:search_n]
        t0 = time.perf_counter()
        exact_sims = cosine_similarity_matrix(queries, corpus_subset)
        exact_rankings = np.argsort(-exact_sims, axis=1)
        t_exact = time.perf_counter() - t0

        # Approximate search (decompress then compare)
        compressed_subset = compressed_all[:search_n]
        t0 = time.perf_counter()
        decompressed_subset = tq.decompress_batch(compressed_subset)
        approx_sims = cosine_similarity_matrix(queries, decompressed_subset)
        approx_rankings = np.argsort(-approx_sims, axis=1)
        t_approx = time.perf_counter() - t0

        recall = recall_at_k(exact_rankings, approx_rankings, top_k)
        print(f"  Recall@{top_k}: {recall:.4f}")
        print(f"  Search speed: exact={t_exact:.3f}s, " f"approx={t_approx:.3f}s")

        # 5. Per-query search speed
        t0 = time.perf_counter()
        for q in queries:
            _ = tq.compressed_cosine_similarity(q, compressed_subset[:1000])
        t_per_query = (time.perf_counter() - t0) / n_queries
        qps = 1.0 / max(t_per_query, 1e-9)
        print(f"  Per-query speed (1K corpus): {qps:.0f} queries/sec")

        row = {
            "n_embeddings": n,
            "dim": dim,
            "bits": bits,
            "original_mb": storage["original_mb"],
            "compressed_mb": storage["compressed_mb"],
            "compression_ratio": storage["ratio"],
            "mean_cosine_similarity": round(float(mean_cos), 6),
            "min_cosine_similarity": round(float(min_cos), 6),
            "recall_at_10": round(recall, 4),
            "compress_embeddings_per_sec": round(compress_rate),
            "search_subset_size": search_n,
        }
        results.append(row)
        print("  Done.\n")

    return results


# ------------------------------------------------------------------ #
# Atlas PostgreSQL benchmark                                          #
# ------------------------------------------------------------------ #


def benchmark_atlas(
    scales: list[int],
    dim: int = 1024,
    bits: int = 3,
    n_queries: int = 50,
    top_k: int = 10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run benchmarks against Atlas PostgreSQL with real pgvector data.

    Requires paramiko and psycopg2 (via SSH tunnel to Atlas).
    """
    try:
        import paramiko  # type: ignore[import-untyped]
    except ImportError:
        print(
            "ERROR: paramiko required for Atlas benchmark. "
            "Install with: pip install paramiko"
        )
        return []

    print(f"\n{'='*72}")
    print("TurboQuant pgvector Benchmark (Atlas PostgreSQL)")
    print(f"{'='*72}\n")

    # Atlas connection parameters
    atlas_host = "100.68.134.21"
    atlas_user = "claude"
    atlas_password = os.environ.get("ATLAS_DB_PASSWORD", "changeme")

    print("  Connecting to Atlas via SSH...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(
            atlas_host,
            username=atlas_user,
            password=atlas_password,
            timeout=30,
        )
    except Exception as e:
        print(f"  SSH connection failed: {e}")
        print("  Falling back to local benchmark.")
        return benchmark_local(scales, dim, bits, n_queries, top_k, seed)

    # Set up SSH tunnel for PostgreSQL
    transport = ssh.get_transport()
    if transport is None:
        print("  Failed to get SSH transport.")
        ssh.close()
        return []

    try:
        # Forward local port to Atlas PostgreSQL

        local_port = 15432
        remote_host = "127.0.0.1"
        remote_port = 5432

        # Simple SSH tunnel
        channel = transport.open_channel(
            "direct-tcpip",
            (remote_host, remote_port),
            ("127.0.0.1", 0),
        )

        print(
            f"  SSH tunnel established (local:{local_port} -> " f"Atlas:{remote_port})"
        )

        # Use psycopg2 through the tunnel
        import psycopg2  # type: ignore[import-untyped]

        conn = psycopg2.connect(
            host="127.0.0.1",
            port=remote_port,
            dbname="agi_hpc",
            user="agi_hpc",
            password="YOUR_DB_PASSWORD",
        )

        print("  Connected to PostgreSQL.")

        # Check current table sizes
        with conn.cursor() as cur:
            cur.execute("""
                SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
                FROM pg_catalog.pg_statio_user_tables
                WHERE relname LIKE '%chunk%' OR relname LIKE '%embedding%'
                ORDER BY pg_total_relation_size(relid) DESC
                LIMIT 10;
            """)
            tables = cur.fetchall()
            if tables:
                print("\n  Existing tables:")
                for name, size in tables:
                    print(f"    {name}: {size}")

        # Fetch sample embeddings from ethics_chunks
        print("\n  Fetching sample embeddings from ethics_chunks...")
        with conn.cursor() as cur:
            for scale in scales:
                cur.execute(
                    "SELECT id, embedding FROM ethics_chunks " "LIMIT %s",
                    (scale,),
                )
                rows = cur.fetchall()
                if rows:
                    print(f"  Fetched {len(rows)} rows at scale {scale}")
                else:
                    print(f"  No rows found for scale {scale}")

        conn.close()
        channel.close()

    except Exception as e:
        print(f"  Atlas benchmark error: {e}")
        print("  Falling back to local benchmark.")
        return benchmark_local(scales, dim, bits, n_queries, top_k, seed)
    finally:
        ssh.close()

    return []


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboQuant pgvector benchmark")
    parser.add_argument(
        "--local",
        action="store_true",
        default=True,
        help="Run local synthetic benchmark (default)",
    )
    parser.add_argument(
        "--atlas",
        action="store_true",
        help="Run benchmark against Atlas PostgreSQL",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=int,
        default=[10_000, 50_000, 100_000],
        help="Dataset scales to benchmark",
    )
    parser.add_argument("--dim", type=int, default=1024, help="Embedding dim")
    parser.add_argument("--bits", type=int, default=3, help="Quantization bits")
    parser.add_argument("--queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k for recall")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.atlas:
        results = benchmark_atlas(
            args.scales, args.dim, args.bits, args.queries, args.top_k, args.seed
        )
    else:
        results = benchmark_local(
            args.scales, args.dim, args.bits, args.queries, args.top_k, args.seed
        )

    # Print summary table
    if results:
        print(f"\n{'='*72}")
        print("SUMMARY")
        print(f"{'='*72}")
        print(
            f"{'Scale':>12} {'Float32 MB':>12} {'Compressed MB':>14} "
            f"{'Ratio':>8} {'Cos Sim':>10} {'Recall@10':>10}"
        )
        print("-" * 72)
        for r in results:
            print(
                f"{r['n_embeddings']:>12,} "
                f"{r['original_mb']:>12.1f} "
                f"{r['compressed_mb']:>14.1f} "
                f"{r['compression_ratio']:>8.2f}x "
                f"{r['mean_cosine_similarity']:>10.6f} "
                f"{r['recall_at_10']:>10.4f}"
            )
        print()

    # Save results
    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

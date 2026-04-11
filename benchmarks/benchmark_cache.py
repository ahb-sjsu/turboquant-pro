"""
Cache hit rate benchmark: compressed vs uncompressed at fixed memory budget.

Demonstrates that TurboQuant-compressed caching fits ~10x more embeddings
in the same memory, dramatically improving cache hit rates under realistic
Zipf-distributed access patterns.

Usage:
    python benchmarks/benchmark_cache.py
    python benchmarks/benchmark_cache.py --corpus-size 200000 --cache-mb 50
"""

from __future__ import annotations

import argparse
from collections import OrderedDict

import numpy as np


def _compressed_size_per_embedding(dim: int, bits: int) -> int:
    """Approximate compressed size per embedding in bytes."""
    if bits == 3:
        packed_bytes = (dim * 3 + 7) // 8
    elif bits == 2:
        packed_bytes = (dim + 3) // 4
    elif bits == 4:
        packed_bytes = (dim + 1) // 2
    else:
        packed_bytes = dim
    return packed_bytes + 4  # +4 for float32 norm


def simulate_lru_cache(
    capacity: int,
    query_stream: np.ndarray,
) -> dict:
    """Simulate an LRU cache and return hit/miss statistics.

    Args:
        capacity: Max number of entries the cache can hold.
        query_stream: 1-D array of integer IDs to look up.

    Returns:
        Dict with hits, misses, hit_rate.
    """
    cache: OrderedDict[int, bool] = OrderedDict()
    hits = 0
    misses = 0

    for q_id in query_stream:
        q_id = int(q_id)
        if q_id in cache:
            cache.move_to_end(q_id)
            hits += 1
        else:
            misses += 1
            if len(cache) >= capacity:
                cache.popitem(last=False)
            cache[q_id] = True

    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / total if total > 0 else 0.0,
        "capacity": capacity,
    }


def benchmark_cache_hit_rate(
    corpus_size: int = 100_000,
    cache_memory_mb: int = 100,
    n_queries: int = 50_000,
    zipf_param: float = 1.2,
    dim: int = 1024,
    bits: int = 3,
    seed: int = 42,
) -> dict:
    """Compare cache hit rates: uncompressed vs compressed at same memory.

    Returns dict with both strategies' results.
    """
    rng = np.random.default_rng(seed)
    cache_bytes = cache_memory_mb * 1024 * 1024

    # How many embeddings fit in each strategy
    float32_size = dim * 4
    compressed_size = _compressed_size_per_embedding(dim, bits)

    uncompressed_capacity = cache_bytes // float32_size
    compressed_capacity = cache_bytes // compressed_size

    # Generate Zipf-distributed query stream
    # Zipf gives heavy-tailed access: a few IDs are very popular
    query_stream = rng.zipf(zipf_param, size=n_queries)
    query_stream = query_stream % corpus_size  # map to valid IDs

    # Simulate both
    uncomp = simulate_lru_cache(uncompressed_capacity, query_stream)
    comp = simulate_lru_cache(compressed_capacity, query_stream)

    return {
        "corpus_size": corpus_size,
        "cache_memory_mb": cache_memory_mb,
        "n_queries": n_queries,
        "zipf_param": zipf_param,
        "dim": dim,
        "bits": bits,
        "float32_bytes_per_emb": float32_size,
        "compressed_bytes_per_emb": compressed_size,
        "uncompressed_capacity": uncompressed_capacity,
        "compressed_capacity": compressed_capacity,
        "capacity_ratio": compressed_capacity / max(uncompressed_capacity, 1),
        "uncompressed_hit_rate": uncomp["hit_rate"],
        "compressed_hit_rate": comp["hit_rate"],
        "hit_rate_improvement": comp["hit_rate"] - uncomp["hit_rate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache hit rate benchmark")
    parser.add_argument("--corpus-size", type=int, default=100_000)
    parser.add_argument("--cache-mb", type=int, default=100)
    parser.add_argument("--n-queries", type=int, default=50_000)
    parser.add_argument("--zipf", type=float, default=1.2)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    args = parser.parse_args()

    print("=" * 65)
    print("TurboQuant Pro: Cache Hit Rate Benchmark")
    print("=" * 65)
    print(f"  Corpus: {args.corpus_size:,} embeddings")
    print(f"  Cache budget: {args.cache_mb} MB")
    print(f"  Queries: {args.n_queries:,} (Zipf a={args.zipf})")
    print(f"  Embedding: dim={args.dim}, bits={args.bits}")
    print()

    results = benchmark_cache_hit_rate(
        corpus_size=args.corpus_size,
        cache_memory_mb=args.cache_mb,
        n_queries=args.n_queries,
        zipf_param=args.zipf,
        dim=args.dim,
        bits=args.bits,
    )

    print(f"{'':>28s}  {'Uncompressed':>14s}  {'Compressed':>14s}")
    print("-" * 62)
    print(
        f"{'Bytes per embedding':>28s}  "
        f"{results['float32_bytes_per_emb']:>14,}  "
        f"{results['compressed_bytes_per_emb']:>14,}"
    )
    print(
        f"{'Cache capacity (entries)':>28s}  "
        f"{results['uncompressed_capacity']:>14,}  "
        f"{results['compressed_capacity']:>14,}"
    )
    print(
        f"{'Cache hit rate':>28s}  "
        f"{results['uncompressed_hit_rate']:>13.1%}  "
        f"{results['compressed_hit_rate']:>13.1%}"
    )
    print()
    print(f"  Capacity ratio: {results['capacity_ratio']:.1f}x more embeddings")
    print(f"  Hit rate improvement: +{results['hit_rate_improvement']:.1%}")

    # Sweep multiple memory budgets
    print()
    print("--- Memory Budget Sweep ---")
    print(
        f"{'Budget MB':>10s}  {'Uncomp cap':>12s}  {'Comp cap':>12s}  "
        f"{'Uncomp hit%':>12s}  {'Comp hit%':>12s}  {'Delta':>8s}"
    )
    print("-" * 72)

    for mb in [10, 25, 50, 100, 200, 500]:
        r = benchmark_cache_hit_rate(
            corpus_size=args.corpus_size,
            cache_memory_mb=mb,
            n_queries=args.n_queries,
            zipf_param=args.zipf,
            dim=args.dim,
            bits=args.bits,
        )
        print(
            f"{mb:>10d}  "
            f"{r['uncompressed_capacity']:>12,}  "
            f"{r['compressed_capacity']:>12,}  "
            f"{r['uncompressed_hit_rate']:>11.1%}  "
            f"{r['compressed_hit_rate']:>11.1%}  "
            f"+{r['hit_rate_improvement']:>6.1%}"
        )


if __name__ == "__main__":
    main()

"""
Compressed HNSW recall, memory, and latency benchmark.

Compares CompressedHNSW against FAISS IndexHNSWFlat and brute-force
exact search on random embeddings.

Usage:
    python benchmarks/benchmark_hnsw.py
    python benchmarks/benchmark_hnsw.py --n-corpus 20000 --dim 256
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from turboquant_pro.hnsw import CompressedHNSW
from turboquant_pro.pgvector import TurboQuantPGVector

try:
    from turboquant_pro.faiss_index import TurboQuantFAISS  # type: ignore
    from turboquant_pro.pca import PCAMatryoshka  # type: ignore

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


def _brute_force_knn(
    query: np.ndarray,
    corpus: np.ndarray,
    k: int,
) -> list[int]:
    """Brute-force top-k by cosine similarity."""
    q_norm = np.linalg.norm(query)
    sims = corpus @ query / (np.linalg.norm(corpus, axis=1) * q_norm + 1e-30)
    return np.argsort(-sims)[:k].tolist()


def recall_at_k(
    approx_ids: list[int] | set[int],
    exact_ids: list[int] | set[int],
    k: int,
) -> float:
    return len(set(approx_ids) & set(exact_ids)) / k


def benchmark_compressed_hnsw(
    n_corpus: int = 10_000,
    dim: int = 128,
    bits: int = 3,
    n_queries: int = 50,
    k: int = 10,
    M_values: list[int] | None = None,
    ef_values: list[int] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Benchmark CompressedHNSW across M and ef configurations."""
    if M_values is None:
        M_values = [8, 16, 32]
    if ef_values is None:
        ef_values = [50, 100, 200]

    rng = np.random.default_rng(seed)
    corpus = rng.standard_normal((n_corpus, dim)).astype(np.float32)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)

    # Pre-compute exact results
    exact_results = []
    for q in queries:
        exact_results.append(_brute_force_knn(q, corpus, k))

    tq = TurboQuantPGVector(dim=dim, bits=bits, seed=seed)
    results = []

    for M in M_values:
        # Build index
        t0 = time.perf_counter()
        index = CompressedHNSW(tq, M=M, ef_construction=200, seed=seed)
        index.batch_insert(list(range(n_corpus)), corpus)
        build_time = time.perf_counter() - t0

        mem = index.memory_usage_bytes()
        float32_mem = n_corpus * dim * 4

        for ef in ef_values:
            # Measure query latency and recall
            total_recall = 0.0
            t0 = time.perf_counter()
            for i, q in enumerate(queries):
                approx = index.search(q, k=k, ef=ef, rerank=True)
                approx_ids = [r[0] for r in approx]
                total_recall += recall_at_k(approx_ids, exact_results[i], k)
            query_time = time.perf_counter() - t0

            avg_recall = total_recall / n_queries
            avg_latency_ms = (query_time / n_queries) * 1000

            row = {
                "index": "CompressedHNSW",
                "M": M,
                "ef": ef,
                "n_corpus": n_corpus,
                "dim": dim,
                "bits": bits,
                "build_time_s": round(build_time, 2),
                "memory_bytes": mem,
                "memory_vs_float32": round(mem / float32_mem, 3),
                "recall_at_k": round(avg_recall, 4),
                "avg_latency_ms": round(avg_latency_ms, 2),
            }
            results.append(row)

    return results


def benchmark_faiss_hnsw(
    n_corpus: int = 10_000,
    dim: int = 128,
    n_queries: int = 50,
    k: int = 10,
    M_values: list[int] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Benchmark FAISS IndexHNSWFlat for comparison."""
    if not _HAS_FAISS:
        return []
    if M_values is None:
        M_values = [8, 16, 32]

    rng = np.random.default_rng(seed)
    corpus = rng.standard_normal((n_corpus, dim)).astype(np.float32)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)

    exact_results = [_brute_force_knn(q, corpus, k) for q in queries]

    pca = PCAMatryoshka(input_dim=dim, output_dim=dim)
    pca.fit(corpus[:1000])

    results = []
    for M in M_values:
        faiss_idx = TurboQuantFAISS(pca, index_type="hnsw", hnsw_m=M)
        t0 = time.perf_counter()
        faiss_idx.add(corpus)
        build_time = time.perf_counter() - t0

        total_recall = 0.0
        t0 = time.perf_counter()
        for i, q in enumerate(queries):
            dists, ids = faiss_idx.search(q.reshape(1, -1), k=k)
            approx_ids = ids[0].tolist()
            total_recall += recall_at_k(approx_ids, exact_results[i], k)
        query_time = time.perf_counter() - t0

        avg_recall = total_recall / n_queries
        avg_latency_ms = (query_time / n_queries) * 1000
        float32_mem = n_corpus * dim * 4

        results.append(
            {
                "index": "FAISS_HNSW",
                "M": M,
                "ef": "-",
                "n_corpus": n_corpus,
                "dim": dim,
                "bits": 32,
                "build_time_s": round(build_time, 2),
                "memory_bytes": float32_mem,
                "memory_vs_float32": 1.0,
                "recall_at_k": round(avg_recall, 4),
                "avg_latency_ms": round(avg_latency_ms, 2),
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compressed HNSW benchmark")
    parser.add_argument("--n-corpus", type=int, default=5_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--n-queries", type=int, default=30)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    print("=" * 80)
    print("TurboQuant Pro: Compressed HNSW Benchmark")
    print("=" * 80)
    print(f"  Corpus: {args.n_corpus:,} vectors, dim={args.dim}, bits={args.bits}")
    print(f"  Queries: {args.n_queries}, k={args.k}")
    print(f"  FAISS available: {_HAS_FAISS}")
    print()

    # Compressed HNSW
    print("--- CompressedHNSW ---")
    c_results = benchmark_compressed_hnsw(
        n_corpus=args.n_corpus,
        dim=args.dim,
        bits=args.bits,
        n_queries=args.n_queries,
        k=args.k,
    )

    print(
        f"{'M':>4s}  {'ef':>5s}  {'Recall@k':>10s}  {'Latency ms':>12s}  "
        f"{'Memory':>12s}  {'vs float32':>11s}  {'Build s':>8s}"
    )
    print("-" * 72)
    for r in c_results:
        print(
            f"{r['M']:>4d}  {r['ef']:>5d}  {r['recall_at_k']:>10.4f}  "
            f"{r['avg_latency_ms']:>12.2f}  "
            f"{r['memory_bytes']:>12,}  "
            f"{r['memory_vs_float32']:>10.3f}x  "
            f"{r['build_time_s']:>8.2f}"
        )

    # FAISS comparison
    if _HAS_FAISS:
        print()
        print("--- FAISS IndexHNSWFlat (float32, for comparison) ---")
        f_results = benchmark_faiss_hnsw(
            n_corpus=args.n_corpus,
            dim=args.dim,
            n_queries=args.n_queries,
            k=args.k,
        )
        for r in f_results:
            print(
                f"{r['M']:>4d}  {'  -':>5s}  {r['recall_at_k']:>10.4f}  "
                f"{r['avg_latency_ms']:>12.2f}  "
                f"{r['memory_bytes']:>12,}  "
                f"{r['memory_vs_float32']:>10.3f}x  "
                f"{r['build_time_s']:>8.2f}"
            )
    else:
        print("\nFAISS not installed — skipping comparison benchmark.")


if __name__ == "__main__":
    main()

"""
Tests for the Compressed HNSW index.

Usage:
    pytest tests/test_hnsw.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro.hnsw import CompressedHNSW
from turboquant_pro.pgvector import TurboQuantPGVector

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _brute_force_knn(
    query: np.ndarray,
    corpus: np.ndarray,
    k: int,
) -> list[int]:
    """Return indices of top-k nearest by cosine similarity."""
    q_norm = np.linalg.norm(query)
    sims = corpus @ query / (np.linalg.norm(corpus, axis=1) * q_norm + 1e-30)
    top_k = np.argsort(-sims)[:k]
    return top_k.tolist()


@pytest.fixture()
def tq() -> TurboQuantPGVector:
    return TurboQuantPGVector(dim=64, bits=3, seed=0)


# ------------------------------------------------------------------ #
# Basic functionality                                                  #
# ------------------------------------------------------------------ #


class TestCompressedHNSWBasic:
    """Basic insertion and search tests."""

    def test_insert_and_search_basic(self, tq: TurboQuantPGVector) -> None:
        index = CompressedHNSW(tq, M=8, ef_construction=50, seed=42)
        rng = np.random.default_rng(42)
        corpus = rng.standard_normal((100, 64)).astype(np.float32)

        for i, emb in enumerate(corpus):
            index.insert(i, emb)

        assert index.n_vectors == 100
        results = index.search(corpus[0], k=5)
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_empty_index(self, tq: TurboQuantPGVector) -> None:
        index = CompressedHNSW(tq, M=8, seed=42)
        results = index.search(np.zeros(64, dtype=np.float32), k=5)
        assert results == []

    def test_single_vector(self, tq: TurboQuantPGVector) -> None:
        index = CompressedHNSW(tq, M=8, seed=42)
        emb = np.random.default_rng(0).standard_normal(64).astype(np.float32)
        index.insert(0, emb)
        results = index.search(emb, k=1)
        assert len(results) == 1
        assert results[0][0] == 0  # should find itself

    def test_batch_insert(self, tq: TurboQuantPGVector) -> None:
        index = CompressedHNSW(tq, M=8, ef_construction=50, seed=42)
        rng = np.random.default_rng(42)
        corpus = rng.standard_normal((500, 64)).astype(np.float32)
        ids = list(range(500))

        count = index.batch_insert(ids, corpus)
        assert count == 500
        assert index.n_vectors == 500


# ------------------------------------------------------------------ #
# Self-retrieval                                                       #
# ------------------------------------------------------------------ #


class TestSelfRetrieval:
    """Query with a vector that's in the index — it should be in top results."""

    def test_self_in_top5(self, tq: TurboQuantPGVector) -> None:
        index = CompressedHNSW(tq, M=16, ef_construction=100, seed=42)
        rng = np.random.default_rng(42)
        corpus = rng.standard_normal((300, 64)).astype(np.float32)

        for i, emb in enumerate(corpus):
            index.insert(i, emb)

        # Check first 10 vectors
        for i in range(10):
            results = index.search(corpus[i], k=5, rerank=True)
            result_ids = [r[0] for r in results]
            assert (
                i in result_ids
            ), f"Vector {i} not found in top-5 results: {result_ids}"


# ------------------------------------------------------------------ #
# Recall                                                               #
# ------------------------------------------------------------------ #


class TestRecall:
    """Recall@k against brute-force exact search."""

    def test_recall_at_10(self, tq: TurboQuantPGVector) -> None:
        index = CompressedHNSW(tq, M=16, ef_construction=200, seed=42)
        rng = np.random.default_rng(42)
        corpus = rng.standard_normal((2000, 64)).astype(np.float32)

        for i, emb in enumerate(corpus):
            index.insert(i, emb)

        k = 10
        n_queries = 30
        queries = rng.standard_normal((n_queries, 64)).astype(np.float32)

        total_recall = 0.0
        for q in queries:
            exact_ids = set(_brute_force_knn(q, corpus, k))
            approx_results = index.search(q, k=k, rerank=True)
            approx_ids = {r[0] for r in approx_results}
            total_recall += len(exact_ids & approx_ids) / k

        avg_recall = total_recall / n_queries
        assert avg_recall > 0.70, f"Recall@{k} = {avg_recall:.3f}, expected > 0.70"

    def test_rerank_improves_recall(self, tq: TurboQuantPGVector) -> None:
        index = CompressedHNSW(tq, M=16, ef_construction=200, seed=42)
        rng = np.random.default_rng(42)
        corpus = rng.standard_normal((1000, 64)).astype(np.float32)

        for i, emb in enumerate(corpus):
            index.insert(i, emb)

        k = 10
        queries = rng.standard_normal((20, 64)).astype(np.float32)

        recall_no_rerank = 0.0
        recall_rerank = 0.0
        for q in queries:
            exact_ids = set(_brute_force_knn(q, corpus, k))

            results_no = index.search(q, k=k, rerank=False)
            ids_no = {r[0] for r in results_no}
            recall_no_rerank += len(exact_ids & ids_no) / k

            results_yes = index.search(q, k=k, rerank=True)
            ids_yes = {r[0] for r in results_yes}
            recall_rerank += len(exact_ids & ids_yes) / k

        # Reranking should be equal or better
        assert recall_rerank >= recall_no_rerank - 1e-9


# ------------------------------------------------------------------ #
# Memory                                                               #
# ------------------------------------------------------------------ #


class TestMemory:
    """Memory usage should be much smaller than float32 equivalent."""

    def test_memory_smaller_than_float32(self) -> None:
        # Use dim=256 to demonstrate compression advantage.
        # At dim=64, cached indices (64 bytes) and neighbor lists
        # dominate the per-node budget.  At dim=256+, the packed
        # embedding savings outweigh the overhead.
        dim = 256
        tq_256 = TurboQuantPGVector(dim=dim, bits=3, seed=0)
        index = CompressedHNSW(tq_256, M=8, ef_construction=50, seed=42)
        rng = np.random.default_rng(42)
        corpus = rng.standard_normal((500, dim)).astype(np.float32)

        for i, emb in enumerate(corpus):
            index.insert(i, emb)

        compressed_bytes = index.memory_usage_bytes()
        float32_bytes = 500 * dim * 4

        ratio = compressed_bytes / float32_bytes
        assert ratio < 1.0, (
            f"Compressed index ({compressed_bytes:,} bytes) is larger than "
            f"float32 ({float32_bytes:,} bytes), ratio={ratio:.2f}"
        )


# ------------------------------------------------------------------ #
# Stats and different M values                                         #
# ------------------------------------------------------------------ #


class TestStatsAndConfig:
    """Stats reporting and configuration variants."""

    def test_stats(self, tq: TurboQuantPGVector) -> None:
        index = CompressedHNSW(tq, M=8, ef_construction=50, seed=42)
        rng = np.random.default_rng(42)
        for i in range(100):
            index.insert(i, rng.standard_normal(64).astype(np.float32))

        stats = index.stats()
        assert stats["n_vectors"] == 100
        assert stats["top_layer"] >= 0
        assert stats["memory_bytes"] > 0
        assert stats["avg_neighbors_layer0"] > 0
        assert stats["entry_point"] is not None

    @pytest.mark.parametrize("M", [4, 8, 16])
    def test_different_M_values(self, tq: TurboQuantPGVector, M: int) -> None:
        index = CompressedHNSW(tq, M=M, ef_construction=100, seed=42)
        rng = np.random.default_rng(42)
        corpus = rng.standard_normal((500, 64)).astype(np.float32)

        for i, emb in enumerate(corpus):
            index.insert(i, emb)

        results = index.search(corpus[0], k=5, rerank=True)
        assert len(results) >= 1
        # Self should be in results
        result_ids = [r[0] for r in results]
        assert 0 in result_ids

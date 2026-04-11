# TurboQuant Pro: Open-source PolarQuant+QJL for LLM KV cache compression
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""
Tests for incremental persistence of the CompressedHNSW index.

Usage:
    pytest tests/test_incremental_hnsw.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro.hnsw import CompressedHNSW
from turboquant_pro.pgvector import TurboQuantPGVector

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


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
# Incremental persistence tests                                        #
# ------------------------------------------------------------------ #


class TestIncrementalHNSW:
    """Tests for open / append / sync incremental persistence."""

    def test_open_loads_existing(
        self, tq: TurboQuantPGVector, tmp_path: object
    ) -> None:
        """save(), then open() — n_vectors should match."""
        index = CompressedHNSW(tq, M=8, ef_construction=50, seed=42)
        rng = np.random.default_rng(42)
        corpus = rng.standard_normal((100, 64)).astype(np.float32)

        for i, emb in enumerate(corpus):
            index.insert(i, emb)

        path = str(tmp_path / "index.tqhnsw")  # type: ignore[operator]
        index.save(path)

        loaded = CompressedHNSW.open(path, tq)
        assert loaded.n_vectors == 100
        assert loaded._persist_path == path
        assert loaded._last_saved_count == 100

    def test_insert_after_open(self, tq: TurboQuantPGVector, tmp_path: object) -> None:
        """open(), insert new vectors, search finds them."""
        index = CompressedHNSW(tq, M=8, ef_construction=50, seed=42)
        rng = np.random.default_rng(42)
        corpus = rng.standard_normal((50, 64)).astype(np.float32)

        for i, emb in enumerate(corpus):
            index.insert(i, emb)

        path = str(tmp_path / "index.tqhnsw")  # type: ignore[operator]
        index.save(path)

        loaded = CompressedHNSW.open(path, tq)
        # Insert new vectors after opening.
        extra = rng.standard_normal((10, 64)).astype(np.float32)
        for i, emb in enumerate(extra):
            loaded.insert(50 + i, emb)

        assert loaded.n_vectors == 60

        # Search should find at least one of the newly inserted vectors
        # when querying with one of them.
        results = loaded.search(extra[0], k=5, rerank=True)
        result_ids = [r[0] for r in results]
        assert 50 in result_ids

    def test_sync_persists_new(self, tq: TurboQuantPGVector, tmp_path: object) -> None:
        """open(), insert, sync(), open() again — all vectors present."""
        index = CompressedHNSW(tq, M=8, ef_construction=50, seed=42)
        rng = np.random.default_rng(42)
        corpus = rng.standard_normal((50, 64)).astype(np.float32)

        for i, emb in enumerate(corpus):
            index.insert(i, emb)

        path = str(tmp_path / "index.tqhnsw")  # type: ignore[operator]
        index.save(path)

        # Open, insert more, sync.
        loaded = CompressedHNSW.open(path, tq)
        extra = rng.standard_normal((20, 64)).astype(np.float32)
        for i, emb in enumerate(extra):
            loaded.insert(50 + i, emb)

        loaded.sync()

        # Re-open and verify all 70 vectors are present.
        reloaded = CompressedHNSW.open(path, tq)
        assert reloaded.n_vectors == 70

        # Verify that both original and new node IDs are present.
        all_ids = set(reloaded._nodes.keys())
        for i in range(70):
            assert i in all_ids, f"Node {i} missing after sync + open"

    def test_sync_on_new_file(self, tq: TurboQuantPGVector, tmp_path: object) -> None:
        """sync() creates a new file if none exists."""
        index = CompressedHNSW(tq, M=8, ef_construction=50, seed=42)
        rng = np.random.default_rng(42)
        corpus = rng.standard_normal((30, 64)).astype(np.float32)

        for i, emb in enumerate(corpus):
            index.insert(i, emb)

        path = str(tmp_path / "new_index.tqhnsw")  # type: ignore[operator]
        index.sync(path)

        # File should exist and be loadable.
        loaded = CompressedHNSW.open(path, tq)
        assert loaded.n_vectors == 30

    def test_multiple_syncs(self, tq: TurboQuantPGVector, tmp_path: object) -> None:
        """Insert batch, sync, insert more, sync again — all present."""
        index = CompressedHNSW(tq, M=8, ef_construction=50, seed=42)
        rng = np.random.default_rng(42)

        path = str(tmp_path / "index.tqhnsw")  # type: ignore[operator]

        # First batch.
        batch1 = rng.standard_normal((40, 64)).astype(np.float32)
        for i, emb in enumerate(batch1):
            index.insert(i, emb)
        index.sync(path)  # Full save (new file).

        # Second batch.
        batch2 = rng.standard_normal((30, 64)).astype(np.float32)
        for i, emb in enumerate(batch2):
            index.insert(40 + i, emb)
        index.sync()  # Incremental append.

        # Third batch.
        batch3 = rng.standard_normal((20, 64)).astype(np.float32)
        for i, emb in enumerate(batch3):
            index.insert(70 + i, emb)
        index.sync()  # Incremental append.

        # Reload and verify everything.
        reloaded = CompressedHNSW.open(path, tq)
        assert reloaded.n_vectors == 90

        all_ids = set(reloaded._nodes.keys())
        for i in range(90):
            assert i in all_ids, f"Node {i} missing after multiple syncs"

    def test_search_after_incremental(
        self, tq: TurboQuantPGVector, tmp_path: object
    ) -> None:
        """Search quality after incremental persist + reload is reasonable.

        A full save() followed by open() preserves exact graph quality
        because all neighbor lists are written.  The incremental append
        path only writes *new* nodes, so backward edges on previously
        saved nodes are stale on disk.  This test verifies that:

        1. A full-save round-trip preserves recall against brute force.
        2. An incremental round-trip still achieves reasonable recall
           (> 0.40 against brute force).
        """
        rng = np.random.default_rng(42)
        corpus = rng.standard_normal((200, 64)).astype(np.float32)

        # --- Full-save round-trip (reference) ---
        full_index = CompressedHNSW(tq, M=8, ef_construction=100, seed=42)
        for i, emb in enumerate(corpus):
            full_index.insert(i, emb)

        full_path = str(tmp_path / "full.tqhnsw")  # type: ignore[operator]
        full_index.save(full_path)
        full_reloaded = CompressedHNSW.open(full_path, tq)

        # --- Incremental round-trip ---
        split = 100
        inc_index = CompressedHNSW(tq, M=8, ef_construction=100, seed=42)
        for i, emb in enumerate(corpus[:split]):
            inc_index.insert(i, emb)

        inc_path = str(tmp_path / "inc.tqhnsw")  # type: ignore[operator]
        inc_index.save(inc_path)

        inc_index = CompressedHNSW.open(inc_path, tq)
        for i, emb in enumerate(corpus[split:]):
            inc_index.insert(split + i, emb)
        inc_index.sync()

        inc_reloaded = CompressedHNSW.open(inc_path, tq)

        # Compare both against brute force.
        k = 10
        queries = rng.standard_normal((20, 64)).astype(np.float32)

        recall_full = 0.0
        recall_inc = 0.0
        for q in queries:
            exact_ids = set(_brute_force_knn(q, corpus, k))

            full_results = full_reloaded.search(q, k=k, rerank=True)
            full_ids = {r[0] for r in full_results}
            recall_full += len(exact_ids & full_ids) / k

            inc_results = inc_reloaded.search(q, k=k, rerank=True)
            inc_ids = {r[0] for r in inc_results}
            recall_inc += len(exact_ids & inc_ids) / k

        avg_recall_full = recall_full / len(queries)
        avg_recall_inc = recall_inc / len(queries)

        # Full-save round-trip should preserve good recall.
        assert (
            avg_recall_full > 0.60
        ), f"Full-save recall ({avg_recall_full:.3f}) too low"

        # Incremental round-trip loses backward edges on existing nodes
        # but should still achieve usable recall.
        assert (
            avg_recall_inc > 0.40
        ), f"Incremental recall ({avg_recall_inc:.3f}) too low"

"""
Tests for the compressed embedding cache adapter.

Usage:
    pytest tests/test_cache_adapter.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro.cache_adapter import (
    CompressedEmbeddingCache,
    InMemoryCacheBackend,
)
from turboquant_pro.pgvector import TurboQuantPGVector


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


@pytest.fixture()
def tq() -> TurboQuantPGVector:
    return TurboQuantPGVector(dim=128, bits=3, seed=0)


def _random_embedding(dim: int = 128, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ------------------------------------------------------------------ #
# InMemoryCacheBackend                                                 #
# ------------------------------------------------------------------ #


class TestInMemoryCacheBackend:
    """Tests for the in-memory LRU backend."""

    def test_put_get_roundtrip(self) -> None:
        backend = InMemoryCacheBackend()
        data = b"\x01\x02\x03\x04"
        backend.set("k1", data)
        assert backend.get("k1") == data

    def test_miss_returns_none(self) -> None:
        backend = InMemoryCacheBackend()
        assert backend.get("nonexistent") is None

    def test_delete(self) -> None:
        backend = InMemoryCacheBackend()
        backend.set("k1", b"data")
        assert backend.delete("k1") is True
        assert backend.get("k1") is None
        assert backend.delete("k1") is False

    def test_exists(self) -> None:
        backend = InMemoryCacheBackend()
        assert backend.exists("k1") is False
        backend.set("k1", b"data")
        assert backend.exists("k1") is True

    def test_size(self) -> None:
        backend = InMemoryCacheBackend()
        assert backend.size() == 0
        backend.set("k1", b"a")
        backend.set("k2", b"b")
        assert backend.size() == 2

    def test_lru_eviction(self) -> None:
        backend = InMemoryCacheBackend(max_entries=3)
        backend.set("k1", b"a")
        backend.set("k2", b"b")
        backend.set("k3", b"c")

        # Access k1 to make it recently used
        backend.get("k1")

        # Insert k4 — should evict k2 (the least recently used)
        backend.set("k4", b"d")

        assert backend.get("k2") is None  # evicted
        assert backend.get("k1") == b"a"  # kept (was accessed)
        assert backend.get("k3") == b"c"  # kept
        assert backend.get("k4") == b"d"  # kept

    def test_clear(self) -> None:
        backend = InMemoryCacheBackend()
        backend.set("k1", b"a")
        backend.set("k2", b"b")
        backend.clear()
        assert backend.size() == 0
        assert backend.get("k1") is None

    def test_memory_usage(self) -> None:
        backend = InMemoryCacheBackend()
        backend.set("k1", b"\x00" * 100)
        backend.set("k2", b"\x00" * 200)
        assert backend.memory_usage_bytes() == 300

    def test_overwrite_existing_key(self) -> None:
        backend = InMemoryCacheBackend(max_entries=2)
        backend.set("k1", b"old")
        backend.set("k2", b"data")
        # Overwriting k1 should not count as a new entry
        backend.set("k1", b"new")
        assert backend.size() == 2
        assert backend.get("k1") == b"new"


# ------------------------------------------------------------------ #
# CompressedEmbeddingCache                                             #
# ------------------------------------------------------------------ #


class TestCompressedEmbeddingCache:
    """Tests for the high-level compressed cache."""

    def test_put_get_roundtrip(self, tq: TurboQuantPGVector) -> None:
        backend = InMemoryCacheBackend()
        cache = CompressedEmbeddingCache(tq, backend)
        emb = _random_embedding(dim=128, seed=42)
        cache.put("doc:1", emb)

        result = cache.get("doc:1")
        assert result is not None
        assert result.shape == (128,)
        sim = _cosine_similarity(emb, result)
        assert sim > 0.95, f"Cosine similarity {sim} too low"

    def test_cache_miss(self, tq: TurboQuantPGVector) -> None:
        backend = InMemoryCacheBackend()
        cache = CompressedEmbeddingCache(tq, backend)
        assert cache.get("nonexistent") is None

    def test_batch_put_get(self, tq: TurboQuantPGVector) -> None:
        backend = InMemoryCacheBackend()
        cache = CompressedEmbeddingCache(tq, backend)

        rng = np.random.default_rng(42)
        embs = rng.standard_normal((20, 128)).astype(np.float32)
        keys = [f"doc:{i}" for i in range(20)]

        cache.put_batch(keys, embs)

        results = cache.get_batch(keys)
        assert len(results) == 20
        for i, r in enumerate(results):
            assert r is not None
            sim = _cosine_similarity(embs[i], r)
            assert sim > 0.95

    def test_stats(self, tq: TurboQuantPGVector) -> None:
        backend = InMemoryCacheBackend()
        cache = CompressedEmbeddingCache(tq, backend)

        # One hit, one miss
        emb = _random_embedding(dim=128)
        cache.put("k1", emb)
        cache.get("k1")       # hit
        cache.get("missing")  # miss

        stats = cache.stats()
        assert stats["n_entries"] == 1
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["memory_bytes"] > 0
        assert stats["equivalent_float32_bytes"] == 128 * 4
        assert stats["effective_compression_ratio"] > 5.0

    def test_effective_compression(self, tq: TurboQuantPGVector) -> None:
        backend = InMemoryCacheBackend()
        cache = CompressedEmbeddingCache(tq, backend)

        rng = np.random.default_rng(0)
        for i in range(100):
            cache.put(f"k{i}", rng.standard_normal(128).astype(np.float32))

        stats = cache.stats()
        # 3-bit compression on 128-dim should give ~8-10x ratio
        assert stats["effective_compression_ratio"] > 5.0
        assert stats["n_entries"] == 100

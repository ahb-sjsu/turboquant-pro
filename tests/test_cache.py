"""
Streaming KV cache tests for TurboQuant-KV.

Tests the TurboQuantKVCache with tiered hot/cold storage:
- Append increases length
- Hot window stays bounded
- Query spanning cold and hot storage
- Query hot-only and cold-only ranges
- 2D input convenience
- Memory stats
- Clear operation

Usage:
    pytest tests/test_cache.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_kv import TurboQuantKVCache


class TestTurboQuantKVCache:
    """Tests for the streaming TurboQuantKVCache."""

    def test_append_increases_length(self) -> None:
        """Appending tokens increases cache length."""
        cache = TurboQuantKVCache(
            head_dim=64,
            n_heads=4,
            bits=3,
            hot_window=8,
            use_gpu=False,
            seed=0,
        )
        assert cache.length == 0
        k = np.random.randn(1, 4, 1, 64).astype(np.float32)
        v = np.random.randn(1, 4, 1, 64).astype(np.float32)
        cache.append(k, v)
        assert cache.length == 1
        cache.append(k, v)
        assert cache.length == 2

    def test_hot_window_stays_bounded(self) -> None:
        """Hot window flushes to cold when exceeding limit."""
        hot_window = 8
        cache = TurboQuantKVCache(
            head_dim=64,
            n_heads=2,
            bits=3,
            hot_window=hot_window,
            use_gpu=False,
            seed=0,
        )
        rng = np.random.default_rng(42)
        for _ in range(20):
            k = rng.standard_normal((1, 2, 1, 64)).astype(np.float32)
            v = rng.standard_normal((1, 2, 1, 64)).astype(np.float32)
            cache.append(k, v)

        assert cache.length == 20
        assert cache.hot_length <= hot_window
        assert cache.cold_length > 0
        assert cache.cold_length + cache.hot_length == 20

    def test_query_full_range(self) -> None:
        """Can query entire cache spanning cold and hot."""
        cache = TurboQuantKVCache(
            head_dim=64,
            n_heads=2,
            bits=3,
            hot_window=4,
            use_gpu=False,
            seed=0,
        )
        rng = np.random.default_rng(42)
        for _ in range(10):
            k = rng.standard_normal((1, 2, 1, 64)).astype(np.float32)
            v = rng.standard_normal((1, 2, 1, 64)).astype(np.float32)
            cache.append(k, v)

        keys = cache.get_keys(0, cache.length)
        values = cache.get_values(0, cache.length)
        assert keys.shape == (1, 2, 10, 64)
        assert values.shape == (1, 2, 10, 64)

    def test_query_hot_only(self) -> None:
        """Can query just the hot window."""
        cache = TurboQuantKVCache(
            head_dim=64,
            n_heads=2,
            bits=3,
            hot_window=4,
            use_gpu=False,
            seed=0,
        )
        rng = np.random.default_rng(42)
        for _ in range(10):
            k = rng.standard_normal((1, 2, 1, 64)).astype(np.float32)
            v = rng.standard_normal((1, 2, 1, 64)).astype(np.float32)
            cache.append(k, v)

        hot_start = cache.length - cache.hot_length
        keys = cache.get_keys(hot_start, cache.length)
        assert keys.shape[2] == cache.hot_length

    def test_query_cold_only(self) -> None:
        """Can query just cold storage."""
        cache = TurboQuantKVCache(
            head_dim=64,
            n_heads=2,
            bits=3,
            hot_window=4,
            use_gpu=False,
            seed=0,
        )
        rng = np.random.default_rng(42)
        for _ in range(10):
            k = rng.standard_normal((1, 2, 1, 64)).astype(np.float32)
            v = rng.standard_normal((1, 2, 1, 64)).astype(np.float32)
            cache.append(k, v)

        cold_len = cache.cold_length
        if cold_len > 0:
            keys = cache.get_keys(0, cold_len)
            assert keys.shape[2] == cold_len

    def test_2d_input(self) -> None:
        """Append accepts (n_heads, head_dim) 2D input."""
        cache = TurboQuantKVCache(
            head_dim=64,
            n_heads=4,
            bits=3,
            hot_window=8,
            use_gpu=False,
            seed=0,
        )
        k = np.random.randn(4, 64).astype(np.float32)
        v = np.random.randn(4, 64).astype(np.float32)
        cache.append(k, v)
        assert cache.length == 1
        keys = cache.get_keys(0, 1)
        assert keys.shape == (1, 4, 1, 64)

    def test_memory_stats(self) -> None:
        """Memory stats return reasonable values."""
        cache = TurboQuantKVCache(
            head_dim=64,
            n_heads=2,
            bits=3,
            hot_window=4,
            use_gpu=False,
            seed=0,
        )
        rng = np.random.default_rng(42)
        for _ in range(10):
            k = rng.standard_normal((1, 2, 1, 64)).astype(np.float32)
            v = rng.standard_normal((1, 2, 1, 64)).astype(np.float32)
            cache.append(k, v)

        stats = cache.memory_stats()
        assert stats["total_bytes"] > 0
        assert stats["uncompressed_equivalent_bytes"] > stats["total_bytes"]
        assert stats["effective_ratio"] > 1.0

    def test_clear(self) -> None:
        """Clear resets cache to empty."""
        cache = TurboQuantKVCache(
            head_dim=64,
            n_heads=2,
            bits=3,
            hot_window=4,
            use_gpu=False,
            seed=0,
        )
        rng = np.random.default_rng(42)
        for _ in range(5):
            k = rng.standard_normal((1, 2, 1, 64)).astype(np.float32)
            v = rng.standard_normal((1, 2, 1, 64)).astype(np.float32)
            cache.append(k, v)

        assert cache.length > 0
        cache.clear()
        assert cache.length == 0
        assert cache.hot_length == 0
        assert cache.cold_length == 0

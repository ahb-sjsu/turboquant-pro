"""Tests for the vLLM KV cache plugin."""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro.vllm_plugin import (
    TurboQuantKVBackend,
    TurboQuantKVManager,
)


def _random_kv(n_heads=4, head_dim=64, seed=42):
    """Generate random key and value tensors."""
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((n_heads, head_dim)).astype(np.float32)
    v = rng.standard_normal((n_heads, head_dim)).astype(np.float32)
    return k, v


class TestKVManager:
    """Tests for TurboQuantKVManager."""

    def test_create(self) -> None:
        mgr = TurboQuantKVManager(n_layers=4, n_kv_heads=4, head_dim=64, bits=3)
        assert mgr.n_layers == 4
        assert mgr.length(0) == 0

    def test_store_and_load(self) -> None:
        mgr = TurboQuantKVManager(
            n_layers=2,
            n_kv_heads=4,
            head_dim=64,
            bits=3,
            hot_window=8,
        )

        # Store a few tokens
        for i in range(5):
            k, v = _random_kv(n_heads=4, head_dim=64, seed=i)
            mgr.store(0, k, v)

        assert mgr.length(0) == 5

        # Load back
        keys, values = mgr.load(0, 0, 5)
        assert keys.shape[-1] == 64  # head_dim preserved

    def test_multi_layer(self) -> None:
        mgr = TurboQuantKVManager(n_layers=3, n_kv_heads=2, head_dim=32, bits=3)

        k, v = _random_kv(n_heads=2, head_dim=32)
        mgr.store(0, k, v)
        mgr.store(1, k, v)
        # Layer 2 untouched

        assert mgr.length(0) == 1
        assert mgr.length(1) == 1
        assert mgr.length(2) == 0

    def test_memory_stats(self) -> None:
        mgr = TurboQuantKVManager(n_layers=2, n_kv_heads=4, head_dim=64, bits=3)

        for i in range(10):
            k, v = _random_kv(n_heads=4, head_dim=64, seed=i)
            mgr.store(0, k, v)

        stats = mgr.memory_stats()
        assert stats["n_layers"] == 2
        assert stats["total_tokens"] == 10
        assert stats["total_bytes"] > 0
        assert "compression_ratio" in stats

    def test_clear(self) -> None:
        mgr = TurboQuantKVManager(n_layers=2, n_kv_heads=4, head_dim=64, bits=3)
        k, v = _random_kv(n_heads=4, head_dim=64)
        mgr.store(0, k, v)
        assert mgr.length(0) == 1

        mgr.clear()
        assert mgr.length(0) == 0

    def test_estimate_capacity(self) -> None:
        mgr = TurboQuantKVManager(
            n_layers=32,
            n_kv_heads=8,
            head_dim=128,
            bits=3,
            hot_window=512,
        )
        cap = mgr.estimate_capacity(max_memory_gb=4.0)
        assert cap > 512  # More than just the hot window
        assert cap < 1_000_000  # Sanity check

    def test_layer_stats(self) -> None:
        mgr = TurboQuantKVManager(n_layers=2, n_kv_heads=4, head_dim=64, bits=3)
        k, v = _random_kv(n_heads=4, head_dim=64)
        mgr.store(0, k, v)

        ls = mgr.layer_stats(0)
        assert ls.layer_id == 0
        assert ls.length == 1
        assert ls.total_bytes > 0


class TestKVBackend:
    """Tests for the vLLM backend wrapper."""

    def test_initialize(self) -> None:
        backend = TurboQuantKVBackend(bits=3, hot_window=256)
        backend.initialize(n_layers=4, n_kv_heads=4, head_dim=64)
        assert backend._manager is not None

    def test_store_and_load(self) -> None:
        backend = TurboQuantKVBackend(bits=3, hot_window=256)
        backend.initialize(n_layers=2, n_kv_heads=4, head_dim=64)

        k, v = _random_kv(n_heads=4, head_dim=64)
        backend.store_block(0, k, v)

        keys, values = backend.load_block(0, 0, 1)
        assert keys.shape[-1] == 64

    def test_not_initialized_raises(self) -> None:
        backend = TurboQuantKVBackend()
        with pytest.raises(RuntimeError, match="not initialized"):
            k, v = _random_kv()
            backend.store_block(0, k, v)

    def test_memory_stats_empty(self) -> None:
        backend = TurboQuantKVBackend()
        assert backend.memory_stats() == {}

    def test_supports_async(self) -> None:
        backend = TurboQuantKVBackend()
        assert backend.supports_async() is False

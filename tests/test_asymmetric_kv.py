"""
Tests for asymmetric K/V bit allocation (Issue #10).

Verifies that keys and values can use different bit-widths, that the
asymmetric mode produces better key quality than uniform lower-bit,
and that backward compatibility is preserved.

Usage:
    pytest tests/test_asymmetric_kv.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro import TurboQuantKV, TurboQuantKVCache


def _random_kv(
    batch: int = 1,
    n_heads: int = 4,
    seq_len: int = 32,
    head_dim: int = 64,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, n_heads, seq_len, head_dim)).astype("float32")


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    dot = np.sum(a_flat * b_flat, axis=-1)
    norm_a = np.linalg.norm(a_flat, axis=-1)
    norm_b = np.linalg.norm(b_flat, axis=-1)
    return float(np.mean(dot / np.maximum(norm_a * norm_b, 1e-30)))


# ------------------------------------------------------------------ #
# TurboQuantKV asymmetric construction                                 #
# ------------------------------------------------------------------ #


class TestAsymmetricConstruction:
    """Verify TurboQuantKV accepts and stores asymmetric bit params."""

    def test_default_symmetric(self) -> None:
        tq = TurboQuantKV(head_dim=64, bits=3, use_gpu=False, seed=0)
        assert tq.key_bits == 3
        assert tq.value_bits == 3
        assert tq.bits == 3

    def test_explicit_asymmetric(self) -> None:
        tq = TurboQuantKV(
            head_dim=64, bits=3, key_bits=4, value_bits=2, use_gpu=False, seed=0
        )
        assert tq.key_bits == 4
        assert tq.value_bits == 2
        assert tq.bits == 3  # default preserved

    def test_key_bits_only(self) -> None:
        tq = TurboQuantKV(head_dim=64, bits=3, key_bits=4, use_gpu=False, seed=0)
        assert tq.key_bits == 4
        assert tq.value_bits == 3  # falls back to bits

    def test_value_bits_only(self) -> None:
        tq = TurboQuantKV(head_dim=64, bits=3, value_bits=2, use_gpu=False, seed=0)
        assert tq.key_bits == 3  # falls back to bits
        assert tq.value_bits == 2

    def test_invalid_key_bits_raises(self) -> None:
        with pytest.raises(ValueError, match="key_bits"):
            TurboQuantKV(head_dim=64, bits=3, key_bits=5, use_gpu=False)

    def test_invalid_value_bits_raises(self) -> None:
        with pytest.raises(ValueError, match="value_bits"):
            TurboQuantKV(head_dim=64, bits=3, value_bits=1, use_gpu=False)


# ------------------------------------------------------------------ #
# Compress with kind parameter                                         #
# ------------------------------------------------------------------ #


class TestCompressWithKind:
    """Verify compress(kind=...) uses the correct bit-width."""

    def test_key_uses_key_bits(self) -> None:
        tq = TurboQuantKV(head_dim=64, key_bits=4, value_bits=2, use_gpu=False, seed=0)
        tensor = _random_kv(head_dim=64)
        c_key = tq.compress(tensor, packed=True, kind="key")
        assert c_key.bits == 4

    def test_value_uses_value_bits(self) -> None:
        tq = TurboQuantKV(head_dim=64, key_bits=4, value_bits=2, use_gpu=False, seed=0)
        tensor = _random_kv(head_dim=64)
        c_val = tq.compress(tensor, packed=True, kind="value")
        assert c_val.bits == 2

    def test_none_kind_uses_default_bits(self) -> None:
        tq = TurboQuantKV(
            head_dim=64, bits=3, key_bits=4, value_bits=2, use_gpu=False, seed=0
        )
        tensor = _random_kv(head_dim=64)
        c = tq.compress(tensor, packed=True, kind=None)
        assert c.bits == 3

    def test_key_larger_than_value(self) -> None:
        """Key at 4-bit should be larger (packed) than value at 2-bit."""
        tq = TurboQuantKV(head_dim=64, key_bits=4, value_bits=2, use_gpu=False, seed=0)
        tensor = _random_kv(head_dim=64)
        c_key = tq.compress(tensor, packed=True, kind="key")
        c_val = tq.compress(tensor, packed=True, kind="value")
        assert c_key.indices.nbytes > c_val.indices.nbytes


# ------------------------------------------------------------------ #
# Decompress roundtrip                                                 #
# ------------------------------------------------------------------ #


class TestAsymmetricRoundtrip:
    """Verify compress -> decompress roundtrip with asymmetric bits."""

    @pytest.mark.parametrize(
        "key_bits,value_bits",
        [(4, 3), (4, 2), (3, 2)],
    )
    def test_roundtrip_shape(self, key_bits: int, value_bits: int) -> None:
        tq = TurboQuantKV(
            head_dim=64,
            key_bits=key_bits,
            value_bits=value_bits,
            use_gpu=False,
            seed=0,
        )
        tensor = _random_kv(head_dim=64)

        for kind in ("key", "value"):
            compressed = tq.compress(tensor, packed=True, kind=kind)
            recon = tq.decompress(compressed)
            assert recon.shape == tensor.shape

    def test_key_higher_quality_than_value(self) -> None:
        """K4/V2 should give better key quality than value quality."""
        tq = TurboQuantKV(head_dim=128, key_bits=4, value_bits=2, use_gpu=False, seed=0)
        tensor = _random_kv(head_dim=128)

        c_key = tq.compress(tensor, packed=True, kind="key")
        c_val = tq.compress(tensor, packed=True, kind="value")

        key_recon = tq.decompress(c_key)
        val_recon = tq.decompress(c_val)

        key_sim = _cosine_similarity(tensor, key_recon)
        val_sim = _cosine_similarity(tensor, val_recon)

        # 4-bit keys should have higher cosine sim than 2-bit values
        assert (
            key_sim > val_sim
        ), f"Key sim ({key_sim:.4f}) should be > value sim ({val_sim:.4f})"

    def test_asymmetric_better_than_uniform_low(self) -> None:
        """K4/V3 key quality should beat uniform K3/V3 key quality."""
        tensor = _random_kv(head_dim=128, seq_len=64)

        tq_asym = TurboQuantKV(
            head_dim=128, key_bits=4, value_bits=3, use_gpu=False, seed=0
        )
        tq_uniform = TurboQuantKV(head_dim=128, bits=3, use_gpu=False, seed=0)

        c_asym = tq_asym.compress(tensor, packed=True, kind="key")
        c_uniform = tq_uniform.compress(tensor, packed=True, kind="key")

        recon_asym = tq_asym.decompress(c_asym)
        recon_uniform = tq_uniform.decompress(c_uniform)

        sim_asym = _cosine_similarity(tensor, recon_asym)
        sim_uniform = _cosine_similarity(tensor, recon_uniform)

        assert sim_asym > sim_uniform, (
            f"Asymmetric K4 ({sim_asym:.4f}) should beat uniform K3 "
            f"({sim_uniform:.4f})"
        )


# ------------------------------------------------------------------ #
# KVCache with asymmetric bits                                         #
# ------------------------------------------------------------------ #


class TestAsymmetricKVCache:
    """Verify TurboQuantKVCache works with asymmetric bits."""

    def test_cache_construction(self) -> None:
        cache = TurboQuantKVCache(
            head_dim=64,
            n_heads=4,
            bits=3,
            key_bits=4,
            value_bits=3,
            hot_window=8,
            use_gpu=False,
            seed=0,
        )
        assert cache.key_bits == 4
        assert cache.value_bits == 3

    def test_cache_append_and_retrieve(self) -> None:
        cache = TurboQuantKVCache(
            head_dim=64,
            n_heads=4,
            key_bits=4,
            value_bits=3,
            hot_window=4,
            use_gpu=False,
            seed=0,
        )
        rng = np.random.default_rng(42)
        for _ in range(10):
            k = rng.standard_normal((1, 4, 1, 64)).astype(np.float32)
            v = rng.standard_normal((1, 4, 1, 64)).astype(np.float32)
            cache.append(k, v)

        assert cache.length == 10
        keys = cache.get_keys(0, 10)
        values = cache.get_values(0, 10)
        assert keys.shape == (1, 4, 10, 64)
        assert values.shape == (1, 4, 10, 64)

    def test_cold_storage_uses_correct_bits(self) -> None:
        """After flushing to cold, K chunks should use key_bits."""
        cache = TurboQuantKVCache(
            head_dim=64,
            n_heads=4,
            key_bits=4,
            value_bits=2,
            hot_window=4,
            use_gpu=False,
            seed=0,
        )
        rng = np.random.default_rng(42)
        # Insert enough to trigger flush
        for _ in range(10):
            k = rng.standard_normal((1, 4, 1, 64)).astype(np.float32)
            v = rng.standard_normal((1, 4, 1, 64)).astype(np.float32)
            cache.append(k, v)

        # Cold storage should exist
        assert cache.cold_length > 0
        # Check that cold key chunks use 4-bit
        for ck in cache._cold_keys:
            assert ck.bits == 4
        # Check that cold value chunks use 2-bit
        for cv in cache._cold_values:
            assert cv.bits == 2


# ------------------------------------------------------------------ #
# Memory estimation                                                    #
# ------------------------------------------------------------------ #


class TestAsymmetricMemoryEstimation:
    """Verify estimate_memory supports asymmetric K/V bits."""

    def test_asymmetric_smaller_than_uniform_high(self) -> None:
        """K4/V2 packed should use less memory than uniform K4/V4."""
        est_asym = TurboQuantKV.estimate_memory(
            n_layers=32,
            n_kv_heads=8,
            head_dim=128,
            seq_len=4096,
            key_bits=4,
            value_bits=2,
            bit_packed=True,
        )
        est_uniform = TurboQuantKV.estimate_memory(
            n_layers=32,
            n_kv_heads=8,
            head_dim=128,
            seq_len=4096,
            bits=4,
            bit_packed=True,
        )
        assert est_asym["compressed_gb"] < est_uniform["compressed_gb"]

    def test_symmetric_backward_compat(self) -> None:
        """Without key_bits/value_bits, should match old behavior."""
        est_new = TurboQuantKV.estimate_memory(
            n_layers=32,
            n_kv_heads=8,
            head_dim=128,
            seq_len=4096,
            bits=3,
            bit_packed=True,
        )
        # The old formula: 2 * n_elements * 3/8 + n_vectors * 4
        n_elements = 32 * 8 * 4096 * 128
        n_vectors = 32 * 8 * 4096
        expected = ((2 * n_elements * 3 + 7) // 8 + 2 * n_vectors * 4) / (1024**3)
        # Should be close (within rounding)
        assert abs(est_new["compressed_gb"] - round(expected, 3)) < 0.01

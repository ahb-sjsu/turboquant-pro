"""
Tests for RoPE-aware KV cache quantization (Issue #9).

Verifies that the RoPE frequency analyzer correctly identifies
low-frequency dimensions and that the RoPE-aware quantizer
provides higher-precision compression for those dimensions.

Usage:
    pytest tests/test_rope.py -v
"""

from __future__ import annotations

import numpy as np

from turboquant_pro.rope import RoPEAwareQuantizer, RoPEFrequencyAnalyzer

# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


def _random_kv(
    batch: int = 1,
    n_heads: int = 4,
    seq_len: int = 32,
    head_dim: int = 128,
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
# TestRoPEFrequencyAnalyzer                                           #
# ------------------------------------------------------------------ #


class TestRoPEFrequencyAnalyzer:
    """Tests for the RoPE frequency spectrum analyzer."""

    def test_frequencies_shape(self) -> None:
        """Frequencies array has shape (head_dim // 2,)."""
        analyzer = RoPEFrequencyAnalyzer(head_dim=128)
        assert analyzer.frequencies.shape == (64,)

    def test_frequencies_decreasing(self) -> None:
        """Higher pair indices have lower frequency."""
        analyzer = RoPEFrequencyAnalyzer(head_dim=128)
        diffs = np.diff(analyzer.frequencies)
        assert np.all(diffs < 0), "Frequencies should be strictly decreasing"

    def test_wavelengths_increasing(self) -> None:
        """Higher pair indices have longer wavelength."""
        analyzer = RoPEFrequencyAnalyzer(head_dim=128)
        diffs = np.diff(analyzer.wavelengths)
        assert np.all(diffs > 0), "Wavelengths should be strictly increasing"

    def test_bit_allocation_shape(self) -> None:
        """Bit allocation has shape (head_dim,)."""
        analyzer = RoPEFrequencyAnalyzer(head_dim=128)
        alloc = analyzer.bit_allocation()
        assert alloc.shape == (128,)
        assert alloc.dtype == np.uint8

    def test_bit_allocation_pairs_match(self) -> None:
        """Dimensions 2i and 2i+1 always have the same bit width."""
        analyzer = RoPEFrequencyAnalyzer(head_dim=128, max_seq_len=8192)
        alloc = analyzer.bit_allocation(default_bits=3, boost_bits=4)
        for i in range(64):
            assert alloc[2 * i] == alloc[2 * i + 1], (
                f"Pair {i}: dim {2*i} has {alloc[2*i]} bits "
                f"but dim {2*i+1} has {alloc[2*i+1]} bits"
            )

    def test_boost_low_frequency(self) -> None:
        """Low-frequency dims (wavelength > max_seq_len) get boost_bits."""
        analyzer = RoPEFrequencyAnalyzer(head_dim=128, max_seq_len=8192)
        alloc = analyzer.bit_allocation(default_bits=3, boost_bits=4)

        for pair_idx in range(64):
            wavelength = analyzer.wavelengths[pair_idx]
            expected_bits = 4 if wavelength > 8192 else 3
            assert alloc[2 * pair_idx] == expected_bits, (
                f"Pair {pair_idx} (wavelength={wavelength:.1f}): "
                f"expected {expected_bits} bits, got {alloc[2 * pair_idx]}"
            )

    def test_summary_keys(self) -> None:
        """Summary returns expected dict keys."""
        analyzer = RoPEFrequencyAnalyzer(head_dim=128)
        s = analyzer.summary()
        expected_keys = {
            "n_boosted_dims",
            "n_default_dims",
            "avg_bits",
            "frequencies_range",
            "wavelengths_range",
        }
        assert set(s.keys()) == expected_keys


# ------------------------------------------------------------------ #
# TestRoPEAwareQuantizer                                              #
# ------------------------------------------------------------------ #


class TestRoPEAwareQuantizer:
    """Tests for the RoPE-aware KV cache quantizer."""

    def test_construction(self) -> None:
        """RoPEAwareQuantizer creates successfully with default args."""
        raq = RoPEAwareQuantizer(head_dim=128, n_heads=4, use_gpu=False, seed=0)
        assert raq.head_dim == 128
        assert raq.n_heads == 4
        assert raq.default_bits == 3
        assert raq.boost_bits == 4
        assert raq._n_boost + raq._n_default == 128

    def test_compress_decompress_roundtrip(self) -> None:
        """Compressed-then-decompressed tensor preserves shape."""
        head_dim = 128
        tensor = _random_kv(batch=1, n_heads=4, seq_len=16, head_dim=head_dim)
        raq = RoPEAwareQuantizer(head_dim=head_dim, n_heads=4, use_gpu=False, seed=0)
        compressed = raq.compress(tensor)
        reconstructed = raq.decompress(compressed)
        assert reconstructed.shape == tensor.shape
        assert reconstructed.dtype == np.float32

    def test_cosine_similarity(self) -> None:
        """Round-trip reconstruction has cosine similarity > 0.95."""
        head_dim = 128
        tensor = _random_kv(batch=2, n_heads=8, seq_len=64, head_dim=head_dim, seed=123)
        raq = RoPEAwareQuantizer(head_dim=head_dim, n_heads=8, use_gpu=False, seed=0)
        compressed = raq.compress(tensor)
        reconstructed = raq.decompress(compressed)
        cos_sim = _cosine_similarity(tensor, reconstructed)
        assert cos_sim > 0.95, f"Cosine similarity {cos_sim:.4f} too low"

    def test_avg_bits_between_default_and_boost(self) -> None:
        """Average bits is between default_bits and boost_bits."""
        raq = RoPEAwareQuantizer(
            head_dim=128,
            n_heads=4,
            default_bits=3,
            boost_bits=4,
            max_seq_len=8192,
            use_gpu=False,
            seed=0,
        )
        # If there are any boosted dims, avg should be strictly between
        # 3 and 4.  If none are boosted, avg == 3.
        assert raq.avg_bits >= raq.default_bits
        assert raq.avg_bits <= raq.boost_bits

    def test_different_rope_bases(self) -> None:
        """Different RoPE base frequencies change the bit allocation."""
        raq_10k = RoPEAwareQuantizer(
            head_dim=128,
            rope_base=10000.0,
            max_seq_len=8192,
            use_gpu=False,
            seed=0,
        )
        raq_500k = RoPEAwareQuantizer(
            head_dim=128,
            rope_base=500000.0,
            max_seq_len=8192,
            use_gpu=False,
            seed=0,
        )
        # A larger base produces lower frequencies (longer wavelengths),
        # so more dims exceed max_seq_len and get boosted.
        assert raq_500k._n_boost >= raq_10k._n_boost

    def test_long_context_more_boost(self) -> None:
        """Shorter max_seq_len results in more boosted dimensions.

        A shorter context window is easier for wavelengths to exceed,
        so more dimension pairs qualify as "low-frequency" and receive
        the boosted bit width.
        """
        raq_short = RoPEAwareQuantizer(
            head_dim=128,
            max_seq_len=2048,
            use_gpu=False,
            seed=0,
        )
        raq_long = RoPEAwareQuantizer(
            head_dim=128,
            max_seq_len=131072,
            use_gpu=False,
            seed=0,
        )
        assert raq_short._n_boost >= raq_long._n_boost, (
            f"Shorter context should boost at least as many dims: "
            f"short={raq_short._n_boost}, long={raq_long._n_boost}"
        )

"""
Unit tests for TurboQuant-KV core compress/decompress and bit-packing.

Tests the TurboQuantKV compressor on CPU (NumPy) to validate:
- Round-trip compress/decompress (unpacked and packed)
- Bit-packing round-trip (pack -> unpack = original)
- Packed compression ratio matches theoretical
- Reconstruction accuracy (MSE, cosine similarity)
- Compression ratio calculations
- Edge cases (zero vectors, single-element batches)
- Memory estimation utility
- All supported bit widths (2, 3, 4)

Usage:
    pytest tests/test_core.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_kv import CompressedKV, TurboQuantKV


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine similarity between corresponding vectors."""
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])

    dot = np.sum(a_flat * b_flat, axis=-1)
    norm_a = np.linalg.norm(a_flat, axis=-1)
    norm_b = np.linalg.norm(b_flat, axis=-1)
    denom = np.maximum(norm_a * norm_b, 1e-30)
    return float(np.mean(dot / denom))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two arrays."""
    return float(np.mean((a - b) ** 2))


def _random_kv(
    batch: int = 1,
    n_heads: int = 4,
    seq_len: int = 32,
    head_dim: int = 64,
    dtype: str = "float32",
    seed: int = 42,
) -> np.ndarray:
    """Generate a random KV-like tensor."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, n_heads, seq_len, head_dim)).astype(dtype)


# ------------------------------------------------------------------ #
# Basic round-trip tests                                              #
# ------------------------------------------------------------------ #


class TestRoundTrip:
    """Compress then decompress and verify reconstruction quality."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_round_trip_shape(self, bits: int) -> None:
        """Decompressed tensor has the same shape as the original."""
        tensor = _random_kv(head_dim=64)
        tq = TurboQuantKV(head_dim=64, n_heads=4, bits=bits, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        reconstructed = tq.decompress(compressed)
        assert reconstructed.shape == tensor.shape

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_round_trip_dtype(self, bits: int) -> None:
        """Decompressed tensor is float32."""
        tensor = _random_kv(head_dim=64)
        tq = TurboQuantKV(head_dim=64, n_heads=4, bits=bits, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        reconstructed = tq.decompress(compressed)
        assert reconstructed.dtype == np.float32

    def test_3bit_cosine_similarity(self) -> None:
        """3-bit quantisation preserves cosine similarity > 0.95."""
        tensor = _random_kv(batch=2, n_heads=8, seq_len=64, head_dim=128, seed=123)
        tq = TurboQuantKV(head_dim=128, n_heads=8, bits=3, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        reconstructed = tq.decompress(compressed)
        cos_sim = _cosine_similarity(tensor, reconstructed)
        assert cos_sim > 0.95, f"Cosine similarity {cos_sim:.4f} too low for 3-bit"

    def test_4bit_cosine_similarity(self) -> None:
        """4-bit quantisation preserves cosine similarity > 0.98."""
        tensor = _random_kv(batch=2, n_heads=8, seq_len=64, head_dim=128, seed=456)
        tq = TurboQuantKV(head_dim=128, n_heads=8, bits=4, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        reconstructed = tq.decompress(compressed)
        cos_sim = _cosine_similarity(tensor, reconstructed)
        assert cos_sim > 0.98, f"Cosine similarity {cos_sim:.4f} too low for 4-bit"

    def test_2bit_cosine_similarity(self) -> None:
        """2-bit quantisation preserves cosine similarity > 0.85."""
        tensor = _random_kv(batch=2, n_heads=8, seq_len=64, head_dim=128, seed=789)
        tq = TurboQuantKV(head_dim=128, n_heads=8, bits=2, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        reconstructed = tq.decompress(compressed)
        cos_sim = _cosine_similarity(tensor, reconstructed)
        assert cos_sim > 0.85, f"Cosine similarity {cos_sim:.4f} too low for 2-bit"

    def test_mse_decreases_with_more_bits(self) -> None:
        """Higher bit widths give lower reconstruction error."""
        tensor = _random_kv(batch=1, n_heads=4, seq_len=32, head_dim=128, seed=42)
        mse_values = {}
        for bits in [2, 3, 4]:
            tq = TurboQuantKV(
                head_dim=128, n_heads=4, bits=bits, use_gpu=False, seed=0
            )
            compressed = tq.compress(tensor)
            reconstructed = tq.decompress(compressed)
            mse_values[bits] = _mse(tensor, reconstructed)

        assert mse_values[4] < mse_values[3] < mse_values[2], (
            f"MSE should decrease with more bits: {mse_values}"
        )


# ------------------------------------------------------------------ #
# Packed round-trip tests                                             #
# ------------------------------------------------------------------ #


class TestPackedRoundTrip:
    """Compress with packed=True then decompress and verify quality."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_packed_round_trip_shape(self, bits: int) -> None:
        """Packed round-trip preserves shape."""
        tensor = _random_kv(head_dim=64)
        tq = TurboQuantKV(head_dim=64, n_heads=4, bits=bits, use_gpu=False, seed=0)
        compressed = tq.compress(tensor, packed=True)
        assert compressed.packed
        reconstructed = tq.decompress(compressed)
        assert reconstructed.shape == tensor.shape

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_packed_matches_unpacked(self, bits: int) -> None:
        """Packed and unpacked produce identical reconstructions."""
        tensor = _random_kv(head_dim=64, seed=99)
        tq = TurboQuantKV(head_dim=64, n_heads=4, bits=bits, use_gpu=False, seed=0)
        c_unpacked = tq.compress(tensor, packed=False)
        c_packed = tq.compress(tensor, packed=True)
        r_unpacked = tq.decompress(c_unpacked)
        r_packed = tq.decompress(c_packed)
        np.testing.assert_array_equal(r_unpacked, r_packed)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_packed_smaller_than_unpacked(self, bits: int) -> None:
        """Packed indices use fewer bytes than unpacked."""
        tensor = _random_kv(head_dim=128, seq_len=64)
        tq = TurboQuantKV(head_dim=128, n_heads=4, bits=bits, use_gpu=False, seed=0)
        c_unpacked = tq.compress(tensor, packed=False)
        c_packed = tq.compress(tensor, packed=True)
        assert c_packed.indices.nbytes < c_unpacked.indices.nbytes

    def test_packed_3bit_compression_ratio(self) -> None:
        """3-bit packed at head_dim=256 achieves > 4x compression."""
        tensor = _random_kv(batch=1, n_heads=16, seq_len=128, head_dim=256)
        tq = TurboQuantKV(
            head_dim=256, n_heads=16, bits=3, use_gpu=False, seed=0
        )
        compressed = tq.compress(tensor, packed=True)
        ratio = compressed.compression_ratio(256)
        assert ratio > 4.0, f"Expected ratio > 4.0, got {ratio:.2f}"


# ------------------------------------------------------------------ #
# Bit-packing unit tests                                              #
# ------------------------------------------------------------------ #


class TestBitPacking:
    """Test raw bit-packing round-trip (pack -> unpack = original)."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_pack_unpack_roundtrip(self, bits: int) -> None:
        """Pack then unpack recovers original indices exactly."""
        tq = TurboQuantKV(head_dim=64, bits=bits, use_gpu=False, seed=0)
        rng = np.random.default_rng(42)
        n = 1000
        indices = rng.integers(0, 2**bits, size=n, dtype=np.uint8)
        packed = tq._pack_bits(indices)
        unpacked = tq._unpack_bits(packed, n)
        np.testing.assert_array_equal(indices, unpacked)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_pack_unpack_non_aligned(self, bits: int) -> None:
        """Round-trip works when n_values is not a multiple of the group size."""
        tq = TurboQuantKV(head_dim=64, bits=bits, use_gpu=False, seed=0)
        rng = np.random.default_rng(123)
        for n in [1, 7, 13, 31, 100, 257]:
            indices = rng.integers(0, 2**bits, size=n, dtype=np.uint8)
            packed = tq._pack_bits(indices)
            unpacked = tq._unpack_bits(packed, n)
            np.testing.assert_array_equal(
                indices,
                unpacked,
                err_msg=f"Failed for bits={bits}, n={n}",
            )

    @pytest.mark.parametrize(
        "bits,group_size,packed_per_group",
        [
            (2, 4, 1),
            (3, 8, 3),
            (4, 2, 1),
        ],
    )
    def test_pack_size_correct(
        self, bits: int, group_size: int, packed_per_group: int
    ) -> None:
        """Packed array has the expected byte count."""
        tq = TurboQuantKV(head_dim=64, bits=bits, use_gpu=False, seed=0)
        n = group_size * 10  # Exactly aligned
        indices = np.zeros(n, dtype=np.uint8)
        packed = tq._pack_bits(indices)
        expected_bytes = 10 * packed_per_group
        assert len(packed) == expected_bytes

    def test_3bit_known_values(self) -> None:
        """Verify 3-bit packing against manually computed values."""
        tq = TurboQuantKV(head_dim=64, bits=3, use_gpu=False, seed=0)
        indices = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8)
        packed = tq._pack_bits(indices)
        assert len(packed) == 3

        bits24 = (
            0
            | (1 << 3)
            | (2 << 6)
            | (3 << 9)
            | (4 << 12)
            | (5 << 15)
            | (6 << 18)
            | (7 << 21)
        )
        expected = np.array(
            [
                bits24 & 0xFF,
                (bits24 >> 8) & 0xFF,
                (bits24 >> 16) & 0xFF,
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(packed, expected)

        unpacked = tq._unpack_bits(packed, 8)
        np.testing.assert_array_equal(indices, unpacked)


# ------------------------------------------------------------------ #
# Compressed container tests                                          #
# ------------------------------------------------------------------ #


class TestCompressedKV:
    """Tests for the CompressedKV dataclass."""

    def test_indices_dtype(self) -> None:
        """Indices are stored as uint8."""
        tensor = _random_kv(head_dim=64)
        tq = TurboQuantKV(head_dim=64, n_heads=4, bits=3, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        assert compressed.indices.dtype == np.uint8

    def test_norms_shape(self) -> None:
        """Norms shape is (B, H, S) -- one norm per vector."""
        tensor = _random_kv(batch=2, n_heads=4, seq_len=16, head_dim=64)
        tq = TurboQuantKV(head_dim=64, n_heads=4, bits=3, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        assert compressed.norms.shape == (2, 4, 16)

    def test_nbytes_less_than_original(self) -> None:
        """Compressed representation uses less memory."""
        tensor = _random_kv(head_dim=128)
        tq = TurboQuantKV(head_dim=128, n_heads=4, bits=3, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        assert compressed.nbytes() < tensor.nbytes

    def test_compression_ratio(self) -> None:
        """Compression ratio is > 1 (actual savings)."""
        tensor = _random_kv(head_dim=128)
        tq = TurboQuantKV(head_dim=128, n_heads=4, bits=3, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        ratio = compressed.compression_ratio(128)
        assert ratio > 1.0, f"Expected ratio > 1, got {ratio}"

    def test_bits_preserved(self) -> None:
        """The bits field is correctly stored."""
        tensor = _random_kv(head_dim=64)
        tq = TurboQuantKV(head_dim=64, n_heads=4, bits=3, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        assert compressed.bits == 3

    def test_packed_flag(self) -> None:
        """Packed flag is correctly set."""
        tensor = _random_kv(head_dim=64)
        tq = TurboQuantKV(head_dim=64, n_heads=4, bits=3, use_gpu=False, seed=0)
        c_unpacked = tq.compress(tensor, packed=False)
        c_packed = tq.compress(tensor, packed=True)
        assert not c_unpacked.packed
        assert c_packed.packed


# ------------------------------------------------------------------ #
# Edge cases                                                          #
# ------------------------------------------------------------------ #


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_zero_vector(self) -> None:
        """Zero vectors compress and decompress without error."""
        tensor = np.zeros((1, 1, 1, 64), dtype=np.float32)
        tq = TurboQuantKV(head_dim=64, n_heads=1, bits=3, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        reconstructed = tq.decompress(compressed)
        assert np.allclose(reconstructed, 0.0, atol=1e-6)

    def test_zero_vector_packed(self) -> None:
        """Zero vectors work with packed compression."""
        tensor = np.zeros((1, 1, 1, 64), dtype=np.float32)
        tq = TurboQuantKV(head_dim=64, n_heads=1, bits=3, use_gpu=False, seed=0)
        compressed = tq.compress(tensor, packed=True)
        reconstructed = tq.decompress(compressed)
        assert np.allclose(reconstructed, 0.0, atol=1e-6)

    def test_single_element_batch(self) -> None:
        """Works with batch=1, seq_len=1."""
        tensor = _random_kv(batch=1, n_heads=1, seq_len=1, head_dim=64)
        tq = TurboQuantKV(head_dim=64, n_heads=1, bits=3, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        reconstructed = tq.decompress(compressed)
        assert reconstructed.shape == (1, 1, 1, 64)

    def test_large_head_dim(self) -> None:
        """Works with head_dim > 4096 (uses structured rotation)."""
        head_dim = 5000
        tensor = _random_kv(
            batch=1, n_heads=1, seq_len=2, head_dim=head_dim, seed=99
        )
        tq = TurboQuantKV(
            head_dim=head_dim, n_heads=1, bits=3, use_gpu=False, seed=0
        )
        compressed = tq.compress(tensor)
        reconstructed = tq.decompress(compressed)
        assert reconstructed.shape == tensor.shape
        cos_sim = _cosine_similarity(tensor, reconstructed)
        assert cos_sim > 0.90

    def test_fp16_input(self) -> None:
        """Accepts float16 input tensors."""
        tensor = _random_kv(head_dim=64, dtype="float16")
        tq = TurboQuantKV(head_dim=64, n_heads=4, bits=3, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        assert compressed.original_dtype == np.dtype("float16")
        reconstructed = tq.decompress(compressed)
        assert reconstructed.shape == tensor.shape

    def test_invalid_bits_raises(self) -> None:
        """Unsupported bit width raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported bits=5"):
            TurboQuantKV(head_dim=64, bits=5, use_gpu=False)

    def test_reproducibility_with_seed(self) -> None:
        """Same seed produces identical compressed output."""
        tensor = _random_kv(head_dim=64, seed=42)
        tq1 = TurboQuantKV(head_dim=64, n_heads=4, bits=3, use_gpu=False, seed=0)
        tq2 = TurboQuantKV(head_dim=64, n_heads=4, bits=3, use_gpu=False, seed=0)
        c1 = tq1.compress(tensor)
        c2 = tq2.compress(tensor)
        np.testing.assert_array_equal(c1.indices, c2.indices)
        np.testing.assert_array_equal(c1.norms, c2.norms)


# ------------------------------------------------------------------ #
# Memory estimation                                                   #
# ------------------------------------------------------------------ #


class TestMemoryEstimation:
    """Tests for the static memory estimation utility."""

    def test_gemma4_8k(self) -> None:
        """Gemma 4 27B at 8K context: compressed < original."""
        est = TurboQuantKV.estimate_memory(
            n_layers=36,
            n_kv_heads=16,
            head_dim=256,
            seq_len=8192,
            bits=3,
            original_dtype="float16",
        )
        assert est["compressed_gb"] < est["original_gb"]
        assert est["ratio"] > 1.0
        assert est["saved_gb"] > 0

    def test_compression_ratio_positive(self) -> None:
        """Ratio is always > 1 for reasonable configs."""
        est = TurboQuantKV.estimate_memory(
            n_layers=32,
            n_kv_heads=8,
            head_dim=128,
            seq_len=4096,
            bits=3,
        )
        assert est["ratio"] > 1.0

    def test_4bit_less_compression(self) -> None:
        """4-bit gives same uint8 storage as 3-bit (difference is quality)."""
        est3 = TurboQuantKV.estimate_memory(
            n_layers=32, n_kv_heads=8, head_dim=128, seq_len=4096, bits=3
        )
        est4 = TurboQuantKV.estimate_memory(
            n_layers=32, n_kv_heads=8, head_dim=128, seq_len=4096, bits=4
        )
        assert est3["compressed_gb"] == est4["compressed_gb"]

    def test_bit_packed_better_than_uint8(self) -> None:
        """Bit-packed estimate has better ratio than uint8."""
        est_uint8 = TurboQuantKV.estimate_memory(
            n_layers=36,
            n_kv_heads=16,
            head_dim=256,
            seq_len=8192,
            bits=3,
            bit_packed=False,
        )
        est_packed = TurboQuantKV.estimate_memory(
            n_layers=36,
            n_kv_heads=16,
            head_dim=256,
            seq_len=8192,
            bits=3,
            bit_packed=True,
        )
        assert est_packed["ratio"] > est_uint8["ratio"]
        assert est_packed["compressed_gb"] < est_uint8["compressed_gb"]


# ------------------------------------------------------------------ #
# Theoretical compression ratio                                       #
# ------------------------------------------------------------------ #


class TestTheoreticalRatio:
    """Test the convenience compression_ratio method."""

    def test_ratio_greater_than_one(self) -> None:
        tq = TurboQuantKV(head_dim=128, bits=3, use_gpu=False, seed=0)
        assert tq.compression_ratio() > 1.0

    def test_ratio_increases_with_head_dim(self) -> None:
        """Larger head_dim means less norm overhead per element."""
        tq_small = TurboQuantKV(head_dim=64, bits=3, use_gpu=False, seed=0)
        tq_large = TurboQuantKV(head_dim=256, bits=3, use_gpu=False, seed=0)
        assert tq_large.compression_ratio() > tq_small.compression_ratio()

    def test_packed_ratio_higher_than_unpacked(self) -> None:
        """Packed ratio is significantly higher than unpacked."""
        tq = TurboQuantKV(head_dim=256, bits=3, use_gpu=False, seed=0)
        unpacked_ratio = tq.compression_ratio(packed=False)
        packed_ratio = tq.compression_ratio(packed=True)
        assert packed_ratio > unpacked_ratio
        assert packed_ratio > 4.5

    def test_packed_ratio_matches_theoretical(self) -> None:
        """Packed ratio should match theoretical_compression_ratio."""
        tq = TurboQuantKV(head_dim=256, bits=3, use_gpu=False, seed=0)
        packed_ratio = tq.compression_ratio(packed=True)
        theoretical = tq.theoretical_compression_ratio()
        assert abs(packed_ratio - theoretical) < 0.01

"""
Unit tests for TurboQuant NATS codec.

Tests the TurboQuantNATSCodec for compressed embedding transport:
- Single encode/decode round-trip
- Batch encode/decode round-trip
- Size reduction vs float32
- Wire format validation
- Error handling for invalid payloads
- Cosine similarity preservation

Usage:
    pytest tests/test_nats_codec.py -v
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from turboquant_kv.nats_codec import TurboQuantNATSCodec

# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-30 or norm_b < 1e-30:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _random_embeddings(
    n: int = 10,
    dim: int = 128,
    seed: int = 42,
) -> np.ndarray:
    """Generate random embeddings."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


# ------------------------------------------------------------------ #
# Single encode/decode tests                                          #
# ------------------------------------------------------------------ #


class TestSingleEncodeDecode:
    """Encode and decode single embeddings."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_round_trip_shape(self, bits: int) -> None:
        """Decoded embedding has the same shape as original."""
        codec = TurboQuantNATSCodec(dim=128, bits=bits, seed=42)
        emb = _random_embeddings(1, 128, seed=0)[0]
        payload = codec.encode(emb)
        decoded = codec.decode(payload)
        assert decoded.shape == emb.shape

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_round_trip_dtype(self, bits: int) -> None:
        """Decoded embedding is float32."""
        codec = TurboQuantNATSCodec(dim=128, bits=bits, seed=42)
        emb = _random_embeddings(1, 128, seed=0)[0]
        payload = codec.encode(emb)
        decoded = codec.decode(payload)
        assert decoded.dtype == np.float32

    def test_3bit_cosine_similarity(self) -> None:
        """3-bit encode/decode preserves cosine similarity > 0.95."""
        codec = TurboQuantNATSCodec(dim=256, bits=3, seed=42)
        emb = _random_embeddings(1, 256, seed=123)[0]
        payload = codec.encode(emb)
        decoded = codec.decode(payload)
        sim = _cosine_similarity(emb, decoded)
        assert sim > 0.95, f"Cosine similarity {sim:.4f} too low"

    def test_4bit_cosine_similarity(self) -> None:
        """4-bit encode/decode preserves cosine similarity > 0.98."""
        codec = TurboQuantNATSCodec(dim=256, bits=4, seed=42)
        emb = _random_embeddings(1, 256, seed=456)[0]
        payload = codec.encode(emb)
        decoded = codec.decode(payload)
        sim = _cosine_similarity(emb, decoded)
        assert sim > 0.98, f"Cosine similarity {sim:.4f} too low"


# ------------------------------------------------------------------ #
# Size reduction tests                                                #
# ------------------------------------------------------------------ #


class TestSizeReduction:
    """Verify payload sizes and compression ratios."""

    def test_payload_smaller_than_float32(self) -> None:
        """Encoded payload is much smaller than float32 embedding."""
        codec = TurboQuantNATSCodec(dim=1024, bits=3, seed=42)
        emb = _random_embeddings(1, 1024, seed=0)[0]
        payload = codec.encode(emb)
        float32_size = 1024 * 4  # 4096 bytes
        assert len(payload) < float32_size / 5

    def test_payload_size_matches_expected(self) -> None:
        """Payload size matches the codec's reported payload_size."""
        codec = TurboQuantNATSCodec(dim=1024, bits=3, seed=42)
        emb = _random_embeddings(1, 1024, seed=0)[0]
        payload = codec.encode(emb)
        assert len(payload) == codec.payload_size()

    def test_compression_ratio_1024_3bit(self) -> None:
        """1024-dim 3-bit achieves > 5x compression ratio."""
        codec = TurboQuantNATSCodec(dim=1024, bits=3, seed=42)
        ratio = codec.compression_ratio()
        assert ratio > 5.0, f"Expected ratio > 5.0, got {ratio:.2f}"

    def test_stats_keys(self) -> None:
        """Stats returns expected keys."""
        codec = TurboQuantNATSCodec(dim=1024, bits=3, seed=42)
        stats = codec.stats()
        expected_keys = {
            "dim",
            "bits",
            "payload_bytes",
            "float32_bytes",
            "compression_ratio",
            "header_overhead_bytes",
        }
        assert set(stats.keys()) == expected_keys


# ------------------------------------------------------------------ #
# Wire format tests                                                   #
# ------------------------------------------------------------------ #


class TestWireFormat:
    """Validate the wire format structure."""

    def test_header_present(self) -> None:
        """Payload starts with the 8-byte header."""
        codec = TurboQuantNATSCodec(dim=128, bits=3, seed=42)
        emb = _random_embeddings(1, 128, seed=0)[0]
        payload = codec.encode(emb)
        assert len(payload) >= 8

    def test_header_version(self) -> None:
        """First byte is wire version 0x01."""
        codec = TurboQuantNATSCodec(dim=128, bits=3, seed=42)
        emb = _random_embeddings(1, 128, seed=0)[0]
        payload = codec.encode(emb)
        assert payload[0] == 0x01

    def test_header_bits(self) -> None:
        """Second byte is the bits parameter."""
        codec = TurboQuantNATSCodec(dim=128, bits=3, seed=42)
        emb = _random_embeddings(1, 128, seed=0)[0]
        payload = codec.encode(emb)
        assert payload[1] == 3

    def test_header_dim(self) -> None:
        """Bytes 2-3 are the dim as uint16 LE."""
        codec = TurboQuantNATSCodec(dim=1024, bits=3, seed=42)
        emb = _random_embeddings(1, 1024, seed=0)[0]
        payload = codec.encode(emb)
        dim = struct.unpack("<H", payload[2:4])[0]
        assert dim == 1024

    def test_header_norm(self) -> None:
        """Bytes 4-7 are the norm as float32 LE."""
        codec = TurboQuantNATSCodec(dim=128, bits=3, seed=42)
        emb = _random_embeddings(1, 128, seed=0)[0]
        payload = codec.encode(emb)
        norm = struct.unpack("<f", payload[4:8])[0]
        expected_norm = float(np.linalg.norm(emb))
        assert abs(norm - expected_norm) < 1e-5


# ------------------------------------------------------------------ #
# Batch tests                                                         #
# ------------------------------------------------------------------ #


class TestBatchOperations:
    """Batch encode/decode operations."""

    def test_batch_count(self) -> None:
        """Batch encode returns correct number of payloads."""
        codec = TurboQuantNATSCodec(dim=128, bits=3, seed=42)
        embs = _random_embeddings(10, 128, seed=0)
        payloads = codec.encode_batch(embs)
        assert len(payloads) == 10

    def test_batch_decode_shape(self) -> None:
        """Batch decode returns correct shape."""
        codec = TurboQuantNATSCodec(dim=128, bits=3, seed=42)
        embs = _random_embeddings(10, 128, seed=0)
        payloads = codec.encode_batch(embs)
        decoded = codec.decode_batch(payloads)
        assert decoded.shape == (10, 128)

    def test_batch_matches_single(self) -> None:
        """Batch encode gives same payload as single encode."""
        codec = TurboQuantNATSCodec(dim=128, bits=3, seed=42)
        embs = _random_embeddings(5, 128, seed=0)
        batch_payloads = codec.encode_batch(embs)
        for i in range(5):
            single_payload = codec.encode(embs[i])
            assert batch_payloads[i] == single_payload

    def test_batch_quality(self) -> None:
        """Batch round-trip preserves cosine similarity."""
        codec = TurboQuantNATSCodec(dim=256, bits=3, seed=42)
        embs = _random_embeddings(20, 256, seed=0)
        payloads = codec.encode_batch(embs)
        decoded = codec.decode_batch(payloads)
        for i in range(20):
            sim = _cosine_similarity(embs[i], decoded[i])
            assert sim > 0.93, f"Embedding {i}: cos_sim {sim:.4f} too low"


# ------------------------------------------------------------------ #
# Error handling                                                      #
# ------------------------------------------------------------------ #


class TestErrorHandling:
    """Error handling for invalid inputs."""

    def test_payload_too_short(self) -> None:
        """Short payload raises ValueError."""
        codec = TurboQuantNATSCodec(dim=128, bits=3, seed=42)
        with pytest.raises(ValueError, match="Payload too short"):
            codec.decode(b"\x01\x03")

    def test_wrong_version(self) -> None:
        """Wrong wire version raises ValueError."""
        codec = TurboQuantNATSCodec(dim=128, bits=3, seed=42)
        # Create a payload with wrong version byte
        bad_payload = b"\x99" + b"\x03" + struct.pack("<H", 128) + b"\x00" * 100
        with pytest.raises(ValueError, match="Unsupported wire version"):
            codec.decode(bad_payload)

    def test_parameter_mismatch(self) -> None:
        """Mismatched parameters raise ValueError."""
        codec = TurboQuantNATSCodec(dim=128, bits=3, seed=42)
        # Create header with dim=256
        header = struct.pack("<BBHf", 0x01, 3, 256, 1.0)
        payload = header + b"\x00" * 100
        with pytest.raises(ValueError, match="Parameter mismatch"):
            codec.decode(payload)

    def test_bits_mismatch(self) -> None:
        """Mismatched bits raise ValueError."""
        codec = TurboQuantNATSCodec(dim=128, bits=3, seed=42)
        # Create header with bits=4
        header = struct.pack("<BBHf", 0x01, 4, 128, 1.0)
        payload = header + b"\x00" * 100
        with pytest.raises(ValueError, match="Parameter mismatch"):
            codec.decode(payload)

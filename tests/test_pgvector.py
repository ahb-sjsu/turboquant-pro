"""
Unit tests for TurboQuant pgvector embedding compression.

Tests the TurboQuantPGVector compressor on CPU (NumPy) to validate:
- Single embedding compress/decompress round-trip
- Batch compress/decompress round-trip
- Cosine similarity preservation
- Compression ratio calculations
- PostgreSQL bytea serialization round-trip
- Storage estimation
- Edge cases (zero vector, large dim, dimension mismatch)
- Compressed cosine similarity search

Usage:
    pytest tests/test_pgvector.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro.pgvector import (
    CompressedEmbedding,
    TurboQuantPGVector,
)

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
# Single embedding tests                                              #
# ------------------------------------------------------------------ #


class TestSingleEmbedding:
    """Compress and decompress single embeddings."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_round_trip_shape(self, bits: int) -> None:
        """Decompressed embedding has the same shape as original."""
        tq = TurboQuantPGVector(dim=128, bits=bits, seed=42)
        emb = _random_embeddings(1, 128, seed=0)[0]
        comp = tq.compress_embedding(emb)
        recon = tq.decompress_embedding(comp)
        assert recon.shape == emb.shape

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_round_trip_dtype(self, bits: int) -> None:
        """Decompressed embedding is float32."""
        tq = TurboQuantPGVector(dim=128, bits=bits, seed=42)
        emb = _random_embeddings(1, 128, seed=0)[0]
        comp = tq.compress_embedding(emb)
        recon = tq.decompress_embedding(comp)
        assert recon.dtype == np.float32

    def test_3bit_cosine_similarity(self) -> None:
        """3-bit compression preserves cosine similarity > 0.95."""
        tq = TurboQuantPGVector(dim=256, bits=3, seed=42)
        emb = _random_embeddings(1, 256, seed=123)[0]
        comp = tq.compress_embedding(emb)
        recon = tq.decompress_embedding(comp)
        sim = _cosine_similarity(emb, recon)
        assert sim > 0.95, f"Cosine similarity {sim:.4f} too low"

    def test_4bit_cosine_similarity(self) -> None:
        """4-bit compression preserves cosine similarity > 0.98."""
        tq = TurboQuantPGVector(dim=256, bits=4, seed=42)
        emb = _random_embeddings(1, 256, seed=456)[0]
        comp = tq.compress_embedding(emb)
        recon = tq.decompress_embedding(comp)
        sim = _cosine_similarity(emb, recon)
        assert sim > 0.98, f"Cosine similarity {sim:.4f} too low"

    def test_norm_preserved(self) -> None:
        """Compressed embedding stores correct norm."""
        tq = TurboQuantPGVector(dim=128, bits=3, seed=42)
        emb = _random_embeddings(1, 128, seed=0)[0]
        comp = tq.compress_embedding(emb)
        expected_norm = float(np.linalg.norm(emb))
        assert abs(comp.norm - expected_norm) < 1e-5

    def test_compression_ratio_positive(self) -> None:
        """Compression ratio is > 1."""
        tq = TurboQuantPGVector(dim=1024, bits=3, seed=42)
        emb = _random_embeddings(1, 1024, seed=0)[0]
        comp = tq.compress_embedding(emb)
        assert comp.compression_ratio > 1.0

    def test_3bit_1024dim_ratio(self) -> None:
        """1024-dim at 3-bit should achieve > 8x compression vs float32."""
        tq = TurboQuantPGVector(dim=1024, bits=3, seed=42)
        emb = _random_embeddings(1, 1024, seed=0)[0]
        comp = tq.compress_embedding(emb)
        # float32: 4096 bytes, 3-bit packed: 384 bytes + 4 norm = 388
        assert (
            comp.compression_ratio > 8.0
        ), f"Expected ratio > 8.0, got {comp.compression_ratio:.2f}"


# ------------------------------------------------------------------ #
# Batch tests                                                         #
# ------------------------------------------------------------------ #


class TestBatchOperations:
    """Compress and decompress batches of embeddings."""

    def test_batch_count(self) -> None:
        """Batch compress returns correct number of results."""
        tq = TurboQuantPGVector(dim=128, bits=3, seed=42)
        embs = _random_embeddings(10, 128, seed=0)
        results = tq.compress_batch(embs)
        assert len(results) == 10

    def test_batch_matches_single(self) -> None:
        """Batch compress gives same result as single compress."""
        tq = TurboQuantPGVector(dim=128, bits=3, seed=42)
        embs = _random_embeddings(5, 128, seed=0)

        batch_results = tq.compress_batch(embs)
        for i in range(5):
            single_result = tq.compress_embedding(embs[i])
            assert batch_results[i].packed_bytes == single_result.packed_bytes
            assert abs(batch_results[i].norm - single_result.norm) < 1e-5

    def test_batch_decompress_shape(self) -> None:
        """Batch decompress returns correct shape."""
        tq = TurboQuantPGVector(dim=128, bits=3, seed=42)
        embs = _random_embeddings(10, 128, seed=0)
        compressed = tq.compress_batch(embs)
        decompressed = tq.decompress_batch(compressed)
        assert decompressed.shape == (10, 128)

    def test_batch_quality(self) -> None:
        """Batch round-trip preserves cosine similarity."""
        tq = TurboQuantPGVector(dim=256, bits=3, seed=42)
        embs = _random_embeddings(20, 256, seed=0)
        compressed = tq.compress_batch(embs)
        decompressed = tq.decompress_batch(compressed)

        for i in range(20):
            sim = _cosine_similarity(embs[i], decompressed[i])
            assert sim > 0.93, f"Embedding {i}: cos_sim {sim:.4f} too low"


# ------------------------------------------------------------------ #
# PostgreSQL bytea serialization                                      #
# ------------------------------------------------------------------ #


class TestByteaSerialization:
    """Test serialize/deserialize for PostgreSQL bytea storage."""

    def test_bytea_round_trip(self) -> None:
        """Serialize to bytea and back preserves data."""
        tq = TurboQuantPGVector(dim=128, bits=3, seed=42)
        emb = _random_embeddings(1, 128, seed=0)[0]
        comp = tq.compress_embedding(emb)

        bytea = comp.to_pgbytea()
        restored = CompressedEmbedding.from_pgbytea(bytea, 128, 3)

        assert restored.packed_bytes == comp.packed_bytes
        assert abs(restored.norm - comp.norm) < 1e-6

    def test_bytea_size(self) -> None:
        """Bytea payload is much smaller than float32."""
        tq = TurboQuantPGVector(dim=1024, bits=3, seed=42)
        emb = _random_embeddings(1, 1024, seed=0)[0]
        comp = tq.compress_embedding(emb)

        bytea = comp.to_pgbytea()
        float32_size = 1024 * 4  # 4096 bytes

        # bytea should be < 500 bytes (384 packed + 4 norm = 388)
        assert len(bytea) < 500
        assert len(bytea) < float32_size / 5

    def test_bytea_decompress_matches(self) -> None:
        """Decompressing from bytea gives same result as original."""
        tq = TurboQuantPGVector(dim=256, bits=3, seed=42)
        emb = _random_embeddings(1, 256, seed=0)[0]
        comp = tq.compress_embedding(emb)
        recon_direct = tq.decompress_embedding(comp)

        bytea = comp.to_pgbytea()
        restored = CompressedEmbedding.from_pgbytea(bytea, 256, 3)
        recon_from_bytea = tq.decompress_embedding(restored)

        np.testing.assert_array_equal(recon_direct, recon_from_bytea)


# ------------------------------------------------------------------ #
# Compressed search                                                   #
# ------------------------------------------------------------------ #


class TestCompressedSearch:
    """Test cosine similarity search on compressed embeddings."""

    def test_self_similarity_highest(self) -> None:
        """A vector's own compressed representation is most similar."""
        tq = TurboQuantPGVector(dim=256, bits=3, seed=42)
        embs = _random_embeddings(20, 256, seed=0)
        compressed = tq.compress_batch(embs)

        # Query with the first embedding
        scores = tq.compressed_cosine_similarity(embs[0], compressed)
        assert np.argmax(scores) == 0

    def test_search_ranking_preserved(self) -> None:
        """Compressed search preserves approximate ranking."""
        tq = TurboQuantPGVector(dim=256, bits=3, seed=42)
        rng = np.random.default_rng(42)

        # Create a query and some embeddings with known similarity
        query = rng.standard_normal(256).astype(np.float32)
        # Make a near-duplicate of the query
        near_dup = query + rng.standard_normal(256).astype(np.float32) * 0.1
        # And some random ones
        random_embs = rng.standard_normal((18, 256)).astype(np.float32)

        all_embs = np.vstack([near_dup[np.newaxis], random_embs])
        compressed = tq.compress_batch(all_embs)

        scores = tq.compressed_cosine_similarity(query, compressed)
        # The near-duplicate should be in top-3
        top_3 = np.argsort(-scores)[:3]
        assert 0 in top_3, "Near-duplicate not in top-3 results"


# ------------------------------------------------------------------ #
# Storage estimation                                                  #
# ------------------------------------------------------------------ #


class TestStorageEstimation:
    """Test static storage estimation utility."""

    def test_rag_chunks(self) -> None:
        """112K RAG chunks at 1024-dim, 3-bit."""
        est = TurboQuantPGVector.estimate_storage(
            n_embeddings=112_000, dim=1024, bits=3
        )
        assert est["compressed_mb"] < est["original_mb"]
        assert est["ratio"] > 5.0
        assert est["original_mb"] > 400  # ~437 MB

    def test_ethics_chunks(self) -> None:
        """2.4M ethics chunks at 1024-dim, 3-bit."""
        est = TurboQuantPGVector.estimate_storage(
            n_embeddings=2_400_000, dim=1024, bits=3
        )
        assert est["compressed_mb"] < est["original_mb"]
        assert est["ratio"] > 5.0
        assert est["original_mb"] > 9000  # ~9375 MB

    def test_ratio_increases_with_dim(self) -> None:
        """Higher dimensions give better compression ratio."""
        est_256 = TurboQuantPGVector.estimate_storage(
            n_embeddings=1000, dim=256, bits=3
        )
        est_1024 = TurboQuantPGVector.estimate_storage(
            n_embeddings=1000, dim=1024, bits=3
        )
        assert est_1024["ratio"] > est_256["ratio"]


# ------------------------------------------------------------------ #
# Edge cases                                                          #
# ------------------------------------------------------------------ #


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_zero_vector(self) -> None:
        """Zero vector compresses without error."""
        tq = TurboQuantPGVector(dim=128, bits=3, seed=42)
        emb = np.zeros(128, dtype=np.float32)
        comp = tq.compress_embedding(emb)
        recon = tq.decompress_embedding(comp)
        assert np.allclose(recon, 0.0, atol=1e-6)

    def test_wrong_dim_raises(self) -> None:
        """Wrong embedding dimension raises ValueError."""
        tq = TurboQuantPGVector(dim=128, bits=3, seed=42)
        emb = np.ones(256, dtype=np.float32)
        with pytest.raises(ValueError, match="Expected dim=128"):
            tq.compress_embedding(emb)

    def test_invalid_bits_raises(self) -> None:
        """Unsupported bit width raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported bits=5"):
            TurboQuantPGVector(dim=128, bits=5, seed=42)

    def test_reproducibility(self) -> None:
        """Same seed produces identical results."""
        tq1 = TurboQuantPGVector(dim=128, bits=3, seed=42)
        tq2 = TurboQuantPGVector(dim=128, bits=3, seed=42)
        emb = _random_embeddings(1, 128, seed=0)[0]
        c1 = tq1.compress_embedding(emb)
        c2 = tq2.compress_embedding(emb)
        assert c1.packed_bytes == c2.packed_bytes
        assert c1.norm == c2.norm

    def test_different_seeds_different_results(self) -> None:
        """Different seeds produce different compressions."""
        tq1 = TurboQuantPGVector(dim=128, bits=3, seed=42)
        tq2 = TurboQuantPGVector(dim=128, bits=3, seed=99)
        emb = _random_embeddings(1, 128, seed=0)[0]
        c1 = tq1.compress_embedding(emb)
        c2 = tq2.compress_embedding(emb)
        assert c1.packed_bytes != c2.packed_bytes

    def test_1d_input_to_batch(self) -> None:
        """1D input to compress_batch is handled correctly."""
        tq = TurboQuantPGVector(dim=128, bits=3, seed=42)
        emb = _random_embeddings(1, 128, seed=0)[0]
        results = tq.compress_batch(emb)
        assert len(results) == 1

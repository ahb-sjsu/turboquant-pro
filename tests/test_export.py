"""
Unit tests for cross-framework vector database export.

Tests the VectorDBExporter adapters for Milvus, Qdrant, Weaviate,
Pinecone, and the generic portable format.

Usage:
    pytest tests/test_export.py -v
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from turboquant_pro.export import (
    GenericExporter,
    MilvusExporter,
    PineconeExporter,
    QdrantExporter,
    WeaviateExporter,
    export_compressed,
    supported_formats,
)
from turboquant_pro.pgvector import CompressedEmbedding, TurboQuantPGVector

# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

DIM = 128
BITS = 3
SEED = 42
N = 5


@pytest.fixture()
def tq() -> TurboQuantPGVector:
    return TurboQuantPGVector(dim=DIM, bits=BITS, seed=SEED)


@pytest.fixture()
def embeddings() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((N, DIM)).astype(np.float32)


@pytest.fixture()
def ids() -> list[int]:
    return list(range(N))


# ------------------------------------------------------------------ #
# GenericExporter                                                     #
# ------------------------------------------------------------------ #


class TestGenericExporter:
    """Tests for the portable generic export format."""

    def test_export_batch_length(
        self, ids: list[int], embeddings: np.ndarray, tq: TurboQuantPGVector
    ) -> None:
        """Correct number of results."""
        exporter = GenericExporter()
        records = exporter.export_batch(ids, embeddings, tq)
        assert len(records) == N

    def test_export_fields_present(
        self, ids: list[int], embeddings: np.ndarray, tq: TurboQuantPGVector
    ) -> None:
        """Each record has id, vector_f32, compressed_b64, norm, dim, bits."""
        exporter = GenericExporter()
        records = exporter.export_batch(ids, embeddings, tq)
        for rec in records:
            assert "id" in rec
            assert "vector_f32" in rec
            assert "compressed_b64" in rec
            assert "norm" in rec
            assert "dim" in rec
            assert "bits" in rec

    def test_roundtrip_via_base64(
        self, ids: list[int], embeddings: np.ndarray, tq: TurboQuantPGVector
    ) -> None:
        """Decode base64, from_pgbytea, decompress matches export vector."""
        exporter = GenericExporter()
        records = exporter.export_batch(ids, embeddings, tq)

        for rec in records:
            bytea = base64.b64decode(rec["compressed_b64"])
            comp = CompressedEmbedding.from_pgbytea(bytea, rec["dim"], rec["bits"])
            decompressed = tq.decompress_embedding(comp)
            np.testing.assert_allclose(
                decompressed, np.array(rec["vector_f32"], dtype=np.float32), atol=1e-6
            )


# ------------------------------------------------------------------ #
# MilvusExporter                                                      #
# ------------------------------------------------------------------ #


class TestMilvusExporter:
    """Tests for the Milvus export adapter."""

    def test_format_name(self) -> None:
        assert MilvusExporter().format_name() == "milvus"

    def test_export_fields(
        self, ids: list[int], embeddings: np.ndarray, tq: TurboQuantPGVector
    ) -> None:
        """id, vector, compressed_bytes present."""
        exporter = MilvusExporter()
        records = exporter.export_batch(ids, embeddings, tq)
        for rec in records:
            assert "id" in rec
            assert "vector" in rec
            assert "compressed_bytes" in rec

    def test_vector_is_list(
        self, ids: list[int], embeddings: np.ndarray, tq: TurboQuantPGVector
    ) -> None:
        """vector field is a Python list of floats."""
        exporter = MilvusExporter()
        records = exporter.export_batch(ids, embeddings, tq)
        for rec in records:
            assert isinstance(rec["vector"], list)
            assert all(isinstance(v, float) for v in rec["vector"])


# ------------------------------------------------------------------ #
# QdrantExporter                                                      #
# ------------------------------------------------------------------ #


class TestQdrantExporter:
    """Tests for the Qdrant export adapter."""

    def test_format_name(self) -> None:
        assert QdrantExporter().format_name() == "qdrant"

    def test_named_vector(
        self, ids: list[int], embeddings: np.ndarray, tq: TurboQuantPGVector
    ) -> None:
        """vector["default"] is a list."""
        exporter = QdrantExporter()
        records = exporter.export_batch(ids, embeddings, tq)
        for rec in records:
            assert isinstance(rec["vector"], dict)
            assert "default" in rec["vector"]
            assert isinstance(rec["vector"]["default"], list)

    def test_payload_metadata(
        self, ids: list[int], embeddings: np.ndarray, tq: TurboQuantPGVector
    ) -> None:
        """payload has tq_compressed, tq_bits, tq_dim."""
        exporter = QdrantExporter()
        records = exporter.export_batch(ids, embeddings, tq)
        for rec in records:
            payload = rec["payload"]
            assert "tq_compressed" in payload
            assert "tq_bits" in payload
            assert "tq_dim" in payload
            assert payload["tq_bits"] == BITS
            assert payload["tq_dim"] == DIM


# ------------------------------------------------------------------ #
# WeaviateExporter                                                    #
# ------------------------------------------------------------------ #


class TestWeaviateExporter:
    """Tests for the Weaviate export adapter."""

    def test_format_name(self) -> None:
        assert WeaviateExporter().format_name() == "weaviate"

    def test_string_id(
        self, ids: list[int], embeddings: np.ndarray, tq: TurboQuantPGVector
    ) -> None:
        """id is a string."""
        exporter = WeaviateExporter()
        records = exporter.export_batch(ids, embeddings, tq)
        for rec in records:
            assert isinstance(rec["id"], str)

    def test_properties(
        self, ids: list[int], embeddings: np.ndarray, tq: TurboQuantPGVector
    ) -> None:
        """properties has tq_compressed, tq_bits."""
        exporter = WeaviateExporter()
        records = exporter.export_batch(ids, embeddings, tq)
        for rec in records:
            props = rec["properties"]
            assert "tq_compressed" in props
            assert "tq_bits" in props
            assert props["tq_bits"] == BITS


# ------------------------------------------------------------------ #
# PineconeExporter                                                    #
# ------------------------------------------------------------------ #


class TestPineconeExporter:
    """Tests for the Pinecone export adapter."""

    def test_format_name(self) -> None:
        assert PineconeExporter().format_name() == "pinecone"

    def test_string_id(
        self, ids: list[int], embeddings: np.ndarray, tq: TurboQuantPGVector
    ) -> None:
        """id is a string."""
        exporter = PineconeExporter()
        records = exporter.export_batch(ids, embeddings, tq)
        for rec in records:
            assert isinstance(rec["id"], str)

    def test_metadata(
        self, ids: list[int], embeddings: np.ndarray, tq: TurboQuantPGVector
    ) -> None:
        """metadata has tq_compressed, tq_bits, tq_dim."""
        exporter = PineconeExporter()
        records = exporter.export_batch(ids, embeddings, tq)
        for rec in records:
            meta = rec["metadata"]
            assert "tq_compressed" in meta
            assert "tq_bits" in meta
            assert "tq_dim" in meta
            assert meta["tq_bits"] == BITS
            assert meta["tq_dim"] == DIM


# ------------------------------------------------------------------ #
# Convenience functions                                               #
# ------------------------------------------------------------------ #


class TestConvenienceFunction:
    """Tests for the module-level helper functions."""

    def test_supported_formats(self) -> None:
        """Returns all 5 format names."""
        formats = supported_formats()
        assert len(formats) == 5
        for name in ["generic", "milvus", "pinecone", "qdrant", "weaviate"]:
            assert name in formats

    def test_export_compressed_generic(
        self, ids: list[int], embeddings: np.ndarray, tq: TurboQuantPGVector
    ) -> None:
        """Works with format='generic'."""
        records = export_compressed(ids, embeddings, tq, format="generic")
        assert len(records) == N
        assert "vector_f32" in records[0]

    def test_export_compressed_all_formats(
        self, ids: list[int], embeddings: np.ndarray, tq: TurboQuantPGVector
    ) -> None:
        """Works with each supported format."""
        for fmt in supported_formats():
            records = export_compressed(ids, embeddings, tq, format=fmt)
            assert len(records) == N, f"format={fmt!r} returned wrong count"

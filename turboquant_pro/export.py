# TurboQuant Pro: Open-source PolarQuant+QJL for LLM KV cache compression
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""
Cross-framework vector database export for TurboQuant-compressed embeddings.

Provides adapters for exporting TurboQuant-compressed embeddings to popular
vector databases.  Each adapter produces dicts (or dict-like structures)
that are ready for the target DB's upsert / insert API.

Every exported record carries **both** the decompressed float32 vector (so
the DB can use it for native ANN indexing) **and** the compressed bytes as
a base64 string (for storage efficiency, bulk transfer, or cold storage).

Supported targets:

  - **Milvus**    — float list vectors, binary data as base64 strings
  - **Qdrant**    — named vectors with arbitrary payloads
  - **Weaviate**  — string UUIDs, properties dict
  - **Pinecone**  — string IDs with metadata
  - **Generic**   — portable JSON-serializable format

Usage::

    from turboquant_pro.export import export_compressed

    records = export_compressed(ids, embeddings, tq, format="qdrant")
    # -> list of dicts ready for qdrant_client.upsert()
"""

from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .pgvector import TurboQuantPGVector

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Abstract base                                                       #
# ------------------------------------------------------------------ #


class VectorDBExporter(ABC):
    """Abstract base for vector DB export adapters."""

    @abstractmethod
    def export_batch(
        self,
        ids: list,
        embeddings: np.ndarray,
        tq: TurboQuantPGVector,
    ) -> list[dict]:
        """Convert embeddings to the target DB's format.

        Args:
            ids: List of vector IDs.
            embeddings: (n, dim) float32 array of original embeddings.
            tq: TurboQuantPGVector instance for compression parameters.

        Returns:
            List of dicts ready for the target DB's upsert API.
        """

    @abstractmethod
    def format_name(self) -> str:
        """Return the target format name."""


# ------------------------------------------------------------------ #
# Concrete adapters                                                   #
# ------------------------------------------------------------------ #


class MilvusExporter(VectorDBExporter):
    """Export as Milvus-compatible dicts.

    Milvus uses float lists for vectors and binary data as base64 strings.
    """

    def format_name(self) -> str:
        return "milvus"

    def export_batch(
        self,
        ids: list,
        embeddings: np.ndarray,
        tq: TurboQuantPGVector,
    ) -> list[dict]:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        compressed_list = tq.compress_batch(embeddings)

        records: list[dict] = []
        for i, comp in enumerate(compressed_list):
            decompressed = tq.decompress_embedding(comp)
            bytea = comp.to_pgbytea()
            records.append(
                {
                    "id": ids[i],
                    "vector": decompressed.tolist(),
                    "compressed_bytes": base64.b64encode(bytea).decode("ascii"),
                }
            )
        return records


class QdrantExporter(VectorDBExporter):
    """Export as Qdrant-compatible point dicts.

    Qdrant uses named vectors with arbitrary payloads.
    """

    def format_name(self) -> str:
        return "qdrant"

    def export_batch(
        self,
        ids: list,
        embeddings: np.ndarray,
        tq: TurboQuantPGVector,
    ) -> list[dict]:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        compressed_list = tq.compress_batch(embeddings)

        records: list[dict] = []
        for i, comp in enumerate(compressed_list):
            decompressed = tq.decompress_embedding(comp)
            bytea = comp.to_pgbytea()
            records.append(
                {
                    "id": ids[i],
                    "vector": {
                        "default": decompressed.tolist(),
                    },
                    "payload": {
                        "tq_compressed": base64.b64encode(bytea).decode("ascii"),
                        "tq_bits": comp.bits,
                        "tq_dim": comp.dim,
                    },
                }
            )
        return records


class WeaviateExporter(VectorDBExporter):
    """Export as Weaviate-compatible objects.

    Weaviate uses string UUIDs and a properties dict.
    """

    def format_name(self) -> str:
        return "weaviate"

    def export_batch(
        self,
        ids: list,
        embeddings: np.ndarray,
        tq: TurboQuantPGVector,
    ) -> list[dict]:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        compressed_list = tq.compress_batch(embeddings)

        records: list[dict] = []
        for i, comp in enumerate(compressed_list):
            decompressed = tq.decompress_embedding(comp)
            bytea = comp.to_pgbytea()
            records.append(
                {
                    "id": str(ids[i]),
                    "vector": decompressed.tolist(),
                    "properties": {
                        "tq_compressed": base64.b64encode(bytea).decode("ascii"),
                        "tq_bits": comp.bits,
                    },
                }
            )
        return records


class PineconeExporter(VectorDBExporter):
    """Export as Pinecone-compatible upsert dicts.

    Pinecone uses string IDs with metadata.
    """

    def format_name(self) -> str:
        return "pinecone"

    def export_batch(
        self,
        ids: list,
        embeddings: np.ndarray,
        tq: TurboQuantPGVector,
    ) -> list[dict]:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        compressed_list = tq.compress_batch(embeddings)

        records: list[dict] = []
        for i, comp in enumerate(compressed_list):
            decompressed = tq.decompress_embedding(comp)
            bytea = comp.to_pgbytea()
            records.append(
                {
                    "id": str(ids[i]),
                    "values": decompressed.tolist(),
                    "metadata": {
                        "tq_compressed": base64.b64encode(bytea).decode("ascii"),
                        "tq_bits": comp.bits,
                        "tq_dim": comp.dim,
                    },
                }
            )
        return records


class GenericExporter(VectorDBExporter):
    """Export a portable JSON-serializable format.

    Includes the full float32 vector, compressed bytes as base64, the
    original norm, dimension, and bit width.
    """

    def format_name(self) -> str:
        return "generic"

    def export_batch(
        self,
        ids: list,
        embeddings: np.ndarray,
        tq: TurboQuantPGVector,
    ) -> list[dict]:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        compressed_list = tq.compress_batch(embeddings)

        records: list[dict] = []
        for i, comp in enumerate(compressed_list):
            decompressed = tq.decompress_embedding(comp)
            bytea = comp.to_pgbytea()
            records.append(
                {
                    "id": ids[i],
                    "vector_f32": decompressed.tolist(),
                    "compressed_b64": base64.b64encode(bytea).decode("ascii"),
                    "norm": comp.norm,
                    "dim": comp.dim,
                    "bits": comp.bits,
                }
            )
        return records


# ------------------------------------------------------------------ #
# Exporter registry                                                   #
# ------------------------------------------------------------------ #

_EXPORTERS: dict[str, type[VectorDBExporter]] = {
    "milvus": MilvusExporter,
    "qdrant": QdrantExporter,
    "weaviate": WeaviateExporter,
    "pinecone": PineconeExporter,
    "generic": GenericExporter,
}


# ------------------------------------------------------------------ #
# Convenience helpers                                                 #
# ------------------------------------------------------------------ #


def supported_formats() -> list[str]:
    """Return the list of supported export format names."""
    return sorted(_EXPORTERS.keys())


def export_compressed(
    ids: list,
    embeddings: np.ndarray,
    tq: TurboQuantPGVector,
    format: str = "generic",  # noqa: A002
) -> list[dict]:
    """Convenience function: export compressed embeddings in one call.

    Creates the appropriate exporter for *format* and calls
    :meth:`VectorDBExporter.export_batch`.

    Args:
        ids: List of vector IDs.
        embeddings: (n, dim) float32 array of original embeddings.
        tq: TurboQuantPGVector instance for compression parameters.
        format: Target format name (see :func:`supported_formats`).

    Returns:
        List of dicts ready for the target DB's upsert API.

    Raises:
        ValueError: If *format* is not one of the supported names.
    """
    if format not in _EXPORTERS:
        raise ValueError(
            f"Unknown export format {format!r}; choose from {supported_formats()}"
        )
    exporter = _EXPORTERS[format]()
    return exporter.export_batch(ids, embeddings, tq)

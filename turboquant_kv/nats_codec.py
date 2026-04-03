# TurboQuant-KV: Open-source PolarQuant+QJL for LLM KV cache compression
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""
TurboQuant NATS codec for compressed embedding transport.

Provides serialization/deserialization of embeddings for transmission
over NATS JetStream (or any message bus). Reduces bandwidth by ~5-10x
for embedding vectors, which is especially useful for:

  - AGI-HPC subsystem communication (LH <-> Memory <-> RH)
  - JetBot / edge device communication (limited network)
  - Streaming embedding pipelines

The codec serializes compressed embeddings into a compact binary format
that includes all metadata needed for decompression. The receiver must
share the same (dim, bits, seed) parameters to decompress.

Wire format::

    [1 byte]  version (0x01)
    [1 byte]  bits (2, 3, or 4)
    [2 bytes] dim (uint16, little-endian, max 65535)
    [4 bytes] norm (float32, little-endian)
    [N bytes] packed indices (bit-packed)

Total overhead: 8 bytes header + packed payload.

Usage::

    from turboquant_kv.nats_codec import TurboQuantNATSCodec

    codec = TurboQuantNATSCodec(dim=1024, bits=3, seed=42)

    # Encode for transport
    payload = codec.encode(embedding_float32)

    # Decode on the other end
    embedding_approx = codec.decode(payload)

    # Batch encode/decode
    payloads = codec.encode_batch(embeddings_2d)
    embeddings = codec.decode_batch(payloads)
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Sequence

import numpy as np

from .pgvector import CompressedEmbedding, TurboQuantPGVector

logger = logging.getLogger(__name__)

# Wire format version
_WIRE_VERSION = 0x01

# Header struct: version(1) + bits(1) + dim(2) + norm(4) = 8 bytes
_HEADER_FMT = "<BBHf"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


class TurboQuantNATSCodec:
    """Compress/decompress embeddings for NATS transport.

    Wraps TurboQuantPGVector to provide a simple encode/decode API
    for message bus communication. Both sender and receiver must use
    the same (dim, bits, seed) parameters.

    Args:
        dim: Embedding dimension (e.g., 1024 for BGE-M3).
        bits: Quantization width -- 2, 3, or 4.
        seed: Random seed for the rotation matrix.
    """

    def __init__(
        self,
        dim: int = 1024,
        bits: int = 3,
        seed: int = 42,
    ) -> None:
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self._tq = TurboQuantPGVector(dim=dim, bits=bits, seed=seed)

        # Precompute expected payload size
        self._packed_bytes_per_emb = (dim * bits + 7) // 8
        self._total_bytes_per_emb = _HEADER_SIZE + self._packed_bytes_per_emb

        logger.info(
            "TurboQuantNATSCodec: dim=%d, bits=%d, "
            "payload=%d bytes (%.1fx reduction from float32)",
            dim,
            bits,
            self._total_bytes_per_emb,
            (dim * 4) / self._total_bytes_per_emb,
        )

    # ------------------------------------------------------------------ #
    # Encode / Decode                                                     #
    # ------------------------------------------------------------------ #

    def encode(self, embedding: np.ndarray) -> bytes:
        """Compress and serialize an embedding for transport.

        Args:
            embedding: Float32 embedding of shape (dim,).

        Returns:
            Compact bytes payload for NATS publish.
        """
        compressed = self._tq.compress_embedding(embedding)
        header = struct.pack(
            _HEADER_FMT,
            _WIRE_VERSION,
            self.bits,
            self.dim,
            compressed.norm,
        )
        return header + compressed.packed_bytes

    def decode(self, payload: bytes) -> np.ndarray:
        """Decompress a payload back to a float32 embedding.

        Args:
            payload: Bytes from encode().

        Returns:
            Approximate float32 embedding of shape (dim,).

        Raises:
            ValueError: If the payload header is invalid.
        """
        if len(payload) < _HEADER_SIZE:
            raise ValueError(f"Payload too short: {len(payload)} < {_HEADER_SIZE}")

        version, bits, dim, norm = struct.unpack(_HEADER_FMT, payload[:_HEADER_SIZE])

        if version != _WIRE_VERSION:
            raise ValueError(
                f"Unsupported wire version: {version} (expected {_WIRE_VERSION})"
            )

        if bits != self.bits or dim != self.dim:
            raise ValueError(
                f"Parameter mismatch: payload has (dim={dim}, bits={bits}), "
                f"codec has (dim={self.dim}, bits={self.bits})"
            )

        packed_bytes = payload[_HEADER_SIZE:]
        compressed = CompressedEmbedding(
            packed_bytes=packed_bytes,
            norm=norm,
            dim=dim,
            bits=bits,
        )

        return self._tq.decompress_embedding(compressed)

    # ------------------------------------------------------------------ #
    # Batch operations                                                    #
    # ------------------------------------------------------------------ #

    def encode_batch(self, embeddings: np.ndarray) -> list[bytes]:
        """Compress and serialize a batch of embeddings.

        Args:
            embeddings: 2D float32 array of shape (n, dim).

        Returns:
            List of bytes payloads, one per embedding.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        compressed_list = self._tq.compress_batch(embeddings)
        payloads = []
        for comp in compressed_list:
            header = struct.pack(
                _HEADER_FMT,
                _WIRE_VERSION,
                self.bits,
                self.dim,
                comp.norm,
            )
            payloads.append(header + comp.packed_bytes)
        return payloads

    def decode_batch(self, payloads: Sequence[bytes]) -> np.ndarray:
        """Decompress a batch of payloads back to float32.

        Args:
            payloads: Sequence of bytes from encode() or encode_batch().

        Returns:
            2D float32 array of shape (n, dim).
        """
        result = np.empty((len(payloads), self.dim), dtype=np.float32)
        for i, payload in enumerate(payloads):
            result[i] = self.decode(payload)
        return result

    # ------------------------------------------------------------------ #
    # Statistics                                                          #
    # ------------------------------------------------------------------ #

    def payload_size(self) -> int:
        """Return the expected payload size in bytes for one embedding."""
        return self._total_bytes_per_emb

    def float32_size(self) -> int:
        """Return the float32 size in bytes for one embedding."""
        return self.dim * 4

    def compression_ratio(self) -> float:
        """Return the compression ratio vs float32."""
        return self.float32_size() / self.payload_size()

    def stats(self) -> dict[str, object]:
        """Return codec statistics.

        Returns:
            Dict with payload_bytes, float32_bytes, compression_ratio,
            and header_overhead_bytes.
        """
        return {
            "dim": self.dim,
            "bits": self.bits,
            "payload_bytes": self.payload_size(),
            "float32_bytes": self.float32_size(),
            "compression_ratio": round(self.compression_ratio(), 2),
            "header_overhead_bytes": _HEADER_SIZE,
        }

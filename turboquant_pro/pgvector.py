# TurboQuant Pro: Open-source PolarQuant+QJL for LLM KV cache compression
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""
TurboQuant compression for pgvector embeddings.

Applies PolarQuant + Lloyd-Max quantization (Zandieh et al., ICLR 2026)
to compress high-dimensional embeddings stored in PostgreSQL pgvector,
reducing storage from float32 (4 bytes/dim) to ~3 bits/dim (0.375 bytes/dim).

For 1024-dim BGE-M3 embeddings this means:
  - float32: 4096 bytes per embedding
  - TurboQuant 3-bit packed: ~388 bytes per embedding + 4 bytes norm
  - Compression ratio: ~10.5x (float32 baseline) or ~5.2x (float16 baseline)

The compressed representation is stored as bytea in PostgreSQL alongside
the original vector (or replacing it). Similarity search works by:
  1. Compressing the query embedding with the same rotation matrix.
  2. Computing approximate distances in compressed space.
  3. Optionally re-ranking top candidates with exact float32 vectors.

Usage::

    from turboquant_pro.pgvector import TurboQuantPGVector

    tq = TurboQuantPGVector(dim=1024, bits=3, seed=42)

    # Compress a single embedding
    compressed = tq.compress_embedding(embedding_float32)
    # compressed.packed_bytes: bytes object for PostgreSQL bytea
    # compressed.norm: float32 scalar

    # Batch compress
    results = tq.compress_batch(embeddings_array)

    # Search: approximate cosine similarity on compressed vectors
    scores = tq.compressed_cosine_similarity(query_emb, stored_compressed)

    # PostgreSQL integration
    tq.create_compressed_table(conn, "rag_chunks_compressed")
    tq.insert_compressed(conn, "rag_chunks_compressed", ids, embeddings)
    results = tq.search_compressed(conn, "rag_chunks_compressed",
                                    query_emb, top_k=10)
"""

from __future__ import annotations

import logging
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cupy as cp  # type: ignore[import-untyped]

    _HAS_CUPY = True
except ImportError:
    cp = None  # type: ignore[assignment]
    _HAS_CUPY = False

logger = logging.getLogger(__name__)

# Lloyd-Max codebook centroids for standard normal distribution
# (same as core.py but included here for standalone usage)
_CODEBOOKS: dict[int, np.ndarray] = {
    2: np.array([-1.510, -0.453, 0.453, 1.510]),
    3: np.array([-1.748, -1.050, -0.500, -0.069, 0.069, 0.500, 1.050, 1.748]),
    4: np.array(
        [
            -2.401,
            -1.844,
            -1.437,
            -1.099,
            -0.800,
            -0.524,
            -0.262,
            -0.066,
            0.066,
            0.262,
            0.524,
            0.800,
            1.099,
            1.437,
            1.844,
            2.401,
        ]
    ),
}


@dataclass
class CompressedEmbedding:
    """A single compressed embedding.

    Attributes:
        packed_bytes: Bit-packed quantization indices as raw bytes.
        norm: L2 norm of the original embedding (float32).
        dim: Original embedding dimension.
        bits: Quantization bit width used.
    """

    packed_bytes: bytes
    norm: float
    dim: int
    bits: int

    @property
    def size_bytes(self) -> int:
        """Total size of the compressed representation in bytes."""
        return len(self.packed_bytes) + 4  # +4 for float32 norm

    @property
    def original_size_bytes(self) -> int:
        """Size of the original float32 embedding in bytes."""
        return self.dim * 4

    @property
    def compression_ratio(self) -> float:
        """Ratio of original to compressed size."""
        return self.original_size_bytes / max(self.size_bytes, 1)

    def to_pgbytea(self) -> bytes:
        """Serialize to a single bytes object for PostgreSQL bytea storage.

        Format: [4 bytes float32 norm][packed_bytes]
        """
        return struct.pack("<f", self.norm) + self.packed_bytes

    @classmethod
    def from_pgbytea(cls, data: bytes, dim: int, bits: int) -> CompressedEmbedding:
        """Deserialize from PostgreSQL bytea.

        Args:
            data: Raw bytes from PostgreSQL bytea column.
            dim: Original embedding dimension.
            bits: Quantization bit width.

        Returns:
            Reconstructed CompressedEmbedding.
        """
        norm = struct.unpack("<f", data[:4])[0]
        packed_bytes = data[4:]
        return cls(packed_bytes=packed_bytes, norm=norm, dim=dim, bits=bits)


class TurboQuantPGVector:
    """Compress pgvector embeddings using TurboQuant.

    Applies the same PolarQuant + Lloyd-Max algorithm as TurboQuantKV
    but adapted for standalone embedding vectors (not KV cache tensors).

    The random rotation matrix is deterministic given a seed, so all
    embeddings compressed with the same (dim, bits, seed) parameters
    share the same quantization space and can be compared directly.

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
        if bits not in _CODEBOOKS:
            raise ValueError(
                f"Unsupported bits={bits}; choose from {sorted(_CODEBOOKS)}"
            )

        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.n_centroids = 2**bits

        # Codebook scaled by 1/sqrt(dim)
        raw = _CODEBOOKS[bits]
        scale = 1.0 / math.sqrt(dim)
        self.centroids = (raw * scale).astype(np.float32)
        self.boundaries = (self.centroids[:-1] + self.centroids[1:]) / 2.0

        # Random rotation matrix
        rng = np.random.default_rng(seed)
        if dim <= 4096:
            G = rng.standard_normal((dim, dim)).astype(np.float32)
            Q, _ = np.linalg.qr(G)
            self._Pi = Q
            self._Pi_T = Q.T.copy()
            self._structured = False
        else:
            self._sign_flip = rng.choice([-1.0, 1.0], size=dim).astype(np.float32)
            self._perm = rng.permutation(dim)
            self._inv_perm = np.argsort(self._perm)
            self._structured = True

        logger.info(
            "TurboQuantPGVector: dim=%d, bits=%d, seed=%d, rotation=%s",
            dim,
            bits,
            seed,
            "structured" if self._structured else "full_QR",
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _rotate(self, x: np.ndarray) -> np.ndarray:
        """Apply random rotation along the last axis."""
        if self._structured:
            return (x * self._sign_flip)[..., self._perm]
        return x @ self._Pi_T

    def _unrotate(self, y: np.ndarray) -> np.ndarray:
        """Inverse rotation along the last axis."""
        if self._structured:
            return y[..., self._inv_perm] / self._sign_flip
        return y @ self._Pi

    def _pack_bits_cpu(self, indices: np.ndarray) -> np.ndarray:
        """Pack b-bit indices into bytes (NumPy, CPU)."""
        flat = indices.ravel().astype(np.uint32)
        n = len(flat)

        if self.bits == 2:
            pad = (4 - n % 4) % 4
            if pad:
                flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint32)])
            flat = flat.reshape(-1, 4)
            packed = (
                flat[:, 0] | (flat[:, 1] << 2) | (flat[:, 2] << 4) | (flat[:, 3] << 6)
            )
            return packed.astype(np.uint8)

        elif self.bits == 3:
            pad = (8 - n % 8) % 8
            if pad:
                flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint32)])
            flat = flat.reshape(-1, 8)
            bits24 = (
                flat[:, 0]
                | (flat[:, 1] << 3)
                | (flat[:, 2] << 6)
                | (flat[:, 3] << 9)
                | (flat[:, 4] << 12)
                | (flat[:, 5] << 15)
                | (flat[:, 6] << 18)
                | (flat[:, 7] << 21)
            )
            b0 = (bits24 & 0xFF).astype(np.uint8)
            b1 = ((bits24 >> 8) & 0xFF).astype(np.uint8)
            b2 = ((bits24 >> 16) & 0xFF).astype(np.uint8)
            packed = np.column_stack([b0, b1, b2]).ravel()
            return packed

        elif self.bits == 4:
            pad = (2 - n % 2) % 2
            if pad:
                flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint32)])
            flat = flat.reshape(-1, 2)
            packed = flat[:, 0] | (flat[:, 1] << 4)
            return packed.astype(np.uint8)

        else:
            raise ValueError(f"Unsupported bits={self.bits} for packing")

    def _unpack_bits_cpu(self, packed: np.ndarray, n_values: int) -> np.ndarray:
        """Unpack bytes back to b-bit indices (NumPy, CPU)."""
        packed = packed.ravel()

        if self.bits == 2:
            b = packed.astype(np.uint32)
            v0 = b & 0x3
            v1 = (b >> 2) & 0x3
            v2 = (b >> 4) & 0x3
            v3 = (b >> 6) & 0x3
            out = np.column_stack([v0, v1, v2, v3]).ravel()
            return out[:n_values].astype(np.uint8)

        elif self.bits == 3:
            packed = packed.reshape(-1, 3)
            b0 = packed[:, 0].astype(np.uint32)
            b1 = packed[:, 1].astype(np.uint32)
            b2 = packed[:, 2].astype(np.uint32)
            bits24 = b0 | (b1 << 8) | (b2 << 16)
            v0 = bits24 & 0x7
            v1 = (bits24 >> 3) & 0x7
            v2 = (bits24 >> 6) & 0x7
            v3 = (bits24 >> 9) & 0x7
            v4 = (bits24 >> 12) & 0x7
            v5 = (bits24 >> 15) & 0x7
            v6 = (bits24 >> 18) & 0x7
            v7 = (bits24 >> 21) & 0x7
            out = np.column_stack([v0, v1, v2, v3, v4, v5, v6, v7]).ravel()
            return out[:n_values].astype(np.uint8)

        elif self.bits == 4:
            b = packed.astype(np.uint32)
            v0 = b & 0xF
            v1 = (b >> 4) & 0xF
            out = np.column_stack([v0, v1]).ravel()
            return out[:n_values].astype(np.uint8)

        else:
            raise ValueError(f"Unsupported bits={self.bits} for unpacking")

    # ------------------------------------------------------------------ #
    # Public API: Single embedding                                        #
    # ------------------------------------------------------------------ #

    def compress_embedding(self, embedding: np.ndarray) -> CompressedEmbedding:
        """Compress a single embedding vector.

        Args:
            embedding: 1D float32 array of shape (dim,).

        Returns:
            CompressedEmbedding with packed bytes and norm.
        """
        embedding = np.asarray(embedding, dtype=np.float32).ravel()
        if len(embedding) != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {len(embedding)}")

        # L2 norm
        norm = float(np.linalg.norm(embedding))

        # Normalize to unit vector
        if norm > 1e-30:
            unit = embedding / norm
        else:
            unit = embedding

        # Rotate
        rotated = self._rotate(unit)

        # Quantize
        indices = np.searchsorted(self.boundaries, rotated).astype(np.uint8)

        # Pack
        packed = self._pack_bits_cpu(indices)

        return CompressedEmbedding(
            packed_bytes=packed.tobytes(),
            norm=norm,
            dim=self.dim,
            bits=self.bits,
        )

    def decompress_embedding(self, compressed: CompressedEmbedding) -> np.ndarray:
        """Decompress a single embedding back to float32.

        Args:
            compressed: CompressedEmbedding from compress_embedding.

        Returns:
            Approximate float32 embedding of shape (dim,).
        """
        packed = np.frombuffer(compressed.packed_bytes, dtype=np.uint8)
        indices = self._unpack_bits_cpu(packed, compressed.dim)

        # Look up centroids
        y_hat = self.centroids[indices]

        # Inverse rotation
        x_hat = self._unrotate(y_hat)

        # Scale by norm
        x_hat = x_hat * compressed.norm

        return x_hat.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Public API: Batch operations                                        #
    # ------------------------------------------------------------------ #

    def compress_batch(
        self,
        embeddings: np.ndarray,
        *,
        use_gpu: bool = False,
    ) -> list[CompressedEmbedding]:
        """Compress a batch of embeddings.

        Args:
            embeddings: 2D array of shape (n, dim).
            use_gpu: If True and CuPy is available, use GPU-accelerated
                rotation, quantization, and bit-packing kernels.  Falls
                back to CPU automatically if CuPy is not installed.

        Returns:
            List of CompressedEmbedding objects.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {embeddings.shape[1]}")

        gpu = use_gpu and _HAS_CUPY

        if gpu:
            return self._compress_batch_gpu(embeddings)

        results = []
        # Vectorized norms
        norms = np.linalg.norm(embeddings, axis=1)

        # Vectorized normalize
        safe_norms = np.maximum(norms, 1e-30)[:, np.newaxis]
        units = embeddings / safe_norms

        # Vectorized rotate
        rotated = self._rotate(units)

        # Vectorized quantize
        # searchsorted works on flattened, then reshape
        indices_flat = np.searchsorted(self.boundaries, rotated.ravel()).astype(
            np.uint8
        )
        indices_2d = indices_flat.reshape(embeddings.shape)

        for i in range(len(embeddings)):
            packed = self._pack_bits_cpu(indices_2d[i])
            results.append(
                CompressedEmbedding(
                    packed_bytes=packed.tobytes(),
                    norm=float(norms[i]),
                    dim=self.dim,
                    bits=self.bits,
                )
            )

        return results

    def _compress_batch_gpu(self, embeddings: np.ndarray) -> list[CompressedEmbedding]:
        """GPU-accelerated batch compression (internal)."""
        from .cuda_kernels import (
            gpu_batch_quantize,
            gpu_batch_rotate_quantize,
        )

        N = embeddings.shape[0]

        # Transfer to GPU
        emb_d = cp.asarray(embeddings)
        norms_d = cp.linalg.norm(emb_d, axis=1)
        safe_norms_d = cp.maximum(norms_d, 1e-30)[:, cp.newaxis]
        units_d = emb_d / safe_norms_d

        # Fused rotation + quantization (or structured + standalone quantize)
        if not self._structured:
            Pi_T_d = cp.asarray(self._Pi_T)
            bounds_d = cp.asarray(self.boundaries)
            indices_d = gpu_batch_rotate_quantize(units_d, Pi_T_d, bounds_d, self.bits)
        else:
            sign_d = cp.asarray(self._sign_flip)
            perm_d = cp.asarray(self._perm)
            rotated_d = (units_d * sign_d)[:, perm_d]
            bounds_d = cp.asarray(self.boundaries)
            indices_d = gpu_batch_quantize(rotated_d, bounds_d, self.bits)

        # GPU bit-packing per embedding
        norms_h = cp.asnumpy(norms_d)
        indices_h = cp.asnumpy(indices_d)

        results = []
        for i in range(N):
            packed = self._pack_bits_cpu(indices_h[i])
            results.append(
                CompressedEmbedding(
                    packed_bytes=packed.tobytes(),
                    norm=float(norms_h[i]),
                    dim=self.dim,
                    bits=self.bits,
                )
            )

        return results

    def decompress_batch(
        self, compressed_list: Sequence[CompressedEmbedding]
    ) -> np.ndarray:
        """Decompress a batch of embeddings.

        Args:
            compressed_list: Sequence of CompressedEmbedding objects.

        Returns:
            2D float32 array of shape (n, dim).
        """
        results = np.empty((len(compressed_list), self.dim), dtype=np.float32)
        for i, comp in enumerate(compressed_list):
            results[i] = self.decompress_embedding(comp)
        return results

    # ------------------------------------------------------------------ #
    # Similarity search on compressed representations                     #
    # ------------------------------------------------------------------ #

    def compressed_cosine_similarity(
        self,
        query: np.ndarray,
        compressed_list: Sequence[CompressedEmbedding],
    ) -> np.ndarray:
        """Compute approximate cosine similarity between a query and
        a set of compressed embeddings.

        This decompresses on-the-fly and computes cosine similarity.
        For large-scale search, use the PostgreSQL integration methods.

        Args:
            query: Float32 query embedding of shape (dim,).
            compressed_list: Sequence of compressed embeddings.

        Returns:
            1D array of cosine similarities, shape (n,).
        """
        query = np.asarray(query, dtype=np.float32).ravel()
        decompressed = self.decompress_batch(compressed_list)

        # Cosine similarity
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-30:
            return np.zeros(len(compressed_list), dtype=np.float32)

        doc_norms = np.linalg.norm(decompressed, axis=1)
        dots = decompressed @ query
        denom = np.maximum(doc_norms * query_norm, 1e-30)
        return (dots / denom).astype(np.float32)

    def compressed_inner_product_search(
        self,
        query_indices: np.ndarray,
        query_norm: float,
        stored_indices_list: list[np.ndarray],
        stored_norms: np.ndarray,
    ) -> np.ndarray:
        """Approximate inner product search entirely in compressed
        (index) space.

        Uses centroid lookup to approximate the inner product without
        full decompression. This is faster for large-scale search.

        Args:
            query_indices: Quantized indices of the query, shape (dim,).
            query_norm: L2 norm of the original query.
            stored_indices_list: List of quantized index arrays.
            stored_norms: Array of stored embedding norms, shape (n,).

        Returns:
            Approximate inner product scores, shape (n,).
        """
        # Precompute centroid-centroid inner products
        # For b-bit, this is a (2^b x 2^b) matrix
        cc = np.outer(self.centroids, self.centroids)

        query_codes = query_indices.astype(np.int32)
        scores = np.empty(len(stored_indices_list), dtype=np.float32)

        for i, stored_idx in enumerate(stored_indices_list):
            stored_codes = stored_idx.astype(np.int32)
            # Sum of centroid products for each dimension
            approx_dot = np.sum(cc[query_codes, stored_codes])
            scores[i] = approx_dot * query_norm * stored_norms[i]

        return scores

    # ------------------------------------------------------------------ #
    # PostgreSQL integration                                              #
    # ------------------------------------------------------------------ #

    def create_compressed_table(
        self,
        conn: Any,
        table_name: str,
        id_type: str = "INTEGER",
    ) -> None:
        """Create a PostgreSQL table for compressed embeddings.

        Args:
            conn: psycopg2 connection object.
            table_name: Name of the table to create.
            id_type: SQL type for the id column.
        """
        sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id {id_type} PRIMARY KEY,
            compressed_embedding BYTEA NOT NULL,
            embedding_dim INTEGER NOT NULL DEFAULT {self.dim},
            embedding_bits INTEGER NOT NULL DEFAULT {self.bits}
        );
        """
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        logger.info("Created compressed table: %s", table_name)

    def insert_compressed(
        self,
        conn: Any,
        table_name: str,
        ids: Sequence[int],
        embeddings: np.ndarray,
        batch_size: int = 1000,
    ) -> int:
        """Compress and insert embeddings into a PostgreSQL table.

        Args:
            conn: psycopg2 connection object.
            table_name: Target table name.
            ids: Sequence of integer IDs.
            embeddings: 2D float32 array, shape (n, dim).
            batch_size: Number of rows per INSERT batch.

        Returns:
            Number of rows inserted.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        n = len(embeddings)
        inserted = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_embs = embeddings[start:end]
            batch_ids = ids[start:end]
            compressed = self.compress_batch(batch_embs)

            with conn.cursor() as cur:
                args_list = []
                for j, comp in enumerate(compressed):
                    bytea = comp.to_pgbytea()
                    args_list.append((batch_ids[j], bytea, self.dim, self.bits))

                from psycopg2.extras import (
                    execute_values,  # type: ignore[import-untyped]
                )

                execute_values(
                    cur,
                    f"INSERT INTO {table_name} "
                    f"(id, compressed_embedding, embedding_dim, embedding_bits) "
                    f"VALUES %s "
                    f"ON CONFLICT (id) DO UPDATE SET "
                    f"compressed_embedding = EXCLUDED.compressed_embedding",
                    args_list,
                )

            conn.commit()
            inserted += end - start
            if inserted % 10000 == 0:
                logger.info("Inserted %d / %d rows", inserted, n)

        logger.info("Inserted %d compressed embeddings into %s", inserted, table_name)
        return inserted

    def search_compressed(
        self,
        conn: Any,
        table_name: str,
        query: np.ndarray,
        top_k: int = 10,
        fetch_limit: int = 0,
    ) -> list[tuple[int, float]]:
        """Search compressed embeddings in PostgreSQL.

        Fetches all compressed embeddings (or up to fetch_limit),
        decompresses and computes cosine similarity, then returns
        top_k results.

        For production use with millions of vectors, combine with
        a pre-filter or an approximate index.

        Args:
            conn: psycopg2 connection object.
            table_name: Table with compressed embeddings.
            query: Float32 query embedding, shape (dim,).
            top_k: Number of results to return.
            fetch_limit: Max rows to fetch (0 = all).

        Returns:
            List of (id, similarity_score) tuples, sorted descending.
        """
        query = np.asarray(query, dtype=np.float32).ravel()

        limit_clause = f" LIMIT {fetch_limit}" if fetch_limit > 0 else ""
        sql = f"SELECT id, compressed_embedding " f"FROM {table_name}" f"{limit_clause}"

        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

        if not rows:
            return []

        # Decompress all and compute cosine similarity
        ids = []
        compressed_list = []
        for row_id, bytea_data in rows:
            ids.append(row_id)
            # Handle memoryview from psycopg2
            if isinstance(bytea_data, memoryview):
                bytea_data = bytes(bytea_data)
            comp = CompressedEmbedding.from_pgbytea(bytea_data, self.dim, self.bits)
            compressed_list.append(comp)

        similarities = self.compressed_cosine_similarity(query, compressed_list)

        # Sort by similarity descending, take top_k
        top_indices = np.argsort(-similarities)[:top_k]
        results = [(ids[i], float(similarities[i])) for i in top_indices]

        return results

    # ------------------------------------------------------------------ #
    # Storage statistics                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def estimate_storage(
        n_embeddings: int,
        dim: int = 1024,
        bits: int = 3,
    ) -> dict[str, float]:
        """Estimate storage savings for compressed embeddings.

        Args:
            n_embeddings: Number of embedding vectors.
            dim: Embedding dimension.
            bits: Quantization bit width.

        Returns:
            Dict with storage estimates in MB and compression ratio.
        """
        # float32 original
        original_bytes = n_embeddings * dim * 4
        # Compressed: packed bits + float32 norm per embedding
        packed_bytes_per = (dim * bits + 7) // 8
        compressed_bytes = n_embeddings * (packed_bytes_per + 4)

        return {
            "n_embeddings": n_embeddings,
            "original_mb": round(original_bytes / (1024**2), 1),
            "compressed_mb": round(compressed_bytes / (1024**2), 1),
            "ratio": round(original_bytes / max(compressed_bytes, 1), 2),
            "saved_mb": round((original_bytes - compressed_bytes) / (1024**2), 1),
        }

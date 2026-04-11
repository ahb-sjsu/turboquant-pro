# TurboQuant Pro: Open-source PolarQuant+QJL for LLM KV cache compression
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""
Compressed HNSW (Hierarchical Navigable Small World) graph index.

A pure-Python HNSW implementation where each node stores a
``CompressedEmbedding`` instead of a float32 vector.  Distance computation
during graph traversal uses a precomputed centroid-centroid inner-product
lookup table, avoiding full decompression and making traversal proportional
to the number of *index lookups* rather than floating-point MACs.

When reranking is enabled (the default), the top candidates are
decompressed to float32 and scored with exact cosine similarity against
the raw query, recovering most of the accuracy lost to quantization.

Usage::

    from turboquant_pro.pgvector import TurboQuantPGVector
    from turboquant_pro.hnsw import CompressedHNSW

    tq = TurboQuantPGVector(dim=1024, bits=3, seed=42)
    index = CompressedHNSW(tq, M=16, ef_construction=200)

    index.insert(0, embedding_0)
    index.insert(1, embedding_1)

    results = index.search(query_embedding, k=10)
    # [(id, score), ...] sorted by descending similarity
"""

from __future__ import annotations

import heapq
import logging
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .ans_codec import ANSCodec
from .pgvector import CompressedEmbedding, TurboQuantPGVector

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Data structures                                                     #
# ------------------------------------------------------------------ #


@dataclass
class HNSWNode:
    """A single node in the HNSW graph.

    Attributes:
        id: External vector ID.
        compressed: Stored compressed embedding.
        indices: Cached unpacked uint8 quantization indices of shape
            ``(dim,)`` used for fast lookup-table distance computation.
        norm: L2 norm of the original embedding.
        neighbors: Neighbor lists per layer.  ``neighbors[layer]`` is a
            list of ``(node_id, distance)`` tuples.
    """

    id: int
    compressed: CompressedEmbedding
    indices: np.ndarray
    norm: float
    neighbors: list[list[tuple[int, float]]] = field(default_factory=list)


# ------------------------------------------------------------------ #
# Compressed HNSW index                                               #
# ------------------------------------------------------------------ #


class CompressedHNSW:
    """HNSW graph index backed by TurboQuant compressed embeddings.

    All graph-traversal distances are computed via a centroid-centroid
    inner-product lookup table (size ``n_centroids x n_centroids``),
    making each distance evaluation O(dim) integer table lookups plus
    one multiply -- roughly 4-8x faster than float32 dot products on
    the same dimensionality.

    Args:
        tq: A configured ``TurboQuantPGVector`` instance that defines
            the compression parameters (dim, bits, rotation matrix,
            codebook).
        M: Maximum number of neighbors per node per layer (layer > 0).
            Layer 0 uses ``2 * M``.
        ef_construction: Size of the dynamic candidate list during
            insertion (larger = better recall, slower build).
        max_layer: Not used for clamping; included for future
            compatibility.
        rerank_factor: When reranking, decompress the top
            ``k * rerank_factor`` candidates for exact scoring.
        seed: Random seed for reproducible layer assignment.
    """

    def __init__(
        self,
        tq: TurboQuantPGVector,
        M: int = 16,
        ef_construction: int = 200,
        max_layer: int = 0,
        rerank_factor: int = 3,
        seed: int = 42,
    ) -> None:
        self._tq = tq
        self._M = M
        self._ef_construction = ef_construction
        self._rerank_factor = rerank_factor

        # Layer-0 gets double the connections for better recall.
        self._M_max: int = M
        self._M_max0: int = 2 * M

        # Level generation factor (Malkov & Yashunin, 2018).
        self._ml: float = 1.0 / math.log(M)

        # Precomputed centroid-centroid inner-product table.
        # For 3-bit quantization this is an 8x8 matrix.
        self._cc_table: np.ndarray = np.outer(tq.centroids, tq.centroids).astype(
            np.float32
        )

        # Node storage
        self._nodes: dict[int, HNSWNode] = {}
        self._entry_point: int | None = None
        self._top_layer: int = -1

        # Deterministic RNG for layer selection.
        self._rng: np.random.Generator = np.random.default_rng(seed)

        logger.info(
            "CompressedHNSW: dim=%d, bits=%d, M=%d, ef_construction=%d, "
            "cc_table=%s, rerank_factor=%d",
            tq.dim,
            tq.bits,
            M,
            ef_construction,
            self._cc_table.shape,
            rerank_factor,
        )

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def n_vectors(self) -> int:
        """Number of vectors currently stored in the index."""
        return len(self._nodes)

    # ------------------------------------------------------------------ #
    # Distance helpers                                                    #
    # ------------------------------------------------------------------ #

    def _compressed_distance(
        self,
        indices_a: np.ndarray,
        norm_a: float,
        indices_b: np.ndarray,
        norm_b: float,
    ) -> float:
        """Approximate cosine distance using the lookup table.

        The centroid-centroid table stores ``c_i * c_j`` for every pair
        of codebook entries.  Because embeddings are normalised to unit
        vectors before rotation and quantization, summing the table
        entries along the dimension axis directly yields the approximate
        cosine similarity of the two original vectors::

            approx_cosine = sum(cc_table[a_i, b_i])

        The norms stored in ``CompressedEmbedding`` are retained for
        optional inner-product scoring but are **not** used in cosine
        distance computation.

        Returns:
            Approximate cosine distance in ``[0, 2]``.
        """
        approx_ip = float(np.sum(self._cc_table[indices_a, indices_b]))
        # The centroids already incorporate 1/sqrt(dim) scaling and the
        # embeddings are normalised to unit vectors before rotation, so
        # approx_ip is already the approximate cosine similarity.  The
        # norms are only needed for inner-product distance (not used here).
        return 1.0 - approx_ip

    # ------------------------------------------------------------------ #
    # Layer selection                                                      #
    # ------------------------------------------------------------------ #

    def _select_layer(self) -> int:
        """Sample a random insertion layer.

        Uses the standard HNSW formula:
        ``floor(-ln(uniform(0,1)) * m_l)``
        """
        return int(-math.log(self._rng.random()) * self._ml)

    # ------------------------------------------------------------------ #
    # Core HNSW search within a single layer                              #
    # ------------------------------------------------------------------ #

    def _search_layer(
        self,
        query_indices: np.ndarray,
        query_norm: float,
        entry_id: int,
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        """Greedy beam search within a single layer.

        Uses a min-heap for candidates (next to explore) and a max-heap
        for the result set (capped at *ef*).

        Args:
            query_indices: Unpacked uint8 quantization indices of the
                query vector.
            query_norm: L2 norm of the original query.
            entry_id: Node ID to start the search from.
            ef: Size of the dynamic candidate list.
            layer: Which layer of each node's neighbor list to traverse.

        Returns:
            List of ``(distance, node_id)`` sorted ascending by
            distance.
        """
        entry_node = self._nodes[entry_id]
        d_entry = self._compressed_distance(
            query_indices,
            query_norm,
            entry_node.indices,
            entry_node.norm,
        )

        # candidates: min-heap  (distance, node_id)
        candidates: list[tuple[float, int]] = [(d_entry, entry_id)]
        # results:   max-heap   (-distance, node_id)  — so we can pop the worst
        results: list[tuple[float, int]] = [(-d_entry, entry_id)]

        visited: set[int] = {entry_id}

        while candidates:
            d_c, c_id = heapq.heappop(candidates)

            # If the closest candidate is farther than the worst result,
            # we can stop.
            worst_dist = -results[0][0]
            if d_c > worst_dist:
                break

            c_node = self._nodes[c_id]
            if layer < len(c_node.neighbors):
                for neighbor_id, _ in c_node.neighbors[layer]:
                    if neighbor_id in visited:
                        continue
                    visited.add(neighbor_id)

                    n_node = self._nodes[neighbor_id]
                    d_n = self._compressed_distance(
                        query_indices,
                        query_norm,
                        n_node.indices,
                        n_node.norm,
                    )

                    worst_dist = -results[0][0]
                    if d_n < worst_dist or len(results) < ef:
                        heapq.heappush(candidates, (d_n, neighbor_id))
                        heapq.heappush(results, (-d_n, neighbor_id))
                        if len(results) > ef:
                            heapq.heappop(results)

        # Convert max-heap to sorted ascending list.
        out = [(abs(neg_d), nid) for neg_d, nid in results]
        out.sort(key=lambda x: x[0])
        return out

    # ------------------------------------------------------------------ #
    # Neighbor selection                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _select_neighbors(
        candidates: list[tuple[float, int]],
        M: int,
    ) -> list[tuple[float, int]]:
        """Simple nearest-M selection from *candidates*.

        Args:
            candidates: List of ``(distance, node_id)``.
            M: Maximum number of neighbors to retain.

        Returns:
            The *M* closest candidates, sorted ascending by distance.
        """
        candidates.sort(key=lambda x: x[0])
        return candidates[:M]

    # ------------------------------------------------------------------ #
    # Insertion                                                            #
    # ------------------------------------------------------------------ #

    def insert(self, id: int, embedding: np.ndarray) -> None:  # noqa: A002
        """Insert a single vector into the HNSW index.

        The embedding is compressed, quantization indices are unpacked
        for fast lookup-table distance, and the node is wired into the
        graph using the standard HNSW algorithm.

        Args:
            id: External vector ID (must be unique).
            embedding: Raw float32 embedding of shape ``(dim,)``.
        """
        embedding = np.asarray(embedding, dtype=np.float32).ravel()

        # 1. Compress
        compressed = self._tq.compress_embedding(embedding)

        # 2. Unpack indices for fast lookup-table distance
        packed = np.frombuffer(compressed.packed_bytes, dtype=np.uint8)
        indices = self._tq._unpack_bits_cpu(packed, self._tq.dim)

        norm = compressed.norm

        # 3. Select random layer for this node
        node_layer = self._select_layer()

        # Create the node with empty neighbor lists for each layer
        node = HNSWNode(
            id=id,
            compressed=compressed,
            indices=indices,
            norm=norm,
            neighbors=[[] for _ in range(node_layer + 1)],
        )
        self._nodes[id] = node

        # 4. Handle first insertion
        if self._entry_point is None:
            self._entry_point = id
            self._top_layer = node_layer
            return

        # 5. Greedy descent from the top layer to one above the
        #    insertion layer.
        current_ep = self._entry_point
        for layer in range(self._top_layer, node_layer, -1):
            results = self._search_layer(
                indices,
                norm,
                current_ep,
                ef=1,
                layer=layer,
            )
            current_ep = results[0][1]  # closest node

        # 6. Insert at each layer from node_layer down to 0.
        for layer in range(min(node_layer, self._top_layer), -1, -1):
            results = self._search_layer(
                indices,
                norm,
                current_ep,
                ef=self._ef_construction,
                layer=layer,
            )

            M_max = self._M_max0 if layer == 0 else self._M_max

            neighbors = self._select_neighbors(results, M_max)

            # Store forward edges (new node -> neighbors).
            node.neighbors[layer] = [(nid, dist) for dist, nid in neighbors]

            # Store backward edges (neighbors -> new node) and prune.
            for dist, nid in neighbors:
                n_node = self._nodes[nid]

                # Ensure the neighbor has enough layers allocated.
                while len(n_node.neighbors) <= layer:
                    n_node.neighbors.append([])

                n_node.neighbors[layer].append((id, dist))

                # Prune if over capacity.
                if len(n_node.neighbors[layer]) > M_max:
                    # Re-sort by distance and keep the closest M_max.
                    n_node.neighbors[layer].sort(key=lambda x: x[1])
                    n_node.neighbors[layer] = n_node.neighbors[layer][:M_max]

            # Use the closest result as the entry point for the next layer.
            current_ep = results[0][1]

        # 7. Update global entry point if this node is the new tallest.
        if node_layer > self._top_layer:
            self._entry_point = id
            self._top_layer = node_layer

    # ------------------------------------------------------------------ #
    # Search                                                               #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: int = 0,
        rerank: bool = True,
    ) -> list[tuple[int, float]]:
        """Search the HNSW index for the *k* nearest neighbors.

        Args:
            query: Raw float32 query embedding of shape ``(dim,)``.
            k: Number of results to return.
            ef: Search beam width.  Defaults to ``max(k, ef_construction)``
                when set to 0.
            rerank: If ``True``, decompress the top
                ``k * rerank_factor`` candidates and compute exact
                cosine similarity against the raw *query*.

        Returns:
            List of ``(id, score)`` tuples sorted by **descending**
            similarity (score = 1 - distance).
        """
        if self._entry_point is None:
            return []

        query = np.asarray(query, dtype=np.float32).ravel()

        if ef == 0:
            ef = max(k, self._ef_construction)

        # 1. Compress the query to get indices.
        compressed_q = self._tq.compress_embedding(query)
        packed_q = np.frombuffer(compressed_q.packed_bytes, dtype=np.uint8)
        q_indices = self._tq._unpack_bits_cpu(packed_q, self._tq.dim)
        q_norm = compressed_q.norm

        # 2. Traverse from the top layer down to layer 1 with ef=1.
        current_ep = self._entry_point
        for layer in range(self._top_layer, 0, -1):
            results = self._search_layer(
                q_indices,
                q_norm,
                current_ep,
                ef=1,
                layer=layer,
            )
            current_ep = results[0][1]

        # 3. Search layer 0 with full ef.
        results = self._search_layer(
            q_indices,
            q_norm,
            current_ep,
            ef=ef,
            layer=0,
        )

        # 4. Optionally rerank with exact cosine similarity.
        if rerank:
            n_rerank = min(len(results), k * self._rerank_factor)
            candidates = results[:n_rerank]

            # Decompress candidates and compute exact cosine.
            query_norm = float(np.linalg.norm(query))
            if query_norm < 1e-30:
                return [(nid, 0.0) for _, nid in candidates[:k]]

            scored: list[tuple[int, float]] = []
            for _, nid in candidates:
                node = self._nodes[nid]
                decompressed = self._tq.decompress_embedding(node.compressed)
                doc_norm = float(np.linalg.norm(decompressed))
                if doc_norm < 1e-30:
                    scored.append((nid, 0.0))
                    continue
                sim = float(np.dot(query, decompressed) / (query_norm * doc_norm))
                scored.append((nid, sim))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:k]

        # No reranking -- return approximate scores.
        out: list[tuple[int, float]] = []
        for dist, nid in results[:k]:
            out.append((nid, 1.0 - dist))

        out.sort(key=lambda x: x[1], reverse=True)
        return out

    # ------------------------------------------------------------------ #
    # Batch insertion                                                      #
    # ------------------------------------------------------------------ #

    def batch_insert(
        self,
        ids: list[int] | np.ndarray,
        embeddings: np.ndarray,
    ) -> int:
        """Insert multiple vectors into the index.

        Args:
            ids: Sequence of external vector IDs (one per embedding).
            embeddings: 2D float32 array of shape ``(n, dim)``.

        Returns:
            Number of vectors successfully inserted.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        count = 0
        for i, vec_id in enumerate(ids):
            self.insert(int(vec_id), embeddings[i])
            count += 1

        logger.info(
            "batch_insert: added %d vectors (total: %d)",
            count,
            self.n_vectors,
        )
        return count

    # ------------------------------------------------------------------ #
    # Memory and statistics                                                #
    # ------------------------------------------------------------------ #

    def memory_usage_bytes(self) -> dict[str, int]:
        """Estimate the in-memory footprint of the index.

        Accounts for compressed embedding bytes, cached uint8 indices,
        and neighbor-list overhead.  Reports both uncompressed and
        compressed graph sizes so users can see the benefit of ANS
        neighbor-list compression.

        Returns:
            Dictionary with ``embeddings``, ``indices``,
            ``graph_uncompressed``, ``graph_compressed``, and ``total``
            (using uncompressed graph size for the in-memory total).
        """
        codec = ANSCodec()
        emb_bytes = 0
        idx_bytes = 0
        graph_uncompressed = 0
        graph_compressed = 0

        for node in self._nodes.values():
            # Compressed packed bytes + 4 bytes for norm
            emb_bytes += len(node.compressed.packed_bytes) + 4
            # Cached indices array (uint8, dim elements)
            idx_bytes += node.indices.nbytes
            # Neighbor lists: each entry is (int id, float dist).
            # Uncompressed: 12 bytes per edge (8-byte int + 4-byte float).
            for layer_neighbors in node.neighbors:
                graph_uncompressed += len(layer_neighbors) * 12
                graph_compressed += codec.compressed_size(layer_neighbors)

        total = emb_bytes + idx_bytes + graph_uncompressed
        return {
            "embeddings": emb_bytes,
            "indices": idx_bytes,
            "graph_uncompressed": graph_uncompressed,
            "graph_compressed": graph_compressed,
            "total": total,
        }

    def stats(self) -> dict:
        """Return summary statistics for the index.

        Returns:
            Dictionary with ``n_vectors``, ``top_layer``,
            ``memory_bytes``, ``avg_neighbors_layer0``, and
            ``entry_point``.
        """
        n = self.n_vectors

        # Average neighbor count at layer 0.
        if n > 0:
            total_neighbors_l0 = sum(
                len(node.neighbors[0])
                for node in self._nodes.values()
                if len(node.neighbors) > 0
            )
            avg_neighbors_l0 = total_neighbors_l0 / n
        else:
            avg_neighbors_l0 = 0.0

        mem = self.memory_usage_bytes()
        return {
            "n_vectors": n,
            "top_layer": self._top_layer,
            "memory_bytes": mem["total"],
            "memory_detail": mem,
            "avg_neighbors_layer0": round(avg_neighbors_l0, 2),
            "entry_point": self._entry_point,
        }

    # ------------------------------------------------------------------ #
    # Save / Load with compressed graph                                    #
    # ------------------------------------------------------------------ #

    _MAGIC = b"TQHNSW01"

    def save(self, path: str) -> None:
        """Serialize the entire index to a binary file.

        Format::

            [8 bytes]  magic ``b"TQHNSW01"``
            [2 bytes]  M (uint16 LE)
            [2 bytes]  ef_construction (uint16 LE)
            [4 bytes]  top_layer (int32 LE)
            [4 bytes]  entry_point (int32 LE, -1 if None)
            [4 bytes]  n_vectors (uint32 LE)
            [2 bytes]  dim (uint16 LE)
            [1 byte]   bits (uint8)
            [4 bytes]  seed (uint32 LE)
            --- per node (repeated n_vectors times) ---
            [4 bytes]  node_id (int32 LE)
            [4 bytes]  norm (float32 LE)
            [2 bytes]  packed_bytes_len (uint16 LE)
            [N bytes]  packed_bytes
            [1 byte]   n_layers (uint8)
            --- per layer ---
            [2 bytes]  encoded_neighbor_bytes_len (uint16 LE)
            [M bytes]  ANS-compressed neighbor list

        Args:
            path: File path to write.
        """
        codec = ANSCodec()

        with open(path, "wb") as f:
            # Header
            f.write(self._MAGIC)
            f.write(struct.pack("<HH", self._M, self._ef_construction))
            f.write(struct.pack("<i", self._top_layer))
            ep = self._entry_point if self._entry_point is not None else -1
            f.write(struct.pack("<i", ep))
            f.write(struct.pack("<I", len(self._nodes)))
            f.write(struct.pack("<HBI", self._tq.dim, self._tq.bits, self._tq.seed))

            # Nodes
            for node_id, node in self._nodes.items():
                f.write(struct.pack("<i", node_id))
                f.write(struct.pack("<f", node.norm))
                packed = node.compressed.packed_bytes
                f.write(struct.pack("<H", len(packed)))
                f.write(packed)

                n_layers = len(node.neighbors)
                f.write(struct.pack("<B", n_layers))
                for layer_neighbors in node.neighbors:
                    encoded = codec.encode_neighbor_list(layer_neighbors)
                    f.write(struct.pack("<H", len(encoded)))
                    f.write(encoded)

        logger.info(
            "Saved HNSW index to %s (%d vectors, %d bytes)",
            path,
            len(self._nodes),
            Path(path).stat().st_size,
        )

    @classmethod
    def load(cls, path: str, tq: TurboQuantPGVector) -> CompressedHNSW:
        """Deserialize an HNSW index from a binary file.

        The caller must provide a ``TurboQuantPGVector`` instance with
        matching ``(dim, bits, seed)`` parameters.  These are validated
        against the file header.

        Args:
            path: File path to read.
            tq: A configured ``TurboQuantPGVector`` instance.

        Returns:
            A fully reconstructed ``CompressedHNSW`` index.

        Raises:
            ValueError: If the file magic or TQ parameters do not match.
        """
        codec = ANSCodec()

        with open(path, "rb") as f:
            magic = f.read(8)
            if magic != cls._MAGIC:
                raise ValueError(f"Invalid magic: {magic!r} (expected {cls._MAGIC!r})")

            M, ef_construction = struct.unpack("<HH", f.read(4))
            top_layer = struct.unpack("<i", f.read(4))[0]
            entry_point_raw = struct.unpack("<i", f.read(4))[0]
            entry_point: int | None = None if entry_point_raw == -1 else entry_point_raw
            n_vectors = struct.unpack("<I", f.read(4))[0]
            dim, bits, seed = struct.unpack("<HBI", f.read(7))

            # Validate TQ parameters.
            if dim != tq.dim or bits != tq.bits or seed != tq.seed:
                raise ValueError(
                    f"TQ parameter mismatch: file has "
                    f"(dim={dim}, bits={bits}, seed={seed}), "
                    f"tq has (dim={tq.dim}, bits={tq.bits}, seed={tq.seed})"
                )

            index = cls(tq, M=M, ef_construction=ef_construction)
            index._top_layer = top_layer
            index._entry_point = entry_point

            for _ in range(n_vectors):
                node_id = struct.unpack("<i", f.read(4))[0]
                norm = struct.unpack("<f", f.read(4))[0]
                packed_len = struct.unpack("<H", f.read(2))[0]
                packed_bytes = f.read(packed_len)

                compressed = CompressedEmbedding(
                    packed_bytes=packed_bytes,
                    norm=norm,
                    dim=dim,
                    bits=bits,
                )

                # Unpack indices for fast lookup-table distance.
                packed_arr = np.frombuffer(packed_bytes, dtype=np.uint8)
                indices = tq._unpack_bits_cpu(packed_arr, dim)

                n_layers = struct.unpack("<B", f.read(1))[0]
                neighbors: list[list[tuple[int, float]]] = []
                for _ in range(n_layers):
                    enc_len = struct.unpack("<H", f.read(2))[0]
                    enc_data = f.read(enc_len)
                    layer_neighbors = codec.decode_neighbor_list(enc_data)
                    neighbors.append(layer_neighbors)

                node = HNSWNode(
                    id=node_id,
                    compressed=compressed,
                    indices=indices,
                    norm=norm,
                    neighbors=neighbors,
                )
                index._nodes[node_id] = node

        logger.info(
            "Loaded HNSW index from %s (%d vectors, top_layer=%d)",
            path,
            n_vectors,
            top_layer,
        )
        return index

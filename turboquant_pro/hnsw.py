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
import sys
from dataclasses import dataclass, field

import numpy as np

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
        self._cc_table: np.ndarray = np.outer(
            tq.centroids, tq.centroids
        ).astype(np.float32)

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
            query_indices, query_norm,
            entry_node.indices, entry_node.norm,
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
                        query_indices, query_norm,
                        n_node.indices, n_node.norm,
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
                indices, norm, current_ep, ef=1, layer=layer,
            )
            current_ep = results[0][1]  # closest node

        # 6. Insert at each layer from node_layer down to 0.
        for layer in range(min(node_layer, self._top_layer), -1, -1):
            results = self._search_layer(
                indices, norm, current_ep, ef=self._ef_construction, layer=layer,
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
                q_indices, q_norm, current_ep, ef=1, layer=layer,
            )
            current_ep = results[0][1]

        # 3. Search layer 0 with full ef.
        results = self._search_layer(
            q_indices, q_norm, current_ep, ef=ef, layer=0,
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

    def memory_usage_bytes(self) -> int:
        """Estimate the in-memory footprint of the index.

        Accounts for compressed embedding bytes, cached uint8 indices,
        and neighbor-list overhead (two ints + one float per edge).

        Returns:
            Estimated memory usage in bytes.
        """
        total = 0
        for node in self._nodes.values():
            # Compressed packed bytes + 4 bytes for norm
            total += len(node.compressed.packed_bytes) + 4
            # Cached indices array (uint8, dim elements)
            total += node.indices.nbytes
            # Neighbor lists: each entry is (int id, float dist).
            # Use 12 bytes per edge (8-byte int + 4-byte float) as the
            # logical data size, excluding Python object overhead.
            for layer_neighbors in node.neighbors:
                total += len(layer_neighbors) * 12
        return total

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

        return {
            "n_vectors": n,
            "top_layer": self._top_layer,
            "memory_bytes": self.memory_usage_bytes(),
            "avg_neighbors_layer0": round(avg_neighbors_l0, 2),
            "entry_point": self._entry_point,
        }

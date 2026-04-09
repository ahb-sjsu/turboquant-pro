"""
FAISS integration for TurboQuant Pro.

Build FAISS indices from PCA-compressed embeddings for fast
approximate nearest-neighbor search with 15-113x smaller memory.

The pipeline: raw embeddings → PCA rotation → truncation → FAISS index.
Queries are automatically PCA-rotated before search.

Usage:
    from turboquant_pro import PCAMatryoshka
    from turboquant_pro.faiss_index import TurboQuantFAISS

    pca = PCAMatryoshka(input_dim=1024, output_dim=384)
    pca.fit(sample_embeddings)

    tqf = TurboQuantFAISS(pca)
    tqf.add(corpus_embeddings)  # PCA-compressed automatically
    distances, indices = tqf.search(query, k=10)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore[assignment]


class TurboQuantFAISS:
    """FAISS index with automatic PCA-Matryoshka compression.

    Wraps a FAISS index so that all add/search operations
    automatically apply PCA rotation and truncation. The index
    stores compressed vectors, saving 2-8x memory vs full-dim.

    Args:
        pca: Fitted PCAMatryoshka instance.
        index_type: FAISS index type — "flat", "ivf", or "hnsw".
        metric: Distance metric — "l2" or "ip" (inner product).
        n_lists: Number of IVF lists (only for index_type="ivf").
        n_probe: Number of IVF probes at search time.
        hnsw_m: HNSW connections per layer (only for "hnsw").
    """

    def __init__(
        self,
        pca: object,
        index_type: str = "flat",
        metric: str = "ip",
        n_lists: int = 100,
        n_probe: int = 10,
        hnsw_m: int = 32,
    ) -> None:
        if faiss is None:
            raise ImportError(
                "faiss required. Install with: " "pip install faiss-cpu  # or faiss-gpu"
            )

        self._pca = pca
        self._index_type = index_type
        self._metric = metric
        self._n_lists = n_lists
        self._n_probe = n_probe
        self._dim = pca.output_dim
        self._n_vectors = 0
        self._trained = False

        # Select FAISS metric
        faiss_metric = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2

        # Build index
        if index_type == "flat":
            self._index = faiss.IndexFlat(self._dim, faiss_metric)
            self._trained = True
        elif index_type == "ivf":
            quantizer = faiss.IndexFlat(self._dim, faiss_metric)
            self._index = faiss.IndexIVFFlat(
                quantizer, self._dim, n_lists, faiss_metric
            )
            self._index.nprobe = n_probe
        elif index_type == "hnsw":
            self._index = faiss.IndexHNSWFlat(self._dim, hnsw_m, faiss_metric)
            self._trained = True
        else:
            raise ValueError(
                f"Unknown index_type: {index_type}. " f"Use 'flat', 'ivf', or 'hnsw'."
            )

        logger.info(
            "TurboQuantFAISS: %s index, %d-dim (from %d), metric=%s",
            index_type,
            self._dim,
            pca.input_dim,
            metric,
        )

    @property
    def n_vectors(self) -> int:
        """Number of vectors in the index."""
        return self._index.ntotal

    @property
    def compression_ratio(self) -> float:
        """Dimension reduction ratio (input_dim / output_dim)."""
        return self._pca.input_dim / self._dim

    def _transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply PCA rotation and L2-normalize for search."""
        projected = self._pca.transform(embeddings)
        if self._metric == "ip":
            # Normalize for cosine similarity via inner product
            norms = np.linalg.norm(projected, axis=1, keepdims=True)
            projected = projected / np.maximum(norms, 1e-30)
        return projected.astype(np.float32)

    def train(self, embeddings: np.ndarray) -> None:
        """Train the index (required for IVF).

        Args:
            embeddings: Training data, shape (n, input_dim).
        """
        if self._trained:
            return
        projected = self._transform(embeddings)
        self._index.train(projected)
        self._trained = True
        logger.info(
            "Trained %s index on %d vectors",
            self._index_type,
            len(embeddings),
        )

    def add(self, embeddings: np.ndarray) -> int:
        """Add embeddings to the index (auto PCA-compressed).

        Args:
            embeddings: shape (n, input_dim) in original space.

        Returns:
            Number of vectors added.
        """
        if not self._trained:
            self.train(embeddings)

        projected = self._transform(embeddings)
        self._index.add(projected)
        self._n_vectors = self._index.ntotal
        logger.info(
            "Added %d vectors (total: %d)",
            len(embeddings),
            self._n_vectors,
        )
        return len(embeddings)

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors.

        Queries are automatically PCA-rotated before search.

        Args:
            queries: shape (n_queries, input_dim) or (input_dim,).
            k: Number of nearest neighbors.

        Returns:
            (distances, indices) — each shape (n_queries, k).
        """
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        projected = self._transform(queries)
        distances, indices = self._index.search(projected, k)
        return distances, indices

    def save(self, path: str | Path) -> None:
        """Save the FAISS index to disk.

        Note: You must also save the PCA model separately
        via pca.save().
        """
        faiss.write_index(self._index, str(path))
        logger.info("Saved FAISS index to %s", path)

    def load(self, path: str | Path) -> None:
        """Load a FAISS index from disk."""
        self._index = faiss.read_index(str(path))
        self._n_vectors = self._index.ntotal
        self._trained = True
        logger.info("Loaded FAISS index: %d vectors", self._n_vectors)

    def stats(self) -> dict:
        """Return index statistics."""
        original_bytes = self._n_vectors * self._pca.input_dim * 4
        compressed_bytes = self._n_vectors * self._dim * 4
        return {
            "n_vectors": self._n_vectors,
            "index_type": self._index_type,
            "input_dim": self._pca.input_dim,
            "index_dim": self._dim,
            "metric": self._metric,
            "dim_compression": round(self.compression_ratio, 1),
            "original_mb": round(original_bytes / 1048576, 1),
            "index_mb": round(compressed_bytes / 1048576, 1),
            "saved_mb": round((original_bytes - compressed_bytes) / 1048576, 1),
        }

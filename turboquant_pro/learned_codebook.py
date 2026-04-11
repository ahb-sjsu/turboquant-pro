# TurboQuant Pro: Open-source PolarQuant+QJL for LLM KV cache compression
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""
Learned codebook fine-tuning for TurboQuant embedding compression.

The default TurboQuant codebooks are Lloyd-Max optimal for N(0, 1/sqrt(d)),
but real embedding models (BGE-M3, E5, ada-002) don't produce perfectly
Gaussian-distributed rotated coordinates.  Fine-tuning codebooks on actual
data can improve cosine similarity from ~0.978 to ~0.99+ at the same
bit-width.

This module provides:

- ``CodebookTrainer``: Train codebooks from data via Lloyd's algorithm.
- ``LearnedQuantizer``: Drop-in replacement for ``TurboQuantPGVector``
  using a data-trained codebook.
- ``fit_codebook``: One-liner convenience function.

Usage::

    from turboquant_pro.learned_codebook import fit_codebook

    lq = fit_codebook(embeddings, dim=1024, bits=3)
    compressed = lq.compress_embedding(embedding)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .pgvector import CompressedEmbedding, TurboQuantPGVector

logger = logging.getLogger(__name__)

_SUPPORTED_BITS = (2, 3, 4)


class CodebookTrainer:
    """Train optimal quantization codebooks from data via Lloyd's algorithm.

    Instead of assuming Gaussian-distributed coordinates, this learns the
    codebook centroids and decision boundaries directly from the rotated
    embedding data, minimizing reconstruction MSE.

    Args:
        bits: Number of quantization bits (2, 3, or 4).
        max_iterations: Maximum Lloyd iterations (default 50).
        tol: Convergence tolerance on centroid movement (default 1e-6).
        seed: Random seed for centroid initialization.
    """

    def __init__(
        self,
        bits: int = 3,
        max_iterations: int = 50,
        tol: float = 1e-6,
        seed: int = 42,
    ) -> None:
        if bits not in _SUPPORTED_BITS:
            raise ValueError(
                f"Unsupported bits={bits}; choose from {sorted(_SUPPORTED_BITS)}"
            )
        self.bits = bits
        self.max_iterations = max_iterations
        self.tol = tol
        self.seed = seed
        self.n_centroids = 2**bits

        self._centroids: np.ndarray | None = None
        self._boundaries: np.ndarray | None = None
        self._n_iterations: int = 0
        self._mse_history: list[float] = []

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #

    def fit(self, rotated_data: np.ndarray) -> CodebookTrainer:
        """Train codebook centroids on flattened rotated coordinates.

        Runs Lloyd's algorithm (iterative 1-D k-means) on the data,
        initializing centroids at evenly-spaced quantiles.

        Args:
            rotated_data: 1-D array of float32 values (flattened rotated
                coordinates from many embeddings).

        Returns:
            Self, for chaining.
        """
        data = np.asarray(rotated_data, dtype=np.float32).ravel()
        if len(data) == 0:
            raise ValueError("rotated_data must not be empty")

        k = self.n_centroids

        # Initialize centroids at evenly-spaced quantiles
        quantile_points = np.linspace(0, 1, k + 2)[1:-1]  # exclude 0 and 1
        centroids = np.quantile(data, quantile_points).astype(np.float32)

        # Ensure unique starting centroids (add tiny jitter if needed)
        rng = np.random.default_rng(self.seed)
        for _attempt in range(10):
            if len(np.unique(centroids)) == k:
                break
            jitter = rng.standard_normal(k).astype(np.float32) * 1e-6
            centroids = centroids + jitter
            centroids = np.sort(centroids)

        self._mse_history = []

        for iteration in range(self.max_iterations):
            # Compute decision boundaries (midpoints between centroids)
            boundaries = (centroids[:-1] + centroids[1:]) / 2.0

            # Assign each data point to nearest centroid via boundaries
            assignments = np.searchsorted(boundaries, data).astype(np.int32)

            # Compute MSE for this iteration
            reconstructed = centroids[assignments]
            mse = float(np.mean((data - reconstructed) ** 2))
            self._mse_history.append(mse)

            # Update centroids as mean of assigned values
            new_centroids = np.empty_like(centroids)
            for j in range(k):
                mask = assignments == j
                if np.any(mask):
                    new_centroids[j] = np.mean(data[mask])
                else:
                    # Empty cluster: keep old centroid
                    new_centroids[j] = centroids[j]

            new_centroids = np.sort(new_centroids)

            # Check convergence
            max_shift = float(np.max(np.abs(new_centroids - centroids)))
            centroids = new_centroids

            if max_shift < self.tol:
                logger.debug(
                    "CodebookTrainer: converged at iteration %d (shift=%.2e)",
                    iteration + 1,
                    max_shift,
                )
                break

        self._n_iterations = iteration + 1  # noqa: B023
        self._centroids = np.sort(centroids)
        self._boundaries = (self._centroids[:-1] + self._centroids[1:]) / 2.0

        logger.info(
            "CodebookTrainer: bits=%d, %d iterations, final MSE=%.6f",
            self.bits,
            self._n_iterations,
            self._mse_history[-1] if self._mse_history else float("nan"),
        )

        return self

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def centroids(self) -> np.ndarray:
        """Trained centroid values, sorted ascending, shape ``(2**bits,)``."""
        if self._centroids is None:
            raise RuntimeError("Call fit() before accessing centroids")
        return self._centroids

    @property
    def boundaries(self) -> np.ndarray:
        """Decision boundaries (midpoints), shape ``(2**bits - 1,)``."""
        if self._boundaries is None:
            raise RuntimeError("Call fit() before accessing boundaries")
        return self._boundaries

    @property
    def n_iterations(self) -> int:
        """Number of Lloyd iterations actually run."""
        return self._n_iterations

    @property
    def mse_history(self) -> list[float]:
        """MSE at each iteration (useful for convergence diagnostics)."""
        return list(self._mse_history)

    # ------------------------------------------------------------------ #
    # Evaluation                                                          #
    # ------------------------------------------------------------------ #

    def mse(self, data: np.ndarray) -> float:
        """Compute mean squared error on *data* using the trained codebook.

        Args:
            data: 1-D array of float32 values.

        Returns:
            Scalar MSE.
        """
        data = np.asarray(data, dtype=np.float32).ravel()
        assignments = np.searchsorted(self.boundaries, data)
        reconstructed = self.centroids[assignments]
        return float(np.mean((data - reconstructed) ** 2))

    # ------------------------------------------------------------------ #
    # Serialization                                                       #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Save trained codebook to a ``.npz`` file.

        Args:
            path: Destination file path.
        """
        if self._centroids is None:
            raise RuntimeError("Call fit() before saving")
        np.savez(
            path,
            centroids=self._centroids,
            boundaries=self._boundaries,
            bits=np.array([self.bits]),
            n_iterations=np.array([self._n_iterations]),
        )
        logger.info("CodebookTrainer: saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> CodebookTrainer:
        """Load a trained codebook from a ``.npz`` file.

        Args:
            path: Source file path.

        Returns:
            A :class:`CodebookTrainer` with centroids and boundaries
            restored (no retraining needed).
        """
        data = np.load(path)
        bits = int(data["bits"][0])
        trainer = cls(bits=bits)
        trainer._centroids = data["centroids"].astype(np.float32)
        trainer._boundaries = data["boundaries"].astype(np.float32)
        trainer._n_iterations = int(data["n_iterations"][0])
        logger.info("CodebookTrainer: loaded from %s (bits=%d)", path, bits)
        return trainer


class LearnedQuantizer:
    """Drop-in replacement for :class:`TurboQuantPGVector` using a trained codebook.

    Uses the same rotation matrix as ``TurboQuantPGVector`` but replaces the
    Lloyd-Max codebook with a data-trained one.

    Args:
        dim: Embedding dimension.
        bits: Quantization bits (must match the ``CodebookTrainer``).
        codebook: A fitted :class:`CodebookTrainer`.
        seed: Rotation-matrix seed (must match the seed used during
            training so that the same rotation is applied).
    """

    def __init__(
        self,
        dim: int,
        bits: int,
        codebook: CodebookTrainer,
        seed: int = 42,
    ) -> None:
        if codebook._centroids is None:
            raise RuntimeError("codebook must be fitted before use")
        if bits != codebook.bits:
            raise ValueError(
                f"bits={bits} does not match codebook.bits={codebook.bits}"
            )

        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.codebook = codebook

        # Create internal TurboQuantPGVector and override its codebook
        self._tq = TurboQuantPGVector(dim=dim, bits=bits, seed=seed)
        self._tq.centroids = codebook.centroids.copy()
        self._tq.boundaries = codebook.boundaries.copy()

        logger.info(
            "LearnedQuantizer: dim=%d, bits=%d, seed=%d",
            dim,
            bits,
            seed,
        )

    # ------------------------------------------------------------------ #
    # Public API — delegates to the patched TurboQuantPGVector             #
    # ------------------------------------------------------------------ #

    def compress_embedding(self, embedding: np.ndarray) -> CompressedEmbedding:
        """Compress a single embedding using the learned codebook.

        Args:
            embedding: 1-D float32 array of shape ``(dim,)``.

        Returns:
            :class:`CompressedEmbedding` with packed bytes and norm.
        """
        return self._tq.compress_embedding(embedding)

    def decompress_embedding(self, compressed: CompressedEmbedding) -> np.ndarray:
        """Decompress a single embedding using the learned codebook.

        Args:
            compressed: A :class:`CompressedEmbedding`.

        Returns:
            Approximate float32 embedding of shape ``(dim,)``.
        """
        return self._tq.decompress_embedding(compressed)

    def compress_batch(
        self,
        embeddings: np.ndarray,
        *,
        use_gpu: bool = False,
    ) -> list[CompressedEmbedding]:
        """Compress a batch of embeddings.

        Args:
            embeddings: 2-D array of shape ``(n, dim)``.
            use_gpu: If True and CuPy is available, use GPU acceleration.

        Returns:
            List of :class:`CompressedEmbedding` objects.
        """
        return self._tq.compress_batch(embeddings, use_gpu=use_gpu)

    def decompress_batch(
        self,
        compressed_list: list[CompressedEmbedding],
    ) -> np.ndarray:
        """Decompress a batch of embeddings.

        Args:
            compressed_list: Sequence of :class:`CompressedEmbedding` objects.

        Returns:
            2-D float32 array of shape ``(n, dim)``.
        """
        return self._tq.decompress_batch(compressed_list)

    def compressed_cosine_similarity(
        self,
        query: np.ndarray,
        compressed_list: list[CompressedEmbedding],
    ) -> np.ndarray:
        """Approximate cosine similarity between query and compressed set.

        Args:
            query: Float32 query embedding of shape ``(dim,)``.
            compressed_list: Sequence of compressed embeddings.

        Returns:
            1-D array of cosine similarities, shape ``(n,)``.
        """
        return self._tq.compressed_cosine_similarity(query, compressed_list)


def fit_codebook(
    embeddings: np.ndarray,
    dim: int | None = None,
    bits: int = 3,
    seed: int = 42,
    n_train: int = 10_000,
    max_iterations: int = 50,
    tol: float = 1e-6,
) -> LearnedQuantizer:
    """One-liner: fit a codebook on embeddings and return a ready quantizer.

    Steps:

    1. Create a ``TurboQuantPGVector`` to obtain the rotation matrix.
    2. Rotate a sample of embeddings.
    3. Train a codebook on the flattened rotated coordinates.
    4. Return a :class:`LearnedQuantizer` with the trained codebook.

    Args:
        embeddings: 2-D float32 array of shape ``(n, dim)``.
        dim: Embedding dimension (inferred from *embeddings* if None).
        bits: Quantization bit-width (2, 3, or 4).
        seed: Rotation-matrix seed.
        n_train: Maximum number of embeddings to use for training.
        max_iterations: Maximum Lloyd iterations.
        tol: Convergence tolerance.

    Returns:
        A :class:`LearnedQuantizer` ready for compression.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    if dim is None:
        dim = embeddings.shape[1]

    if embeddings.shape[1] != dim:
        raise ValueError(f"embeddings have dim={embeddings.shape[1]}, expected {dim}")

    # Sub-sample if needed
    if len(embeddings) > n_train:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(embeddings), size=n_train, replace=False)
        sample = embeddings[indices]
    else:
        sample = embeddings

    # Rotate using the same rotation matrix the quantizer will use
    tq = TurboQuantPGVector(dim=dim, bits=bits, seed=seed)

    # Normalize to unit vectors then rotate
    norms = np.linalg.norm(sample, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, 1e-30)
    units = sample / safe_norms
    rotated = tq._rotate(units)

    # Flatten all rotated coordinates for 1-D codebook training
    flat_rotated = rotated.ravel()

    # Train codebook
    trainer = CodebookTrainer(
        bits=bits,
        max_iterations=max_iterations,
        tol=tol,
        seed=seed,
    )
    trainer.fit(flat_rotated)

    # Build the quantizer
    return LearnedQuantizer(dim=dim, bits=bits, codebook=trainer, seed=seed)

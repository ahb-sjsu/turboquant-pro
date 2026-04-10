# TurboQuant Pro: PCA-Matryoshka dimension reduction
# Copyright (c) 2025 Andrew H. Bond
# MIT License
#
# Implements PCA-Matryoshka (Bond, IEEE TAI 2026): PCA rotation enables
# effective dimension truncation for non-Matryoshka embedding models.

"""
PCA-Matryoshka: training-free dimension reduction for embeddings.

Applies a PCA rotation to any embedding model's output, reordering
dimensions by explained variance so that truncation becomes effective
without retraining.  Composes with :class:`TurboQuantPGVector` to form
the full PCA-Matryoshka + TurboQuant compression pipeline.

Typical usage::

    from turboquant_pro import PCAMatryoshka, TurboQuantPGVector

    # Fit PCA on a sample of embeddings
    pca = PCAMatryoshka(input_dim=1024, output_dim=384)
    pca.fit(sample_embeddings)  # (N, 1024) array

    # Transform + quantize
    tq = TurboQuantPGVector(dim=384, bits=3)
    reduced = pca.transform(embedding)        # 1024 -> 384
    compressed = tq.compress_embedding(reduced)  # 384 -> 3-bit packed

    # Or use the convenience pipeline
    pipeline = pca.with_quantizer(bits=3)
    compressed = pipeline.compress(embedding)
    reconstructed = pipeline.decompress(compressed)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import numpy as np

if TYPE_CHECKING:
    from .pgvector import TurboQuantPGVector

logger = logging.getLogger(__name__)


@dataclass
class PCAFitResult:
    """Statistics from fitting PCA on a corpus sample.

    Attributes:
        n_samples: Number of vectors used for fitting.
        explained_variance_ratio: Per-component variance ratio (output_dim,).
        cumulative_variance: Cumulative variance at each component.
        total_variance_explained: Fraction of total variance retained.
        input_dim: Original embedding dimension.
        output_dim: Retained dimension count.
    """

    n_samples: int
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    total_variance_explained: float
    input_dim: int
    output_dim: int


@dataclass
class PCACompressedEmbedding:
    """A PCA-reduced and optionally quantized embedding.

    Attributes:
        packed_bytes: Bit-packed quantization indices (if quantized).
        norm: L2 norm of the PCA-projected vector.
        pca_dim: Dimension after PCA truncation.
        bits: Quantization bit width (0 if unquantized float16/float32).
        pca_vector: Float vector if not quantized (None if quantized).
    """

    packed_bytes: bytes | None
    norm: float
    pca_dim: int
    bits: int
    pca_vector: np.ndarray | None = field(default=None, repr=False)

    @property
    def size_bytes(self) -> int:
        """Total compressed size in bytes."""
        if self.packed_bytes is not None:
            return len(self.packed_bytes) + 4  # +4 for norm
        if self.pca_vector is not None:
            return self.pca_vector.nbytes + 4
        return 4

    @property
    def original_size_bytes(self) -> int:
        """Size this would be as full-dim float32 (for ratio computation)."""
        # Caller must set this externally if they want the true original dim.
        # By default we report the PCA-dim size.
        return self.pca_dim * 4


class PCAMatryoshka:
    """PCA-Matryoshka dimension reduction for embeddings.

    Fits a PCA rotation on a sample of embeddings, then rotates and
    truncates new vectors to ``output_dim`` dimensions.  The rotation
    matrix concentrates information into leading components, enabling
    effective truncation for models not trained with Matryoshka losses.

    Args:
        input_dim: Original embedding dimension (e.g. 1024 for BGE-M3).
        output_dim: Number of principal components to retain.
        whiten: If True, scale components by 1/sqrt(eigenvalue) after
            projection (standardizes variance across components).
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 384,
        whiten: bool = False,
    ) -> None:
        if output_dim > input_dim:
            raise ValueError(
                f"output_dim ({output_dim}) must be <= input_dim ({input_dim})"
            )
        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {output_dim}")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.whiten = whiten

        # Fitted state (None until fit() is called)
        self._components: np.ndarray | None = None  # (output_dim, input_dim)
        self._mean: np.ndarray | None = None  # (input_dim,)
        self._eigenvalues: np.ndarray | None = None  # (output_dim,)
        self._all_eigenvalues: np.ndarray | None = None  # (input_dim,)
        self._n_samples_seen: int = 0

        # For incremental PCA
        self._sum: np.ndarray | None = None  # running sum for mean
        self._cov_accum: np.ndarray | None = None  # running covariance

    @property
    def is_fitted(self) -> bool:
        """Whether fit() has been called."""
        return self._components is not None

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("PCAMatryoshka is not fitted. Call fit() first.")

    # ------------------------------------------------------------------ #
    # Fitting                                                             #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray) -> PCAFitResult:
        """Fit PCA on a sample of embeddings.

        Args:
            X: Array of shape (n_samples, input_dim).  A sample of 5K-10K
               vectors is sufficient for stable results (see paper Sec. 5.7).

        Returns:
            PCAFitResult with variance statistics.

        Raises:
            ValueError: If X has wrong shape.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"Expected shape (n, {self.input_dim}), got {X.shape}")

        n = X.shape[0]
        if n < self.output_dim:
            raise ValueError(
                f"Need at least {self.output_dim} samples to fit "
                f"{self.output_dim} components, got {n}"
            )

        # Center
        mean = X.mean(axis=0)
        X_centered = X - mean

        # Covariance (use float64 for numerical stability)
        cov = (X_centered.T @ X_centered) / (n - 1)

        # Eigen-decomposition (symmetric, so use eigh for speed + stability)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # eigh returns ascending order; reverse to descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Clamp negative eigenvalues (numerical noise)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # Store top-k components as (output_dim, input_dim)
        self._components = eigenvectors[:, : self.output_dim].T.astype(np.float32)
        self._mean = mean.astype(np.float32)
        self._eigenvalues = eigenvalues[: self.output_dim].astype(np.float32)
        self._all_eigenvalues = eigenvalues.astype(np.float32)
        self._n_samples_seen = n

        # Store for incremental updates
        self._sum = (mean * n).astype(np.float64)
        self._cov_accum = (cov * (n - 1)).astype(np.float64)

        total_var = float(eigenvalues.sum())
        retained_var = float(eigenvalues[: self.output_dim].sum())
        ratio = retained_var / total_var if total_var > 0 else 0.0

        evr = eigenvalues[: self.output_dim] / total_var
        cumvar = np.cumsum(evr)

        result = PCAFitResult(
            n_samples=n,
            explained_variance_ratio=evr.astype(np.float32),
            cumulative_variance=cumvar.astype(np.float32),
            total_variance_explained=ratio,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )

        logger.info(
            "PCAMatryoshka fit: %d samples, %d -> %d dims, "
            "%.1f%% variance explained",
            n,
            self.input_dim,
            self.output_dim,
            ratio * 100,
        )
        return result

    def partial_fit(self, X: np.ndarray) -> PCAFitResult:
        """Incrementally update PCA with a new batch of embeddings.

        Uses the incremental covariance update method.  The new batch is
        merged with the running statistics, and the eigendecomposition is
        recomputed.  This allows the PCA basis to be updated as new data
        arrives without storing the full dataset.

        Args:
            X: Array of shape (n_new, input_dim).

        Returns:
            Updated PCAFitResult.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"Expected shape (n, {self.input_dim}), got {X.shape}")

        n_new = X.shape[0]

        if not self.is_fitted:
            return self.fit(X)

        n_old = self._n_samples_seen
        n_total = n_old + n_new

        # Running sums
        new_sum = X.sum(axis=0)
        combined_sum = self._sum + new_sum
        new_mean = combined_sum / n_total
        old_mean = self._sum / n_old

        # Incremental covariance update (Chan et al., 1979)
        X_centered_new = X - X.mean(axis=0)
        cov_new = X_centered_new.T @ X_centered_new  # unnormalized

        # Mean-shift correction
        delta = X.mean(axis=0) - old_mean
        correction = (n_old * n_new / n_total) * np.outer(delta, delta)

        combined_cov = self._cov_accum + cov_new + correction

        # Re-do eigendecomposition
        cov_normalized = combined_cov / (n_total - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_normalized)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # Update stored state
        self._components = eigenvectors[:, : self.output_dim].T.astype(np.float32)
        self._mean = new_mean.astype(np.float32)
        self._eigenvalues = eigenvalues[: self.output_dim].astype(np.float32)
        self._all_eigenvalues = eigenvalues.astype(np.float32)
        self._n_samples_seen = n_total
        self._sum = combined_sum
        self._cov_accum = combined_cov

        total_var = float(eigenvalues.sum())
        retained_var = float(eigenvalues[: self.output_dim].sum())
        ratio = retained_var / total_var if total_var > 0 else 0.0
        evr = eigenvalues[: self.output_dim] / total_var

        result = PCAFitResult(
            n_samples=n_total,
            explained_variance_ratio=evr.astype(np.float32),
            cumulative_variance=np.cumsum(evr).astype(np.float32),
            total_variance_explained=ratio,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )

        logger.info(
            "PCAMatryoshka partial_fit: +%d samples (total %d), "
            "%.1f%% variance explained",
            n_new,
            n_total,
            ratio * 100,
        )
        return result

    # ------------------------------------------------------------------ #
    # Transform                                                           #
    # ------------------------------------------------------------------ #

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project embeddings into the PCA-truncated space.

        Args:
            X: Array of shape (input_dim,) or (n, input_dim).

        Returns:
            Projected array of shape (output_dim,) or (n, output_dim).
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float32)
        squeeze = X.ndim == 1
        if squeeze:
            X = X[np.newaxis, :]

        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected dim={self.input_dim}, got {X.shape[1]}")

        # Center and project
        X_centered = X - self._mean
        projected = X_centered @ self._components.T  # (n, output_dim)

        if self.whiten:
            # Scale by 1/sqrt(eigenvalue) to standardize variance
            scale = 1.0 / np.sqrt(self._eigenvalues + 1e-10)
            projected = projected * scale

        return projected[0] if squeeze else projected

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct approximate embeddings from PCA-truncated space.

        Args:
            Z: Array of shape (output_dim,) or (n, output_dim).

        Returns:
            Reconstructed array of shape (input_dim,) or (n, input_dim).
        """
        self._check_fitted()
        Z = np.asarray(Z, dtype=np.float32)
        squeeze = Z.ndim == 1
        if squeeze:
            Z = Z[np.newaxis, :]

        if Z.shape[1] != self.output_dim:
            raise ValueError(f"Expected dim={self.output_dim}, got {Z.shape[1]}")

        if self.whiten:
            scale = np.sqrt(self._eigenvalues + 1e-10)
            Z = Z * scale

        # Reconstruct: Z @ components + mean
        reconstructed = Z @ self._components + self._mean

        return reconstructed[0] if squeeze else reconstructed

    def cosine_similarity(
        self, original: np.ndarray, transformed: np.ndarray | None = None
    ) -> np.ndarray:
        """Compute cosine similarity between originals and PCA round-trips.

        If ``transformed`` is None, compute transform + inverse_transform
        internally.

        Args:
            original: Shape (n, input_dim) or (input_dim,).
            transformed: Optional pre-transformed vectors (output_dim).

        Returns:
            Array of cosine similarities, shape (n,) or scalar.
        """
        self._check_fitted()
        original = np.asarray(original, dtype=np.float32)
        squeeze = original.ndim == 1
        if squeeze:
            original = original[np.newaxis, :]

        if transformed is None:
            projected = self.transform(original)
            reconstructed = self.inverse_transform(projected)
        else:
            if transformed.ndim == 1:
                transformed = transformed[np.newaxis, :]
            reconstructed = self.inverse_transform(transformed)

        # Cosine similarity per row
        dot = np.sum(original * reconstructed, axis=1)
        norm_a = np.linalg.norm(original, axis=1)
        norm_b = np.linalg.norm(reconstructed, axis=1)
        denom = norm_a * norm_b
        denom = np.where(denom < 1e-30, 1.0, denom)
        cos = dot / denom

        return float(cos[0]) if squeeze else cos

    # ------------------------------------------------------------------ #
    # Serialization                                                       #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path | BinaryIO) -> None:
        """Save the fitted PCA model to a .npz file.

        Args:
            path: File path or writable binary file object.
        """
        self._check_fitted()
        np.savez_compressed(
            path,
            components=self._components,
            mean=self._mean,
            eigenvalues=self._eigenvalues,
            all_eigenvalues=self._all_eigenvalues,
            n_samples_seen=np.array([self._n_samples_seen]),
            input_dim=np.array([self.input_dim]),
            output_dim=np.array([self.output_dim]),
            whiten=np.array([self.whiten]),
            # Incremental state
            sum=self._sum,
            cov_accum=self._cov_accum,
        )
        logger.info("PCAMatryoshka saved to %s", path)

    @classmethod
    def load(cls, path: str | Path | BinaryIO) -> PCAMatryoshka:
        """Load a fitted PCA model from a .npz file.

        Args:
            path: File path or readable binary file object.

        Returns:
            Fitted PCAMatryoshka instance.
        """
        data = np.load(path)
        input_dim = int(data["input_dim"][0])
        output_dim = int(data["output_dim"][0])
        whiten = bool(data["whiten"][0])

        obj = cls(input_dim=input_dim, output_dim=output_dim, whiten=whiten)
        obj._components = data["components"]
        obj._mean = data["mean"]
        obj._eigenvalues = data["eigenvalues"]
        obj._all_eigenvalues = data["all_eigenvalues"]
        obj._n_samples_seen = int(data["n_samples_seen"][0])
        obj._sum = data["sum"]
        obj._cov_accum = data["cov_accum"]

        logger.info(
            "PCAMatryoshka loaded: %d -> %d dims, %d samples, whiten=%s",
            input_dim,
            output_dim,
            obj._n_samples_seen,
            whiten,
        )
        return obj

    # ------------------------------------------------------------------ #
    # Pipeline composition                                                #
    # ------------------------------------------------------------------ #

    def with_quantizer(self, bits: int = 3, seed: int = 42) -> PCAMatryoshkaPipeline:
        """Create a full PCA + TurboQuant compression pipeline.

        Args:
            bits: Quantization bit width (2, 3, or 4).
            seed: Random seed for the TurboQuant rotation matrix.

        Returns:
            PCAMatryoshkaPipeline that composes PCA reduction with
            TurboQuant quantization.
        """
        self._check_fitted()
        from .pgvector import TurboQuantPGVector

        tq = TurboQuantPGVector(dim=self.output_dim, bits=bits, seed=seed)
        return PCAMatryoshkaPipeline(pca=self, quantizer=tq)

    def with_weighted_quantizer(
        self,
        bit_schedule: list[tuple[int, int]] | None = None,
        avg_bits: float = 3.0,
        seed: int = 42,
    ) -> "EigenweightedPipeline":
        """Create a pipeline with eigenvalue-weighted bit allocation.

        Allocates more bits to high-eigenvalue (more important) dimensions
        and fewer bits to low-eigenvalue dimensions. Motivated by
        Varici et al. (2025), who prove eigenvalues are theoretically
        grounded importance scores for ordered representations.

        Args:
            bit_schedule: Explicit list of (n_dims, bits) pairs.
                E.g., [(64, 4), (128, 3), (64, 2)] allocates 4 bits
                to the top 64 PCA dims, 3 bits to the next 128, etc.
                If None, auto-computed from eigenvalue spectrum.
            avg_bits: Target average bits per dimension when auto-computing
                the schedule. Ignored if bit_schedule is provided.
            seed: Random seed for quantizer rotation matrices.

        Returns:
            EigenweightedPipeline with per-segment quantization.
        """
        self._check_fitted()
        from .pgvector import TurboQuantPGVector

        if bit_schedule is None:
            bit_schedule = self._auto_bit_schedule(avg_bits)

        # Validate schedule covers output_dim
        total_dims = sum(n for n, _ in bit_schedule)
        if total_dims != self.output_dim:
            raise ValueError(
                f"Bit schedule covers {total_dims} dims "
                f"but PCA output is {self.output_dim}"
            )

        # Create one quantizer per segment
        segments = []
        offset = 0
        for n_dims, bits in bit_schedule:
            tq = TurboQuantPGVector(dim=n_dims, bits=bits, seed=seed + offset)
            segments.append((offset, n_dims, bits, tq))
            offset += n_dims

        return EigenweightedPipeline(pca=self, segments=segments)

    def _auto_bit_schedule(self, avg_bits: float) -> list[tuple[int, int]]:
        """Compute bit allocation from eigenvalue importance scores.

        Dimensions with eigenvalues above the median get more bits,
        dimensions below get fewer bits. The schedule averages to
        approximately avg_bits per dimension.
        """
        eigenvalues = self._eigenvalues
        d = self.output_dim

        # Compute cumulative variance fractions
        total_var = eigenvalues.sum()
        cum_var = np.cumsum(eigenvalues) / total_var

        if avg_bits <= 2.5:
            # Aggressive: 3-bit top quarter, 2-bit rest
            q1 = d // 4
            return [(q1, 3), (d - q1, 2)]
        elif avg_bits <= 3.5:
            # Balanced: 4-bit top quarter, 3-bit middle half, 2-bit bottom quarter
            q1 = d // 4
            q2 = d // 2
            q3 = d - q1 - q2
            return [(q1, 4), (q2, 3), (q3, 2)]
        else:
            # Conservative: 4-bit top half, 3-bit bottom half
            h = d // 2
            return [(h, 4), (d - h, 3)]

    # ------------------------------------------------------------------ #
    # Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def variance_report(self, dims: list[int] | None = None) -> dict:
        """Report variance explained at various truncation levels.

        Args:
            dims: List of dimensions to report.  Defaults to a useful set.

        Returns:
            Dict mapping dim -> cumulative variance explained.
        """
        self._check_fitted()
        if dims is None:
            dims = [
                d
                for d in [64, 128, 256, 384, 512, 768, self.output_dim]
                if d <= self.input_dim
            ]
            dims = sorted(set(dims))

        total = float(self._all_eigenvalues.sum())
        result = {}
        for d in dims:
            if d > len(self._all_eigenvalues):
                continue
            var = float(self._all_eigenvalues[:d].sum())
            result[d] = var / total if total > 0 else 0.0
        return result

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        var = ""
        if self.is_fitted:
            total = float(self._all_eigenvalues.sum())
            retained = float(self._eigenvalues.sum())
            pct = retained / total * 100 if total > 0 else 0
            var = f", {pct:.1f}% var"
        return (
            f"PCAMatryoshka({self.input_dim} -> {self.output_dim}, " f"{status}{var})"
        )


class PCAMatryoshkaPipeline:
    """Combined PCA-Matryoshka + TurboQuant compression pipeline.

    Composes :class:`PCAMatryoshka` (rotation + truncation) with
    :class:`TurboQuantPGVector` (scalar quantization + bit-packing)
    for end-to-end embedding compression.

    At 384 dims + 3-bit quantization on BGE-M3 (1024-dim), this achieves
    ~27x compression at 0.979 cosine similarity.

    Args:
        pca: Fitted PCAMatryoshka instance.
        quantizer: TurboQuantPGVector configured for the PCA output dimension.
    """

    def __init__(
        self,
        pca: PCAMatryoshka,
        quantizer: TurboQuantPGVector,
    ) -> None:

        if not pca.is_fitted:
            raise RuntimeError("PCA must be fitted before creating pipeline")
        if quantizer.dim != pca.output_dim:
            raise ValueError(
                f"Quantizer dim ({quantizer.dim}) must match "
                f"PCA output_dim ({pca.output_dim})"
            )

        self.pca = pca
        self.quantizer = quantizer

    @property
    def input_dim(self) -> int:
        """Original embedding dimension."""
        return self.pca.input_dim

    @property
    def output_dim(self) -> int:
        """PCA-truncated dimension."""
        return self.pca.output_dim

    @property
    def bits(self) -> int:
        """Quantization bit width."""
        return self.quantizer.bits

    @property
    def compression_ratio(self) -> float:
        """Theoretical compression ratio vs. full-dim float32."""
        original_bits = self.input_dim * 32
        compressed_bits = self.output_dim * self.bits + 32  # +32 for norm
        return original_bits / compressed_bits

    def compress(self, embedding: np.ndarray) -> PCACompressedEmbedding:
        """Compress a single embedding through the full pipeline.

        Pipeline: center → PCA rotate → truncate → quantize → bit-pack.

        Args:
            embedding: 1D float32 array of shape (input_dim,).

        Returns:
            PCACompressedEmbedding with packed bytes and metadata.
        """
        # PCA transform
        reduced = self.pca.transform(embedding)

        # TurboQuant compress
        tq_compressed = self.quantizer.compress_embedding(reduced)

        return PCACompressedEmbedding(
            packed_bytes=tq_compressed.packed_bytes,
            norm=tq_compressed.norm,
            pca_dim=self.output_dim,
            bits=self.bits,
        )

    def decompress(self, compressed: PCACompressedEmbedding) -> np.ndarray:
        """Decompress back to the original embedding space.

        Pipeline: unpack → dequantize → PCA inverse-rotate → uncenter.

        Args:
            compressed: PCACompressedEmbedding from compress().

        Returns:
            Approximate float32 embedding of shape (input_dim,).
        """
        from .pgvector import CompressedEmbedding

        # TurboQuant decompress
        tq_compressed = CompressedEmbedding(
            packed_bytes=compressed.packed_bytes,
            norm=compressed.norm,
            dim=compressed.pca_dim,
            bits=compressed.bits,
        )
        reduced = self.quantizer.decompress_embedding(tq_compressed)

        # PCA inverse transform
        return self.pca.inverse_transform(reduced)

    def compress_batch(self, embeddings: np.ndarray) -> list[PCACompressedEmbedding]:
        """Compress a batch of embeddings.

        Args:
            embeddings: 2D array of shape (n, input_dim).

        Returns:
            List of PCACompressedEmbedding.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected shape (n, {self.input_dim}), " f"got {embeddings.shape}"
            )

        # Batch PCA transform
        reduced = self.pca.transform(embeddings)

        # Batch TurboQuant compress
        tq_list = self.quantizer.compress_batch(reduced)

        return [
            PCACompressedEmbedding(
                packed_bytes=tq.packed_bytes,
                norm=tq.norm,
                pca_dim=self.output_dim,
                bits=self.bits,
            )
            for tq in tq_list
        ]

    def decompress_batch(
        self, compressed_list: list[PCACompressedEmbedding]
    ) -> np.ndarray:
        """Decompress a batch back to the original embedding space.

        Args:
            compressed_list: List of PCACompressedEmbedding.

        Returns:
            Array of shape (n, input_dim).
        """
        from .pgvector import CompressedEmbedding

        tq_list = [
            CompressedEmbedding(
                packed_bytes=c.packed_bytes,
                norm=c.norm,
                dim=c.pca_dim,
                bits=c.bits,
            )
            for c in compressed_list
        ]
        reduced_batch = self.quantizer.decompress_batch(tq_list)
        return self.pca.inverse_transform(reduced_batch)

    def batch_cosine_similarity(
        self, originals: np.ndarray
    ) -> tuple[float, float, float]:
        """Measure round-trip cosine similarity on a batch.

        Args:
            originals: Shape (n, input_dim).

        Returns:
            Tuple of (mean_cosine, min_cosine, std_cosine).
        """
        compressed = self.compress_batch(originals)
        reconstructed = self.decompress_batch(compressed)

        dot = np.sum(originals * reconstructed, axis=1)
        norm_a = np.linalg.norm(originals, axis=1)
        norm_b = np.linalg.norm(reconstructed, axis=1)
        denom = norm_a * norm_b
        denom = np.where(denom < 1e-30, 1.0, denom)
        cos = dot / denom

        return float(cos.mean()), float(cos.min()), float(cos.std())

    @staticmethod
    def estimate_storage(
        n_embeddings: int,
        input_dim: int = 1024,
        output_dim: int = 384,
        bits: int = 3,
    ) -> dict[str, float]:
        """Estimate storage requirements.

        Args:
            n_embeddings: Number of embeddings.
            input_dim: Original dimension.
            output_dim: PCA-truncated dimension.
            bits: Quantization bit width.

        Returns:
            Dict with original_mb, compressed_mb, ratio, saved_mb.
        """
        original = n_embeddings * input_dim * 4  # float32
        # Packed: ceil(output_dim * bits / 8) + 4 bytes norm
        packed_per = math.ceil(output_dim * bits / 8) + 4
        compressed = n_embeddings * packed_per

        original_mb = original / (1024 * 1024)
        compressed_mb = compressed / (1024 * 1024)

        return {
            "n_embeddings": n_embeddings,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "bits": bits,
            "original_mb": round(original_mb, 1),
            "compressed_mb": round(compressed_mb, 1),
            "ratio": round(original_mb / compressed_mb, 1) if compressed_mb > 0 else 0,
            "saved_mb": round(original_mb - compressed_mb, 1),
        }

    def __repr__(self) -> str:
        return (
            f"PCAMatryoshkaPipeline("
            f"{self.input_dim} -> PCA-{self.output_dim} -> "
            f"TQ{self.bits}-bit, "
            f"~{self.compression_ratio:.1f}x compression)"
        )


class EigenweightedPipeline:
    """PCA + eigenvalue-weighted quantization pipeline.

    Allocates more bits to high-eigenvalue PCA dimensions (more important
    features) and fewer bits to low-eigenvalue dimensions. Based on the
    spectral importance scores proven by Varici et al. (2025).

    This improves compression ratio at the same quality, or improves
    quality at the same compression ratio, compared to uniform bit allocation.

    Args:
        pca: Fitted PCAMatryoshka instance.
        segments: List of (offset, n_dims, bits, quantizer) tuples.
    """

    def __init__(
        self,
        pca: PCAMatryoshka,
        segments: list[tuple[int, int, int, "TurboQuantPGVector"]],
    ) -> None:
        self.pca = pca
        self.segments = segments

    @property
    def input_dim(self) -> int:
        return self.pca.input_dim

    @property
    def output_dim(self) -> int:
        return self.pca.output_dim

    @property
    def bit_schedule(self) -> list[tuple[int, int]]:
        """The (n_dims, bits) schedule."""
        return [(n, b) for _, n, b, _ in self.segments]

    @property
    def avg_bits(self) -> float:
        """Weighted average bits per dimension."""
        total_bits = sum(n * b for _, n, b, _ in self.segments)
        total_dims = sum(n for _, n, _, _ in self.segments)
        return total_bits / max(total_dims, 1)

    @property
    def compression_ratio(self) -> float:
        """Theoretical compression ratio vs. full-dim float32."""
        original_bits = self.input_dim * 32
        compressed_bits = sum(n * b for _, n, b, _ in self.segments) + 32
        return original_bits / compressed_bits

    def compress(self, embedding: np.ndarray) -> dict:
        """Compress a single embedding with per-segment quantization."""
        reduced = self.pca.transform(embedding)

        packed_segments = []
        for offset, n_dims, bits, tq in self.segments:
            segment = reduced[offset : offset + n_dims]
            compressed = tq.compress_embedding(segment)
            packed_segments.append(compressed)

        return {
            "segments": packed_segments,
            "pca_dim": self.output_dim,
            "schedule": self.bit_schedule,
        }

    def decompress(self, compressed: dict) -> np.ndarray:
        """Decompress back to original space."""
        reduced = np.zeros(self.output_dim, dtype=np.float32)

        for i, (offset, n_dims, bits, tq) in enumerate(self.segments):
            segment = tq.decompress_embedding(compressed["segments"][i])
            reduced[offset : offset + n_dims] = segment

        return self.pca.inverse_transform(reduced)

    def batch_cosine_similarity(
        self, originals: np.ndarray
    ) -> tuple[float, float, float]:
        """Measure round-trip cosine similarity on a batch."""
        reconstructed = []
        for emb in originals:
            c = self.compress(emb)
            r = self.decompress(c)
            reconstructed.append(r)
        reconstructed = np.array(reconstructed)

        dot = np.sum(originals * reconstructed, axis=1)
        norm_a = np.linalg.norm(originals, axis=1)
        norm_b = np.linalg.norm(reconstructed, axis=1)
        denom = norm_a * norm_b
        denom = np.where(denom < 1e-30, 1.0, denom)
        cos = dot / denom

        return float(cos.mean()), float(cos.min()), float(cos.std())

    def __repr__(self) -> str:
        schedule_str = "+".join(f"{n}d@{b}b" for n, b in self.bit_schedule)
        return (
            f"EigenweightedPipeline("
            f"{self.input_dim} -> PCA-{self.output_dim} -> "
            f"[{schedule_str}], "
            f"avg={self.avg_bits:.1f}b, "
            f"~{self.compression_ratio:.1f}x)"
        )

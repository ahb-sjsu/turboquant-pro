"""
Unit tests for learned codebook fine-tuning.

Tests the CodebookTrainer, LearnedQuantizer, and fit_codebook convenience
function, validating that data-trained codebooks improve reconstruction
quality over the default Lloyd-Max codebooks.

Usage:
    pytest tests/test_learned_codebook.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from turboquant_pro.learned_codebook import (
    CodebookTrainer,
    LearnedQuantizer,
    fit_codebook,
)
from turboquant_pro.pgvector import TurboQuantPGVector

# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-30 or norm_b < 1e-30:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _random_embeddings(
    n: int = 100,
    dim: int = 128,
    seed: int = 42,
) -> np.ndarray:
    """Generate random embeddings."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _make_rotated_data(
    n: int = 5000,
    dim: int = 128,
    bits: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """Generate flattened rotated coordinates for codebook training."""
    embs = _random_embeddings(n=n, dim=dim, seed=seed)
    tq = TurboQuantPGVector(dim=dim, bits=bits, seed=seed)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, 1e-30)
    units = embs / safe_norms
    rotated = tq._rotate(units)
    return rotated.ravel()


# ------------------------------------------------------------------ #
# CodebookTrainer tests                                               #
# ------------------------------------------------------------------ #


class TestCodebookTrainer:
    """Tests for CodebookTrainer."""

    def test_fit_produces_sorted_centroids(self) -> None:
        """Centroids are ascending after fit."""
        data = _make_rotated_data(n=2000, dim=64, bits=3)
        trainer = CodebookTrainer(bits=3, max_iterations=30)
        trainer.fit(data)
        centroids = trainer.centroids
        assert np.all(centroids[:-1] <= centroids[1:])

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_fit_correct_count(self, bits: int) -> None:
        """Fit produces exactly 2**bits centroids."""
        data = _make_rotated_data(n=2000, dim=64, bits=bits)
        trainer = CodebookTrainer(bits=bits, max_iterations=20)
        trainer.fit(data)
        assert len(trainer.centroids) == 2**bits

    def test_boundaries_between_centroids(self) -> None:
        """Each boundary lies between adjacent centroids."""
        data = _make_rotated_data(n=2000, dim=64, bits=3)
        trainer = CodebookTrainer(bits=3, max_iterations=30)
        trainer.fit(data)
        centroids = trainer.centroids
        boundaries = trainer.boundaries

        assert len(boundaries) == len(centroids) - 1
        for i in range(len(boundaries)):
            assert centroids[i] < boundaries[i] < centroids[i + 1], (
                f"boundary[{i}]={boundaries[i]} not between "
                f"centroids[{i}]={centroids[i]} and "
                f"centroids[{i+1}]={centroids[i+1]}"
            )

    def test_convergence(self) -> None:
        """MSE does not increase over iterations."""
        data = _make_rotated_data(n=3000, dim=64, bits=3)
        trainer = CodebookTrainer(bits=3, max_iterations=50, tol=1e-10)
        trainer.fit(data)
        history = trainer.mse_history
        assert len(history) > 1
        # MSE should be non-increasing (allow tiny float noise)
        for i in range(1, len(history)):
            assert history[i] <= history[i - 1] + 1e-10, (
                f"MSE increased at iteration {i}: "
                f"{history[i-1]:.8f} -> {history[i]:.8f}"
            )

    def test_mse_lower_than_lloyd_max(self) -> None:
        """Trained codebook has lower MSE than default Lloyd-Max on same data."""
        dim = 128
        bits = 3
        data = _make_rotated_data(n=5000, dim=dim, bits=bits, seed=99)

        # Default Lloyd-Max MSE
        tq = TurboQuantPGVector(dim=dim, bits=bits, seed=99)
        default_assignments = np.searchsorted(tq.boundaries, data)
        default_recon = tq.centroids[default_assignments]
        default_mse = float(np.mean((data - default_recon) ** 2))

        # Trained codebook MSE
        trainer = CodebookTrainer(bits=bits, max_iterations=50)
        trainer.fit(data)
        trained_mse = trainer.mse(data)

        assert trained_mse <= default_mse, (
            f"Trained MSE ({trained_mse:.6f}) should be <= "
            f"default MSE ({default_mse:.6f})"
        )

    def test_save_load_roundtrip(self) -> None:
        """Save and load preserves centroids and boundaries."""
        data = _make_rotated_data(n=2000, dim=64, bits=3)
        trainer = CodebookTrainer(bits=3, max_iterations=30)
        trainer.fit(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "codebook.npz"
            trainer.save(path)
            loaded = CodebookTrainer.load(path)

        np.testing.assert_array_almost_equal(
            trainer.centroids, loaded.centroids, decimal=6
        )
        np.testing.assert_array_almost_equal(
            trainer.boundaries, loaded.boundaries, decimal=6
        )
        assert trainer.bits == loaded.bits
        assert trainer.n_iterations == loaded.n_iterations

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_different_bits(self, bits: int) -> None:
        """Trainer works for all supported bit widths."""
        data = _make_rotated_data(n=2000, dim=64, bits=bits)
        trainer = CodebookTrainer(bits=bits, max_iterations=20)
        trainer.fit(data)

        assert len(trainer.centroids) == 2**bits
        assert len(trainer.boundaries) == 2**bits - 1
        assert trainer.n_iterations > 0
        assert trainer.mse(data) >= 0.0

    def test_invalid_bits_raises(self) -> None:
        """Unsupported bit width raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported bits=5"):
            CodebookTrainer(bits=5)

    def test_empty_data_raises(self) -> None:
        """Empty data raises ValueError."""
        trainer = CodebookTrainer(bits=3)
        with pytest.raises(ValueError, match="must not be empty"):
            trainer.fit(np.array([], dtype=np.float32))

    def test_properties_before_fit_raise(self) -> None:
        """Accessing centroids/boundaries before fit raises RuntimeError."""
        trainer = CodebookTrainer(bits=3)
        with pytest.raises(RuntimeError, match="Call fit"):
            _ = trainer.centroids
        with pytest.raises(RuntimeError, match="Call fit"):
            _ = trainer.boundaries


# ------------------------------------------------------------------ #
# LearnedQuantizer tests                                              #
# ------------------------------------------------------------------ #


class TestLearnedQuantizer:
    """Tests for LearnedQuantizer."""

    def test_compress_decompress_shape(self) -> None:
        """Round-trip preserves embedding shape."""
        dim, bits = 128, 3
        embs = _random_embeddings(n=500, dim=dim, seed=10)
        data = _make_rotated_data(n=500, dim=dim, bits=bits, seed=42)

        trainer = CodebookTrainer(bits=bits, max_iterations=30)
        trainer.fit(data)
        lq = LearnedQuantizer(dim=dim, bits=bits, codebook=trainer, seed=42)

        emb = embs[0]
        comp = lq.compress_embedding(emb)
        recon = lq.decompress_embedding(comp)
        assert recon.shape == emb.shape
        assert recon.dtype == np.float32

    def test_better_than_default(self) -> None:
        """Learned quantizer has >= cosine similarity than default on training data."""
        dim, bits = 128, 3
        seed = 42
        embs = _random_embeddings(n=1000, dim=dim, seed=77)

        # Fit codebook on the same data
        lq = fit_codebook(embs, dim=dim, bits=bits, seed=seed, n_train=1000)

        # Default quantizer
        tq = TurboQuantPGVector(dim=dim, bits=bits, seed=seed)

        # Compare average cosine similarity on a test subset
        test_embs = embs[:50]
        learned_sims = []
        default_sims = []
        for emb in test_embs:
            # Learned
            comp_l = lq.compress_embedding(emb)
            recon_l = lq.decompress_embedding(comp_l)
            learned_sims.append(_cosine_similarity(emb, recon_l))

            # Default
            comp_d = tq.compress_embedding(emb)
            recon_d = tq.decompress_embedding(comp_d)
            default_sims.append(_cosine_similarity(emb, recon_d))

        avg_learned = np.mean(learned_sims)
        avg_default = np.mean(default_sims)

        assert avg_learned >= avg_default - 0.001, (
            f"Learned avg cosine sim ({avg_learned:.4f}) should be >= "
            f"default ({avg_default:.4f})"
        )

    def test_compress_batch(self) -> None:
        """Batch compression works and matches single compression."""
        dim, bits = 128, 3
        embs = _random_embeddings(n=500, dim=dim, seed=10)
        data = _make_rotated_data(n=500, dim=dim, bits=bits, seed=42)

        trainer = CodebookTrainer(bits=bits, max_iterations=30)
        trainer.fit(data)
        lq = LearnedQuantizer(dim=dim, bits=bits, codebook=trainer, seed=42)

        batch = embs[:10]
        batch_results = lq.compress_batch(batch)
        assert len(batch_results) == 10

        for i in range(10):
            single = lq.compress_embedding(batch[i])
            assert batch_results[i].packed_bytes == single.packed_bytes
            assert abs(batch_results[i].norm - single.norm) < 1e-5

    def test_decompress_batch(self) -> None:
        """Batch decompression returns correct shape."""
        dim, bits = 128, 3
        embs = _random_embeddings(n=500, dim=dim, seed=10)
        data = _make_rotated_data(n=500, dim=dim, bits=bits, seed=42)

        trainer = CodebookTrainer(bits=bits, max_iterations=30)
        trainer.fit(data)
        lq = LearnedQuantizer(dim=dim, bits=bits, codebook=trainer, seed=42)

        compressed = lq.compress_batch(embs[:20])
        decompressed = lq.decompress_batch(compressed)
        assert decompressed.shape == (20, dim)
        assert decompressed.dtype == np.float32

    def test_unfitted_codebook_raises(self) -> None:
        """Using an unfitted codebook raises RuntimeError."""
        trainer = CodebookTrainer(bits=3)
        with pytest.raises(RuntimeError, match="must be fitted"):
            LearnedQuantizer(dim=128, bits=3, codebook=trainer)

    def test_bits_mismatch_raises(self) -> None:
        """Mismatched bits raises ValueError."""
        data = _make_rotated_data(n=500, dim=64, bits=3)
        trainer = CodebookTrainer(bits=3, max_iterations=10)
        trainer.fit(data)
        with pytest.raises(ValueError, match="does not match"):
            LearnedQuantizer(dim=64, bits=4, codebook=trainer)


# ------------------------------------------------------------------ #
# fit_codebook convenience function tests                             #
# ------------------------------------------------------------------ #


class TestFitCodebook:
    """Tests for the fit_codebook convenience function."""

    def test_convenience_function(self) -> None:
        """fit_codebook returns a LearnedQuantizer."""
        embs = _random_embeddings(n=200, dim=128, seed=42)
        lq = fit_codebook(embs, dim=128, bits=3, seed=42)
        assert isinstance(lq, LearnedQuantizer)
        assert lq.dim == 128
        assert lq.bits == 3

    def test_quality_improvement(self) -> None:
        """fit_codebook produces better or equal cosine sim than default."""
        dim, bits, seed = 128, 3, 42
        embs = _random_embeddings(n=2000, dim=dim, seed=99)

        lq = fit_codebook(embs, dim=dim, bits=bits, seed=seed, n_train=2000)
        tq = TurboQuantPGVector(dim=dim, bits=bits, seed=seed)

        # Evaluate on held-out embeddings
        test_embs = _random_embeddings(n=50, dim=dim, seed=777)

        learned_sims = []
        default_sims = []
        for emb in test_embs:
            comp_l = lq.compress_embedding(emb)
            recon_l = lq.decompress_embedding(comp_l)
            learned_sims.append(_cosine_similarity(emb, recon_l))

            comp_d = tq.compress_embedding(emb)
            recon_d = tq.decompress_embedding(comp_d)
            default_sims.append(_cosine_similarity(emb, recon_d))

        avg_learned = np.mean(learned_sims)
        avg_default = np.mean(default_sims)

        # Learned should be at least as good (allow tiny tolerance)
        assert avg_learned >= avg_default - 0.002, (
            f"Learned avg cosine ({avg_learned:.4f}) should be >= "
            f"default ({avg_default:.4f})"
        )

    def test_infer_dim(self) -> None:
        """fit_codebook infers dim from embeddings when not provided."""
        embs = _random_embeddings(n=100, dim=64, seed=42)
        lq = fit_codebook(embs, bits=3, seed=42)
        assert lq.dim == 64

    def test_subsampling(self) -> None:
        """fit_codebook subsamples when n > n_train."""
        embs = _random_embeddings(n=500, dim=64, seed=42)
        # n_train=100 should subsample
        lq = fit_codebook(embs, bits=3, seed=42, n_train=100)
        assert isinstance(lq, LearnedQuantizer)

    def test_1d_input(self) -> None:
        """fit_codebook handles 1-D input (single embedding)."""
        emb = _random_embeddings(n=1, dim=64, seed=42)[0]
        lq = fit_codebook(emb, dim=64, bits=3, seed=42)
        assert isinstance(lq, LearnedQuantizer)

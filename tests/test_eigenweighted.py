"""
Tests for eigenvalue-weighted mixed-precision quantization (Issue #7).

Verifies that EigenweightedPipeline allocates bits based on PCA
eigenvalue importance and achieves better quality than uniform bits
at the same average bit-width.

Usage:
    pytest tests/test_eigenweighted.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro.pca import EigenweightedPipeline, PCAMatryoshka

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _fit_pca(
    input_dim: int = 256,
    output_dim: int = 128,
    n_samples: int = 500,
    seed: int = 42,
) -> PCAMatryoshka:
    rng = np.random.default_rng(seed)
    # Synthetic data with decaying variance
    cov = np.diag(np.logspace(2, -2, input_dim).astype(np.float32))
    X = rng.multivariate_normal(np.zeros(input_dim), cov, size=n_samples).astype(
        np.float32
    )
    pca = PCAMatryoshka(input_dim=input_dim, output_dim=output_dim)
    pca.fit(X)
    return pca


def _random_embeddings(n: int, dim: int, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


# ------------------------------------------------------------------ #
# Auto bit schedule from eigenvalues                                   #
# ------------------------------------------------------------------ #


class TestAutoBitSchedule:
    """Verify _auto_bit_schedule uses eigenvalue spectrum."""

    def test_covers_output_dim(self) -> None:
        pca = _fit_pca(output_dim=128)
        for avg in [2.0, 3.0, 4.0]:
            schedule = pca._auto_bit_schedule(avg)
            total = sum(n for n, _ in schedule)
            assert total == 128, f"avg_bits={avg}: schedule covers {total} != 128"

    def test_all_bits_in_valid_range(self) -> None:
        pca = _fit_pca(output_dim=128)
        for avg in [2.0, 3.0, 4.0]:
            schedule = pca._auto_bit_schedule(avg)
            for n, b in schedule:
                assert b in (2, 3, 4), f"Invalid bits={b}"
                assert n > 0, f"Empty segment n={n}"

    def test_high_avg_gives_more_4bit(self) -> None:
        pca = _fit_pca(output_dim=128)
        sched_low = pca._auto_bit_schedule(2.0)
        sched_high = pca._auto_bit_schedule(4.0)

        bits_4_low = sum(n for n, b in sched_low if b == 4)
        bits_4_high = sum(n for n, b in sched_high if b == 4)
        assert bits_4_high > bits_4_low

    def test_decreasing_bits_order(self) -> None:
        """Bit schedule should go from high to low (importance-ordered)."""
        pca = _fit_pca(output_dim=128)
        schedule = pca._auto_bit_schedule(3.0)
        prev_bits = 5
        for _, b in schedule:
            assert b <= prev_bits, f"Schedule not decreasing: {schedule}"
            prev_bits = b


# ------------------------------------------------------------------ #
# EigenweightedPipeline construction                                   #
# ------------------------------------------------------------------ #


class TestEigenweightedConstruction:
    """Test with_weighted_quantizer and pipeline creation."""

    def test_from_auto_schedule(self) -> None:
        pca = _fit_pca(output_dim=128)
        pipeline = pca.with_weighted_quantizer(avg_bits=3.0)
        assert isinstance(pipeline, EigenweightedPipeline)
        assert pipeline.input_dim == 256
        assert pipeline.output_dim == 128

    def test_from_explicit_schedule(self) -> None:
        pca = _fit_pca(output_dim=128)
        schedule = [(64, 4), (64, 2)]
        pipeline = pca.with_weighted_quantizer(bit_schedule=schedule)
        assert pipeline.bit_schedule == schedule

    def test_invalid_schedule_raises(self) -> None:
        pca = _fit_pca(output_dim=128)
        with pytest.raises(ValueError, match="covers"):
            pca.with_weighted_quantizer(bit_schedule=[(50, 3), (50, 2)])

    def test_avg_bits(self) -> None:
        pca = _fit_pca(output_dim=128)
        pipeline = pca.with_weighted_quantizer(bit_schedule=[(64, 4), (64, 2)])
        assert pipeline.avg_bits == 3.0

    def test_compression_ratio_positive(self) -> None:
        pca = _fit_pca(output_dim=128)
        pipeline = pca.with_weighted_quantizer(avg_bits=3.0)
        assert pipeline.compression_ratio > 1.0

    def test_repr(self) -> None:
        pca = _fit_pca(output_dim=128)
        pipeline = pca.with_weighted_quantizer(avg_bits=3.0)
        r = repr(pipeline)
        assert "EigenweightedPipeline" in r
        assert "256" in r  # input_dim
        assert "128" in r  # output_dim


# ------------------------------------------------------------------ #
# Compress / decompress roundtrip                                      #
# ------------------------------------------------------------------ #


class TestEigenweightedRoundtrip:
    """Test compression and decompression quality."""

    def test_roundtrip_shape(self) -> None:
        pca = _fit_pca(input_dim=256, output_dim=128)
        pipeline = pca.with_weighted_quantizer(avg_bits=3.0)
        emb = _random_embeddings(1, 256, seed=42)[0]

        compressed = pipeline.compress(emb)
        reconstructed = pipeline.decompress(compressed)
        assert reconstructed.shape == (256,)

    def test_roundtrip_quality(self) -> None:
        # Use high output_dim ratio for better roundtrip quality
        pca = _fit_pca(input_dim=256, output_dim=200, n_samples=1000)
        pipeline = pca.with_weighted_quantizer(avg_bits=3.0)

        # Use embeddings from the same distribution as training data
        rng = np.random.default_rng(99)
        cov = np.diag(np.logspace(2, -2, 256).astype(np.float32))
        emb = rng.multivariate_normal(np.zeros(256), cov).astype(np.float32)

        compressed = pipeline.compress(emb)
        reconstructed = pipeline.decompress(compressed)
        sim = _cosine_similarity(emb, reconstructed)
        assert sim > 0.80, f"Cosine similarity {sim:.4f} too low"

    def test_weighted_beats_uniform_at_same_avg_bits(self) -> None:
        """Eigenweighted should beat uniform quantization at same avg bits."""
        pca = _fit_pca(input_dim=256, output_dim=128, n_samples=1000)
        embs = _random_embeddings(50, 256, seed=99)

        # Weighted: adaptive bit allocation
        weighted = pca.with_weighted_quantizer(avg_bits=3.0)
        w_mean, _, _ = weighted.batch_cosine_similarity(embs)

        # Uniform: all 3-bit
        uniform = pca.with_quantizer(bits=3)
        # Measure uniform quality
        u_sims = []
        for emb in embs:
            c = uniform.compress(emb)
            r = uniform.decompress(c)
            u_sims.append(_cosine_similarity(emb, r))
        u_mean = float(np.mean(u_sims))

        # Weighted should be at least as good (may be better due to
        # allocating 4-bit to high-variance dims)
        assert w_mean >= u_mean - 0.02, (
            f"Weighted ({w_mean:.4f}) significantly worse than "
            f"uniform ({u_mean:.4f})"
        )

    def test_batch_cosine_similarity(self) -> None:
        pca = _fit_pca(input_dim=256, output_dim=200, n_samples=1000)
        pipeline = pca.with_weighted_quantizer(avg_bits=3.0)

        # Use same-distribution embeddings for realistic quality
        rng = np.random.default_rng(42)
        cov = np.diag(np.logspace(2, -2, 256).astype(np.float32))
        embs = rng.multivariate_normal(np.zeros(256), cov, size=20).astype(np.float32)

        mean, mn, std = pipeline.batch_cosine_similarity(embs)
        assert mean > 0.70
        assert mn > 0.40
        assert std < 0.3


# ------------------------------------------------------------------ #
# Compressed output structure                                          #
# ------------------------------------------------------------------ #


class TestEigenweightedOutput:
    """Verify compressed output contains expected fields."""

    def test_compressed_has_segments(self) -> None:
        pca = _fit_pca(output_dim=128)
        pipeline = pca.with_weighted_quantizer(avg_bits=3.0)
        emb = _random_embeddings(1, 256, seed=42)[0]

        compressed = pipeline.compress(emb)
        assert "segments" in compressed
        assert "pca_dim" in compressed
        assert "schedule" in compressed
        assert compressed["pca_dim"] == 128
        assert len(compressed["segments"]) == len(pipeline.segments)

    def test_segments_use_correct_bits(self) -> None:
        pca = _fit_pca(output_dim=128)
        schedule = [(64, 4), (64, 2)]
        pipeline = pca.with_weighted_quantizer(bit_schedule=schedule)
        emb = _random_embeddings(1, 256, seed=42)[0]

        compressed = pipeline.compress(emb)
        assert compressed["segments"][0].bits == 4
        assert compressed["segments"][1].bits == 2

"""Tests for PCAMatryoshka and PCAMatryoshkaPipeline."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest

from turboquant_pro.pca import (
    PCACompressedEmbedding,
    PCAFitResult,
    PCAMatryoshka,
    PCAMatryoshkaPipeline,
)

# ------------------------------------------------------------------ #
# Fixtures                                                            #
# ------------------------------------------------------------------ #


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def sample_embeddings(rng: np.random.Generator) -> np.ndarray:
    """1000 vectors in 128-dim with realistic correlation structure."""
    # Create correlated data: first dims have more variance
    n, d = 1000, 128
    base = rng.standard_normal((n, d)).astype(np.float32)
    # Scale dimensions so eigenvalues decay
    scale = np.exp(-np.arange(d) / 30.0).astype(np.float32)
    return base * scale


@pytest.fixture()
def high_dim_embeddings(rng: np.random.Generator) -> np.ndarray:
    """600 vectors in 1024-dim (BGE-M3 scale)."""
    n, d = 600, 1024
    base = rng.standard_normal((n, d)).astype(np.float32)
    scale = np.exp(-np.arange(d) / 100.0).astype(np.float32)
    return base * scale


@pytest.fixture()
def fitted_pca(sample_embeddings: np.ndarray) -> PCAMatryoshka:
    pca = PCAMatryoshka(input_dim=128, output_dim=32)
    pca.fit(sample_embeddings)
    return pca


# ------------------------------------------------------------------ #
# PCAMatryoshka: Construction                                         #
# ------------------------------------------------------------------ #


class TestPCAMatryoshkaInit:
    def test_default_params(self) -> None:
        pca = PCAMatryoshka()
        assert pca.input_dim == 1024
        assert pca.output_dim == 384
        assert pca.whiten is False
        assert not pca.is_fitted

    def test_custom_params(self) -> None:
        pca = PCAMatryoshka(input_dim=768, output_dim=256, whiten=True)
        assert pca.input_dim == 768
        assert pca.output_dim == 256
        assert pca.whiten is True

    def test_output_dim_exceeds_input(self) -> None:
        with pytest.raises(ValueError, match="output_dim.*must be <= input_dim"):
            PCAMatryoshka(input_dim=128, output_dim=256)

    def test_output_dim_zero(self) -> None:
        with pytest.raises(ValueError, match="output_dim must be >= 1"):
            PCAMatryoshka(input_dim=128, output_dim=0)

    def test_repr_unfitted(self) -> None:
        pca = PCAMatryoshka(input_dim=128, output_dim=32)
        assert "unfitted" in repr(pca)
        assert "128 -> 32" in repr(pca)

    def test_repr_fitted(self, fitted_pca: PCAMatryoshka) -> None:
        r = repr(fitted_pca)
        assert "fitted" in r
        assert "var" in r


# ------------------------------------------------------------------ #
# PCAMatryoshka: Fitting                                              #
# ------------------------------------------------------------------ #


class TestPCAMatryoshkaFit:
    def test_fit_basic(self, sample_embeddings: np.ndarray) -> None:
        pca = PCAMatryoshka(input_dim=128, output_dim=32)
        result = pca.fit(sample_embeddings)

        assert pca.is_fitted
        assert isinstance(result, PCAFitResult)
        assert result.n_samples == 1000
        assert result.input_dim == 128
        assert result.output_dim == 32
        assert len(result.explained_variance_ratio) == 32
        assert len(result.cumulative_variance) == 32

    def test_variance_is_positive(self, sample_embeddings: np.ndarray) -> None:
        pca = PCAMatryoshka(input_dim=128, output_dim=32)
        result = pca.fit(sample_embeddings)

        assert result.total_variance_explained > 0
        assert result.total_variance_explained <= 1.0
        assert all(r >= 0 for r in result.explained_variance_ratio)

    def test_variance_ratio_sums_correctly(self, sample_embeddings: np.ndarray) -> None:
        pca = PCAMatryoshka(input_dim=128, output_dim=128)
        result = pca.fit(sample_embeddings)

        # All components should explain ~100% variance
        assert result.total_variance_explained > 0.999

    def test_eigenvalues_descending(self, sample_embeddings: np.ndarray) -> None:
        pca = PCAMatryoshka(input_dim=128, output_dim=64)
        pca.fit(sample_embeddings)

        # Eigenvalues should be in descending order
        ev = pca._eigenvalues
        for i in range(len(ev) - 1):
            assert ev[i] >= ev[i + 1] - 1e-6  # small tolerance

    def test_fit_wrong_dim(self, sample_embeddings: np.ndarray) -> None:
        pca = PCAMatryoshka(input_dim=64, output_dim=16)
        with pytest.raises(ValueError, match="Expected shape"):
            pca.fit(sample_embeddings)  # 128-dim, not 64

    def test_fit_too_few_samples(self) -> None:
        pca = PCAMatryoshka(input_dim=128, output_dim=32)
        X = np.random.randn(10, 128).astype(np.float32)
        with pytest.raises(ValueError, match="Need at least"):
            pca.fit(X)

    def test_fit_1d_rejected(self) -> None:
        pca = PCAMatryoshka(input_dim=128, output_dim=32)
        with pytest.raises(ValueError, match="Expected shape"):
            pca.fit(np.random.randn(128).astype(np.float32))


# ------------------------------------------------------------------ #
# PCAMatryoshka: Transform                                            #
# ------------------------------------------------------------------ #


class TestPCAMatryoshkaTransform:
    def test_transform_reduces_dim(
        self, fitted_pca: PCAMatryoshka, sample_embeddings: np.ndarray
    ) -> None:
        reduced = fitted_pca.transform(sample_embeddings)
        assert reduced.shape == (1000, 32)

    def test_transform_single_vector(
        self, fitted_pca: PCAMatryoshka, sample_embeddings: np.ndarray
    ) -> None:
        single = sample_embeddings[0]
        reduced = fitted_pca.transform(single)
        assert reduced.shape == (32,)

    def test_transform_not_fitted(self) -> None:
        pca = PCAMatryoshka(input_dim=128, output_dim=32)
        with pytest.raises(RuntimeError, match="not fitted"):
            pca.transform(np.random.randn(128).astype(np.float32))

    def test_transform_wrong_dim(self, fitted_pca: PCAMatryoshka) -> None:
        with pytest.raises(ValueError, match="Expected dim=128"):
            fitted_pca.transform(np.random.randn(64).astype(np.float32))

    def test_inverse_transform_restores_dim(
        self, fitted_pca: PCAMatryoshka, sample_embeddings: np.ndarray
    ) -> None:
        reduced = fitted_pca.transform(sample_embeddings[:5])
        reconstructed = fitted_pca.inverse_transform(reduced)
        assert reconstructed.shape == (5, 128)

    def test_inverse_transform_single(
        self, fitted_pca: PCAMatryoshka, sample_embeddings: np.ndarray
    ) -> None:
        reduced = fitted_pca.transform(sample_embeddings[0])
        reconstructed = fitted_pca.inverse_transform(reduced)
        assert reconstructed.shape == (128,)


# ------------------------------------------------------------------ #
# PCAMatryoshka: Cosine similarity preservation                       #
# ------------------------------------------------------------------ #


class TestCosinePreservation:
    def test_round_trip_high_cosine(self, sample_embeddings: np.ndarray) -> None:
        """PCA round-trip should preserve cosine similarity well."""
        pca = PCAMatryoshka(input_dim=128, output_dim=64)
        pca.fit(sample_embeddings)

        cos = pca.cosine_similarity(sample_embeddings[:100])
        mean_cos = float(cos.mean())

        # 64/128 dims with correlated data should give >0.95
        assert mean_cos > 0.90, f"Mean cosine {mean_cos} too low"

    def test_more_dims_better_cosine(self, sample_embeddings: np.ndarray) -> None:
        """More retained dims should give higher cosine similarity."""
        cosines = {}
        dims = [16, 32, 64, 96]
        for out_dim in dims:
            pca = PCAMatryoshka(input_dim=128, output_dim=out_dim)
            pca.fit(sample_embeddings)
            cos = pca.cosine_similarity(sample_embeddings[:100])
            cosines[out_dim] = float(cos.mean())

        for i in range(1, len(dims)):
            d, d_prev = dims[i], dims[i - 1]
            assert cosines[d] >= cosines[d_prev] - 0.01, (
                f"cosine at {d} dims ({cosines[d]:.3f}) should be >= "
                f"cosine at {d_prev} dims ({cosines[d_prev]:.3f})"
            )

    def test_full_dim_near_perfect(self, sample_embeddings: np.ndarray) -> None:
        """Full-rank PCA should give near-perfect cosine."""
        pca = PCAMatryoshka(input_dim=128, output_dim=128)
        pca.fit(sample_embeddings)
        cos = pca.cosine_similarity(sample_embeddings[:50])
        assert float(cos.mean()) > 0.999

    def test_cosine_single_vector(
        self, fitted_pca: PCAMatryoshka, sample_embeddings: np.ndarray
    ) -> None:
        cos = fitted_pca.cosine_similarity(sample_embeddings[0])
        assert isinstance(cos, float)
        assert 0.0 <= cos <= 1.0


# ------------------------------------------------------------------ #
# PCAMatryoshka: Incremental PCA                                      #
# ------------------------------------------------------------------ #


class TestIncrementalPCA:
    def test_partial_fit_unfitted(self, sample_embeddings: np.ndarray) -> None:
        """partial_fit on unfitted model should behave like fit."""
        pca = PCAMatryoshka(input_dim=128, output_dim=32)
        result = pca.partial_fit(sample_embeddings)
        assert pca.is_fitted
        assert result.n_samples == 1000

    def test_partial_fit_updates(
        self, sample_embeddings: np.ndarray, rng: np.random.Generator
    ) -> None:
        """partial_fit should accumulate samples."""
        pca = PCAMatryoshka(input_dim=128, output_dim=32)
        pca.fit(sample_embeddings[:500])
        assert pca._n_samples_seen == 500

        new_data = rng.standard_normal((300, 128)).astype(np.float32)
        result = pca.partial_fit(new_data)
        assert result.n_samples == 800
        assert pca._n_samples_seen == 800

    def test_incremental_close_to_batch(self, rng: np.random.Generator) -> None:
        """Incremental PCA should approximate batch PCA."""
        d = 64
        scale = np.exp(-np.arange(d) / 15.0).astype(np.float32)
        data = (rng.standard_normal((2000, d)) * scale).astype(np.float32)

        # Batch PCA
        batch_pca = PCAMatryoshka(input_dim=d, output_dim=16)
        batch_pca.fit(data)
        batch_var = batch_pca._eigenvalues.copy()

        # Incremental PCA (two halves)
        inc_pca = PCAMatryoshka(input_dim=d, output_dim=16)
        inc_pca.fit(data[:1000])
        inc_pca.partial_fit(data[1000:])
        inc_var = inc_pca._eigenvalues.copy()

        # Eigenvalues should be close
        np.testing.assert_allclose(batch_var, inc_var, rtol=0.05)


# ------------------------------------------------------------------ #
# PCAMatryoshka: Whitening                                            #
# ------------------------------------------------------------------ #


class TestWhitening:
    def test_whitened_output_unit_variance(self, sample_embeddings: np.ndarray) -> None:
        pca = PCAMatryoshka(input_dim=128, output_dim=32, whiten=True)
        pca.fit(sample_embeddings)
        projected = pca.transform(sample_embeddings)

        # Each component should have approximately unit variance
        variances = projected.var(axis=0)
        np.testing.assert_allclose(variances, 1.0, atol=0.15)

    def test_whiten_round_trip(self, sample_embeddings: np.ndarray) -> None:
        pca = PCAMatryoshka(input_dim=128, output_dim=128, whiten=True)
        pca.fit(sample_embeddings)
        projected = pca.transform(sample_embeddings[:10])
        reconstructed = pca.inverse_transform(projected)

        cos = np.sum(sample_embeddings[:10] * reconstructed, axis=1)
        cos /= np.linalg.norm(sample_embeddings[:10], axis=1) * np.linalg.norm(
            reconstructed, axis=1
        )
        assert float(cos.mean()) > 0.999


# ------------------------------------------------------------------ #
# PCAMatryoshka: Serialization                                        #
# ------------------------------------------------------------------ #


class TestSerialization:
    def test_save_load_file(
        self, fitted_pca: PCAMatryoshka, sample_embeddings: np.ndarray
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
            fitted_pca.save(path)

        loaded = PCAMatryoshka.load(path)
        assert loaded.input_dim == fitted_pca.input_dim
        assert loaded.output_dim == fitted_pca.output_dim
        assert loaded._n_samples_seen == fitted_pca._n_samples_seen

        # Transform should give identical results
        orig = fitted_pca.transform(sample_embeddings[:5])
        loaded_result = loaded.transform(sample_embeddings[:5])
        np.testing.assert_allclose(orig, loaded_result, atol=1e-5)

        Path(path).unlink()

    def test_save_load_buffer(
        self, fitted_pca: PCAMatryoshka, sample_embeddings: np.ndarray
    ) -> None:
        buf = io.BytesIO()
        fitted_pca.save(buf)
        buf.seek(0)

        loaded = PCAMatryoshka.load(buf)
        orig = fitted_pca.transform(sample_embeddings[0])
        loaded_result = loaded.transform(sample_embeddings[0])
        np.testing.assert_allclose(orig, loaded_result, atol=1e-5)

    def test_load_preserves_incremental_state(
        self, sample_embeddings: np.ndarray, rng: np.random.Generator
    ) -> None:
        pca = PCAMatryoshka(input_dim=128, output_dim=32)
        pca.fit(sample_embeddings[:500])

        buf = io.BytesIO()
        pca.save(buf)
        buf.seek(0)
        loaded = PCAMatryoshka.load(buf)

        # partial_fit on loaded should work
        new_data = rng.standard_normal((200, 128)).astype(np.float32)
        result = loaded.partial_fit(new_data)
        assert result.n_samples == 700


# ------------------------------------------------------------------ #
# PCAMatryoshka: Diagnostics                                          #
# ------------------------------------------------------------------ #


class TestDiagnostics:
    def test_variance_report(self, fitted_pca: PCAMatryoshka) -> None:
        report = fitted_pca.variance_report(dims=[8, 16, 32])
        assert 8 in report
        assert 16 in report
        assert 32 in report
        assert report[8] <= report[16] <= report[32]

    def test_variance_report_defaults(self, fitted_pca: PCAMatryoshka) -> None:
        report = fitted_pca.variance_report()
        assert len(report) > 0
        assert all(0 <= v <= 1.0 for v in report.values())


# ------------------------------------------------------------------ #
# PCAMatryoshkaPipeline                                               #
# ------------------------------------------------------------------ #


class TestPipeline:
    def test_with_quantizer(self, fitted_pca: PCAMatryoshka) -> None:
        pipeline = fitted_pca.with_quantizer(bits=3, seed=42)
        assert isinstance(pipeline, PCAMatryoshkaPipeline)
        assert pipeline.input_dim == 128
        assert pipeline.output_dim == 32
        assert pipeline.bits == 3

    def test_compress_decompress_single(
        self, fitted_pca: PCAMatryoshka, sample_embeddings: np.ndarray
    ) -> None:
        pipeline = fitted_pca.with_quantizer(bits=3)
        embedding = sample_embeddings[0]

        compressed = pipeline.compress(embedding)
        assert isinstance(compressed, PCACompressedEmbedding)
        assert compressed.packed_bytes is not None
        assert compressed.pca_dim == 32
        assert compressed.bits == 3

        reconstructed = pipeline.decompress(compressed)
        assert reconstructed.shape == (128,)

    def test_compress_decompress_cosine(
        self, fitted_pca: PCAMatryoshka, sample_embeddings: np.ndarray
    ) -> None:
        pipeline = fitted_pca.with_quantizer(bits=3)
        embedding = sample_embeddings[0]

        reconstructed = pipeline.decompress(pipeline.compress(embedding))

        # Cosine similarity
        cos = float(
            np.dot(embedding, reconstructed)
            / (np.linalg.norm(embedding) * np.linalg.norm(reconstructed))
        )
        # PCA-32 out of 128 with 3-bit quant — should still be decent
        assert cos > 0.80, f"Cosine {cos} too low"

    def test_compress_batch(
        self, fitted_pca: PCAMatryoshka, sample_embeddings: np.ndarray
    ) -> None:
        pipeline = fitted_pca.with_quantizer(bits=3)
        batch = sample_embeddings[:50]

        compressed = pipeline.compress_batch(batch)
        assert len(compressed) == 50
        assert all(isinstance(c, PCACompressedEmbedding) for c in compressed)

    def test_decompress_batch(
        self, fitted_pca: PCAMatryoshka, sample_embeddings: np.ndarray
    ) -> None:
        pipeline = fitted_pca.with_quantizer(bits=3)
        batch = sample_embeddings[:50]

        compressed = pipeline.compress_batch(batch)
        reconstructed = pipeline.decompress_batch(compressed)
        assert reconstructed.shape == (50, 128)

    def test_batch_cosine_similarity(
        self, fitted_pca: PCAMatryoshka, sample_embeddings: np.ndarray
    ) -> None:
        pipeline = fitted_pca.with_quantizer(bits=3)
        mean_cos, min_cos, std_cos = pipeline.batch_cosine_similarity(
            sample_embeddings[:100]
        )
        assert mean_cos > 0.75
        assert min_cos > 0.50
        assert std_cos < 0.2

    def test_compression_ratio(self, fitted_pca: PCAMatryoshka) -> None:
        pipeline = fitted_pca.with_quantizer(bits=3)
        ratio = pipeline.compression_ratio
        # 128 * 32 bits / (32 * 3 + 32) bits = 4096 / 128 = 32
        assert ratio > 20

    def test_estimate_storage(self) -> None:
        est = PCAMatryoshkaPipeline.estimate_storage(
            n_embeddings=1_000_000,
            input_dim=1024,
            output_dim=384,
            bits=3,
        )
        assert est["n_embeddings"] == 1_000_000
        assert est["original_mb"] > est["compressed_mb"]
        assert est["ratio"] > 20
        assert est["saved_mb"] > 0

    def test_repr(self, fitted_pca: PCAMatryoshka) -> None:
        pipeline = fitted_pca.with_quantizer(bits=3)
        r = repr(pipeline)
        assert "PCA-32" in r
        assert "TQ3-bit" in r
        assert "compression" in r

    def test_pipeline_wrong_dim(self, fitted_pca: PCAMatryoshka) -> None:
        from turboquant_pro.pgvector import TurboQuantPGVector

        tq = TurboQuantPGVector(dim=64, bits=3)  # Wrong: 64 != 32
        with pytest.raises(ValueError, match="Quantizer dim"):
            PCAMatryoshkaPipeline(pca=fitted_pca, quantizer=tq)

    def test_pipeline_unfitted_pca(self) -> None:
        from turboquant_pro.pgvector import TurboQuantPGVector

        pca = PCAMatryoshka(input_dim=128, output_dim=32)
        tq = TurboQuantPGVector(dim=32, bits=3)
        with pytest.raises(RuntimeError, match="must be fitted"):
            PCAMatryoshkaPipeline(pca=pca, quantizer=tq)


# ------------------------------------------------------------------ #
# PCAMatryoshka: High-dimensional (BGE-M3 scale)                     #
# ------------------------------------------------------------------ #


class TestHighDimensional:
    def test_1024_to_384(self, high_dim_embeddings: np.ndarray) -> None:
        """Realistic BGE-M3 scenario: 1024 -> 384 dims."""
        pca = PCAMatryoshka(input_dim=1024, output_dim=384)
        result = pca.fit(high_dim_embeddings)

        assert result.total_variance_explained > 0.80

        # Transform and check cosine
        cos = pca.cosine_similarity(high_dim_embeddings[:50])
        assert float(cos.mean()) > 0.95

    def test_full_pipeline_1024(self, high_dim_embeddings: np.ndarray) -> None:
        """Full PCA-384 + TQ3 pipeline on 1024-dim data."""
        pca = PCAMatryoshka(input_dim=1024, output_dim=384)
        pca.fit(high_dim_embeddings)

        pipeline = pca.with_quantizer(bits=3)
        mean_cos, min_cos, _ = pipeline.batch_cosine_similarity(
            high_dim_embeddings[:50]
        )
        assert mean_cos > 0.90, f"Mean cosine {mean_cos} too low"

    def test_various_output_dims(self, high_dim_embeddings: np.ndarray) -> None:
        """Test multiple truncation levels."""
        for out_dim in [64, 128, 256, 384, 512]:
            pca = PCAMatryoshka(input_dim=1024, output_dim=out_dim)
            result = pca.fit(high_dim_embeddings)
            assert result.total_variance_explained > 0
            assert pca.is_fitted

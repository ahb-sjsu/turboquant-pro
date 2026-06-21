"""Tests for PCAMatryoshka.suggest_output_dim (variance-aware truncation)."""

import numpy as np
import pytest

from turboquant_pro import PCAMatryoshka


def test_concentrated_spectrum_truncates():
    # rank-10 signal embedded in 100-d -> ~10 dims carry ~all variance
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((5000, 10))
    W = rng.standard_normal((10, 100))
    X = (Z @ W).astype(np.float32)
    X += 0.001 * rng.standard_normal((5000, 100)).astype(np.float32)
    d = PCAMatryoshka.suggest_output_dim(X, target_variance=0.95)
    assert 8 <= d <= 25  # close to the true rank


def test_isotropic_spectrum_keeps_most_dims():
    # isotropic Gaussian -> no concentration -> need most dims for 95%
    rng = np.random.default_rng(1)
    X = rng.standard_normal((5000, 64)).astype(np.float32)
    d = PCAMatryoshka.suggest_output_dim(X, target_variance=0.95)
    assert d >= 55  # ~95% of 64 roughly-equal dims


def test_monotone_in_target():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((3000, 50)).astype(np.float32)
    assert PCAMatryoshka.suggest_output_dim(
        X, 0.90
    ) <= PCAMatryoshka.suggest_output_dim(X, 0.99)


def test_validation():
    X = np.zeros((10, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        PCAMatryoshka.suggest_output_dim(X, target_variance=1.5)
    with pytest.raises(ValueError):
        PCAMatryoshka.suggest_output_dim(np.zeros(5, dtype=np.float32), 0.95)

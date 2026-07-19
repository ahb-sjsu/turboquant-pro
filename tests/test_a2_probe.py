"""
Tests for the (A2) consumer-metric probe.

The acceptance test is historical: the probe must reproduce the v1.2.0
KV-keys catastrophe on synthetic post-RoPE-like key statistics (large shared
per-channel offsets) -- recommending per-channel for attention logits while
still recommending the polar family for isotropic cosine-ranking workloads.

Usage:
    pytest tests/test_a2_probe.py -v
"""

from __future__ import annotations

import numpy as np

from turboquant_pro.a2_probe import (
    A2ProbeResult,
    displacement_decomposition,
    probe_quotient,
    recommend_key_quantizer,
    tangential_fraction,
)

RNG = np.random.default_rng(42)


def _isotropic_batch(n: int = 256, d: int = 64, seed: int = 42) -> np.ndarray:
    """Direction-dominated data: unit-ish gaussian cloud.

    Self-seeded so batches are identical regardless of test order.
    """
    return np.random.default_rng(seed).standard_normal((n, d))


def _keylike_batch(n: int = 256, d: int = 64, seed: int = 42) -> np.ndarray:
    """Post-RoPE-key-like data: a large shared per-channel DC offset plus a
    small informative residual orthogonal to the offset. This is the
    direction-concentration regime -- every key sits in a tiny cone around
    the offset direction, norms are nearly constant, and the informative
    angular displacement is far below a 4-bit angular quantizer's cell
    size, while per-channel scales resolve the residual easily.

    Self-seeded so batches are identical regardless of test order."""
    rng = np.random.default_rng(seed)
    offset = rng.uniform(-8.0, 8.0, size=d)
    residual = 0.3 * rng.standard_normal((n, d))
    u = offset / np.linalg.norm(offset)
    residual -= np.outer(residual @ u, u)  # informative part is off-axis
    return offset[None, :] + residual


def _correlated_keylike_batch(
    n: int = 512, d: int = 64, r: int = 4, seed: int = 7
) -> np.ndarray:
    """Direction-concentration WITH cross-channel correlation: a large shared
    offset plus a low-rank correlated informative signal. A per-channel
    (diagonal) code cannot decorrelate the signal; ZCA whitening can, so the
    whitened polar family should win here -- where per-channel wins the
    uncorrelated ``_keylike_batch``. Self-seeded for order-independence."""
    rng = np.random.default_rng(seed)
    offset = rng.uniform(-6.0, 6.0, size=d)
    loadings = rng.standard_normal((d, r))  # shared mixing => cross-channel correlation
    factors = rng.standard_normal((n, r))
    signal = factors @ loadings.T
    return offset[None, :] + signal


class TestTangentialFraction:
    """The (A2) quantity itself."""

    def test_pure_rotation_is_one(self) -> None:
        """Equal-norm vectors differing in direction: fraction = 1."""
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        assert abs(tangential_fraction(x, y) - 1.0) < 1e-12

    def test_pure_scaling_is_zero(self) -> None:
        """Same ray, different norms: fraction = 0."""
        x = np.array([1.0, 2.0, 3.0])
        assert abs(tangential_fraction(x, 2.5 * x)) < 1e-12

    def test_range_and_nan(self) -> None:
        """Always within [0, 1]; NaN on coincident vectors."""
        for _ in range(50):
            f = tangential_fraction(RNG.standard_normal(16), RNG.standard_normal(16))
            assert 0.0 <= f <= 1.0
        x = RNG.standard_normal(16)
        assert np.isnan(tangential_fraction(x, x))

    def test_decomposition_separates_regimes(self) -> None:
        """Isotropic data reads tangential; ray data reads radial."""
        iso = displacement_decomposition(_isotropic_batch(), seed=1)
        assert iso["median_tangential_fraction"] > 0.8
        ray = RNG.standard_normal(64)
        ray /= np.linalg.norm(ray)
        scales = RNG.uniform(0.5, 5.0, size=(256, 1))
        radial = displacement_decomposition(scales * ray[None, :], seed=1)
        assert radial["median_tangential_fraction"] < 0.2


class TestProbeQuotient:
    """Calibration-time family selection against the declared consumer."""

    def test_returns_result_dataclass(self) -> None:
        """Probe returns an A2ProbeResult with sane fields."""
        res = probe_quotient(_isotropic_batch(), consumer="cosine")
        assert isinstance(res, A2ProbeResult)
        assert res.recommendation in ("polar", "per_channel")
        assert res.spearman_whitened is None  # opt-in; off by default
        assert 0.0 <= res.median_tangential_fraction <= 1.0
        assert set(res.as_dict().keys()) == {
            "consumer",
            "spearman_polar",
            "spearman_per_channel",
            "spearman_whitened",
            "median_tangential_fraction",
            "median_unit_displacement",
            "recommendation",
            "margin",
        }

    def test_polar_safe_for_isotropic_cosine(self) -> None:
        """Cosine ranking on direction-dominated data: polar preserves it."""
        res = probe_quotient(_isotropic_batch(), consumer="cosine", seed=3)
        assert res.spearman_polar > 0.9

    def test_keys_catastrophe_reproduced(self) -> None:
        """Attention logits on key-like data: polar collapses, per-channel
        holds -- the v1.2.0 incident as a calibration-time verdict."""
        keys = _keylike_batch()
        queries = RNG.standard_normal((32, keys.shape[1]))
        res = probe_quotient(keys, consumer="attention_logits", queries=queries, seed=3)
        assert res.recommendation == "per_channel"
        assert res.spearman_per_channel > 0.9
        assert res.spearman_per_channel - res.spearman_polar > 0.1

    def test_keylike_data_reads_direction_concentrated(self) -> None:
        """Key-like batches are flagged by direction concentration, NOT by
        the tangential fraction: the shared offset cancels in pairwise
        displacements (fraction ~1), while every direction sits in a tiny
        cone (unit displacement near 0). This distinction is the reason the
        end-to-end probe exists alongside the streaming statistic."""
        res = probe_quotient(_keylike_batch(), consumer="attention_logits", seed=3)
        assert res.median_tangential_fraction > 0.8  # blind here, by design
        assert res.median_unit_displacement < 0.4  # the actual flag
        iso = probe_quotient(_isotropic_batch(), consumer="cosine", seed=3)
        assert iso.median_unit_displacement > 1.0

    def test_invalid_consumer_raises(self) -> None:
        """Unknown consumer names are rejected."""
        try:
            probe_quotient(_isotropic_batch(), consumer="bleu")
        except ValueError:
            return
        raise AssertionError("expected ValueError")

    def test_small_batch_raises(self) -> None:
        """Batches below the minimum size are rejected."""
        try:
            probe_quotient(RNG.standard_normal((2, 8)))
        except ValueError:
            return
        raise AssertionError("expected ValueError")

    def test_deterministic(self) -> None:
        """Same seed, same verdict and numbers."""
        batch = _keylike_batch()
        a = probe_quotient(batch, consumer="attention_logits", seed=11)
        b = probe_quotient(batch, consumer="attention_logits", seed=11)
        assert a.as_dict() == b.as_dict()

    def test_whitened_off_by_default(self) -> None:
        """Whitening is opt-in: the field stays None, verdict is two-family."""
        res = probe_quotient(_keylike_batch(), consumer="attention_logits", seed=3)
        assert res.spearman_whitened is None
        assert res.recommendation in ("polar", "per_channel")

    def test_whitened_leaves_two_family_verdict_identical(self) -> None:
        """The dedicated whitened RNG must not perturb the two-family numbers."""
        batch = _correlated_keylike_batch()
        base = probe_quotient(batch, consumer="attention_logits", seed=5)
        withw = probe_quotient(
            batch, consumer="attention_logits", seed=5, include_whitened=True
        )
        assert withw.spearman_polar == base.spearman_polar
        assert withw.spearman_per_channel == base.spearman_per_channel
        assert withw.median_unit_displacement == base.median_unit_displacement

    def test_whitened_wins_on_correlated_nuisance(self) -> None:
        """Cross-channel correlation: ZCA whitening beats the diagonal
        per-channel code and is recommended when included."""
        keys = _correlated_keylike_batch()
        queries = np.random.default_rng(3).standard_normal((32, keys.shape[1]))
        res = probe_quotient(
            keys,
            consumer="attention_logits",
            queries=queries,
            bits=2,
            seed=3,
            include_whitened=True,
        )
        assert res.spearman_whitened is not None
        assert res.recommendation == "whitened"
        assert res.spearman_whitened > res.spearman_per_channel
        assert res.spearman_whitened > res.spearman_polar

    def test_whitened_deterministic(self) -> None:
        """Same seed with whitening on → identical verdict."""
        keys = _correlated_keylike_batch()
        a = probe_quotient(
            keys, consumer="attention_logits", seed=11, include_whitened=True
        )
        b = probe_quotient(
            keys, consumer="attention_logits", seed=11, include_whitened=True
        )
        assert a.as_dict() == b.as_dict()


class TestRecommendKeyQuantizer:
    """The KV-keys wrapper."""

    def test_recommends_per_channel_for_keylike(self) -> None:
        """Key-like statistics with real query traffic get per-channel."""
        keys = _keylike_batch()
        queries = np.random.default_rng(9).standard_normal((32, keys.shape[1]))
        res = recommend_key_quantizer(keys, queries=queries, seed=5)
        assert res.consumer == "attention_logits"
        assert res.recommendation == "per_channel"

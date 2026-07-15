"""
Tests for the distribution-free rank certificate.

The certificate is a theorem, so the tests check the theorem's contract:
the tau floor must never exceed the realized Kendall tau of any
kappa-distorted distance, and vacuity must fire on distance-concentrated
corpora.

Usage:
    pytest tests/test_rank_certificate.py -v
"""

from __future__ import annotations

import numpy as np

from turboquant_pro.rank_certificate import (
    RankCertificate,
    certificate_from_embeddings,
    certify,
    max_certifiable_kappa,
    measure_kappa,
    mu_curve,
    mu_hat,
    pairwise_distances,
)

RNG = np.random.default_rng(42)


def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """O(n^2) reference Kendall tau (no ties assumed)."""
    n = len(a)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    return (conc - disc) / (conc + disc)


class TestKappa:
    """Robust distortion measurement."""

    def test_identity_gives_one(self) -> None:
        """Undistorted distances measure kappa = 1."""
        d = RNG.uniform(0.1, 2.0, size=500)
        assert measure_kappa(d, d) == 1.0

    def test_global_scale_gives_one(self) -> None:
        """A global rescaling is not distortion."""
        d = RNG.uniform(0.1, 2.0, size=500)
        assert abs(measure_kappa(d, 3.7 * d) - 1.0) < 1e-12

    def test_known_two_point_distortion(self) -> None:
        """Half the pairs stretched 2x measures kappa ~ 2."""
        d = RNG.uniform(0.5, 1.5, size=1000)
        approx = d.copy()
        approx[::2] *= 2.0
        k = measure_kappa(d, approx)
        assert 1.8 < k < 2.2

    def test_zero_exact_pairs_excluded(self) -> None:
        """Zero exact distances do not poison the ratio."""
        d = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        approx = np.array([5.0, 5.0, 1.0, 1.0, 1.0])
        assert measure_kappa(d, approx) == 1.0

    def test_too_few_pairs_nan(self) -> None:
        """Fewer than two valid pairs returns NaN."""
        assert np.isnan(measure_kappa(np.array([0.0]), np.array([1.0])))


class TestMuHat:
    """Distance-ratio concentration."""

    def test_kappa_one_is_zero(self) -> None:
        """At kappa = 1 no pair-of-pairs is strictly within ratio."""
        d = RNG.uniform(0.1, 2.0, size=300)
        assert mu_hat(d, 1.0) == 0.0

    def test_huge_kappa_is_one(self) -> None:
        """A kappa exceeding the full range captures every pair-of-pairs."""
        d = RNG.uniform(0.5, 1.5, size=300)
        assert mu_hat(d, 10.0) == 1.0

    def test_matches_bruteforce(self) -> None:
        """The O(P log P) count equals the O(P^2) definition."""
        d = RNG.uniform(0.1, 3.0, size=120)
        kappa = 1.7
        brute = 0
        for i in range(len(d)):
            for j in range(i + 1, len(d)):
                hi, lo = max(d[i], d[j]), min(d[i], d[j])
                if hi / lo < kappa:
                    brute += 1
        total = len(d) * (len(d) - 1) // 2
        assert abs(mu_hat(d, kappa) - brute / total) < 1e-12

    def test_monotone_in_kappa(self) -> None:
        """mu_hat is nondecreasing in kappa."""
        d = RNG.uniform(0.1, 2.0, size=200)
        curve = mu_curve(d, np.linspace(1.0, 5.0, 20))
        assert np.all(np.diff(curve) >= 0)

    def test_concentrated_corpus_saturates(self) -> None:
        """Concentrated distances give mu_hat ~ 1 at small kappa."""
        d = RNG.uniform(0.99, 1.01, size=300)
        assert mu_hat(d, 1.05) > 0.95


class TestCertificate:
    """The floor itself: the theorem's contract."""

    def test_floor_never_exceeds_realized_tau(self) -> None:
        """tau_floor <= realized Kendall tau for a genuinely distorted map."""
        d = RNG.uniform(0.2, 2.0, size=200)
        approx = d * RNG.uniform(1.0, 1.3, size=200)  # kappa ~ 1.3 distortion
        cert = certify(d, approx)
        realized = _kendall_tau(d, approx)
        assert cert.tau_floor <= realized + 1e-9

    def test_undistorted_certifies_perfect(self) -> None:
        """kappa = 1 gives mu_hat = 0 and floors of exactly 1."""
        d = RNG.uniform(0.2, 2.0, size=300)
        cert = certify(d, d)
        assert cert.kappa == 1.0
        assert cert.mu_hat == 0.0
        assert cert.tau_floor == 1.0
        assert cert.spearman_floor == 1.0
        assert not cert.vacuous

    def test_vacuous_on_concentrated_corpus(self) -> None:
        """Distance concentration makes any real distortion uncertifiable."""
        d = RNG.uniform(0.99, 1.01, size=300)
        approx = d * RNG.uniform(1.0, 1.5, size=300)
        cert = certify(d, approx)
        assert cert.vacuous
        assert cert.rerank_required

    def test_as_dict_keys(self) -> None:
        """as_dict exposes the report fields."""
        d = RNG.uniform(0.2, 2.0, size=100)
        cert = certify(d, d)
        assert isinstance(cert, RankCertificate)
        expected = {
            "kappa",
            "mu_hat",
            "tau_floor",
            "spearman_floor",
            "n_pairs",
            "max_certifiable_kappa",
            "vacuous",
        }
        assert set(cert.as_dict().keys()) == expected


class TestMaxCertifiableKappa:
    """The vacuity threshold: 'rerank required beyond kappa*'."""

    def test_inverts_the_bound(self) -> None:
        """mu_hat at the returned kappa sits at the tau = 0 boundary."""
        d = RNG.uniform(0.1, 4.0, size=400)
        k_star = max_certifiable_kappa(d, min_tau=0.0)
        assert mu_hat(d, k_star) <= 0.5 + 1e-6
        assert mu_hat(d, k_star * 1.05) > 0.5 - 0.05

    def test_spread_corpus_tolerates_more(self) -> None:
        """A spread corpus certifies larger distortion than a concentrated one."""
        spread = RNG.uniform(0.05, 5.0, size=300)
        tight = RNG.uniform(0.95, 1.05, size=300)
        assert max_certifiable_kappa(spread) > max_certifiable_kappa(tight)


class TestFromEmbeddings:
    """End-to-end convenience API on embedding matrices."""

    def test_lossless_reconstruction_certifies(self) -> None:
        """Identical embeddings give a non-vacuous perfect certificate."""
        x = RNG.standard_normal((300, 64))
        cert = certificate_from_embeddings(x, x)
        assert cert.kappa == 1.0
        assert cert.tau_floor == 1.0

    def test_light_noise_still_certifies(self) -> None:
        """Mild reconstruction noise on a spread corpus keeps a real floor."""
        # Low-dimensional latent structure -> spread pairwise distances.
        latent = RNG.standard_normal((300, 4))
        lift = RNG.standard_normal((4, 64))
        x = latent @ lift
        recon = x + 0.01 * RNG.standard_normal(x.shape)
        cert = certificate_from_embeddings(x, recon, metric="l2")
        assert not cert.vacuous
        assert cert.tau_floor > 0.5

    def test_shape_mismatch_raises(self) -> None:
        """Mismatched matrices are rejected."""
        x = RNG.standard_normal((10, 8))
        try:
            certificate_from_embeddings(x, x[:5])
        except ValueError:
            return
        raise AssertionError("expected ValueError")

    def test_metrics_agree_with_pairwise(self) -> None:
        """pairwise_distances matches a direct computation."""
        x = RNG.standard_normal((20, 8))
        d = pairwise_distances(x, metric="l2")
        i, j = np.triu_indices(20, k=1)
        direct = np.linalg.norm(x[i] - x[j], axis=1)
        assert np.allclose(d, direct)

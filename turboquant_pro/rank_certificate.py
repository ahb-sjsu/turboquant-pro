# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""
Distribution-free rank certificates for compressed-domain retrieval.

Cosine similarity is a promise; this module ships a *floor*. For any corpus,
measure the distortion ratio kappa of the compressed-domain distance against
the exact distance (the robust 97.5/2.5 percentile protocol), measure the
corpus's distance-ratio concentration mu_hat(kappa) in one pass, and the
following are guaranteed floors on rank agreement between compressed and
exact distances -- with **no distributional assumptions whatsoever**:

    Kendall  tau     >=  1 - 2 * mu_hat(kappa)
    Spearman rho_S   >=  1 - 3 * mu_hat(kappa)      (via Daniels' inequality)

where mu_hat(kappa) is the fraction of unordered pairs-of-pairs whose exact
distances lie within ratio kappa of each other. The argument is elementary:
a rank inversion between two pairs forces their exact distances within ratio
kappa, so the discordant fraction is at most mu_hat(kappa).

Source: A. H. Bond, "Keep the Angle" (v0.8), Proposition `Distortion controls
rank' (https://github.com/ahb-sjsu/the-angular-observer); H. E. Daniels,
J. Royal Stat. Soc. B 12:171-181 (1950).

The certificate's *failure* is as informative as its success: on corpora
whose pairwise distances concentrate (mu_hat -> 1, the high intrinsic
dimension regime), **no finite distortion certifies any rank agreement** --
which is exactly the regime where single-stage retrieval dies and exact
reranking becomes mandatory. ``max_certifiable_kappa`` inverts the bound to
give the largest distortion a corpus can tolerate, so autotune can derive
"rerank required" per corpus instead of offering a fixed oversampling menu.

The floors are worst-case over all kappa-distorted maps and typically sit far
below measured recall (documented in the source paper); a loose floor beats
no floor, and the concentration statistic doubles as an intrinsic-dimension
proxy.

Usage::

    from turboquant_pro.rank_certificate import certificate_from_embeddings

    cert = certificate_from_embeddings(original, reconstructed)
    print(cert.kappa, cert.mu_hat, cert.tau_floor, cert.vacuous)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "RankCertificate",
    "certify",
    "certificate_from_embeddings",
    "measure_kappa",
    "mu_hat",
    "mu_curve",
    "max_certifiable_kappa",
    "pairwise_distances",
]


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------


def measure_kappa(
    exact: np.ndarray,
    approx: np.ndarray,
    lo: float = 2.5,
    hi: float = 97.5,
) -> float:
    """Robust empirical distortion of ``approx`` against ``exact`` distances.

    kappa is the ratio of the ``hi``-th to the ``lo``-th percentile of the
    per-pair ratio ``approx / exact`` -- the percentile-robust bi-Lipschitz
    constant of the source paper's torus protocol. Pairs with zero exact
    distance are excluded.

    Args:
        exact: 1-D array of exact pairwise distances.
        approx: 1-D array of compressed-domain distances, same pairs.
        lo, hi: Percentiles for the robust ratio (defaults 2.5 / 97.5).

    Returns:
        kappa >= 1, or NaN if fewer than 2 valid pairs.
    """
    exact = np.asarray(exact, dtype=np.float64).ravel()
    approx = np.asarray(approx, dtype=np.float64).ravel()
    ok = exact > 0
    if ok.sum() < 2:
        return float("nan")
    r = approx[ok] / exact[ok]
    p_lo = float(np.percentile(r, lo))
    p_hi = float(np.percentile(r, hi))
    if p_lo <= 0:
        return float("inf")
    return max(p_hi / p_lo, 1.0)


def mu_hat(exact: np.ndarray, kappa: float) -> float:
    """Distance-ratio concentration of a corpus at distortion ``kappa``.

    The fraction of unordered pairs-of-pairs whose exact distances lie
    strictly within ratio ``kappa`` of each other (max/min < kappa). This is
    the quantity that bounds the discordant fraction of any kappa-distorted
    distance. Computed in O(P log P) by sorting, not O(P^2).

    Args:
        exact: 1-D array of exact pairwise distances (zeros are excluded).
        kappa: Distortion ratio, >= 1.

    Returns:
        mu_hat in [0, 1], or NaN if fewer than 2 valid pairs or kappa is NaN.
    """
    if not np.isfinite(kappa):
        return float("nan") if np.isnan(kappa) else 1.0
    d = np.sort(np.asarray(exact, dtype=np.float64).ravel())
    d = d[d > 0]
    p = len(d)
    if p < 2:
        return float("nan")
    # For each i (ascending), pairs (i, j) with i < j and d[j] < kappa * d[i].
    idx = np.searchsorted(d, kappa * d, side="left")
    counts = np.maximum(idx - np.arange(1, p + 1), 0)
    total_within = int(counts.sum())
    total = p * (p - 1) // 2
    return float(total_within / total)


def mu_curve(exact: np.ndarray, kappas: np.ndarray) -> np.ndarray:
    """Evaluate ``mu_hat`` at each kappa: the corpus's concentration curve.

    The steepness of this curve near kappa = 1 is a proxy for intrinsic
    dimension (concentrated distances = high-d regime).
    """
    return np.array([mu_hat(exact, float(k)) for k in np.asarray(kappas)])


def max_certifiable_kappa(
    exact: np.ndarray,
    min_tau: float = 0.0,
    kappa_max: float = 64.0,
    tol: float = 1e-3,
) -> float:
    """Largest distortion kappa for which the tau floor stays >= ``min_tau``.

    Inverts ``tau_floor = 1 - 2 * mu_hat(kappa)`` by bisection (mu_hat is
    nondecreasing in kappa). If even kappa = 1 cannot certify ``min_tau``
    (extreme concentration: ties), returns 1.0. If the corpus certifies at
    ``kappa_max``, returns ``kappa_max``.

    A measured kappa above this value is the principled "exact reranking
    required" signal for the corpus.
    """
    mu_target = (1.0 - min_tau) / 2.0
    if mu_hat(exact, 1.0) > mu_target:
        return 1.0
    if mu_hat(exact, kappa_max) <= mu_target:
        return kappa_max
    lo, hi = 1.0, kappa_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if mu_hat(exact, mid) <= mu_target:
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# The certificate
# ---------------------------------------------------------------------------


@dataclass
class RankCertificate:
    """A distribution-free floor on rank agreement.

    Attributes:
        kappa: Measured robust distortion (97.5/2.5 percentile ratio).
        mu_hat: Corpus distance-ratio concentration at kappa.
        tau_floor: Guaranteed Kendall tau, ``1 - 2 * mu_hat``.
        spearman_floor: Guaranteed Spearman rho, ``1 - 3 * mu_hat``
            (Daniels' inequality; exact up to O(1/n_pairs)).
        n_pairs: Number of (nonzero-distance) pairs measured.
        max_certifiable_kappa: Largest distortion this corpus could certify
            at tau >= 0; measured kappa above it means no finite distortion
            certifies rank -- exact reranking is required.
    """

    kappa: float
    mu_hat: float
    tau_floor: float
    spearman_floor: float
    n_pairs: int
    max_certifiable_kappa: float = field(default=float("nan"))

    @property
    def vacuous(self) -> bool:
        """True when the certificate certifies nothing (tau floor <= 0).

        On distance-concentrated corpora this is the expected outcome and is
        itself the signal: single-stage rank fidelity cannot be certified,
        so exact reranking is mandatory.
        """
        return not (self.tau_floor > 0.0)

    @property
    def rerank_required(self) -> bool:
        """Alias for ``vacuous`` in retrieval-configuration language."""
        return self.vacuous

    def as_dict(self) -> dict:
        return {
            "kappa": self.kappa,
            "mu_hat": self.mu_hat,
            "tau_floor": self.tau_floor,
            "spearman_floor": self.spearman_floor,
            "n_pairs": self.n_pairs,
            "max_certifiable_kappa": self.max_certifiable_kappa,
            "vacuous": self.vacuous,
        }


def certify(
    exact: np.ndarray,
    approx: np.ndarray,
    lo: float = 2.5,
    hi: float = 97.5,
) -> RankCertificate:
    """Build a :class:`RankCertificate` from paired distance arrays.

    Args:
        exact: 1-D exact pairwise distances.
        approx: 1-D compressed-domain distances over the same pairs.
        lo, hi: Robust-kappa percentiles.
    """
    exact = np.asarray(exact, dtype=np.float64).ravel()
    n_pairs = int((exact > 0).sum())
    kappa = measure_kappa(exact, approx, lo=lo, hi=hi)
    mu = mu_hat(exact, kappa)
    tau_floor = 1.0 - 2.0 * mu if np.isfinite(mu) else float("nan")
    rho_floor = 1.0 - 3.0 * mu if np.isfinite(mu) else float("nan")
    return RankCertificate(
        kappa=kappa,
        mu_hat=mu,
        tau_floor=tau_floor,
        spearman_floor=rho_floor,
        n_pairs=n_pairs,
        max_certifiable_kappa=max_certifiable_kappa(exact),
    )


# ---------------------------------------------------------------------------
# Convenience: from embedding matrices
# ---------------------------------------------------------------------------


def pairwise_distances(
    x: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """Upper-triangle pairwise distances of the rows of ``x``.

    Metrics: ``"cosine"`` (1 - cosine similarity) or ``"l2"`` (Euclidean).
    """
    x = np.asarray(x, dtype=np.float64)
    if metric == "cosine":
        norms = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)
        u = x / norms
        sim = u @ u.T
        d = 1.0 - sim
    elif metric == "l2":
        sq = (x**2).sum(axis=1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
        d = np.sqrt(np.maximum(d2, 0.0))
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown metric: {metric!r}")
    iu = np.triu_indices(len(x), k=1)
    return np.maximum(d[iu], 0.0)


def certificate_from_embeddings(
    original: np.ndarray,
    reconstructed: np.ndarray,
    n_anchors: int = 200,
    metric: str = "cosine",
    seed: int = 0,
) -> RankCertificate:
    """Certify a compression run directly from embedding matrices.

    Samples ``n_anchors`` rows (deterministically, ``seed``), measures the
    exact and reconstructed pairwise distances over the anchor pairs, and
    returns the distribution-free certificate.

    Args:
        original: (n, d) exact embeddings.
        reconstructed: (n, d) decompressed embeddings, same row order.
        n_anchors: Anchor rows to sample (200 anchors = 19,900 pairs).
        metric: "cosine" or "l2" -- match the index's ranking metric.
        seed: Anchor-sampling seed.
    """
    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)
    if original.shape != reconstructed.shape:
        raise ValueError(f"shape mismatch: {original.shape} vs {reconstructed.shape}")
    n = original.shape[0]
    rng = np.random.default_rng(seed)
    if n > n_anchors:
        anchors = rng.choice(n, size=n_anchors, replace=False)
        original = original[anchors]
        reconstructed = reconstructed[anchors]
    exact = pairwise_distances(original, metric=metric)
    approx = pairwise_distances(reconstructed, metric=metric)
    return certify(exact, approx)

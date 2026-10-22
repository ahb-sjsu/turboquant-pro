# TurboQuant Pro: corpus geometry profile (issue #162)
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Corpus geometry: regularized Mahalanobis centrality against reverse-kNN hubness.

``hub_anatomy`` (:mod:`turboquant_pro.anatomy`) measures centrality as Euclidean
distance to the mean. That treats every direction alike, so a point far out
along a low-variance direction reads as "central". This module measures it in
the whitened space

    z = Sigma_hat^(-1/2) (x - mu_hat),    radius r = ||z||  (the Mahalanobis radius),

and sets it against the reverse-kNN occurrence count N_k (how often a row is
among the k nearest neighbours of the others). The two axes give four regions:

    ===========  =========  =====================================
    centrality   hubness    reading
    ===========  =========  =====================================
    central      high       conventional central hub
    central      low        central anti-hub
    peripheral   low        conventional peripheral point
    peripheral   high       eccentric hub (the high-value case)
    ===========  =========  =====================================

Covariance regularization (docs/FUZZ.md has the long form). The sample
covariance S is singular whenever n <= d and ill-conditioned for most
embedding corpora, and S^(-1/2) then blows up exactly the low-variance
directions a compressor discards first. Two guards, both recorded in the
profile:

* **Shrinkage** toward the scaled identity, Sigma = (1 - a) S + a (tr S / d) I.
  ``shrinkage="ledoit-wolf"`` (default) estimates a analytically (Ledoit and
  Wolf, 2004); a float in [0, 1] fixes it. Any a > 0 makes Sigma full rank: a
  direction outside the data's span gets variance a * tr(S)/d, so the radius
  along it is measured against the average variance, not against zero.
* **Eigenvalue floor**: eigenvalues below ``eig_floor * max(eig)`` are raised
  to it. With a > 0 it rarely binds; with ``shrinkage=0`` on a singular S it is
  what keeps the radius finite, and ``n_floored`` reports how many directions
  it decided.

Report schema ``turboquant-pro/geometry-profile`` version 1. Field names are
API: additions only, never renames (the anatomy report's rule).
"""

from __future__ import annotations

import hashlib

import numpy as np

from .anatomy import knn_exact

__all__ = [
    "Whitener",
    "fit_whitener",
    "reverse_knn_counts",
    "geometry_profile",
    "QUANTILE_GRID",
]

#: Quantiles reported for every distribution in the profile.
QUANTILE_GRID = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def _sha256_rows(x: np.ndarray) -> str:
    a = np.ascontiguousarray(x, dtype=np.float32)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _ledoit_wolf(xc: np.ndarray, s: np.ndarray) -> float:
    """Ledoit-Wolf (2004) shrinkage toward (tr S / d) I for centered rows ``xc``."""
    n, d = xc.shape
    mu_t = float(np.trace(s)) / d
    delta = float(((s - mu_t * np.eye(d)) ** 2).sum()) / d
    if delta <= 0.0:
        return 0.0  # S is already a multiple of the identity
    # sum_k ||x_k x_k^T - S||_F^2 = sum_k ||x_k||^4 - n ||S||_F^2  (S = X^T X / n)
    sq = (xc.astype(np.float64) ** 2).sum(axis=1)
    beta_bar = float((sq**2).sum() - n * (s**2).sum()) / (n * n * d)
    return float(min(max(beta_bar, 0.0), delta) / delta)


class Whitener:
    """The fitted map x -> z = Sigma^(-1/2)(x - mu) and its inverse."""

    def __init__(self, mean, eigvals, eigvecs, shrinkage, eig_floor, n_floored, n_fit):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.eigvals = np.asarray(
            eigvals, dtype=np.float64
        )  # after shrinkage and floor
        self.eigvecs = np.asarray(eigvecs, dtype=np.float64)
        self.shrinkage = float(shrinkage)
        self.eig_floor = float(eig_floor)
        self.n_floored = int(n_floored)
        self.n_fit = int(n_fit)

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    def whiten(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        return ((x - self.mean) @ self.eigvecs) / np.sqrt(self.eigvals)

    def unwhiten(self, z: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(np.asarray(z, dtype=np.float64))
        return (z * np.sqrt(self.eigvals)) @ self.eigvecs.T + self.mean

    def radius(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.whiten(x), axis=1)

    def digest(self) -> str:
        """Content hash of the fitted map (replay refuses a different whitening)."""
        h = hashlib.sha256()
        for a in (self.mean, self.eigvals, self.eigvecs):
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        h.update(f"{self.shrinkage!r}/{self.eig_floor!r}".encode())
        return h.hexdigest()

    def metadata(self) -> dict:
        ev = self.eigvals
        return {
            "shrinkage": self.shrinkage,
            "eig_floor_relative": self.eig_floor,
            "n_floored": self.n_floored,
            "n_fit": self.n_fit,
            "dim": self.dim,
            "condition_number": float(ev.max() / ev.min()),
            "whitener_sha256": self.digest(),
        }

    def spectrum(self) -> dict:
        ev = np.sort(self.eigvals)[::-1]
        p = ev / ev.sum()
        nz = p[p > 0]
        return {
            # exp(Shannon entropy) of the normalized spectrum (Roy and Vetterli, 2007)
            "effective_rank": float(np.exp(-(nz * np.log(nz)).sum())),
            "participation_ratio": float(ev.sum() ** 2 / (ev**2).sum()),
            "top_eigenvalue_share": [float(v) for v in p[:10]],
            "energy_in_top_10pct_dims": float(p[: max(1, len(p) // 10)].sum()),
        }


def fit_whitener(
    x: np.ndarray,
    *,
    shrinkage: float | str = "ledoit-wolf",
    eig_floor: float = 1e-6,
    sample: int | None = None,
    seed: int = 0,
) -> Whitener:
    """Fit the regularized whitening map on ``x`` (or a seeded row sample of it)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"need at least 2 rows of shape (n, d); got {x.shape}")
    if sample is not None and sample < len(x):
        rng = np.random.default_rng(seed)
        x = x[np.sort(rng.choice(len(x), size=sample, replace=False))]
    n, d = x.shape
    mean = x.mean(axis=0)
    xc = x - mean
    s = (xc.T @ xc) / n
    tr = float(np.trace(s))
    if not tr > 0.0:
        raise ValueError("the rows have zero variance: no geometry to profile")
    if shrinkage == "ledoit-wolf":
        a = _ledoit_wolf(xc, s)
    else:
        a = float(shrinkage)
        if not 0.0 <= a <= 1.0:
            raise ValueError(
                f"shrinkage must be in [0, 1] or 'ledoit-wolf'; got {shrinkage!r}"
            )
    sig = (1.0 - a) * s + a * (tr / d) * np.eye(d)
    ev, vec = np.linalg.eigh(sig)
    floor = float(eig_floor) * float(ev.max())
    n_floored = int((ev < floor).sum())
    ev = np.maximum(ev, floor)
    return Whitener(mean, ev, vec, a, eig_floor, n_floored, n)


def reverse_knn_counts(
    base: np.ndarray, k: int, *, sample: int | None = None, seed: int = 0
) -> tuple[np.ndarray, dict]:
    """Reverse-kNN occurrence counts N_k of every base row.

    Exact when ``sample`` is None or >= n: every row queries the rest. Otherwise
    a seeded sample of rows queries the whole corpus and the counts are scaled
    by n / sample, an unbiased estimate of N_k (each row's count is a sum over
    queries, estimated from a uniform sample of them).
    """
    base = np.ascontiguousarray(base, dtype=np.float32)
    n = len(base)
    if k >= n:
        raise ValueError(f"k={k} must be smaller than the corpus ({n} rows)")
    if sample is None or sample >= n:
        _, idx = knn_exact(base, base, k, exclude_self=True)
        counts = np.bincount(idx.ravel(), minlength=n).astype(np.float64)
        return counts, {"estimator": "exact", "n_query_rows": n}
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(n, size=sample, replace=False))
    _, idx = knn_exact(base, base[rows], k + 1)
    # drop each query row's own index (it is its own nearest neighbour)
    keep = idx != rows[:, None]
    picked = np.stack([r[m][:k] for r, m in zip(idx, keep)])
    counts = np.bincount(picked.ravel(), minlength=n).astype(np.float64) * (n / sample)
    return counts, {
        "estimator": "sampled",
        "n_query_rows": int(sample),
        "scale": n / sample,
    }


def _quantiles(v: np.ndarray) -> dict:
    return {
        f"q{int(round(q * 100)):02d}": float(np.quantile(v, q)) for q in QUANTILE_GRID
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from .anatomy import _rankcorr

    return float(_rankcorr(a, b))


def geometry_profile(
    base: np.ndarray,
    *,
    k: int = 10,
    sample: int | None = None,
    shrinkage: float | str = "ledoit-wolf",
    eig_floor: float = 1e-6,
    central_quantile: float = 0.25,
    hub_quantile: float = 0.99,
    anti_quantile: float = 0.10,
    top: int = 20,
    seed: int = 0,
) -> tuple[dict, Whitener, np.ndarray]:
    """Profile a corpus. Returns ``(profile, whitener, counts)``.

    A row is *central* when its Mahalanobis radius is in the lowest
    ``central_quantile`` of the corpus and *peripheral* when it is in the
    highest; a *hub* when its N_k is at or above the ``hub_quantile`` quantile
    and an *anti-hub* when at or below the ``anti_quantile`` quantile.
    """
    base = np.asarray(base, dtype=np.float32)
    n, d = base.shape
    w = fit_whitener(
        base, shrinkage=shrinkage, eig_floor=eig_floor, sample=sample, seed=seed
    )
    radius = w.radius(base)
    counts, est = reverse_knn_counts(base, k, sample=sample, seed=seed)
    counts2k, _ = reverse_knn_counts(base, min(2 * k, n - 1), sample=sample, seed=seed)

    r_lo, r_hi = np.quantile(radius, central_quantile), np.quantile(
        radius, 1.0 - central_quantile
    )
    c_hub, c_anti = np.quantile(counts, hub_quantile), np.quantile(
        counts, anti_quantile
    )
    central, peripheral = radius <= r_lo, radius >= r_hi
    hub, anti = counts >= c_hub, counts <= c_anti
    hub2k = counts2k >= np.quantile(counts2k, hub_quantile)

    def rows(mask: np.ndarray, order: np.ndarray) -> list:
        ids = np.flatnonzero(mask)
        ids = ids[np.argsort(order[ids], kind="stable")][:top]
        return [
            {
                "row": int(i),
                "mahalanobis_radius": float(radius[i]),
                "n_k": float(counts[i]),
            }
            for i in ids
        ]

    union = int((hub | hub2k).sum())
    from turboquant_pro import __version__

    profile = {
        "schema": "turboquant-pro/geometry-profile",
        "schema_version": 1,
        "tool_version": __version__,
        "n_base": int(n),
        "dim": int(d),
        "k": int(k),
        "seed": int(seed),
        "dataset_sha256": _sha256_rows(base),
        "estimator": est,
        "regularization": w.metadata(),
        "spectrum": w.spectrum(),
        "thresholds": {
            "central_quantile": central_quantile,
            "hub_quantile": hub_quantile,
            "anti_quantile": anti_quantile,
            "radius_central_max": float(r_lo),
            "radius_peripheral_min": float(r_hi),
            "n_k_hub_min": float(c_hub),
            "n_k_anti_max": float(c_anti),
        },
        "mahalanobis_radius": _quantiles(radius),
        "n_k": {
            **_quantiles(counts),
            "skew": float(
                ((counts - counts.mean()) ** 3).mean() / max(counts.std() ** 3, 1e-12)
            ),
        },
        # Positive: the more central (smaller radius) a row, the more of a hub it is.
        "corr_n_k_centrality": _spearman(counts, -radius),
        "region_counts": {
            "central_hub": int((central & hub).sum()),
            "central_anti_hub": int((central & anti).sum()),
            "peripheral_hub": int((peripheral & hub).sum()),
            "peripheral_anti_hub": int((peripheral & anti).sum()),
        },
        # Largest N_k first within each region, then the smallest for anti-hubs.
        "central_hubs": rows(central & hub, -counts),
        "peripheral_hubs": rows(peripheral & hub, -counts),
        "central_anti_hubs": rows(central & anti, counts),
        # Do the hubs at k survive at 2k? Jaccard of the two top-quantile hub sets.
        "hub_stability_k_2k": (
            float((hub & hub2k).sum() / union) if union else float("nan")
        ),
    }
    return profile, w, counts

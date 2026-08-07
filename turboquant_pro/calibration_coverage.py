# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Is your calibration set actually representative? Measure it, don't assume.

:func:`~turboquant_pro.calibration.calibrate_key_quantizer` fits a
per-(head, channel) Lloyd-Max codebook from "a representative set of real key
activations". The word *representative* carries the whole load and nothing in
this package checked it. A codebook fitted on a calibration sample that is
shifted or differently shaped from serving traffic is fitted to the wrong
measure, and the failure is quiet: the codebook still reconstructs its own
calibration set beautifully.

That is the same error term an oscilloscope calls **probe loading** — the
instrument's coupling perturbs what it reads — and the same one the companion
instrument package characterizes for read-operator recovery
(https://github.com/ahb-sjsu/readscope, `readscope.loading`). Backported here
because it applies verbatim to a calibration set.

Both distributions are summarized by first and second moments, which is the
right level of description for a single guard number. A heavier divergence
estimator would be more faithful and much harder to act on.

**Thresholds here are conventions, not measurements.** ``COVERAGE_WARN`` and
``COVERAGE_FAIL`` are round numbers chosen so the guard fires before the
codebook is obviously mis-fitted. No sweep in this repository has established
where the real knee is for KV keys. The companion package measured a knee
between 25 and 50 nats of Jeffreys divergence for *subspace recovery* on a
synthetic consumer, which is a different quantity on a different task and is
cited as motivation rather than as a calibrated value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backend import to_numpy

COVERAGE_WARN = 10.0
COVERAGE_FAIL = 50.0

__all__ = [
    "COVERAGE_FAIL",
    "COVERAGE_WARN",
    "CoverageReport",
    "calibration_coverage",
    "check_calibration_coverage",
]


@dataclass
class CoverageReport:
    """How far a calibration sample sits from the serving distribution."""

    jeffreys: float
    """Symmetrized KL between the fitted Gaussians, in nats. The headline."""

    bhattacharyya: float
    mean_shift: float
    """Mahalanobis distance between the means, in the serving metric."""

    spectral_ratio: float
    """Worst-direction variance ratio, ``max(l, 1/l)``."""

    verdict: str
    """``ok``, ``warn`` or ``fail`` against the declared thresholds."""

    n_calibration: int
    n_serving: int
    dim: int

    def to_dict(self) -> dict:
        return {
            "jeffreys": self.jeffreys,
            "bhattacharyya": self.bhattacharyya,
            "mean_shift": self.mean_shift,
            "spectral_ratio": self.spectral_ratio,
            "verdict": self.verdict,
            "n_calibration": self.n_calibration,
            "n_serving": self.n_serving,
            "dim": self.dim,
            "thresholds": {"warn": COVERAGE_WARN, "fail": COVERAGE_FAIL},
        }


def _flatten(x) -> np.ndarray:
    """Pool everything except the channel axis, matching calibrate()."""
    a = np.asarray(to_numpy(x), dtype=np.float64)
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a.reshape(-1, a.shape[-1])


def _moments(x: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    if x.shape[0] < 2:
        raise ValueError("need at least two rows to fit a covariance")
    mu = x.mean(axis=0)
    cov = np.atleast_2d(np.cov(x, rowvar=False))
    return mu, cov + ridge * np.eye(cov.shape[0])


def _logdet(c: np.ndarray) -> float:
    sign, val = np.linalg.slogdet(c)
    if sign <= 0:
        raise ValueError("covariance is not positive definite")
    return float(val)


def _quad(v: np.ndarray, m: np.ndarray) -> float:
    return float((v.T @ m @ v).reshape(()))


def calibration_coverage(
    calibration_samples,
    serving_samples,
    *,
    ridge: float = 1e-9,
) -> CoverageReport:
    """Measure how far a calibration set sits from real serving activations.

    Args:
        calibration_samples: the activations a codebook was or would be fitted
            on. Any shape accepted by ``calibrate``; pooled to ``(N, D)``.
        serving_samples: activations from the distribution that will actually
            be compressed at inference time.
        ridge: diagonal loading for the fitted covariances.

    Returns:
        A :class:`CoverageReport`. Read ``jeffreys`` first and treat the rest
        as the decomposition telling you *why* it is large: a mean shift, a
        variance mismatch, or both.
    """
    cal = _flatten(calibration_samples)
    srv = _flatten(serving_samples)
    if cal.shape[1] != srv.shape[1]:
        raise ValueError(
            f"channel counts differ: calibration {cal.shape[1]} vs "
            f"serving {srv.shape[1]}"
        )
    d = cal.shape[1]

    mu_c, cov_c = _moments(cal, ridge)
    mu_s, cov_s = _moments(srv, ridge)
    inv_c, inv_s = np.linalg.inv(cov_c), np.linalg.inv(cov_s)
    dmu = (mu_c - mu_s).reshape(-1, 1)

    kl_cs = 0.5 * (
        float(np.trace(inv_s @ cov_c))
        + _quad(dmu, inv_s)
        - d
        + _logdet(cov_s)
        - _logdet(cov_c)
    )
    kl_sc = 0.5 * (
        float(np.trace(inv_c @ cov_s))
        + _quad(dmu, inv_c)
        - d
        + _logdet(cov_c)
        - _logdet(cov_s)
    )
    jeffreys = kl_cs + kl_sc

    mix = 0.5 * (cov_c + cov_s)
    bhat = 0.125 * _quad(dmu, np.linalg.inv(mix)) + 0.5 * (
        _logdet(mix) - 0.5 * (_logdet(cov_c) + _logdet(cov_s))
    )

    ratios = np.clip(
        np.linalg.eigvalsh(np.linalg.solve(cov_s, cov_c)).real, 1e-300, None
    )

    if jeffreys >= COVERAGE_FAIL:
        verdict = "fail"
    elif jeffreys >= COVERAGE_WARN:
        verdict = "warn"
    else:
        verdict = "ok"

    return CoverageReport(
        jeffreys=jeffreys,
        bhattacharyya=bhat,
        mean_shift=float(np.sqrt(max(_quad(dmu, inv_s), 0.0))),
        spectral_ratio=float(np.max(np.maximum(ratios, 1.0 / ratios))),
        verdict=verdict,
        n_calibration=int(cal.shape[0]),
        n_serving=int(srv.shape[0]),
        dim=d,
    )


def check_calibration_coverage(
    calibration_samples,
    serving_samples,
    *,
    strict: bool = False,
    **kwargs,
) -> CoverageReport:
    """Measure coverage and complain when it is poor.

    Warns at ``COVERAGE_WARN``. With ``strict=True`` raises at
    ``COVERAGE_FAIL`` instead of warning, for a CI gate that should stop a
    codebook being fitted on the wrong distribution.
    """
    import warnings

    report = calibration_coverage(calibration_samples, serving_samples, **kwargs)
    if report.verdict == "fail":
        msg = (
            f"calibration set is far from serving activations "
            f"(Jeffreys {report.jeffreys:.1f} nats >= {COVERAGE_FAIL}); "
            f"mean shift {report.mean_shift:.2f}, worst variance ratio "
            f"{report.spectral_ratio:.1f}. A codebook fitted here is fitted "
            f"to the wrong measure"
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    elif report.verdict == "warn":
        warnings.warn(
            f"calibration coverage is marginal (Jeffreys "
            f"{report.jeffreys:.1f} nats >= {COVERAGE_WARN})",
            RuntimeWarning,
            stacklevel=2,
        )
    return report

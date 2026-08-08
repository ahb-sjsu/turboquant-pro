# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Conformance kit for read-operator providers.

The executable half of the contract in :mod:`turboquant_pro.read_operators`,
mirroring :mod:`turboquant_pro.plugin_conformance` for quantizers. A provider
that passes this is usable by every consumer-relative instrument in the
toolkit, because those instruments need a valid ``P_C`` and nothing else.

The checks are deliberately about *validity*, not accuracy. Whether a
provider's operator is the right one for your consumer is an empirical
question its own calibration has to answer; whether it is a symmetric PSD
matrix of the right shape that behaves linearly under the distortion
functional is a property this kit can settle in a second.

One check earns its place from experience rather than principle.
``non_degenerate`` exists because an operator of numerical rank zero passes
every algebraic test and reports a distortion of zero against any codec,
which reads as a perfect result. A provider that returns nothing useful must
fail loudly rather than silently certify everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .read_operators import consumer_distortion

__all__ = [
    "ReadOperatorConformanceReport",
    "assert_read_operator_conformance",
    "run_read_operator_conformance",
]


@dataclass
class ReadOperatorConformanceReport:
    """Per-check results, with the reason each failure failed."""

    provider: str
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    @property
    def failures(self) -> list[str]:
        return [k for k, v in self.checks.items() if not v]

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "passed": self.passed,
            "checks": dict(self.checks),
            "failures": self.failures,
            "details": self.details,
        }


def run_read_operator_conformance(
    provider: Any,
    activations: np.ndarray,
    *,
    name: str = "",
    tol: float = 1e-9,
    **context: Any,
) -> ReadOperatorConformanceReport:
    """Run every check against one provider and one activation batch."""
    rep = ReadOperatorConformanceReport(provider=name or type(provider).__name__)
    a = np.asarray(activations, dtype=np.float64)
    d = a.shape[-1]

    try:
        P = np.asarray(provider.operator(a, **context), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - a raising provider fails here
        rep.checks["callable"] = False
        rep.details["error"] = repr(exc)[:300]
        return rep
    rep.checks["callable"] = True

    rep.checks["shape"] = bool(P.shape == (d, d))
    rep.details["shape"] = list(P.shape)
    if not rep.checks["shape"]:
        return rep

    rep.checks["finite"] = bool(np.all(np.isfinite(P)))
    asym = float(np.abs(P - P.T).max()) if P.size else 0.0
    rep.checks["symmetric"] = bool(asym <= 1e-8 * max(1.0, np.abs(P).max()))
    rep.details["max_asymmetry"] = asym

    if rep.checks["finite"]:
        eig = np.linalg.eigvalsh(0.5 * (P + P.T))
        lo = float(eig.min())
        rep.checks["psd"] = bool(lo >= -1e-8 * max(1.0, float(eig.max())))
        rep.details["min_eigenvalue"] = lo
        rank = int(np.linalg.matrix_rank(P, tol=1e-10))
        rep.checks["non_degenerate"] = bool(rank >= 1 and float(np.trace(P)) > 0)
        rep.details["rank"] = rank
        rep.details["trace"] = float(np.trace(P))
    else:
        rep.checks["psd"] = False
        rep.checks["non_degenerate"] = False

    # determinism: the same batch must give the same operator
    try:
        P2 = np.asarray(provider.operator(a, **context), dtype=np.float64)
        drift = float(np.abs(P - P2).max())
        rep.checks["deterministic"] = bool(drift <= tol * max(1.0, np.abs(P).max()))
        rep.details["repeat_drift"] = drift
    except Exception as exc:  # noqa: BLE001
        rep.checks["deterministic"] = False
        rep.details["repeat_error"] = repr(exc)[:200]

    # the distortion functional must be linear in the error covariance
    rng = np.random.default_rng(0)
    e = rng.standard_normal((max(4 * d, 32), d))
    sigma = e.T @ e / e.shape[0]
    d1 = consumer_distortion(P, sigma)
    d2 = consumer_distortion(P, 2.0 * sigma)
    rep.checks["distortion_linear"] = bool(
        abs(d2 - 2.0 * d1) <= 1e-6 * max(1.0, abs(d2))
    )
    rep.checks["distortion_non_negative"] = bool(d1 >= -1e-9)
    rep.details["distortion_unit"] = d1

    return rep


def assert_read_operator_conformance(
    provider: Any, activations: np.ndarray, *, name: str = "", **context: Any
) -> ReadOperatorConformanceReport:
    """Run the kit and raise on any failure, for use in a test."""
    rep = run_read_operator_conformance(provider, activations, name=name, **context)
    if not rep.passed:
        raise AssertionError(
            f"read-operator conformance failed for {rep.provider}: "
            f"{rep.failures}; details {rep.details}"
        )
    return rep

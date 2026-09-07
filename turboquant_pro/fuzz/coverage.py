"""Frozen quantile coverage and deterministic retained-case selection.

Coverage boundaries belong to the immutable campaign geometry, not to a
candidate stream.  This keeps the meaning of a cell stable when the budget,
evaluation order, or set of candidates changes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .oracles import CLASSIFICATIONS

_SIGNAL_NAMES = (
    "mahalanobis_centrality",
    "reverse_knn_hubness",
    "exact_neighbor_margin",
    "consumer_error",
)


@dataclass(frozen=True)
class CoverageCase:
    """One evaluated candidate, represented without mutable campaign state."""

    case_id: str
    signals: Mapping[str, float]
    severity: float
    classification: str


def _validated_edges(name: str, values: Sequence[float]) -> np.ndarray:
    edges = np.asarray(values, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all():
        raise ValueError(f"coverage edges for {name!r} must be finite with two values")
    if np.any(np.diff(edges) < 0.0):
        raise ValueError(f"coverage edges for {name!r} must be sorted")
    return edges


class FrozenQuantileCoverage:
    """Map four fixed metric values to stable integer coverage cells."""

    def __init__(self, quantile_edges: Mapping[str, Sequence[float]]) -> None:
        unknown = set(quantile_edges).difference(_SIGNAL_NAMES)
        missing = set(_SIGNAL_NAMES).difference(quantile_edges)
        if unknown or missing:
            raise ValueError(
                "coverage needs exactly "
                f"{', '.join(_SIGNAL_NAMES)} quantile edge sets"
            )
        self._edges = {
            name: _validated_edges(name, quantile_edges[name]) for name in _SIGNAL_NAMES
        }

    @classmethod
    def from_geometry(cls, profile: Mapping[str, object]) -> FrozenQuantileCoverage:
        """Load frozen campaign cells from a validated geometry profile document."""
        try:
            coverage = profile["coverage"]
            if not isinstance(coverage, Mapping):
                raise TypeError
            if coverage["schema"] != "turboquant-pro/fuzz-frozen-coverage":
                raise ValueError("unsupported frozen coverage schema")
            signals = coverage["signals"]
            if not isinstance(signals, Mapping):
                raise TypeError
        except (KeyError, TypeError) as error:
            raise ValueError(
                "geometry profile lacks frozen coverage signals"
            ) from error
        return cls(signals)

    def signature(self, signals: Mapping[str, float]) -> tuple[int, ...]:
        """Return the frozen bin index of each signal in canonical order.

        Interior boundaries belong to the higher bin.  The first and last
        profile quantiles clamp underflow and overflow respectively, avoiding
        a data-dependent overflow cell.
        """
        if set(signals) != set(_SIGNAL_NAMES):
            raise ValueError(
                f"coverage signals must be exactly {', '.join(_SIGNAL_NAMES)}"
            )
        signature = []
        for name in _SIGNAL_NAMES:
            value = float(signals[name])
            if not np.isfinite(value):
                raise ValueError(f"coverage signal {name!r} must be finite")
            edges = self._edges[name]
            signature.append(
                int(
                    np.clip(
                        np.searchsorted(edges, value, side="right") - 1,
                        0,
                        len(edges) - 2,
                    )
                )
            )
        return tuple(signature)


def retain_cases(
    coverage: FrozenQuantileCoverage,
    cases: Sequence[CoverageCase],
) -> tuple[CoverageCase, ...]:
    """Keep novel cells and the strongest case per class-and-cell deterministically.

    A new classification is retained even when it lands in an existing cell.
    Within each ``(cell, classification)`` group, severity wins and the case ID
    breaks ties.  Sorting before grouping makes the archive independent of
    candidate evaluation order.
    """
    validated: list[tuple[tuple[int, ...], CoverageCase]] = []
    seen_ids: set[str] = set()
    for case in cases:
        if not case.case_id or case.case_id in seen_ids:
            raise ValueError("coverage case IDs must be non-empty and unique")
        if case.classification not in CLASSIFICATIONS:
            raise ValueError("coverage case has an unsupported classification")
        if not np.isfinite(case.severity):
            raise ValueError("coverage case severity must be finite")
        seen_ids.add(case.case_id)
        validated.append((coverage.signature(case.signals), case))
    winners: dict[tuple[tuple[int, ...], str], CoverageCase] = {}
    for signature, case in sorted(
        validated,
        key=lambda item: (item[0], item[1].classification, item[1].case_id),
    ):
        key = (signature, case.classification)
        previous = winners.get(key)
        if previous is None or case.severity > previous.severity:
            winners[key] = case
    return tuple(
        winners[key] for key in sorted(winners, key=lambda item: (item[0], item[1]))
    )

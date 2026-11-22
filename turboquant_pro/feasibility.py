# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Should this be compressed at all, and can the guarantee be met?

The planner answers *which* codec. This answers the question before it: given
a corpus and a declared observer, is the requested guarantee attainable, which
source dimensions does the consumer never read, and is any of what the
consumer needs already missing from the representation (issue #176, phase 1;
``docs/DESIGN_feasibility.md``).

The governing result is Observation Theory's **omission floor** (Paper III,
``thm:omission``): information the observation has already discarded cannot be
recovered downstream. No encoder repairs it. So the honest first question is
whether the consumer's read directions carry any variance in this
representation at all; if they do not, the answer is not "use a better
codec", it is "re-embed, retain the source, or change the upstream encoder".

Four measurements, each from the corpus and the observer's read operator
``P``, with ``Sigma`` the corpus covariance.

- **Observable signal** ``tr(P Sigma)``: what the consumer can distinguish
  rows by at all. Its **observable rank**, the participation ratio of
  ``P^(1/2) Sigma P^(1/2)``, is how many directions carry it.
- **Unread source dimensions**: directions with real variance and negligible
  consumer sensitivity. They are free to drop, and their count is the size of
  the reduction available before any quantizer runs.
- **Omitted sensitivity**: consumer sensitivity lying in the corpus's null
  space, as a fraction of ``tr(P)``. The consumer reads there; the data does
  not vary there; nothing downstream puts row-to-row information back. This is
  the omission floor, measured. Past a majority it is a verdict, below it a
  warning, because constant directions are free for a reconstruction consumer
  and useless only to one that must tell rows apart.
- **The attainable region**: with the widths the scan stores (0 to 4 bits per
  direction) and the Lloyd-Max distortion table, the smallest consumer
  distortion any allocation can reach, and the bytes per vector needed to
  reach a declared one.

Two honest boundaries. A distortion target is stated as a fraction of the
observable signal, because a recall target is not convertible to a distortion
by any distribution-free relation. And a rank target is answered by the rank
certificate's own inversion (``max_certifiable_kappa``), which needs no
conversion: on a distance-concentrated corpus no finite distortion certifies
the requested rank fidelity at any byte budget, and that is a feasibility
answer, not a codec choice.

When the observer's operator is an estimate rather than a closed form, the
report also measures how much source variance lies outside the subspace the
estimate could identify. readscope's recovery cliff is at ``k = d`` and is a
theorem, not a measurement: below full dimension a confined transcript cannot
identify hidden components. Sensitivity that the estimate reports as zero
there may be unread or merely unidentified, and the two are indistinguishable,
so the verdict is ABSTAIN rather than a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from turboquant_pro.refinement import NORM_BYTES, layer_bytes
from turboquant_pro.spectrum import DISTORTION, allocate_bits

PASS, INFEASIBLE, ABSTAIN = "PASS", "INFEASIBLE", "ABSTAIN"
MAX_WIDTH = max(DISTORTION)
VARIANCE_TOL = 1e-8
# A direction is "unread" when dropping it, and every direction weaker than it,
# would cost the consumer less than this fraction of the observable signal. It
# is a stated convention, not a derived constant: it makes the count actionable
# (drop them and lose under a thousandth of what the consumer reads) rather
# than a test of exact zeros, which floating-point eigenvectors never give.
UNREAD_TOL = 1e-3
# The omission verdict is a majority rule, stated rather than tuned: the
# consumer's sensitivity is mostly supported by directions this corpus does not
# vary along, so more than half of what it nominally reads carries no
# row-to-row information and no encoder puts that back. Below the majority the
# same measurement is a warning, because constant directions are not
# necessarily harmful: a reconstruction consumer stores them once for free.
OMISSION_INFEASIBLE = 0.5
OMISSION_WARN = 0.01
NEAR_EXACT_KAPPA = 1.01
MAX_UNIDENTIFIED = 0.05
REFERENCE_DISTORTIONS = (0.10, 0.05, 0.01)

__all__ = [
    "PASS",
    "INFEASIBLE",
    "ABSTAIN",
    "FeasibilityReport",
    "feasibility",
    "min_bytes_for_distortion",
]


def _moments(x: np.ndarray, P: np.ndarray):
    """The corpus spectrum and the consumer's sensitivity along it.

    Both are read in the eigenbasis of the covariance, so "a direction the
    data has" and "a direction the consumer reads" are stated in one frame
    and the null space is exactly where the variance stops.
    """
    a = np.asarray(x, dtype=np.float64)
    a = a.reshape(-1, a.shape[-1])
    centred = a - a.mean(axis=0, keepdims=True)
    cov = (centred.T @ centred) / max(1, centred.shape[0] - 1)
    var, basis = np.linalg.eigh(0.5 * (cov + cov.T))
    order = np.argsort(var)[::-1]
    var = np.clip(var[order], 0.0, None)
    basis = np.ascontiguousarray(basis[:, order])
    A = 0.5 * (np.asarray(P, dtype=np.float64) + np.asarray(P, dtype=np.float64).T)
    sens = np.maximum(np.einsum("ij,jk,ki->i", basis.T, A, basis), 0.0)
    return var, sens, basis, A


def _read_mask(weights: np.ndarray, signal: float) -> np.ndarray:
    """Which directions the consumer effectively reads: all but the weakest
    ones whose contributions together stay under ``UNREAD_TOL`` of the
    observable signal."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    if signal <= 0:
        return np.zeros(w.size, dtype=bool)
    order = np.argsort(w)
    keep = np.ones(w.size, dtype=bool)
    spent = 0.0
    for i in order:
        if spent + w[i] > UNREAD_TOL * signal:
            break
        spent += w[i]
        keep[i] = False
    return keep


def min_bytes_for_distortion(
    weights: np.ndarray, target_fraction: float, *, norm: bool = True
) -> tuple[int, np.ndarray] | tuple[None, None]:
    """Fewest bytes per vector whose best allocation leaves at most
    ``target_fraction`` of the observable signal, or ``(None, None)`` when the
    widest width cannot reach it.

    ``weights`` is the per-direction ``sensitivity * variance``. The greedy
    allocation is optimal for consecutive widths and its distortion falls with
    the budget, so a bisection over total bits finds the smallest one.
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    total = float(w.sum())
    if total <= 0:
        return 0, np.zeros(w.size, dtype=np.int64)
    ceiling = total * target_fraction

    def distortion(bits: np.ndarray) -> float:
        return float(np.sum(w * np.array([DISTORTION[int(b)] for b in bits])))

    widest = np.full(w.size, MAX_WIDTH, dtype=np.int64)
    if distortion(widest) > ceiling:
        return None, None
    lo, hi = 0, int(MAX_WIDTH * w.size)
    best = widest
    while lo < hi:
        mid = (lo + hi) // 2
        bits = allocate_bits(w, mid, tuple(range(MAX_WIDTH + 1)))
        if distortion(bits) <= ceiling:
            hi, best = mid, bits
        else:
            lo = mid + 1
    return layer_bytes(best, norm=norm), best


@dataclass
class FeasibilityReport:
    result: str
    reason: str
    action: str | None
    warnings: list
    source: dict
    observer: dict
    observable: dict
    attainable: dict
    rank: dict
    identification: dict
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "schema": "turboquant-pro/feasibility-report",
            "schema_version": 1,
            "result": self.result,
            "reason": self.reason,
            "action": self.action,
            "warnings": list(self.warnings),
            "source": self.source,
            "observer": self.observer,
            "observable": self.observable,
            "attainable": self.attainable,
            "rank": self.rank,
            "identification": self.identification,
            **self.meta,
        }

    def explain(self) -> str:
        s, o, a = self.source, self.observable, self.attainable
        L = [f"Observer: {self.observer.get('observer') or '<none>'}"]
        if self.observer.get("sha256"):
            L.append(f"  contract {self.observer['sha256'][:16]}")
        L.append(
            f"Source: {s['rows']} rows, {s['dim']} dimensions, "
            f"numerical rank {s['rank']}"
        )
        L.append(f"Observable rank: {o['observable_rank']:.1f} / {s['dim']}")
        L.append(f"Unread source dimensions: {o['unread_dimensions']}")
        L.append(
            "Omitted sensitivity: "
            f"{o['omitted_sensitivity_fraction'] * 100:.2f}% of what the "
            "consumer reads has no variance here"
        )
        floor = a["floor_distortion_fraction"]
        L.append(
            f"Distortion floor at {MAX_WIDTH} bits/direction: {floor:.5f} "
            "of the observable signal"
        )
        if a.get("target_fraction") is not None:
            if a["min_bytes"] is None:
                L.append(
                    f"Target distortion {a['target_fraction']:.5f}: "
                    "NOT REACHABLE at any width"
                )
            else:
                L.append(
                    "Minimum predicted payload at distortion "
                    f"{a['target_fraction']:.5f}: {a['min_bytes']} B/vector"
                )
        for row in a.get("curve", []):
            if row["bytes"] is not None:
                L.append(f"  distortion {row['fraction']:.3f}: {row['bytes']} B/vector")
        if self.rank.get("min_tau") is not None:
            k = self.rank["max_certifiable_kappa"]
            L.append(
                f"Rank: tau >= {self.rank['min_tau']} needs distance distortion "
                f"below kappa {k:.4f}"
                + (
                    "  (no compressed code certifies it)"
                    if self.rank["near_exact"]
                    else ""
                )
            )
        if self.identification.get("checked"):
            frac = self.identification["unidentified_variance_fraction"]
            L.append(
                f"Identification: {frac * 100:.1f}% of source variance lies "
                "outside the subspace the estimated operator could identify"
            )
        for w in self.warnings:
            L.append(f"WARNING: {w}")
        L.append(f"Result: {self.result}")
        L.append(f"  {self.reason}")
        if self.action:
            L.append(f"  Suggested action: {self.action}")
        return "\n".join(L)


def feasibility(
    x: np.ndarray,
    operator: np.ndarray,
    *,
    observer: dict | None = None,
    max_distortion: float | None = None,
    min_tau: float | None = None,
    exact_operator: bool = True,
    metric: str = "cosine",
    rank_sample: int = 256,
    seed: int = 0,
) -> FeasibilityReport:
    """Is the guarantee attainable, and is anything the consumer needs already
    missing?

    Args:
        x: a corpus sample, ``(n, d)``.
        operator: the observer's read operator ``P`` on the channel axis.
        observer: the contract's reference block, for the record.
        max_distortion: the largest acceptable consumer distortion, as a
            fraction of the observable signal ``tr(P Sigma)``.
        min_tau: a Kendall-tau floor to answer through the rank certificate.
        exact_operator: False when the operator is an estimate; the
            identification check then runs and can force ABSTAIN.
        metric, rank_sample, seed: the rank certificate's distance protocol.
    """
    a = np.asarray(x, dtype=np.float64)
    a = a.reshape(-1, a.shape[-1])
    n, d = a.shape
    if np.shape(operator) != (d, d):
        raise ValueError(
            f"operator is {np.shape(operator)}; the corpus has {d} channels"
        )
    var, sens, _basis, A = _moments(a, operator)
    weights = sens * var

    vmax = float(var.max()) if var.size else 0.0
    live = var > VARIANCE_TOL * max(vmax, 1.0)
    numerical_rank = int(live.sum())
    trace_P = float(np.trace(A))
    omitted = float(sens[~live].sum() / trace_P) if trace_P > 0 else 0.0

    signal = float(weights.sum())
    read = _read_mask(weights, signal)
    unread = int((live & ~read).sum())
    s2 = float((weights**2).sum())
    observable_rank = (signal * signal / s2) if s2 > 0 else 0.0

    floor_bits = np.full(d, MAX_WIDTH, dtype=np.int64)
    floor_distortion = float(
        np.sum(weights * np.array([DISTORTION[int(b)] for b in floor_bits]))
    )
    floor_fraction = floor_distortion / signal if signal > 0 else 0.0

    curve = []
    for frac in REFERENCE_DISTORTIONS:
        b, _ = min_bytes_for_distortion(weights, frac)
        curve.append({"fraction": frac, "bytes": b})
    min_bytes = None
    if max_distortion is not None:
        min_bytes, _ = min_bytes_for_distortion(weights, float(max_distortion))

    rank: dict[str, Any] = {"min_tau": min_tau}
    if min_tau is not None:
        from turboquant_pro.rank_certificate import (
            max_certifiable_kappa,
            pairwise_distances,
        )

        rng = np.random.default_rng(seed)
        rows = a if n <= rank_sample else a[rng.choice(n, rank_sample, replace=False)]
        dist = pairwise_distances(rows, metric=metric)
        k = float(max_certifiable_kappa(dist, float(min_tau)))
        rank.update(
            {
                "max_certifiable_kappa": k,
                "vacuous": k <= 1.0,
                # below this ratio only a code that preserves every pairwise
                # distance to within one percent certifies the floor, which no
                # allocation of these widths does; stated as a convention in
                # docs/DESIGN_feasibility.md rather than derived
                "near_exact": k < NEAR_EXACT_KAPPA,
                "near_exact_kappa": NEAR_EXACT_KAPPA,
                "metric": metric,
                "rows": int(rows.shape[0]),
            }
        )

    identification: dict[str, Any] = {
        "checked": not exact_operator,
        "exact": exact_operator,
    }
    if not exact_operator:
        # variance living where the estimate reports no sensitivity: unread, or
        # merely unidentified, and a confined estimate cannot tell them apart
        total_var = float(var.sum())
        unidentified = (
            float(var[live & ~read].sum() / total_var) if total_var > 0 else 0.0
        )
        identification["unidentified_variance_fraction"] = unidentified
        identification["max"] = MAX_UNIDENTIFIED

    warnings: list[str] = []
    if OMISSION_WARN < omitted <= OMISSION_INFEASIBLE:
        warnings.append(
            f"{omitted * 100:.1f}% of what the consumer reads lies in directions this "
            "corpus does not vary along; those directions carry no row-to-row "
            "information, which is free for a reconstruction consumer and useless "
            "for a discriminating one"
        )
    if omitted > OMISSION_INFEASIBLE:
        result = INFEASIBLE
        reason = (
            f"{omitted * 100:.1f}% of what the consumer reads lies in directions this "
            "corpus does not vary along: most of what it reads carries no row-to-row "
            "information here. Either the consumer is declared over directions this "
            "representation does not carry, or the encoder dropped them; compression "
            "repairs neither"
        )
        action = (
            "check the declared consumer against this encoder, then re-embed, retain "
            "the source, or change the upstream encoder"
        )
    elif max_distortion is not None and min_bytes is None:
        result = INFEASIBLE
        reason = (
            f"the widest width ({MAX_WIDTH} bits per direction) leaves "
            f"{floor_fraction:.5f} of the observable signal, above the requested "
            f"{max_distortion:.5f}: no allocation of these widths reaches it"
        )
        action = "raise the distortion target, or keep exact originals for rerank"
    elif min_tau is not None and rank.get("near_exact"):
        k = rank["max_certifiable_kappa"]
        reason = (
            f"tau >= {min_tau} needs distance distortion below kappa {k:.4f} on this "
            "corpus, which no allocation of these widths reaches: its distances are "
            "too concentrated for single-stage ranking at any byte budget"
        )
        result = INFEASIBLE
        action = "uncertified at any code size: validate or rerank"
    elif (
        not exact_operator
        and identification.get("unidentified_variance_fraction", 0.0) > MAX_UNIDENTIFIED
    ):
        result = ABSTAIN
        reason = (
            f"{identification['unidentified_variance_fraction'] * 100:.1f}% of source "
            "variance lies outside the subspace this estimated operator could "
            "identify; "
            "sensitivity reported as zero there may be unread or merely unidentified"
        )
        action = "widen the probe set to full dimension, or declare the operator"
    else:
        result = PASS
        reason = (
            f"the consumer reads {observable_rank:.1f} effective directions of {d}; "
            + (
                f"{min_bytes} B/vector reaches the requested distortion"
                if min_bytes is not None
                else (
                    f"the floor at {MAX_WIDTH} bits is {floor_fraction:.5f} "
                    "of the observable signal"
                )
            )
        )
        action = None

    return FeasibilityReport(
        result=result,
        reason=reason,
        action=action,
        warnings=warnings,
        source={
            "rows": int(n),
            "dim": int(d),
            "rank": numerical_rank,
            "variance_total": float(var.sum()),
        },
        observer=dict(observer or {}),
        observable={
            "signal": signal,
            "observable_rank": observable_rank,
            "unread_dimensions": unread,
            "omitted_sensitivity_fraction": omitted,
            "operator_trace": trace_P,
        },
        attainable={
            "floor_distortion_fraction": floor_fraction,
            "floor_bytes": layer_bytes(floor_bits),
            "target_fraction": max_distortion,
            "min_bytes": min_bytes,
            "curve": curve,
            "widths": list(range(MAX_WIDTH + 1)),
            "norm_bytes": NORM_BYTES,
        },
        rank=rank,
        identification=identification,
        meta={
            "model": (
                "Lloyd-Max distortion table over the corpus spectrum, read through "
                "the consumer's operator; a prediction, not a measured recall"
            )
        },
    )

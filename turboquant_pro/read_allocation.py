# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Spend bits against a measured read operator.

The middle third of the governing principle. ``read_operators`` supplies
``P_C``, the rank certificate judges the result, and this is the step between:
given how much a consumer responds along each direction, decide where the bits
go.

The problem is exactly reverse water-filling. Writing ``lambda_i`` for the
consumer's sensitivity along eigendirection ``i`` and ``sigma_i^2`` for the
source variance there, a quantizer that spends ``b_i`` bits on that direction
leaves the consumer feeling

    D(b) = sum_i lambda_i sigma_i^2 2^(-2 b_i)

and minimizing under a total budget gives

    b_i = max(0, 0.5 log2(lambda_i sigma_i^2 / theta))

with ``theta`` set by the budget. Directions whose sensitivity-weighted
variance falls below the water level get nothing at all.

**This is not an analogy to power allocation across frequency bins; it is the
same optimization.** What changes is the weight: a downstream task's
sensitivity rather than a signal's power. That substitution is the whole
content of consumer-aware compression, and it is why allocating against
``P_C`` is different from allocating against the covariance of the data.

**Two allocators already exist and neither does this.**
``strata_ops.allocate_by_fragility`` spends against measured per-stratum
anti-hub recall, which is an empirical fragility signal rather than a read
operator. ``rope.bit_allocation`` splits bits by RoPE frequency band, which is
a structural prior specific to rotary keys. Both are useful; neither can see a
consumer that was measured rather than assumed.

**A caution the calibration record earns.** The allocation is only as good as
the operator, and a blindly recovered operator has a budget law: recovery
against the probe's direction budget is a cliff at ``k = d`` and the cliff is
rank independent, so an operator recovered sub-dimensionally describes the
dominant direction and little else. Allocating against such an operator will
confidently starve directions it never actually measured, so
:func:`allocate_bits` reports the operator's effective rank alongside the
allocation and :func:`allocation_report` says plainly when the spectrum is too
concentrated for the allocation to mean much.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .read_operators import consumer_distortion

__all__ = [
    "BitAllocation",
    "allocate_bits",
    "allocation_report",
    "predicted_distortion",
    "uniform_bits",
]


@dataclass
class BitAllocation:
    """A per-direction bit allocation and what it is expected to cost."""

    bits: np.ndarray
    """Bits per eigendirection of the read operator, in its eigenbasis."""

    basis: np.ndarray
    """Eigenvectors, columns ordered by descending sensitivity."""

    sensitivity: np.ndarray
    """Eigenvalues of ``P_C``, descending."""

    variance: np.ndarray
    """Source variance along each eigendirection."""

    water_level: float
    budget_bits: float
    predicted_distortion: float
    n_starved: int
    """Directions that received no bits."""

    effective_rank: float
    """Participation ratio of the sensitivity spectrum."""

    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_bits(self) -> float:
        return float(self.bits.mean()) if self.bits.size else 0.0

    def to_dict(self) -> dict:
        return {
            "bits": [float(b) for b in self.bits],
            "sensitivity": [float(v) for v in self.sensitivity],
            "variance": [float(v) for v in self.variance],
            "water_level": self.water_level,
            "budget_bits": self.budget_bits,
            "mean_bits": self.mean_bits,
            "predicted_distortion": self.predicted_distortion,
            "n_starved": self.n_starved,
            "effective_rank": self.effective_rank,
            **self.meta,
        }


def _spectrum(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(P, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("read operator must be square")
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    order = np.argsort(vals)[::-1]
    return np.clip(vals[order], 0.0, None), np.ascontiguousarray(vecs[:, order])


def _variance_along(basis: np.ndarray, activations: np.ndarray | None, d: int):
    if activations is None:
        return np.ones(d)
    a = np.asarray(activations, dtype=np.float64)
    a = a.reshape(-1, a.shape[-1])
    if a.shape[1] != d:
        raise ValueError(
            f"activations have {a.shape[1]} channels but the operator is {d}"
        )
    centred = a - a.mean(axis=0, keepdims=True)
    proj = centred @ basis
    return np.maximum((proj**2).mean(axis=0), 0.0)


def predicted_distortion(
    sensitivity: np.ndarray, variance: np.ndarray, bits: np.ndarray
) -> float:
    """``sum_i lambda_i sigma_i^2 2^(-2 b_i)`` for a given allocation."""
    lam = np.asarray(sensitivity, dtype=np.float64).ravel()
    var = np.asarray(variance, dtype=np.float64).ravel()
    b = np.asarray(bits, dtype=np.float64).ravel()
    return float(np.sum(lam * var * np.power(2.0, -2.0 * b)))


def uniform_bits(n: int, budget_bits: float) -> np.ndarray:
    """Equal bits everywhere: the baseline an allocation has to beat."""
    if n < 1:
        raise ValueError("n must be positive")
    return np.full(n, float(budget_bits) / n)


def allocate_bits(
    read_operator: np.ndarray,
    *,
    budget_bits: float,
    activations: np.ndarray | None = None,
    max_bits: float | None = None,
    tol: float = 1e-12,
) -> BitAllocation:
    """Reverse water-fill ``budget_bits`` against a read operator.

    Args:
        read_operator: ``(D, D)`` PSD ``P_C`` from any registered provider.
        budget_bits: total bits to spend, summed over directions. A codec
            quoting ``b`` bits per component on ``D`` components has a budget
            of ``b * D``.
        activations: optional ``(N, D)`` sample used to measure the source
            variance along each eigendirection. Omitted means whitened, which
            allocates against sensitivity alone.
        max_bits: optional per-direction ceiling for a fixed-width codec.

    Returns:
        A :class:`BitAllocation` in the operator's eigenbasis. Applying it
        means rotating into ``basis``, quantizing component ``i`` at
        ``bits[i]``, and rotating back.
    """
    lam, basis = _spectrum(read_operator)
    d = lam.size
    var = _variance_along(basis, activations, d)
    if budget_bits < 0:
        raise ValueError("budget_bits must be non-negative")

    w = lam * var
    s1, s2 = float(lam.sum()), float((lam**2).sum())
    eff = (s1 * s1 / s2) if s2 > 0 else 0.0

    def bits_at(theta: float) -> np.ndarray:
        b = np.zeros(d)
        live = w > theta
        b[live] = 0.5 * np.log2(w[live] / theta)
        if max_bits is not None:
            b = np.minimum(b, max_bits)
        return b

    if not np.any(w > 0) or budget_bits == 0.0:
        bits = np.zeros(d)
        return BitAllocation(
            bits=bits,
            basis=basis,
            sensitivity=lam,
            variance=var,
            water_level=float(w.max()) if w.size else 0.0,
            budget_bits=float(budget_bits),
            predicted_distortion=predicted_distortion(lam, var, bits),
            n_starved=d,
            effective_rank=eff,
            meta={"degenerate": True},
        )

    hi = float(w.max())
    lo = hi
    while bits_at(lo).sum() < budget_bits:
        lo *= 0.5
        if lo < 1e-300:
            break
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if bits_at(mid).sum() > budget_bits:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(hi, 1.0):
            break
    theta = 0.5 * (lo + hi)
    bits = bits_at(theta)

    return BitAllocation(
        bits=bits,
        basis=basis,
        sensitivity=lam,
        variance=var,
        water_level=float(theta),
        budget_bits=float(budget_bits),
        predicted_distortion=predicted_distortion(lam, var, bits),
        n_starved=int(np.sum(bits <= 0.0)),
        effective_rank=eff,
        meta={"whitened": activations is None},
    )


def allocation_report(
    read_operator: np.ndarray,
    *,
    budget_bits: float,
    activations: np.ndarray | None = None,
    max_bits: float | None = None,
    concentration_warn: float = 2.0,
) -> dict:
    """Allocate, compare against uniform, and say when it does not mean much.

    The gain over uniform allocation is the number worth reporting: it is what
    consumer-awareness buys at a fixed budget. The caution is equally
    important. When the operator's effective rank is very low the allocation
    will pour everything into one or two directions, which is correct if the
    operator is right and catastrophic if it was recovered at a
    sub-dimensional probe budget, where a cliff at ``k = d`` leaves exactly
    that signature whatever the consumer actually reads.
    """
    alloc = allocate_bits(
        read_operator,
        budget_bits=budget_bits,
        activations=activations,
        max_bits=max_bits,
    )
    uni = uniform_bits(alloc.bits.size, budget_bits)
    d_uniform = predicted_distortion(alloc.sensitivity, alloc.variance, uni)
    d_alloc = alloc.predicted_distortion
    gain = (d_uniform / d_alloc) if d_alloc > 0 else float("inf")

    concentrated = alloc.effective_rank < concentration_warn
    return {
        "allocation": alloc.to_dict(),
        "uniform_distortion": d_uniform,
        "allocated_distortion": d_alloc,
        "gain_over_uniform": gain,
        "spectrum_concentrated": bool(concentrated),
        "caution": (
            "the read operator's effective rank is "
            f"{alloc.effective_rank:.2f}, so this allocation rests on one or "
            "two directions. That is correct if the operator is, and is also "
            "exactly what a sub-dimensional probe budget produces regardless "
            "of the consumer: recovery is a cliff at k = d and the cliff is "
            "rank independent. Check how the operator was obtained before "
            "trusting the split."
            if concentrated
            else ""
        ),
    }


def realised_distortion(
    read_operator: np.ndarray,
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> float:
    """What the consumer actually lost, for checking a prediction.

    :func:`allocate_bits` predicts a distortion from a rate-distortion model.
    This measures the real one from a codec's actual output, so the two can be
    compared rather than the model being trusted.
    """
    from .read_operators import error_covariance

    return consumer_distortion(read_operator, error_covariance(original, reconstructed))

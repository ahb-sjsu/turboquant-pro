# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""What a learned read operator costs: a confidence band and a price in bits.

Every consumer-aware allocation in this package spends bits against a weight
``Q`` (a read operator ``P_C``), and almost every such weight is an average of
samples: the Jacobian Gram of an attention head over probed keys
(``attention_analytic``), ``E[q q^T]`` over a sample of queries for a
retrieval consumer, a blind recovery by finite differences (readscope). The
allocation then treats the average as the truth. This module prices that.

The source is the T1 note of the rate-leakage-distortion series,
geometric-observation ``paper/t1-learn-then-serve`` (v4.1). Its proofs have
fresh-context audits on record (VI-23, VI-24); its custody table is still
owed, so it is cited as a working note. Statement numbers below are the
note's.

**The band (Lemma 1).** Let ``P_1 .. P_m`` be i.i.d. PSD samples with mean
``Q``, and suppose it is known in advance that
``(1 - nu_minus) Q <= P_i <= (1 + nu_plus) Q``. With probability at least
``1 - delta`` the sample mean ``Q_hat`` satisfies

    (1 - eps_minus) Q  <=  Q_hat  <=  (1 + eps_plus) Q,
    eps_pm = sqrt(2 nu_plus nu_minus L / m) + 2 nu_pm L / (3 m),
    L = ln(2 p / delta).

A read metric is PSD, so the lower half holds with ``nu_minus = 1`` for every
consumer: only ``nu_plus`` must be declared.

**The learning price** is

    tau = (p / 2) log2((1 + eps_plus) / (1 - eps_minus))

bits per vector, and what it bounds depends on the consumer. Samples come in
two kinds:

* ``kind="linear"``: terms of a *linear* consumer's weight, such as the
  ``q q^T`` of a retrieval consumer's queries. The design for ``Q_hat`` at
  distortion ``D (1 - eps_minus)`` is safe for every linear consumer
  consistent with the band, and its regret is at most ``tau`` in rate and in
  leakage at once (Proposition 12, its closing remark, by (P1)-(P3)).
* ``kind="points"``: read metrics ``P_C(x_i)`` of a nonlinear consumer at probe
  points (T1's assumption (A2)). Then ``tau`` is the *learning part* of
  Corollary 8's gap. The whole gap is ``tau_bar = tau + O(D)`` at the design
  distortion ``D_hat = (1 - eps_minus)(D - rho_bar(D))``, and it needs the
  consumer's derivative bounds: :meth:`LearnedWeight.gap` computes it. The
  corollary's setting is paper I's (memoryless Gaussian source, the decoder
  reproducing it, the probe law the source law); in the process settings of
  papers II-IV the transfer to codes is open (Remark 6).

For an encoder given only ``Q_hat`` and the band, ``tau`` is sharp
(Proposition 12).

**What is new here, beyond transcription.** The declared ``nu_plus`` is a
claim, and a sample can refute it. On the band's event
``Q <= Q_hat / (1 - eps_minus)``, so a sample with
``lambda_max(Q_hat^-1/2 P_i Q_hat^-1/2) > (1 + nu_plus) / (1 - eps_minus)``
is impossible unless the event failed, which happens with probability at most
``delta``. :func:`learn_weight` runs that test (and its mirror for a declared
``nu_minus < 1``) and returns ``REFUTED`` rather than a certificate. A sample
can refute a constant; it can never confirm one, so the observed spread is
reported as a diagnostic and never substituted for the declared value.

**Lower rank is not priced.** Lemma 1 and Corollary 8 need ``Q > 0``. The
note's Remark 14 extends only Theorem 11, under the sandwich (A1) with both
constants below one and the loss sandwich (A1a) declared in advance; a kernel
seen to be fixed in a sample does not supply that, because a sample can
refute a constant but never confirm one. So a ``Q_hat`` whose numerical rank
falls short of ``p`` (including an ill-conditioned one whose smallest
eigenvalues fall under the rank cut) gets ``NO_CERTIFICATE``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "CERTIFIED",
    "NO_CERTIFICATE",
    "REFUTED",
    "CalibrationMessage",
    "KernelGeometry",
    "LearnedWeight",
    "calibration_message",
    "combine_bands",
    "KINDS",
    "confidence_band",
    "corollary8_gap",
    "finite_difference_error",
    "kernel_geometry",
    "learn_weight",
    "learning_price",
    "max_step_for_price",
    "price_of_samples",
    "probes_for_price",
    "quantize_weight",
]

CERTIFIED = "CERTIFIED"
REFUTED = "REFUTED"
NO_CERTIFICATE = "NO_CERTIFICATE"

NOTE = "geometric-observation paper/t1-learn-then-serve v4.1"
ADDENDUM = "geometric-observation paper/t1-moving-kernel"
_ROUNDING = 1e-9


# ---------------------------------------------------------------------------
# Lemma 1, the price, Corollary 3
# ---------------------------------------------------------------------------


def _check_constants(m: int, dim: int, delta: float, nu_plus, nu_minus) -> None:
    if int(m) != m or m < 1:
        raise ValueError(f"m must be a positive integer, got {m!r}")
    if int(dim) != dim or dim < 1:
        raise ValueError(f"dim must be a positive integer, got {dim!r}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie in (0, 1), got {delta!r}")
    if not nu_plus >= 0.0 or not math.isfinite(nu_plus):
        raise ValueError(f"nu_plus must be finite and >= 0, got {nu_plus!r}")
    if not 0.0 <= nu_minus <= 1.0:
        raise ValueError(f"nu_minus must lie in [0, 1], got {nu_minus!r}")


def confidence_band(
    m: int,
    dim: int,
    delta: float,
    nu_plus: float,
    nu_minus: float = 1.0,
    *,
    exact_roots: bool = False,
) -> tuple[float, float]:
    """``(eps_plus, eps_minus)`` of Lemma 1 for ``m`` samples in dimension ``dim``.

    ``exact_roots=True`` returns the roots ``t_pm`` of the Bernstein exponents
    from the lemma's proof, which the note allows in place of ``eps_pm``:
    each is at most the stated value and each tail is still at most
    ``delta / 2`` there.
    """
    _check_constants(m, dim, delta, nu_plus, nu_minus)
    L = math.log(2.0 * dim / delta)
    var = 2.0 * nu_plus * nu_minus * L / m
    if exact_roots:
        a_p, a_m = nu_plus * L / (3.0 * m), nu_minus * L / (3.0 * m)
        return a_p + math.sqrt(a_p * a_p + var), a_m + math.sqrt(a_m * a_m + var)
    root = math.sqrt(var)
    return root + 2.0 * nu_plus * L / (3.0 * m), root + 2.0 * nu_minus * L / (3.0 * m)


def learning_price(eps_plus: float, eps_minus: float, dim: int) -> float:
    """``tau = (dim/2) log2((1 + eps_plus) / (1 - eps_minus))`` bits per vector.

    Infinite when ``eps_minus >= 1``: the band then gives no lower bound on
    the true weight, and no finite price follows.
    """
    if eps_plus < 0 or eps_minus < 0:
        raise ValueError("relative errors must be non-negative")
    if eps_minus >= 1.0:
        return math.inf
    return 0.5 * dim * math.log2((1.0 + eps_plus) / (1.0 - eps_minus))


def combine_bands(bands: Iterable[tuple[float, float]]) -> tuple[float, float]:
    """The band of a non-negative weighted sum of weights, each with its own band.

    If ``(1 - e_c-) Q_c <= Q_hat_c <= (1 + e_c+) Q_c`` for every consumer
    ``c``, the Loewner order survives non-negative sums, so the sum obeys the
    band with the largest of each error. The events must hold together, so
    split ``delta`` across the estimated consumers (a union bound) before
    computing each band.
    """
    ep = em = 0.0
    for p, m in bands:
        ep, em = max(ep, p), max(em, m)
    return ep, em


def probes_for_price(
    eta: float,
    dim: int,
    delta: float,
    nu_plus: float,
    nu_minus: float = 1.0,
    *,
    finite_differences: bool = False,
) -> dict:
    """How many samples make the learning price at most ``eta`` bits.

    Returns ``corollary``, the closed-form sufficient count of Corollary 3
    ((i) exact read metrics, (ii) with central differences, which leaves a
    third of each error for the step), and ``minimal``, the least ``m`` at
    which Lemma 1's own band gives a price of at most ``eta`` (with exact
    read metrics only; with differences use ``corollary`` together with
    :func:`max_step_for_price`). Both are rigorous; ``minimal`` is smaller.
    """
    if not 0.0 < eta <= dim:
        raise ValueError(f"eta must lie in (0, dim], got {eta!r}")
    _check_constants(1, dim, delta, nu_plus, nu_minus)
    L = math.log(2.0 * dim / delta)
    nu = max(nu_plus, nu_minus)
    a, b = (50.0, 10.0 / 3.0) if finite_differences else (200.0 / 9.0, 20.0 / 9.0)
    m_cor = max(
        1,
        math.ceil(
            L * max(a * nu_plus * nu_minus * dim**2 / eta**2, b * nu * dim / eta)
        ),
    )
    out = {
        "eta_bits": float(eta),
        "dim": int(dim),
        "delta": float(delta),
        "nu_plus": float(nu_plus),
        "nu_minus": float(nu_minus),
        "corollary": int(m_cor),
        "statement": "Corollary 3(ii)" if finite_differences else "Corollary 3(i)",
    }
    if finite_differences:
        out["minimal"] = None
        return out

    def price(m: int) -> float:
        return learning_price(*confidence_band(m, dim, delta, nu_plus, nu_minus), dim)

    lo, hi = 0, m_cor  # price(m_cor) <= eta by the corollary; find the least m
    while price(hi) > eta:  # not expected (the note's check C3), but never trust it
        lo, hi = hi, 2 * hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if price(mid) <= eta:
            hi = mid
        else:
            lo = mid
    out["minimal"] = int(hi)
    return out


# ---------------------------------------------------------------------------
# Proposition 2 (finite differences) and Proposition 4 (the calibration message)
# ---------------------------------------------------------------------------


def finite_difference_error(
    dim: int,
    h: float,
    *,
    lipschitz: float,
    lam_min: float,
    hessian_bound: float | None = None,
    third_bound: float | None = None,
) -> float:
    """``eps_h`` of Proposition 2: what central differences of step ``h`` add.

    ``lipschitz`` is ``M1`` (the weighted map's Lipschitz constant),
    ``hessian_bound`` ``M2`` and ``third_bound`` ``M3`` bound its second and
    third derivatives (give at least one), and ``lam_min`` is an a priori
    ``lambda <= lambda_min(Q)``. The error is deterministic, so it adds to
    both ``eps_pm`` rather than to the variance. Noiseless queries only:
    output noise adds a bias of order ``sigma^2 / h^2`` (Remark 17).
    """
    if h <= 0 or lam_min <= 0 or lipschitz < 0:
        raise ValueError("need h > 0, lam_min > 0, lipschitz >= 0")
    betas = []
    if hessian_bound is not None:
        betas.append(hessian_bound * h / 2.0)
    if third_bound is not None:
        betas.append(third_bound * h * h / 6.0)
    if not betas:
        raise ValueError("give hessian_bound (M2) or third_bound (M3)")
    beta = min(betas)
    return (2.0 * lipschitz * math.sqrt(dim) * beta + dim * beta * beta) / lam_min


def max_step_for_price(
    eta: float, dim: int, *, lipschitz: float, third_bound: float, lam_min: float
) -> float:
    """The largest step of Corollary 3(ii): ``h^2 <= (6/M3) min{...}``."""
    if not 0.0 < eta <= dim or third_bound <= 0 or lam_min <= 0:
        raise ValueError("need 0 < eta <= dim, third_bound > 0, lam_min > 0")
    first = lam_min * eta / (20.0 * lipschitz * dim**1.5) if lipschitz > 0 else math.inf
    second = math.sqrt(lam_min * eta / (10.0 * dim * dim))
    return math.sqrt(6.0 / third_bound * min(first, second))


@dataclass(frozen=True)
class CalibrationMessage:
    """Proposition 4: the weight sent once, on a grid, and what that adds."""

    dim: int
    step: float
    levels: int
    """``N_q``: the grid is ``{j * step : |j| <= N_q}``."""
    bits: int
    """``B``: the message size, sent once per session."""
    added_price_bits: float
    """``(p/2) log2((1 + eps_q) / (1 - eps_q))``, added to the learning price."""

    def as_dict(self) -> dict:
        return {
            "dim": self.dim,
            "step": self.step,
            "levels": self.levels,
            "bits": self.bits,
            "added_price_bits": self.added_price_bits,
            "statement": "Proposition 4",
        }


def calibration_message(
    dim: int,
    *,
    lam_min: float,
    lam_max: float,
    eps_q: float,
    eps_plus: float,
    eps_minus: float,
) -> CalibrationMessage:
    """Size of the one-time message that hands a decoder the learned weight.

    The weight is a function of the probes and the consumer only, so the
    message carries no information about the source: when the realized
    calibration is known to every party, as a code is, the leakage
    ``I(T; Q_tilde, M | S)`` is that of the code designed for ``Q_tilde``. In
    the entropy form ``H(Q_tilde, M | S) / K``, which also counts the
    message's own entropy, it adds at most ``bits / K`` per vector, as it
    does to the rate, over a session of ``K`` vectors.
    """
    if not (0 < lam_min <= lam_max) or not 0 < eps_q < 1 or not 0 <= eps_minus < 1:
        raise ValueError("need 0 < lam_min <= lam_max, 0 < eps_q < 1, eps_minus < 1")
    step = 2.0 * (1.0 - eps_minus) * lam_min * eps_q / dim
    kappa = lam_max / lam_min
    levels = math.ceil(
        dim * kappa * (1.0 + eps_plus) / (2.0 * (1.0 - eps_minus) * eps_q)
    )
    bits = dim * (dim + 1) // 2 * math.ceil(math.log2(2 * levels + 1))
    return CalibrationMessage(
        dim=int(dim),
        step=float(step),
        levels=int(levels),
        bits=int(bits),
        added_price_bits=0.5 * dim * math.log2((1.0 + eps_q) / (1.0 - eps_q)),
    )


def quantize_weight(Q_hat: np.ndarray, message: CalibrationMessage) -> np.ndarray:
    """The rounded weight ``Q_tilde`` of Proposition 4, symmetric by construction.

    Raises if an entry falls outside the grid, which the proposition rules
    out on the band's event: off the event the message cannot be formed.
    """
    A = _symmetric(np.asarray(Q_hat, dtype=np.float64))
    if A.shape != (message.dim, message.dim):
        raise ValueError(f"weight is {A.shape}, message is for dim {message.dim}")
    j = np.rint(A / message.step)
    if np.abs(j).max() > message.levels:
        raise ValueError(
            "an entry of the weight lies outside the message grid; the declared "
            "lam_max or the band does not hold for this weight"
        )
    iu = np.triu_indices(message.dim)
    out = np.zeros_like(A)
    out[iu] = j[iu] * message.step
    return out + np.triu(out, 1).T


# ---------------------------------------------------------------------------
# Per-sample spectra relative to Q_hat, and the kernel geometry (addendum)
# ---------------------------------------------------------------------------


def _symmetric(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + np.swapaxes(A, -1, -2))


def _as_factor_list(factors: Any) -> list[np.ndarray]:
    """Sample ``i`` as a factor ``A_i`` with ``P_i = A_i^T A_i``.

    A 2-D ``(m, p)`` array is ``m`` rank-one factors (a query sample: the
    retrieval weight ``E[q q^T]``); a 3-D ``(m, k, p)`` array is ``m`` stacked
    Jacobians; a sequence may mix row counts.
    """
    if isinstance(factors, np.ndarray):
        a = np.asarray(factors, dtype=np.float64)
        if a.ndim == 2:
            return [a[i : i + 1] for i in range(a.shape[0])]
        if a.ndim == 3:
            return [a[i] for i in range(a.shape[0])]
        raise ValueError(f"factors must be 2-D or 3-D, got shape {a.shape}")
    out = [np.atleast_2d(np.asarray(f, dtype=np.float64)) for f in factors]
    if not out:
        raise ValueError("no samples")
    return out


@dataclass(frozen=True)
class KernelGeometry:
    """Whether the directions a consumer ignores move with the sample point.

    From the moving-kernel addendum, for read metrics at points of the source
    (``kind="points"``): ``q = rank Q`` counts the directions read at some
    point (Lemma 1), ``r`` the rank of the read metric at a point, and the
    kernel moves when ``q > r``. Then fixed-covariance Gaussian test channels,
    the class a single weight serves, exceed the consumer's own
    rate-distortion function by ``((q - k) / 2) log2(1/D) + O(1)`` bits as the
    loss ``D -> 0``, ``k`` the consumer's output dimension (Corollary 9).

    The corollary's hypotheses are the caller's to establish: a measurable
    right inverse on a closed convex image, an output with a density, finite
    differential entropy and positive definite covariance, bounded
    derivatives, ``Q`` under the source law. The coefficient is reported only
    when some sample reaches rank ``k``, a necessary condition for a density,
    not a sufficient one. ``q`` here is the rank of the *sample* mean, which
    is at most ``rank Q``, so the coefficient is a lower bound on the
    corollary's. It concerns one class of test channels, not every code.
    """

    q: int
    point_rank_min: int
    point_rank_max: int
    kernel_moves: bool
    output_dim: int | None
    penalty_coefficient: float | None
    """``(q - k) / 2`` from the sample's ``q``, bits per halving of ``D``; at
    most the corollary's coefficient. ``None`` without ``k``."""

    def as_dict(self) -> dict:
        return {
            "q_rank_of_mean": self.q,
            "point_rank_min": self.point_rank_min,
            "point_rank_max": self.point_rank_max,
            "kernel_moves": self.kernel_moves,
            "output_dim": self.output_dim,
            "fixed_weight_penalty_coefficient": self.penalty_coefficient,
            "statement": f"{ADDENDUM}, Corollary 9",
            "scope": "fixed-covariance Gaussian test channels against the "
            "consumer's own rate-distortion function, as D -> 0; not every code; "
            "the corollary's hypotheses are the caller's; q is the sample mean's "
            "rank, so the coefficient is a lower bound",
        }


@dataclass
class _Spectra:
    Q_hat: np.ndarray
    basis: np.ndarray  # (p, r) orthonormal basis of range(Q_hat)
    rank: int
    lam_max: np.ndarray  # per-sample lambda_max of Q_hat^-1/2 P_i Q_hat^-1/2
    lam_min: np.ndarray  # per-sample lambda_min on the range
    point_rank: np.ndarray
    kernel_energy: float  # largest fraction of a sample's trace off the range


def _spectra(
    factors: list[np.ndarray] | None,
    metrics: np.ndarray | None,
    rank_rtol: float,
    point_tol: float,
) -> _Spectra:
    if metrics is not None:
        P = _symmetric(np.asarray(metrics, dtype=np.float64))
        if P.ndim != 3 or P.shape[1] != P.shape[2]:
            raise ValueError(f"metrics must be (m, p, p), got {P.shape}")
        scale = float(np.abs(P).max()) or 1.0
        if float(np.linalg.eigvalsh(P).min()) < -1e-9 * scale:
            raise ValueError("a sample read metric is not PSD")
        Q_hat = P.mean(axis=0)
        p = P.shape[1]
    else:
        p = factors[0].shape[1]
        if any(f.shape[1] != p for f in factors):
            raise ValueError("every factor must have the same number of columns")
        Q_hat = sum(f.T @ f for f in factors) / len(factors)
    w, V = np.linalg.eigh(_symmetric(Q_hat))
    top = float(w.max())
    if top <= 0.0:
        raise ValueError("the mean read metric is zero: the consumer reads nothing")
    keep = w > rank_rtol * top
    r = int(keep.sum())
    U = V[:, keep]
    Wr = U / np.sqrt(w[keep])  # (p, r): maps the range to whitened coordinates

    lam_max, lam_min, ranks, off = [], [], [], 0.0
    items = range(P.shape[0]) if metrics is not None else range(len(factors))
    for i in items:
        if metrics is not None:
            Y = _symmetric(Wr.T @ P[i] @ Wr)
            ev = np.linalg.eigvalsh(Y)
            tr_all = float(np.trace(P[i]))
            tr_rng = float(np.trace(U.T @ P[i] @ U))
        else:
            B = factors[i] @ Wr  # (k_i, r)
            s2 = np.linalg.svd(B, compute_uv=False) ** 2
            ev = np.zeros(r)
            ev[: min(r, s2.size)] = np.sort(s2)[::-1][:r]
            ev = np.sort(ev)
            tr_all = float(np.sum(factors[i] ** 2))
            tr_rng = float(np.sum((factors[i] @ U) ** 2))
        lam_max.append(float(ev[-1]))
        lam_min.append(float(ev[0]))
        ranks.append(int(np.sum(ev > point_tol)))
        if tr_all > 0:
            off = max(off, (tr_all - tr_rng) / tr_all)
    return _Spectra(
        Q_hat=Q_hat,
        basis=U,
        rank=r,
        lam_max=np.asarray(lam_max),
        lam_min=np.asarray(lam_min),
        point_rank=np.asarray(ranks),
        kernel_energy=float(max(off, 0.0)),
    )


def kernel_geometry(
    factors: Any = None,
    *,
    metrics: np.ndarray | None = None,
    output_dim: int | None = None,
    rank_rtol: float = 1e-9,
    point_tol: float = 1e-8,
) -> KernelGeometry:
    """The addendum's diagnostic from the same samples :func:`learn_weight` takes."""
    sp = _spectra(
        None if factors is None else _as_factor_list(factors),
        metrics,
        rank_rtol,
        point_tol,
    )
    return _geometry(sp, output_dim)


def _geometry(sp: _Spectra, output_dim: int | None) -> KernelGeometry:
    rmin, rmax = int(sp.point_rank.min()), int(sp.point_rank.max())
    coef = None
    if output_dim is not None:
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        # Corollary 9 needs an output with a density in R^k, so the read
        # metric must reach rank k somewhere; otherwise its hypotheses fail
        # and no coefficient is stated.
        if rmax >= output_dim:
            coef = (sp.rank - int(output_dim)) / 2.0
    return KernelGeometry(
        q=sp.rank,
        point_rank_min=rmin,
        point_rank_max=rmax,
        # every sample's range lies in range(Q_hat), so the kernel is fixed
        # exactly when every sample has the mean's rank
        kernel_moves=bool(rmin < sp.rank),
        output_dim=None if output_dim is None else int(output_dim),
        penalty_coefficient=coef,
    )


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

KINDS = ("points", "linear")

_SCOPE = {
    "linear": "regret bound: the design for Q_hat at distortion D (1 - eps_minus) "
    "is safe for every linear consumer consistent with the band and within tau "
    "bits per vector of the true weight's frontier, in rate and in leakage "
    "(Proposition 12, its closing remark, by (P1)-(P3))",
    "points": "the learning part of Corollary 8's gap: the whole gap is "
    "tau_bar = tau + O(D), computed by LearnedWeight.gap from the consumer's "
    "derivative bounds; paper I's memoryless setting with the probe law the "
    "source law (in the process settings the transfer to codes is open)",
}


def corollary8_gap(
    *,
    dim: int,
    m: int,
    delta: float,
    nu_plus: float,
    D: float,
    lam_min_q_hat: float,
    lipschitz: float,
    hessian_bound: float,
    third_bound: float,
    fd_error: float = 0.0,
    exact_roots: bool = False,
) -> dict:
    """Corollary 8: the whole certified gap for a smooth nonlinear consumer.

    With ``nu_minus = 1`` (the corollary's), ``s = D / lambda_min(Q_hat)`` and

        rho_bar(D) = (7/4 M2^2 + 2 M1 M3) s^2 + (sqrt5 / 2) M2 M3 s^(5/2)
                     + (5/12) M3^2 s^3,

    if ``rho_bar(D) < D`` then on the band's event every Gaussian test channel
    with ``tr(Q_hat Delta) <= D_hat = (1 - eps-)(D - rho_bar(D))`` has consumer
    loss at most ``D``, the codes of paper I achieve its rate and leakage at
    that loss, and

        tau_bar = (p/2) log2((1 + eps+) / ((1 - eps-)(1 - rho_bar(D)/D))).

    Raises when ``rho_bar(D) >= D``: the corollary then certifies nothing at
    this distortion. For a linear consumer ``rho_bar = 0`` and this is the
    learning price itself.
    """
    if D <= 0 or lam_min_q_hat <= 0:
        raise ValueError("need D > 0 and lam_min_q_hat > 0")
    if min(lipschitz, hessian_bound, third_bound) < 0:
        raise ValueError("derivative bounds must be non-negative")
    ep, em = confidence_band(m, dim, delta, nu_plus, 1.0, exact_roots=exact_roots)
    ep, em = ep + fd_error, em + fd_error
    if em >= 1.0:
        raise ValueError(f"eps_minus = {em:.3g} >= 1: the band has no lower bound")
    M1, M2, M3 = lipschitz, hessian_bound, third_bound
    s = D / lam_min_q_hat
    rho = (
        (1.75 * M2 * M2 + 2.0 * M1 * M3) * s * s
        + 0.5 * math.sqrt(5.0) * M2 * M3 * s**2.5
        + (5.0 / 12.0) * M3 * M3 * s**3
    )
    if rho >= D:
        raise ValueError(
            f"rho_bar(D) = {rho:.4g} >= D = {D:.4g}: Corollary 8 certifies nothing "
            "at this distortion; take D smaller"
        )
    return {
        "D": float(D),
        "rho_bar": float(rho),
        "design_distortion": float((1.0 - em) * (D - rho)),
        "tau_bar_bits": 0.5
        * dim
        * math.log2((1.0 + ep) / ((1.0 - em) * (1.0 - rho / D))),
        "learning_part_bits": learning_price(ep, em, dim),
        "eps_plus": float(ep),
        "eps_minus": float(em),
        "statement": "Corollary 8 (nu_minus = 1)",
        "scope": _SCOPE["points"],
    }


@dataclass
class LearnedWeight:
    """A learned weight, its band, its learning price, and whether it may be used."""

    Q_hat: np.ndarray
    kind: str
    status: str
    reasons: list[str]
    m: int
    dim: int
    rank: int
    delta: float
    nu_plus: float
    nu_minus: float
    eps_plus: float
    eps_minus: float
    fd_error: float
    price_bits: float
    """``tau`` in bits per vector when ``CERTIFIED``, else ``inf``. What it
    bounds depends on ``kind``: see ``price_scope``."""
    observed_plus: float
    """``max_i lambda_max - 1`` relative to ``Q_hat``: a diagnostic only."""
    observed_minus: float
    """``1 - min_i lambda_min`` on the range, relative to ``Q_hat``."""
    refutation_bounds: tuple[float, float]
    """What the declared constants allow a sample to show, on the band's event."""
    geometry: KernelGeometry | None
    """The moving-kernel diagnostic, for ``kind="points"`` only."""
    exact_roots: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def certified(self) -> bool:
        return self.status == CERTIFIED

    @property
    def price_scope(self) -> str:
        return _SCOPE[self.kind]

    def gap(
        self, D: float, *, lipschitz: float, hessian_bound: float, third_bound: float
    ) -> dict:
        """Corollary 8's whole gap at loss ``D`` (:func:`corollary8_gap`)."""
        if self.kind != "points":
            raise ValueError(
                "Corollary 8 is for read metrics at points (kind='points')"
            )
        if not self.certified:
            raise ValueError(f"no gap without a certificate ({self.status})")
        lam = float(np.linalg.eigvalsh(_symmetric(self.Q_hat)).min())
        return corollary8_gap(
            dim=self.dim,
            m=self.m,
            delta=self.delta,
            nu_plus=self.nu_plus,
            D=D,
            lam_min_q_hat=lam,
            lipschitz=lipschitz,
            hessian_bound=hessian_bound,
            third_bound=third_bound,
            fd_error=self.fd_error,
            exact_roots=self.exact_roots,
        )

    def as_dict(self) -> dict:
        finite = math.isfinite(self.price_bits)
        return {
            "status": self.status,
            "kind": self.kind,
            "reasons": list(self.reasons),
            "m": self.m,
            "dim": self.dim,
            "rank": self.rank,
            "delta": self.delta,
            "declared": {"nu_plus": self.nu_plus, "nu_minus": self.nu_minus},
            "band": {
                "eps_plus": self.eps_plus,
                "eps_minus": self.eps_minus,
                "fd_error": self.fd_error,
                "exact_roots": self.exact_roots,
                "statement": "Lemma 1" + (" + Proposition 2" if self.fd_error else ""),
            },
            "price_bits_per_vector": self.price_bits if finite else None,
            "price_scope": self.price_scope,
            "observed_spread": {
                "plus": self.observed_plus,
                "minus": self.observed_minus,
                "note": "diagnostic only: a sample can refute a declared constant "
                "but never confirm one",
            },
            "refutation_bounds": {
                "lambda_max_allowed": self.refutation_bounds[0],
                "lambda_min_allowed": self.refutation_bounds[1],
            },
            "kernel_geometry": (
                None if self.geometry is None else self.geometry.as_dict()
            ),
            "source": NOTE,
            **self.meta,
        }

    def explain(self) -> str:
        lines = [f"learned weight ({self.kind}): {self.status}"]
        lines.append(
            f"  {self.m} samples, dimension {self.dim}, rank {self.rank}; "
            f"declared nu_plus={self.nu_plus:g}, nu_minus={self.nu_minus:g}, "
            f"delta={self.delta:g}"
        )
        fd = (
            f", with finite-difference error {self.fd_error:.4f}"
            if self.fd_error
            else ""
        )
        lines.append(
            f"  band (Lemma 1): (1 - {self.eps_minus:.4f}) Q <= Q_hat <= "
            f"(1 + {self.eps_plus:.4f}) Q{fd}"
        )
        if math.isfinite(self.price_bits):
            what = "regret bound" if self.kind == "linear" else "learning part"
            lines.append(
                f"  learning price: {self.price_bits:.4g} bits per vector ({what})"
            )
        else:
            lines.append("  learning price: none, for the reasons below")
        lines.append(
            f"  observed spread relative to Q_hat: +{self.observed_plus:.4g} / "
            f"-{self.observed_minus:.4g} (diagnostic)"
        )
        g = self.geometry
        if g is not None:
            lines.append(
                f"  kernel: rank of mean {g.q}, per-sample rank {g.point_rank_min}.."
                f"{g.point_rank_max}"
                + (" (the kernel moves)" if g.kernel_moves else "")
            )
            if g.penalty_coefficient:
                lines.append(
                    f"  fixed-covariance test channels exceed the consumer's own rate "
                    f"by at least about {g.penalty_coefficient:g} bits per halving of "
                    "the loss, under Corollary 9's hypotheses (addendum)"
                )
        lines.extend(f"  - {r}" for r in self.reasons)
        return "\n".join(lines)


def learn_weight(
    factors: Any = None,
    *,
    metrics: np.ndarray | None = None,
    kind: str = "points",
    delta: float = 0.05,
    nu_plus: float,
    nu_minus: float = 1.0,
    fd_error: float = 0.0,
    exact_roots: bool = False,
    output_dim: int | None = None,
    rank_rtol: float = 1e-9,
    point_tol: float = 1e-8,
) -> LearnedWeight:
    """Average the samples, attach Lemma 1's band, price it, and judge it.

    Give the samples as ``factors`` (``P_i = A_i^T A_i``: a query matrix, or
    per-point Jacobians; the memory-light form) or as ``metrics``, an
    ``(m, p, p)`` stack, and say what they are: ``kind="points"`` for read
    metrics of a nonlinear consumer at probe points, ``kind="linear"`` for the
    terms of a linear consumer's weight (queries). ``nu_plus`` is required and
    must be fixed before the sample is seen; ``fd_error`` is Proposition 2's
    ``eps_h`` when the samples came from central differences.

    Status:

    * ``CERTIFIED``: ``Q_hat`` has full rank, the band has a lower bound, and
      the sample is consistent with the declared constants. ``price_bits`` is
      then ``tau``, with the scope ``price_scope`` states.
    * ``REFUTED``: some sample is outside what the declared constants allow on
      the band's event; at level ``delta`` the constants are wrong.
    * ``NO_CERTIFICATE``: ``Q_hat`` is rank-deficient (after the numerical rank
      cut), or the samples are too few for a lower bound (``eps_minus >= 1``).
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if factors is None and metrics is None:
        raise ValueError("give factors or metrics")
    if factors is not None and metrics is not None:
        raise ValueError("give factors or metrics, not both")
    if fd_error < 0:
        raise ValueError("fd_error must be non-negative")
    flist = None if factors is None else _as_factor_list(factors)
    sp = _spectra(flist, metrics, rank_rtol, point_tol)
    m = int(sp.lam_max.size)
    p = int(sp.Q_hat.shape[0])
    eps_p, eps_m = confidence_band(
        m, p, delta, nu_plus, nu_minus, exact_roots=exact_roots
    )
    ep, em = eps_p + fd_error, eps_m + fd_error
    geom = _geometry(sp, output_dim) if kind == "points" else None

    hi = (1.0 + nu_plus + fd_error) / (1.0 - em) if em < 1.0 else math.inf
    lo = (1.0 - nu_minus - fd_error) / (1.0 + ep)
    reasons: list[str] = []
    status = CERTIFIED
    obs_hi, obs_lo = float(sp.lam_max.max()), float(sp.lam_min.min())
    if sp.rank < p:
        # Lemma 1 and Corollary 8 need Q > 0, and Remark 14 extends only
        # Theorem 11 under (A1): no certificate, and no refutation either,
        # since the band's event is not available to test against.
        status = NO_CERTIFICATE
        reasons.append(
            f"Q_hat has numerical rank {sp.rank} < {p} (rank_rtol={rank_rtol:g}; up "
            f"to {sp.kernel_energy:.2g} of a sample's trace lies off its range): "
            "Lemma 1 and Corollary 8 need Q > 0, and the note's Remark 14 extends "
            "only Theorem 11, under the sandwich (A1) with both constants below one "
            "and the loss sandwich (A1a) declared in advance; a kernel that looks "
            "fixed in a sample does not supply that"
        )
    else:
        # a relative slack of _ROUNDING keeps floating-point rounding in the
        # whitening from refuting a constant that holds with equality
        if obs_hi > hi * (1.0 + _ROUNDING):
            status = REFUTED
            reasons.append(
                f"a sample reads {obs_hi:.4g} times Q_hat along some direction, more "
                f"than the {hi:.4g} that nu_plus={nu_plus:g} allows on the band's "
                f"event: the declared constant is wrong at level delta={delta:g}"
            )
        if lo > 0.0 and obs_lo < lo * (1.0 - _ROUNDING):
            status = REFUTED
            reasons.append(
                f"a sample reads only {obs_lo:.4g} times Q_hat along some direction, "
                f"less than the {lo:.4g} that nu_minus={nu_minus:g} allows: the "
                f"declared constant is wrong at level delta={delta:g}"
            )
        if status == CERTIFIED and em >= 1.0:
            status = NO_CERTIFICATE
            if fd_error >= 1.0:
                hint = "the finite-difference error alone is >= 1: take a smaller step"
            else:
                need = probes_for_price(min(1.0, p), p, delta, nu_plus, nu_minus)
                hint = (
                    f"about {need['minimal']} exact samples give a price of at most "
                    "1 bit at these constants"
                    + (", before the finite-difference error" if fd_error else "")
                )
            reasons.append(
                f"eps_minus = {em:.3g} >= 1: {m} samples bound the weight from "
                f"above only; {hint}"
            )
    price = learning_price(ep, em, p) if status == CERTIFIED else math.inf
    if status == CERTIFIED:
        reasons.append("the sample is consistent with the declared constants")
    return LearnedWeight(
        Q_hat=sp.Q_hat,
        kind=kind,
        status=status,
        reasons=reasons,
        m=m,
        dim=p,
        rank=sp.rank,
        delta=float(delta),
        nu_plus=float(nu_plus),
        nu_minus=float(nu_minus),
        eps_plus=float(ep),
        eps_minus=float(em),
        fd_error=float(fd_error),
        price_bits=float(price),
        observed_plus=obs_hi - 1.0,
        observed_minus=1.0 - obs_lo,
        refutation_bounds=(hi, lo),
        geometry=geom,
        exact_roots=bool(exact_roots),
    )


def price_of_samples(
    samples: Sequence[Any], *, delta: float, nu_plus: Sequence[float], **kw: Any
) -> tuple[list[LearnedWeight], tuple[float, float]]:
    """Several consumers' samples at once, with ``delta`` split across them.

    Returns each consumer's record (each at ``delta / n``) and the combined
    band of their weighted sum (:func:`combine_bands`), which holds with
    probability at least ``1 - delta`` by the union bound.
    """
    if len(samples) != len(nu_plus):
        raise ValueError("one nu_plus per consumer")
    n = max(len(samples), 1)
    recs = [
        learn_weight(s, delta=delta / n, nu_plus=v, **kw)
        for s, v in zip(samples, nu_plus)
    ]
    return recs, combine_bands((r.eps_plus, r.eps_minus) for r in recs)

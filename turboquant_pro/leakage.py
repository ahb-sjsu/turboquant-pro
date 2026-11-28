# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Leakage to a holder: what a stored code tells a party that already knows something.

A code is read by its consumer, and it is also seen by parties that never
decode it but already hold something correlated with the data: an index host
that has public-model embeddings of the same documents, a tenant with the
metadata, a worker that holds part of the context. What the code adds to
what such a *holder* already knows is its leakage. Every allocation in this
package so far optimizes the consumer's distortion at a rate and is blind to
it.

**The model** (rate-leakage-distortion series, geometric-observation; paper I
``paper/tit-rate-leakage``, in review at IEEE T-IT). The source is
``T ~ N(0, Sigma)`` in ``R^p``, reproduced under the weight ``Q`` (a read
operator), and the holder observes ``S = H T + U``, ``U ~ N(0, Sigma_U)``,
independent of ``T``. Per vector a description costs its rate
``I(T; T_hat)`` and its leakage ``I(T; T_hat | S)``. By Theorem 3 the
achievable pairs at distortion ``D`` are those above
``(r(Delta), l(Delta))`` for an error covariance ``Delta`` with
``tr(Q Delta) <= D``. In the whitened coordinate
``X = Sigma^-1/2 Delta Sigma^-1/2``:

    r(X) = -1/2 log2 det X
    l(X) = r(X) - 1/2 log2 det(Ht Ht^T + Sigma_U) + 1/2 log2 det(Ht X Ht^T + Sigma_U)
    d(X) = tr(Qt X),   Qt = Sigma^1/2 Q Sigma^1/2,   Ht = H Sigma^1/2.

**The frontier** (Theorem 7). The point of weight ``alpha`` (minimizing
``alpha r + (1 - alpha) l``) is the unique ``X`` with ``d(X) = D`` and

    X = Phi(2 nu Qt + (1 - alpha) Ht^T (Ht X Ht^T + Sigma_U)^-1 Ht),

``Phi`` reverse water-filling at level one. At ``alpha = 1`` this is the
consumer-only design :mod:`turboquant_pro.read_allocation` approximates; for
``alpha < 1`` the holder tilts the weight. It is computed by Proposition 11's
majorize-minimize iteration, whose cost is monotone and which converges; a
point counts as converged only when its fixed-point residual is small too.

**Blocks.** The program separates over independent blocks that share one
water level ``nu``: frequencies of a stationary process (paper II, Theorem
4), regimes or strata (paper III, Proposition 23). So one solver serves all
three: :func:`frontier_point` takes a list of :class:`Block` with weights.
Corollary 6 of paper III then says that pooling the blocks' covariances never
costs less than coding them apart, which :func:`pooling_gain` measures.

**What the numbers are, and are not.** They are the Gaussian-model frontier
and its test channels. Paper III, Theorem 2, needs only second moments: for a
source that is *not* Gaussian, the Gaussian region with the same second
moments is still achievable, so the frontier computed from a sample
covariance *estimates* an achievable target for real data. Theorem 3 (every
code is bounded below by that frontier less two relative entropies) needs
more: a holder ``S = H T + U`` with ``U`` independent of ``T``, a source with
a density and ``I(T; S)`` finite. A least-squares holder fit only makes the
residual uncorrelated with ``T``, so the lower bound applies only when the
holder really is such a channel. A practical scalar quantizer in the frontier
basis approaches the frontier but is not certified by it: its leakage is
measured, by :func:`linear_leakage`, as a labelled proxy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "NO_TRADEOFF",
    "WITHIN_NOISE",
    "TRADEOFF",
    "Block",
    "FrontierPoint",
    "Holder",
    "described_components",
    "fit_holder",
    "frontier",
    "frontier_point",
    "linear_leakage",
    "noise_floor",
    "pooling_gain",
    "sequence_gain",
    "spectral_blocks",
    "spectral_density",
    "tradeoff_report",
    "water_fill",
]

TRADEOFF = "TRADEOFF"
NO_TRADEOFF = "NO_TRADEOFF"
WITHIN_NOISE = "WITHIN_NOISE"
PAPER_I = "geometric-observation paper/tit-rate-leakage (in review, IEEE T-IT)"
LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------


def _herm(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.conj().T)


def _psd_sqrt(A: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(_herm(A))
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.conj().T


def _logdet2(A: np.ndarray) -> float:
    sign, ld = np.linalg.slogdet(_herm(A))
    if np.real(sign) <= 0:
        return -math.inf
    return float(ld) / LN2


def water_fill(A: np.ndarray) -> np.ndarray:
    """``Phi(A)``: reverse water-filling at level one on the spectrum of ``A >= 0``.

    ``Phi(A) = sum_i min(1, 1/lambda_i) e_i e_i^*``, with ``1`` where
    ``lambda_i <= 1`` (paper I, before Theorem 7).
    """
    w, V = np.linalg.eigh(_herm(A))
    x = np.where(w > 1.0, 1.0 / np.maximum(w, 1e-300), 1.0)
    return _herm((V * x) @ V.conj().T)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Holder:
    """A party that observes ``S = H T + U``, ``U ~ N(0, noise)``, and never decodes."""

    H: np.ndarray
    noise: np.ndarray

    def __post_init__(self) -> None:
        H = np.atleast_2d(np.asarray(self.H))
        N = np.atleast_2d(np.asarray(self.noise))
        if N.shape != (H.shape[0], H.shape[0]):
            raise ValueError(f"noise must be {H.shape[0]}x{H.shape[0]}, got {N.shape}")
        if float(np.linalg.eigvalsh(_herm(N)).min()) <= 0.0:
            raise ValueError(
                "the holder's noise covariance is not positive definite: it observes "
                "some combination of the source without noise, and any description "
                "of that combination leaks without bound"
            )
        object.__setattr__(self, "H", H)
        object.__setattr__(self, "noise", _herm(N))

    @property
    def information(self) -> np.ndarray:
        """``J = H^T Sigma_U^-1 H``: what ``S`` tells about ``T``. Leakage depends on
        the holder only through ``J``."""
        return _herm(self.H.conj().T @ np.linalg.solve(self.noise, self.H))

    @classmethod
    def blind(cls, dim: int) -> Holder:
        """A holder that observes nothing about the source (``J = 0``)."""
        return cls(np.zeros((1, dim)), np.eye(1))


def fit_holder(
    source: np.ndarray, observation: np.ndarray
) -> tuple[Holder, np.ndarray]:
    """Least-squares holder model from paired rows, and the source covariance.

    ``H = C_ST Sigma^-1`` and ``Sigma_U = C_SS - H C_TS``: the linear model with
    the same second moments as the pair, which is the Gaussian problem paper
    III's Theorem 2 (achievability) needs. The fitted residual is only
    uncorrelated with ``T``; Theorem 3's lower bound needs it independent.
    """
    T = np.asarray(source, dtype=np.float64)
    S = np.asarray(observation, dtype=np.float64)
    if T.ndim != 2 or S.ndim != 2 or T.shape[0] != S.shape[0]:
        raise ValueError("source and observation must be 2-D with the same rows")
    n, p = T.shape
    r = S.shape[1]
    if n <= p + r:
        raise ValueError(
            f"{n} paired rows cannot estimate a {p}-dimensional source with a "
            f"{r}-dimensional holder; need more than {p + r}"
        )
    T = T - T.mean(axis=0)
    S = S - S.mean(axis=0)
    Sigma = _herm(T.T @ T / (n - 1))
    C_st = S.T @ T / (n - 1)
    C_ss = _herm(S.T @ S / (n - 1))
    H = np.linalg.solve(Sigma, C_st.T).T
    return Holder(H, _herm(C_ss - H @ C_st.T)), Sigma


@dataclass(frozen=True)
class Block:
    """One independent component of the source, in whitened coordinates.

    ``Qt = Sigma^1/2 Q Sigma^1/2``, ``Ht = H Sigma^1/2``, ``noise = Sigma_U``,
    and ``root = Sigma^1/2`` to map designs back. Hermitian blocks (frequencies
    of a real process) are allowed (paper II, Lemma 1).
    """

    Qt: np.ndarray
    Ht: np.ndarray
    noise: np.ndarray
    root: np.ndarray

    @classmethod
    def from_model(
        cls, Q: np.ndarray, Sigma: np.ndarray, holder: Holder | None = None
    ) -> Block:
        Sigma = _herm(np.asarray(Sigma))
        Q = _herm(np.asarray(Q))
        p = Sigma.shape[0]
        if Q.shape != (p, p):
            raise ValueError(f"weight is {Q.shape}, source is {p}-dimensional")
        if float(np.linalg.eigvalsh(Sigma).min()) <= 0.0:
            raise ValueError("source covariance must be positive definite")
        if float(np.linalg.eigvalsh(Q).min()) < -1e-9 * max(
            1.0, float(np.abs(Q).max())
        ):
            raise ValueError("weight must be positive semidefinite")
        holder = holder or Holder.blind(p)
        if holder.H.shape[1] != p:
            raise ValueError(f"holder reads {holder.H.shape[1]} dims, source has {p}")
        R = _psd_sqrt(Sigma)
        return cls(_herm(R @ Q @ R), holder.H @ R, holder.noise, R)

    @property
    def dim(self) -> int:
        return int(self.Qt.shape[0])

    @property
    def d_max(self) -> float:
        """The distortion of describing nothing, ``tr(Q Sigma)``."""
        return float(np.real(np.trace(self.Qt)))

    def tilt(self, X: np.ndarray, alpha: float) -> np.ndarray:
        M = self.Ht @ X @ self.Ht.conj().T + self.noise
        return (1.0 - alpha) * _herm(self.Ht.conj().T @ np.linalg.solve(M, self.Ht))

    def rate(self, X: np.ndarray) -> float:
        return -0.5 * _logdet2(X)

    def leakage(self, X: np.ndarray) -> float:
        Hh = self.Ht.conj().T
        info = 0.5 * _logdet2(self.Ht @ Hh + self.noise) - 0.5 * _logdet2(
            self.Ht @ X @ Hh + self.noise
        )
        return self.rate(X) - info

    def distortion(self, X: np.ndarray) -> float:
        return float(np.real(np.trace(self.Qt @ X)))


# ---------------------------------------------------------------------------
# The solver: Proposition 11 over blocks with one level
# ---------------------------------------------------------------------------


@dataclass
class FrontierPoint:
    """The frontier point of weight ``alpha`` at distortion ``D``."""

    alpha: float
    D: float
    nu: float
    X: list[np.ndarray]
    rate: float
    leakage: float
    distortion: float
    rate_blocks: np.ndarray
    leakage_blocks: np.ndarray
    distortion_blocks: np.ndarray
    iterations: int
    residual: float
    """``max_b ||X_b - Phi(2 nu Qt_b + tilt(X_b))||_F``: Theorem 7's fixed point."""
    monotone: bool
    """The weighted cost never rose along the iteration (Proposition 11)."""
    converged: bool
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "D": self.D,
            "nu": self.nu,
            "rate_bits": self.rate,
            "leakage_bits": self.leakage,
            "distortion": self.distortion,
            "rate_blocks": [float(v) for v in self.rate_blocks],
            "leakage_blocks": [float(v) for v in self.leakage_blocks],
            "iterations": self.iterations,
            "fixed_point_residual": self.residual,
            "monotone": self.monotone,
            "converged": self.converged,
            **self.meta,
        }


def _weights(blocks: Sequence[Block], weights) -> np.ndarray:
    c = np.ones(len(blocks)) if weights is None else np.asarray(weights, float)
    if c.shape != (len(blocks),) or np.any(c <= 0):
        raise ValueError("one positive weight per block")
    return c


def _level(blocks, c, T, D, nu_hint):
    """``nu >= 0`` with ``sum_b c_b d_b(Phi(2 nu Qt_b + T_b)) = D`` (or 0 if slack).

    The distortion is continuous and nonincreasing in ``nu`` (Proposition 11's
    proof) and tends to zero, so a bracket exists; Illinois regula falsi finds
    the root in a few evaluations instead of bisection's dozens."""

    def dist(nu):
        Xs = [water_fill(2.0 * nu * b.Qt + t) for b, t in zip(blocks, T)]
        return float(sum(ci * b.distortion(x) for ci, b, x in zip(c, blocks, Xs))), Xs

    d0, X0 = dist(0.0)
    if d0 <= D:
        return 0.0, X0
    lo, f_lo = 0.0, d0 - D
    hi = nu_hint if nu_hint and nu_hint > 0 else 1.0
    d_hi, X_hi = dist(hi)
    while d_hi > D:
        lo, f_lo = hi, d_hi - D
        hi *= 4.0
        d_hi, X_hi = dist(hi)
    f_hi, side = d_hi - D, 0
    best = (hi, X_hi)
    for _ in range(200):
        nu = (lo * f_hi - hi * f_lo) / (f_hi - f_lo)
        if not lo < nu < hi:
            nu = 0.5 * (lo + hi)
        d_nu, X_nu = dist(nu)
        f = d_nu - D
        if f <= 0:
            best = (nu, X_nu)
        if abs(f) <= 1e-13 * max(D, 1e-300) or hi - lo <= 1e-15 * hi:
            return nu, X_nu
        if f > 0:
            lo, f_lo = nu, f
            if side == -1:
                f_hi *= 0.5
            side = -1
        else:
            hi, f_hi = nu, f
            if side == 1:
                f_lo *= 0.5
            side = 1
    return best


def frontier_point(
    blocks: Sequence[Block],
    weights: Sequence[float] | None = None,
    *,
    alpha: float,
    D: float,
    tol: float = 1e-10,
    max_iter: int = 5000,
    residual_tol: float = 1e-7,
) -> FrontierPoint:
    """Minimize ``alpha R + (1 - alpha) L`` subject to ``sum_b c_b d_b <= D``.

    ``R``, ``L`` and the distortion are the ``c``-weighted sums over blocks; for
    one block they are paper I's costs per vector. Requires
    ``0 < D < sum_b c_b tr(Qt_b)``; at or above that, describing nothing is
    free. ``converged`` needs both a step below ``tol`` and a fixed-point
    residual below ``residual_tol``.
    """
    blocks = list(blocks)
    if not blocks:
        raise ValueError("no blocks")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    c = _weights(blocks, weights)
    d_max = float(sum(ci * b.d_max for ci, b in zip(c, blocks)))
    if not 0.0 < D < d_max:
        raise ValueError(
            f"D must lie in (0, {d_max:.6g}), the distortion of no description"
        )

    def cost(Xs):
        return float(
            sum(
                ci * (alpha * b.rate(x) + (1.0 - alpha) * b.leakage(x))
                for ci, b, x in zip(c, blocks, Xs)
            )
        )

    X = [np.eye(b.dim, dtype=np.result_type(b.Qt, b.Ht)) for b in blocks]
    nu, prev, monotone, converged, it = None, math.inf, True, False, 0
    for it in range(1, max_iter + 1):
        T = [b.tilt(x, alpha) for b, x in zip(blocks, X)]
        nu, Xn = _level(blocks, c, T, D, nu)
        now = cost(Xn)
        if it > 1 and now > prev + 1e-10 * max(1.0, abs(prev)):
            monotone = False
        step = max(float(np.linalg.norm(a - b)) for a, b in zip(Xn, X))
        X, prev = Xn, now
        if step < tol:
            converged = True
            break
    T = [b.tilt(x, alpha) for b, x in zip(blocks, X)]
    residual = max(
        float(np.linalg.norm(x - water_fill(2.0 * nu * b.Qt + t)))
        for b, x, t in zip(blocks, X, T)
    )
    # a small step is not enough: the point must also be Theorem 7's fixed point
    converged = converged and residual <= residual_tol
    r = np.array([b.rate(x) for b, x in zip(blocks, X)])
    lk = np.array([b.leakage(x) for b, x in zip(blocks, X)])
    dv = np.array([b.distortion(x) for b, x in zip(blocks, X)])
    return FrontierPoint(
        alpha=float(alpha),
        D=float(D),
        nu=float(nu),
        X=X,
        rate=float(c @ r),
        leakage=float(c @ lk),
        distortion=float(c @ dv),
        rate_blocks=r,
        leakage_blocks=lk,
        distortion_blocks=dv,
        iterations=it,
        residual=residual,
        monotone=monotone,
        converged=converged,
    )


def frontier(
    blocks: Sequence[Block],
    weights: Sequence[float] | None = None,
    *,
    D: float,
    alphas: Sequence[float] = (1.0, 0.75, 0.5, 0.25, 0.0),
    **kw: Any,
) -> list[FrontierPoint]:
    """Frontier points at several weights: rate-optimal (1) to leakage-optimal (0)."""
    return [frontier_point(blocks, weights, alpha=a, D=D, **kw) for a in alphas]


def tradeoff_report(
    points: Sequence[FrontierPoint],
    *,
    tol_bits: float = 1e-6,
    noise_floor: dict | None = None,
    noise_factor: float = 10.0,
) -> dict:
    """Does guarding against the holder cost rate here; what does ignoring it leak?

    ``points`` are frontier points at one distortion (from :func:`frontier`),
    including ``alpha = 1`` and ``alpha = 0``.

    ``excess_leakage_bits`` is what the consumer-only design (``alpha = 1``)
    leaks beyond the least possible leakage at the same distortion;
    ``rate_premium_bits`` is what the least-leaking design costs in rate
    beyond the least rate. Both are zero exactly when one description is
    optimal for both costs, as when the holder learns nothing (``J = 0``);
    when the rate-optimal description is one-dimensional along ``b_R``, a
    strictly positive excess on both sides follows as soon as
    ``J Sigma b_R`` is not parallel to ``b_R`` (paper I, Theorem 22).

    For a model that is known exactly, the verdict is ``NO_TRADEOFF`` when
    both are below ``tol_bits``. For a model *estimated* from a sample, give
    ``noise_floor`` (from :func:`noise_floor`, a holder that knows nothing,
    fitted to the same sample): the verdict is then ``TRADEOFF`` only when
    both exceed ``noise_factor`` times the floor, and ``WITHIN_NOISE``
    otherwise, because an estimated holder always shows a little information.
    """
    pts = list(points)
    by = {p.alpha: p for p in pts}
    if 1.0 not in by or 0.0 not in by:
        raise ValueError("the points must include alpha = 1 and alpha = 0")
    if len({p.D for p in pts}) != 1:
        raise ValueError("the points must share one distortion")
    r_opt, l_opt = by[1.0], by[0.0]
    excess = max(r_opt.leakage - l_opt.leakage, 0.0)
    premium = max(l_opt.rate - r_opt.rate, 0.0)
    if noise_floor is None:
        verdict = NO_TRADEOFF if max(excess, premium) <= tol_bits else TRADEOFF
        threshold = tol_bits
    else:
        floor = max(
            noise_floor["excess_leakage_bits"], noise_floor["rate_premium_bits"]
        )
        threshold = max(tol_bits, noise_factor * floor)
        verdict = TRADEOFF if min(excess, premium) > threshold else WITHIN_NOISE
    ok = all(p.converged and p.monotone for p in pts)
    if noise_floor is not None:
        ok = ok and bool(noise_floor.get("solver_ok", True))
    return {
        "verdict": verdict,
        "D": float(r_opt.D),
        "excess_leakage_bits": excess,
        "rate_premium_bits": premium,
        "threshold_bits": threshold,
        "noise_floor": noise_floor,
        "noise_factor": noise_factor if noise_floor is not None else None,
        "rate_optimal": r_opt.as_dict(),
        "leakage_optimal": l_opt.as_dict(),
        "frontier": [p.as_dict() for p in pts],
        "solver_ok": ok,
        "source": f"{PAPER_I}, Theorems 3 and 7, Proposition 11",
        "scope": "Gaussian-model frontier of test channels from second moments; "
        "achievable for a non-Gaussian source with these moments (paper III, "
        "Theorem 2); a scalar quantizer's own leakage must be measured",
    }


def noise_floor(
    control: Block, rate_optimal: FrontierPoint, *, D: float, **kw: Any
) -> dict:
    """The excess and premium a holder that knows nothing shows in the same sample.

    ``control`` is the block fitted with the holder's rows shuffled against
    the source's, so its true information is zero and anything it shows is
    estimation noise. The rate-optimal design does not depend on the holder,
    so only the least-leaking one is solved again.
    """
    X_r = rate_optimal.X[0]
    lk = frontier_point([control], alpha=0.0, D=D, **kw)
    return {
        "excess_leakage_bits": max(control.leakage(X_r) - lk.leakage, 0.0),
        "rate_premium_bits": max(lk.rate - control.rate(X_r), 0.0),
        "solver_ok": bool(lk.converged and lk.monotone),
        "kind": "holder rows shuffled against the source's: no true information",
    }


def described_components(point: FrontierPoint, block_index: int, block: Block) -> dict:
    """The frontier design of one block as a code: directions and bits.

    In whitened coordinates ``Z = Sigma^-1/2 T`` the design is ``X = V diag(x) V^*``:
    component ``i`` is ``v_i^* Z``, of unit variance, described with error
    variance ``x_i`` at ``-1/2 log2 x_i`` bits (a Gaussian test channel; a
    scalar quantizer at about that width approaches it). ``directions`` are
    the analysis functionals in source coordinates, ``Sigma^-1/2 v_i``, and
    ``reduction`` the directions in which the error covariance falls below the
    source's, the eigenvectors of ``Sigma - Delta`` with nonzero eigenvalue.
    """
    X = point.X[block_index]
    x, V = np.linalg.eigh(_herm(X))
    live = x < 1.0 - 1e-12
    bits = -0.5 * np.log2(np.clip(x[live], 1e-300, None))
    order = np.argsort(bits)[::-1]
    inv_root = np.linalg.inv(block.root)
    R = block.root
    gap = _herm(R @ (np.eye(block.dim) - X) @ R)  # Sigma - Delta
    w, U = np.linalg.eigh(gap)
    keep = w > 1e-12 * max(float(w.max()), 1e-300)
    return {
        "bits": bits[order],
        "directions": (inv_root @ V[:, live])[:, order],
        "reduction_values": w[keep][::-1],
        "reduction_directions": U[:, keep][:, ::-1],
        "components": int(live.sum()),
    }


# ---------------------------------------------------------------------------
# Paper III, Corollary 6: pooling never costs less
# ---------------------------------------------------------------------------


def pooling_gain(
    covariances: Sequence[np.ndarray],
    proportions: Sequence[float],
    Q: np.ndarray,
    *,
    D: float,
    alpha: float = 1.0,
    holder: Holder | None = None,
    **kw: Any,
) -> dict:
    """What coding strata apart, under one water level, saves over pooling them.

    With strata of covariance ``K_c`` in proportions ``lambda_c`` and a common
    weight and holder, Corollary 6 of paper III gives

        V(alpha, D; sum_c lambda_c K_c) >= min { sum_c lambda_c V(alpha, D_c; K_c) :
                                                  sum_c lambda_c D_c <= D },

    and the right side is the block program with one level (Proposition 23),
    so ``gain_bits >= 0``. A stratum label the decoder must be told costs at
    most ``H(lambda)`` bits more; one implicit in where the vector is stored
    (a shard, a partition) costs nothing.
    """
    lam = np.asarray(proportions, float)
    if lam.ndim != 1 or len(covariances) != lam.size or np.any(lam <= 0):
        raise ValueError("one positive proportion per covariance")
    lam = lam / lam.sum()
    Ks = [_herm(np.asarray(K, float)) for K in covariances]
    pooled_K = sum(li * K for li, K in zip(lam, Ks))
    pooled = frontier_point(
        [Block.from_model(Q, pooled_K, holder)], alpha=alpha, D=D, **kw
    )
    apart = frontier_point(
        [Block.from_model(Q, K, holder) for K in Ks], lam, alpha=alpha, D=D, **kw
    )

    def v(p):
        return alpha * p.rate + (1.0 - alpha) * p.leakage

    label_bits = float(-np.sum(lam * np.log2(lam)))
    return {
        "alpha": float(alpha),
        "D": float(D),
        "pooled_cost_bits": v(pooled),
        "stratified_cost_bits": v(apart),
        "gain_bits": v(pooled) - v(apart),
        "shared_level": apart.nu,
        "rate_per_stratum": [float(x) for x in apart.rate_blocks],
        "distortion_per_stratum": [float(x) for x in apart.distortion_blocks],
        "label_entropy_bits": label_bits,
        "solver_ok": bool(
            pooled.converged and pooled.monotone and apart.converged and apart.monotone
        ),
        "source": "geometric-observation paper/tit-beyond-stationary, Corollary 6 "
        "and Proposition 23 (draft, not submitted)",
    }


# ---------------------------------------------------------------------------
# Papers II and III: a sequence of vectors coded along its length
# ---------------------------------------------------------------------------


def spectral_density(
    sequence: np.ndarray, nfft: int, *, cross: np.ndarray | None = None
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None, int]:
    """Welch estimate of the spectral density matrix of a vector sequence.

    Rows are time steps (tokens, samples). Hann-windowed segments of length
    ``nfft`` at half overlap; the normalization makes the mean over the
    ``nfft`` frequencies equal the windowed lag-0 covariance, so a block of
    weight ``1 / nfft`` per frequency is paper II's spectral program. With
    ``cross`` (a second, row-paired sequence ``s``) it also returns
    ``(S_sx, S_ss)``, from which a filtered holder is fitted frequency by
    frequency. Returns ``(S_xx, (S_sx, S_ss) or None, segments)``.
    """
    x = np.asarray(sequence, float)
    if x.ndim != 2:
        raise ValueError("sequence must be (time, dim)")
    if nfft < 2 or nfft > x.shape[0]:
        raise ValueError(f"nfft must lie in [2, {x.shape[0]}]")
    hop = nfft // 2
    starts = range(0, x.shape[0] - nfft + 1, hop)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nfft) / nfft)  # periodic Hann
    norm = float(np.sum(w * w))

    def segs(a):
        a = a - a.mean(axis=0)
        return np.fft.fft(
            np.stack([w[:, None] * a[s : s + nfft] for s in starts]), axis=1
        )

    F = segs(x)  # (segments, nfft, p)
    n_seg = F.shape[0]
    S_xx = np.einsum("skp,skq->kpq", F, F.conj()) / (n_seg * norm)
    pair = None
    if cross is not None:
        G = segs(np.asarray(cross, float).reshape(x.shape[0], -1))
        S_sx = np.einsum("skr,skq->krq", G, F.conj()) / (n_seg * norm)
        S_ss = np.einsum("skr,skt->krt", G, G.conj()) / (n_seg * norm)
        pair = (S_sx, S_ss)
    return S_xx, pair, n_seg


def spectral_blocks(
    sequence: np.ndarray,
    Q: np.ndarray,
    nfft: int,
    *,
    holder_sequence: np.ndarray | None = None,
) -> tuple[list[Block], np.ndarray]:
    """Per-frequency blocks of a stationary vector sequence, with weights ``1/nfft``.

    Paper II, Theorem 4: the region of a stationary Gaussian process is
    tilted water-filling frequency by frequency under one level, and Theorems
    16 to 23 make it operational for a holder that observes the process
    through a filter in colored noise. With ``holder_sequence`` that holder
    is fitted per frequency: ``H(w) = S_sx S_xx^-1`` and noise
    ``S_ss - H S_xs``.
    """
    S_xx, cross, n_seg = spectral_density(sequence, nfft, cross=holder_sequence)
    p = S_xx.shape[1]
    if n_seg < p:
        raise ValueError(
            f"{n_seg} segments cannot estimate a {p}x{p} spectral density; give at "
            f"least {(p + 1) * (nfft // 2)} rows or a smaller nfft"
        )
    blocks = []
    for k in range(nfft):
        holder = None
        if cross is not None:
            S_sx, S_ss = cross[0][k], cross[1][k]
            H = np.linalg.solve(S_xx[k].T, S_sx.T).T
            holder = Holder(H, _herm(S_ss - H @ S_sx.conj().T))
        blocks.append(Block.from_model(np.asarray(Q, float), S_xx[k], holder))
    return blocks, np.full(nfft, 1.0 / nfft)


def sequence_gain(
    sequence: np.ndarray, Q: np.ndarray, *, D: float, nfft: int, **kw: Any
) -> dict:
    """Rate saved by coding along the sequence instead of each vector alone.

    A memoryless code sees only the lag-0 covariance, which is the mean of
    the spectrum, and a memoryless channel's block information is at most the
    sum of its per-letter informations, so ``gain_bits >= 0``. The sequence
    rate is paper II's spectral rate (Theorem 4 at ``alpha = 1``, Kolmogorov's
    formula); the per-frequency split says where the bits go. Rate only: the
    leakage comparison against a holder with memory needs paper II's holder
    model on both sides, which :func:`spectral_blocks` provides for the
    sequence code alone.
    """
    blocks, c = spectral_blocks(sequence, Q, nfft)
    lag0 = _herm(sum(ci * b.root @ b.root for ci, b in zip(c, blocks)))
    alone = frontier_point([Block.from_model(Q, np.real(lag0))], alpha=1.0, D=D, **kw)
    along = frontier_point(blocks, c, alpha=1.0, D=D, **kw)
    freqs = np.fft.fftfreq(nfft)
    return {
        "D": float(D),
        "nfft": int(nfft),
        "memoryless_rate_bits": alone.rate,
        "sequence_rate_bits": along.rate,
        "gain_bits": alone.rate - along.rate,
        "rate_by_frequency": [
            {"cycles_per_step": float(f), "bits": float(ci * r)}
            for f, ci, r in zip(freqs, c, along.rate_blocks)
        ],
        "shared_level": along.nu,
        "solver_ok": bool(alone.converged and along.converged),
        "source": "geometric-observation paper/tit-stationary (submitted to IEEE "
        "T-IT, IT-26-1961), Theorem 4",
        "scope": "Gaussian-model rates from estimated second-order statistics; "
        "achievable for these statistics, not a codec measurement",
    }


# ---------------------------------------------------------------------------
# A measured proxy for a real code
# ---------------------------------------------------------------------------


def _residual_cov(Y: np.ndarray, X: np.ndarray | None) -> np.ndarray:
    Yc = Y - Y.mean(axis=0)
    if X is None:
        return _herm(Yc.T @ Yc / (len(Y) - 1))
    Xc = X - X.mean(axis=0)
    B, *_ = np.linalg.lstsq(Xc, Yc, rcond=None)
    E = Yc - Xc @ B
    return _herm(E.T @ E / (len(Y) - 1))


def linear_leakage(
    source: np.ndarray, observation: np.ndarray, reconstruction: np.ndarray
) -> dict:
    """Linear-Gaussian proxies for a real code's rate and leakage, in bits per vector.

    ``leakage = 1/2 log2 det Cov(T | S) - 1/2 log2 det Cov(T | S, T_hat)`` with
    the conditional covariances of the best linear estimates, and the rate
    proxy likewise without ``S``. For jointly Gaussian ``(T, S, T_hat)`` these
    are the information quantities exactly, and when ``T`` and ``S`` follow
    paper I's Gaussian model and the code is formed from ``T`` alone, Lemma 1
    makes them *lower bounds* on the true rate and leakage of any code. For a
    quantizer's output on real data they measure what a *linear* estimator at
    the holder gains from the code, not the information-theoretic leakage, so
    they are a labelled proxy, never an acceptance gate on their own.
    """
    T = np.asarray(source, float)
    S = np.asarray(observation, float)
    Th = np.asarray(reconstruction, float)
    n, p = T.shape
    if S.shape[0] != n or Th.shape != T.shape:
        raise ValueError("source, observation and reconstruction must share rows")
    if n <= p + S.shape[1] + 1:
        raise ValueError("too few rows for the conditional covariances")
    both = np.concatenate([S, Th], axis=1)
    ld = {
        "T": _logdet2(_residual_cov(T, None)),
        "T|S": _logdet2(_residual_cov(T, S)),
        "T|Th": _logdet2(_residual_cov(T, Th)),
        "T|S,Th": _logdet2(_residual_cov(T, both)),
    }
    return {
        "rate_proxy_bits": 0.5 * (ld["T"] - ld["T|Th"]),
        "leakage_proxy_bits": 0.5 * (ld["T|S"] - ld["T|S,Th"]),
        "kind": "linear-Gaussian proxy: exact for jointly Gaussian variables, a "
        "measure of a linear holder's gain otherwise",
    }

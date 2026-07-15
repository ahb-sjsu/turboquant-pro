# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""
Decision-level behavioral metrics for quantized-vs-base model comparison.

Aggregate metrics (accuracy, perplexity) can stay flat while a quantized model
*flips individual answers* -- the "illusion of equivalency" of Rababah, Akcora &
Leung (arXiv:2607.08734). They introduce **correctness agreement** (CA): the
fraction of examples where base and quantized are *both* correct. CA is a step in
the right direction, but has two defects this module fixes:

1. **CA only counts joint-correct.** It is bounded by ``min(acc_base, acc_quant)``
   and is blind to base-wrong -> quant-right recoveries; it conflates *accuracy
   loss* with *answer churn*. Two very different models (one that drops accuracy
   uniformly, one that swaps which items it gets right) can score the same CA.
2. **No noise floor.** Near-threshold multiple-choice items flip under *any*
   perturbation. In the paper's own table, CA is already ~14 points below base
   accuracy at near-lossless Q8_0 and is nearly flat from Q8 down to Q4 -- so
   most of the "moderate quantization breaks behavior" signal is item
   brittleness, not quantization. Without a control that measures the *free*
   flip rate between two behaviorally-equivalent runs, a raw flip count cannot
   be attributed to quantization.

This module therefore reports:

* :func:`correctness_agreement` -- the paper's CA, exactly, for comparison.
* :func:`flip_rate` -- the symmetric McNemar split: base-right -> quant-wrong
  (regressions) *and* base-wrong -> quant-right (recoveries), plus the McNemar
  statistic for whether the two directions are imbalanced.
* :func:`behavioral_agreement` -- prediction-level agreement: do the two models
  return the *same answer*, right or wrong? The purest behavioral-drift signal,
  independent of the gold label.
* :func:`noise_floor` -- the disagreement between two behaviorally-equivalent
  reference runs (e.g. two near-lossless requantization seeds), so a quantized
  model's disagreement can be reported as *excess over the floor* with a z-score.
* :func:`evaluate` -> :class:`BehavioralReport` -- all of the above in one pass.

scipy-free (matching :mod:`turboquant_pro.a2_probe`): the McNemar test uses the
exact binomial two-sided p-value for small discordant counts and the normal
approximation with continuity correction otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "correctness_agreement",
    "flip_rate",
    "behavioral_agreement",
    "noise_floor",
    "evaluate",
    "FlipResult",
    "NoiseFloor",
    "BehavioralReport",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_bool(a) -> np.ndarray:
    arr = np.asarray(a)
    if arr.dtype == bool:
        return arr.ravel()
    return arr.ravel().astype(bool)


def _check_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape != b.shape:
        raise ValueError(f"length mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("empty input")
    return a, b


def _binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value for k successes in n trials (p=0.5).

    Used for McNemar's exact test on the discordant pairs. scipy-free: sums
    binomial tail masses no larger than the observed point mass.
    """
    if n == 0:
        return 1.0
    from math import comb

    pmf = [comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(n + 1)]
    obs = pmf[k]
    tol = 1e-12
    return float(min(1.0, sum(m for m in pmf if m <= obs + tol)))


# ---------------------------------------------------------------------------
# the paper's metric
# ---------------------------------------------------------------------------


def correctness_agreement(base_correct, quant_correct) -> float:
    """Correctness Agreement (Rababah et al.): fraction both models get right.

    ``CA = mean( base_correct AND quant_correct )``. Bounded above by
    ``min(acc_base, acc_quant)``. Provided for exact comparison; prefer
    :func:`flip_rate` / :func:`behavioral_agreement`, which separate accuracy
    loss from answer churn.
    """
    bc, qc = _check_pair(_as_bool(base_correct), _as_bool(quant_correct))
    return float(np.mean(bc & qc))


# ---------------------------------------------------------------------------
# the corrected metric: symmetric flips + McNemar
# ---------------------------------------------------------------------------


@dataclass
class FlipResult:
    """Symmetric correctness-flip breakdown between base and quantized models.

    Attributes:
        n: Number of examples.
        acc_base: Base-model accuracy.
        acc_quant: Quantized-model accuracy.
        correctness_agreement: Fraction both correct (the paper's CA).
        regressions: Fraction base-correct that became quant-wrong (b10 / n).
        recoveries: Fraction base-wrong that became quant-right (b01 / n).
        churn: regressions + recoveries -- total correctness changes. Unlike
            ``acc_base - acc_quant`` (which nets the two directions to zero when
            they balance), churn exposes swapped decisions that accuracy hides.
        net_delta: acc_quant - acc_base ( = recoveries - regressions ).
        mcnemar_stat: McNemar chi-square statistic (with continuity correction)
            on the discordant pairs; large -> the flip directions are imbalanced.
        mcnemar_p: Two-sided p-value (exact binomial for small discordant n,
            normal approx otherwise). Small p -> quantization systematically
            regresses (or recovers), not a symmetric wash.
    """

    n: int
    acc_base: float
    acc_quant: float
    correctness_agreement: float
    regressions: float
    recoveries: float
    churn: float
    net_delta: float
    mcnemar_stat: float
    mcnemar_p: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def flip_rate(base_correct, quant_correct) -> FlipResult:
    """Symmetric correctness-flip analysis (McNemar) between base and quant.

    Splits the change in correctness into *regressions* (base-right -> quant-
    wrong) and *recoveries* (base-wrong -> quant-right), whose sum (``churn``)
    is the behavioral movement that accuracy delta nets away. Reports the
    McNemar statistic for directional imbalance.
    """
    bc, qc = _check_pair(_as_bool(base_correct), _as_bool(quant_correct))
    n = int(bc.size)
    b10 = int(np.sum(bc & ~qc))  # base right, quant wrong  (regression)
    b01 = int(np.sum(~bc & qc))  # base wrong, quant right  (recovery)
    disc = b10 + b01
    if disc == 0:
        stat, p = 0.0, 1.0
    else:
        # continuity-corrected McNemar chi-square (1 dof)
        stat = (abs(b10 - b01) - 1.0) ** 2 / disc
        stat = max(stat, 0.0)
        if disc <= 25:
            p = _binom_two_sided_p(min(b10, b01), disc, 0.5)
        else:
            z = (abs(b10 - b01) - 1.0) / math.sqrt(disc)
            p = math.erfc(z / math.sqrt(2.0))  # two-sided normal tail
    return FlipResult(
        n=n,
        acc_base=float(np.mean(bc)),
        acc_quant=float(np.mean(qc)),
        correctness_agreement=float(np.mean(bc & qc)),
        regressions=b10 / n,
        recoveries=b01 / n,
        churn=disc / n,
        net_delta=(b01 - b10) / n,
        mcnemar_stat=float(stat),
        mcnemar_p=float(min(1.0, p)),
    )


def behavioral_agreement(base_pred, quant_pred) -> float:
    """Prediction-level agreement: fraction where the two models pick the same
    answer, *regardless of correctness*.

    This is the label-independent behavioral-drift signal. Two models can have
    identical accuracy and high correctness agreement yet low behavioral
    agreement (they swap which items they answer which way). ``base_pred`` /
    ``quant_pred`` are arrays of predicted labels (any hashable/orderable dtype).
    """
    bp = np.asarray(base_pred).ravel()
    qp = np.asarray(quant_pred).ravel()
    _check_pair(bp, qp)
    return float(np.mean(bp == qp))


# ---------------------------------------------------------------------------
# noise floor: the free churn between behaviorally-equivalent runs
# ---------------------------------------------------------------------------


@dataclass
class NoiseFloor:
    """Baseline disagreement between two behaviorally-equivalent reference runs.

    Attributes:
        floor_disagreement: Prediction-level disagreement between the two
            reference runs (e.g. two near-lossless requantization seeds). The
            "free" flip rate that any quantization inherits from item brittleness.
        floor_churn: Correctness churn between the two reference runs, if gold
            labels were supplied (else NaN).
        n: Number of examples.
    """

    floor_disagreement: float
    floor_churn: float
    n: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def noise_floor(ref_pred_a, ref_pred_b, gold=None) -> NoiseFloor:
    """Disagreement between two behaviorally-equivalent runs (the control).

    Feed predictions from two runs that *should* behave identically -- e.g. the
    base model quantized at two different near-lossless seeds, or base vs base
    under two decoding seeds. The resulting disagreement is the floor a genuine
    quantization effect must clear. When ``gold`` is provided, also reports the
    correctness churn floor.
    """
    a = np.asarray(ref_pred_a).ravel()
    b = np.asarray(ref_pred_b).ravel()
    _check_pair(a, b)
    dis = float(np.mean(a != b))
    churn = float("nan")
    if gold is not None:
        g = np.asarray(gold).ravel()
        ac = a == g
        bc = b == g
        churn = float(np.mean(ac != bc))
    return NoiseFloor(floor_disagreement=dis, floor_churn=churn, n=int(a.size))


# ---------------------------------------------------------------------------
# one-pass report
# ---------------------------------------------------------------------------


@dataclass
class BehavioralReport:
    """Full behavioral comparison of a quantized model against its base.

    Combines the paper's CA, the symmetric flip analysis, prediction-level
    behavioral agreement, and (when a control is supplied) the excess
    disagreement over the noise floor with a z-score. ``excess_z`` > ~2 is the
    signal that behavioral drift exceeds free item brittleness.
    """

    flips: FlipResult
    behavioral_agreement: float
    disagreement: float
    floor: NoiseFloor | None
    excess_disagreement: float
    excess_z: float

    def as_dict(self) -> dict:
        d = {
            "flips": self.flips.as_dict(),
            "behavioral_agreement": self.behavioral_agreement,
            "disagreement": self.disagreement,
            "excess_disagreement": self.excess_disagreement,
            "excess_z": self.excess_z,
        }
        d["floor"] = self.floor.as_dict() if self.floor is not None else None
        return d

    def summary(self) -> str:
        f = self.flips
        lines = [
            f"n={f.n}  acc_base={f.acc_base:.3f}  acc_quant={f.acc_quant:.3f}  "
            f"net={f.net_delta:+.3f}",
            f"correctness_agreement(paper)={f.correctness_agreement:.3f}   "
            f"<= min(acc)={min(f.acc_base, f.acc_quant):.3f}",
            f"regressions={f.regressions:.3f}  recoveries={f.recoveries:.3f}  "
            f"churn={f.churn:.3f}  (McNemar p={f.mcnemar_p:.3g})",
            f"behavioral_agreement={self.behavioral_agreement:.3f}  "
            f"disagreement={self.disagreement:.3f}",
        ]
        if self.floor is not None:
            lines.append(
                f"noise floor={self.floor.floor_disagreement:.3f}  "
                f"excess={self.excess_disagreement:+.3f}  z={self.excess_z:.2f}"
            )
        return "\n".join(lines)


def evaluate(
    base_pred,
    quant_pred,
    gold,
    floor: NoiseFloor | None = None,
) -> BehavioralReport:
    """Full behavioral report from predicted labels + gold.

    Args:
        base_pred: Predicted labels from the base model.
        quant_pred: Predicted labels from the quantized model.
        gold: Gold labels.
        floor: Optional :class:`NoiseFloor` from :func:`noise_floor` for
            excess-over-floor attribution.
    """
    bp = np.asarray(base_pred).ravel()
    qp = np.asarray(quant_pred).ravel()
    g = np.asarray(gold).ravel()
    _check_pair(bp, qp)
    _check_pair(bp, g)
    bc = bp == g
    qc = qp == g
    flips = flip_rate(bc, qc)
    ba = float(np.mean(bp == qp))
    disagree = 1.0 - ba
    excess = float("nan")
    z = float("nan")
    if floor is not None:
        excess = disagree - floor.floor_disagreement
        # z of observed disagreement vs a Binomial(n, floor) null
        n = bp.size
        p0 = min(max(floor.floor_disagreement, 1e-9), 1 - 1e-9)
        se = math.sqrt(p0 * (1 - p0) / n)
        z = excess / se if se > 0 else float("nan")
    return BehavioralReport(
        flips=flips,
        behavioral_agreement=ba,
        disagreement=disagree,
        floor=floor,
        excess_disagreement=excess,
        excess_z=z,
    )

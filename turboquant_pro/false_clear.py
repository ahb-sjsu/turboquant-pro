# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""How often does a cheap nominal metric clear a result the consumer rejects?

The rank certificate (:mod:`~turboquant_pro.rank_certificate`) ships a
distribution-free *floor* on rank agreement. This module ships its empirical
companion: the *false-clear rate*, the measured frequency at which a cheap
nominal accept-metric (cosine similarity, reconstruction MSE) approves a
compression that the downstream consumer (retrieval recall, perplexity, task
score) actually fails.

It quantifies the phenomenon this package documents qualitatively as the KV-keys
finding: a per-channel key compression can hold cosine similarity at 0.995 and
still send perplexity to ~1e4. Cosine cleared; the consumer failed. That is a
false clear, and until now the package named it without measuring its rate.

The report is **directional**, matching the runtime policy's one-sided design
(:mod:`~turboquant_pro.runtime_policy`): the *false-clear* is the harmful
direction (you were reassured and the consumer broke), and a *conservative miss*
is the harmless one (you flagged a result the consumer would have accepted). The
verdict gates on the false-clear direction only.

**Thresholds here are conventions, not measurements.** ``FALSE_CLEAR_WARN`` and
``FALSE_CLEAR_FAIL`` are round numbers on the conditional false-clear rate
(``P(consumer fails | nominal cleared)``), chosen so the guard fires before a
"clear" is obviously untrustworthy. No sweep in this repository has established
where the operational knee sits for a given consumer; pass your own ``warn`` /
``fail`` when you have calibrated them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backend import to_numpy

FALSE_CLEAR_WARN = 0.05
FALSE_CLEAR_FAIL = 0.20

__all__ = [
    "FALSE_CLEAR_FAIL",
    "FALSE_CLEAR_WARN",
    "FalseClearReport",
    "check_false_clear",
    "false_clear",
    "false_clear_from_scores",
]


@dataclass
class FalseClearReport:
    """How often a nominal accept-metric disagrees with the consumer outcome."""

    false_clear_rate: float
    """``P(nominal clears AND consumer fails)`` over all items: how often, in
    total, a cheap approval left you wrongly reassured. The harmful direction."""

    false_clear_given_cleared: float
    """``P(consumer fails | nominal cleared)``: the untrustworthiness of a
    "clear". The headline the verdict gates on. ``0.0`` when nothing cleared."""

    conservative_miss_rate: float
    """``P(nominal rejects AND consumer ok)`` over all items: the harmless
    direction -- results the nominal metric flagged that were actually fine."""

    agreement: float
    """``P(nominal accept == consumer ok)``: overall agreement of the two."""

    verdict: str
    """``ok``, ``warn`` or ``fail`` against the thresholds, on the conditional
    false-clear rate (the harmful direction)."""

    n: int
    n_cleared: int
    n_consumer_fail: int

    def to_dict(self) -> dict:
        return {
            "false_clear_rate": self.false_clear_rate,
            "false_clear_given_cleared": self.false_clear_given_cleared,
            "conservative_miss_rate": self.conservative_miss_rate,
            "agreement": self.agreement,
            "verdict": self.verdict,
            "n": self.n,
            "n_cleared": self.n_cleared,
            "n_consumer_fail": self.n_consumer_fail,
            "thresholds": {"warn": FALSE_CLEAR_WARN, "fail": FALSE_CLEAR_FAIL},
        }


def _verdict(cond_rate: float, warn: float, fail: float) -> str:
    if cond_rate >= fail:
        return "fail"
    if cond_rate >= warn:
        return "warn"
    return "ok"


def false_clear(
    nominal_accept,
    consumer_ok,
    *,
    warn: float = FALSE_CLEAR_WARN,
    fail: float = FALSE_CLEAR_FAIL,
) -> FalseClearReport:
    """Measure the false-clear rate from two per-item boolean outcomes.

    Args:
        nominal_accept: Boolean per item -- the cheap nominal metric approved
            this item (e.g. cosine similarity above threshold).
        consumer_ok: Boolean per item -- the true downstream consumer outcome
            was good (e.g. exact neighbor preserved, perplexity within budget).
        warn, fail: Thresholds on the conditional false-clear rate for the
            tri-state verdict. Defaults are conventions (see module docstring).

    Returns:
        A :class:`FalseClearReport`.

    Raises:
        ValueError: if the two arrays differ in length or are empty.
    """
    na = np.asarray(to_numpy(nominal_accept)).astype(bool).ravel()
    co = np.asarray(to_numpy(consumer_ok)).astype(bool).ravel()
    if na.shape != co.shape:
        raise ValueError(f"length mismatch: {na.shape} vs {co.shape}")
    n = int(na.size)
    if n == 0:
        raise ValueError("need at least one item")
    n_cleared = int(na.sum())
    n_fail = int((~co).sum())
    fc = int((na & ~co).sum())              # cleared, but consumer failed
    miss = int((~na & co).sum())            # rejected, but consumer was fine
    fc_rate = fc / n
    fc_given = fc / n_cleared if n_cleared else 0.0
    return FalseClearReport(
        false_clear_rate=fc_rate,
        false_clear_given_cleared=fc_given,
        conservative_miss_rate=miss / n,
        agreement=float((na == co).mean()),
        verdict=_verdict(fc_given, warn, fail),
        n=n,
        n_cleared=n_cleared,
        n_consumer_fail=n_fail,
    )


def false_clear_from_scores(
    nominal,
    consumer,
    *,
    nominal_threshold: float,
    consumer_threshold: float,
    nominal_higher_is_better: bool = True,
    consumer_higher_is_better: bool = True,
    warn: float = FALSE_CLEAR_WARN,
    fail: float = FALSE_CLEAR_FAIL,
) -> FalseClearReport:
    """Threshold continuous per-item scores into accept / ok, then measure.

    A convenience for the common case where you have a cheap nominal score
    (cosine similarity, reconstruction distance) and an expensive consumer score
    (recall@k, negative perplexity, task metric) per item, and a threshold on
    each that defines "accepted" and "consumer ok".

    Args:
        nominal: Per-item nominal score.
        consumer: Per-item consumer score, same length.
        nominal_threshold: Accept iff score is on the good side of this.
        consumer_threshold: Consumer ok iff score is on the good side of this.
        nominal_higher_is_better: If True, accept when ``score >= threshold``;
            if False (e.g. a distance / MSE), accept when ``score <= threshold``.
        consumer_higher_is_better: Same convention for the consumer score.
        warn, fail: Verdict thresholds (conventions).

    Returns:
        A :class:`FalseClearReport`.
    """
    nom = np.asarray(to_numpy(nominal), dtype=np.float64).ravel()
    con = np.asarray(to_numpy(consumer), dtype=np.float64).ravel()
    accept = nom >= nominal_threshold if nominal_higher_is_better else nom <= nominal_threshold
    ok = con >= consumer_threshold if consumer_higher_is_better else con <= consumer_threshold
    return false_clear(accept, ok, warn=warn, fail=fail)


def check_false_clear(
    nominal_accept,
    consumer_ok,
    *,
    strict: bool = False,
    warn: float = FALSE_CLEAR_WARN,
    fail: float = FALSE_CLEAR_FAIL,
) -> FalseClearReport:
    """Measure the false-clear rate and warn (or raise) on the verdict.

    Emits a :class:`UserWarning` at ``warn`` and at ``fail``; with
    ``strict=True`` a ``fail`` verdict raises :class:`ValueError` instead of
    warning. Returns the report either way (unless it raises).
    """
    import warnings

    report = false_clear(nominal_accept, consumer_ok, warn=warn, fail=fail)
    if report.verdict == "fail":
        msg = (
            f"false clear: a nominal 'clear' is wrong {report.false_clear_given_cleared:.1%} "
            f"of the time (>= fail={fail:.0%}); the consumer rejects what the metric accepts"
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
    elif report.verdict == "warn":
        warnings.warn(
            f"false clear: a nominal 'clear' is wrong {report.false_clear_given_cleared:.1%} "
            f"of the time (>= warn={warn:.0%})",
            stacklevel=2,
        )
    return report

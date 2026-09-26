# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Adaptive certified refinement: rerank only where the compressed ranking is
not decisive, with a declared recall that holds (issue #175, phase 1).

A fixed ``rerank=r`` reads ``k * r`` original rows for every query, whether the
compressed scores already separated the top ``k`` or not. This module replaces
it with a per-query band. After the compressed scan, a candidate enters the
band when its score is within ``epsilon`` of the ``k``-th compressed score:

    s_hat_j  >=  s_hat_(k)  -  epsilon * scale(q)

with ``scale(q) = 1`` under cosine and ``||q||`` under inner product and l2.
If the band holds exactly ``k`` rows, the top-``k`` set is returned as it is
and no original row is read (stage ``scan``). Otherwise the band is rescored
exactly and the best ``k`` are returned (stage ``rerank``). The band grows
with ``epsilon``, so recall is non-decreasing in it, and the number of rows
read is set per query by how crowded the ranking is at rank ``k``.

``band="rank"`` is the same procedure with a rank cutoff in place of the score
margin: the first ``k + epsilon`` candidates, for every query. Calibrated the
same way, it is the smallest fixed rerank depth that certifies the target, the
baseline an adaptive band has to beat.

**Where epsilon comes from, and what it guarantees.** :func:`calibrate` runs
the same procedure on ``n`` calibration queries against the exact top ``k``
and picks the smallest ``epsilon`` with

    ( n * R_n(epsilon) + 1 ) / ( n + 1 )  <=  1 - target_recall

where ``R_n`` is the mean of ``1 - recall@k`` over the calibration queries.
Conformal risk control (Angelopoulos, Bates, Fisch, Lei and Schuster,
*Conformal Risk Control*, ICLR 2024) then gives, for a loss bounded by one
and non-increasing in ``epsilon``,

    E[ recall@k of the returned set ]  >=  target_recall

for a new query exchangeable with the calibration queries, searched against
the same index with the same ``k`` and candidate cap. The statement is
marginal over queries, not per query. It needs no bound on the codec's
error, no distributional assumption on the corpus, and at least
``1 / (1 - target) - 1`` calibration queries (999 for 0.999); fewer, or a
target the candidate cap cannot reach, is refused with the reason
(:class:`InfeasibleTarget`) instead of returning a weaker policy.

**Provenance.** The guarantee is about one index, one ``k`` and one cap. A
policy records them and a fingerprint of the index (its identity and the
sha256 of its stored row norms), and :func:`search` refuses a policy made for
another index. Exchangeability of the query stream is the caller's claim;
when the queries drift, the certificate expiry of #177 is the check.

Usage::

    from turboquant_pro import adaptive_rerank as AR

    policy = AR.calibrate(index, cal_queries, originals, k=10, target_recall=0.99)
    ids, report = AR.search(index, queries, originals, policy)
    report.mean_rows_read, report.stage_fractions
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from .metrics import COSINE, check_metric, exact_scores

__all__ = [
    "SCHEMA",
    "AdaptivePolicy",
    "AdaptiveReport",
    "BANDS",
    "InfeasibleTarget",
    "calibrate",
    "index_fingerprint",
    "search",
]

SCHEMA = "turboquant-pro/adaptive-policy"
SCHEMA_VERSION = 1
GUARANTEE = (
    "expected recall@k >= target_recall over queries exchangeable with the "
    "calibration queries, on this index with this k and candidate cap "
    "(conformal risk control)"
)


class InfeasibleTarget(ValueError):
    """The declared recall cannot be certified; the message says why."""


# --------------------------------------------------------------------------- #
# The index a policy belongs to                                               #
# --------------------------------------------------------------------------- #


def index_fingerprint(index) -> dict:
    """What a policy binds to: the index's identity and a content hash.

    The hash covers the stored per-row norms (four bytes a row), which change
    with any change to the indexed rows, their order or the projection; it is
    a fingerprint of this index, not of the codec in general.
    """
    ident = getattr(index, "_trace_identity", None)
    norms = getattr(index, "_cnorm", None)
    if ident is None or norms is None:
        raise TypeError(
            "adaptive refinement needs an ADCIndex (identity and stored norms); "
            f"got {type(index).__name__}"
        )
    out = {k: v for k, v in ident().items() if k != "kernel"}
    h = hashlib.sha256(json.dumps(out, sort_keys=True).encode())
    h.update(np.ascontiguousarray(norms, dtype=np.float32).tobytes())
    out["fingerprint_sha256"] = h.hexdigest()
    return out


# --------------------------------------------------------------------------- #
# The band                                                                    #
# --------------------------------------------------------------------------- #


def _scale(queries: np.ndarray, metric: str) -> np.ndarray:
    """Per-query unit of the margin: one under cosine (scores are cosines), the
    query norm under inner product and l2 (scores scale with it)."""
    if metric == COSINE:
        return np.ones(len(queries), dtype=np.float64)
    n = np.linalg.norm(np.asarray(queries, dtype=np.float64), axis=1)
    return np.where(n > 0, n, 1.0)


BANDS = ("score", "rank")


def _entry(sc: np.ndarray, k: int, scale: np.ndarray, band: str) -> np.ndarray:
    """``(n_q, cap)``: the smallest epsilon at which each scanned candidate
    enters its query's band (a score margin in units of ``scale``, or a count
    of extra ranks). The first ``k`` enter at zero. Calibration and search both
    decide membership as ``_entry(...) <= epsilon``, so they cannot disagree
    by rounding."""
    if band == "rank":
        e = np.maximum(np.arange(sc.shape[1], dtype=np.float64) - (k - 1), 0.0)
        return np.broadcast_to(e, sc.shape)
    s = sc.astype(np.float64)
    e = (s[:, k - 1 : k] - s) / scale[:, None]
    e[:, :k] = 0.0
    return np.maximum(e, 0.0)


def _scan(index, queries: np.ndarray, cap: int):
    idx, sc = index.search(queries, k=cap)
    return np.asarray(idx), np.asarray(sc)


# --------------------------------------------------------------------------- #
# Policy                                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AdaptivePolicy:
    """A calibrated band: ``epsilon`` for one index, ``k`` and candidate cap.

    ``risk_bound`` is the left side of the calibration inequality at
    ``epsilon`` (at most ``1 - target_recall``); ``calibration_recall``,
    ``calibration_rows_read`` and ``calibration_scan_fraction`` are what the
    procedure did on the calibration queries, for comparison with a fixed
    rerank (which reads ``k * r`` rows a query).
    """

    k: int
    max_candidates: int
    band: str
    epsilon: float
    target_recall: float
    n_calibration: int
    risk_bound: float
    calibration_recall: float
    calibration_rows_read: float
    calibration_scan_fraction: float
    index: dict

    def as_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "guarantee": GUARANTEE,
            "k": self.k,
            "max_candidates": self.max_candidates,
            "band": self.band,
            "epsilon": self.epsilon,
            "target_recall": self.target_recall,
            "calibration": {
                "n_queries": self.n_calibration,
                "risk_bound": self.risk_bound,
                "recall": self.calibration_recall,
                "rows_read_mean": self.calibration_rows_read,
                "scan_stop_fraction": self.calibration_scan_fraction,
            },
            "index": dict(self.index),
        }

    @classmethod
    def from_dict(cls, d: dict) -> AdaptivePolicy:
        if not isinstance(d, dict) or d.get("schema") != SCHEMA:
            raise ValueError(f"not an adaptive policy (schema must be {SCHEMA!r})")
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"adaptive policy schema_version must be {SCHEMA_VERSION}")
        c = d["calibration"]
        p = cls(
            k=int(d["k"]),
            max_candidates=int(d["max_candidates"]),
            band=str(d["band"]),
            epsilon=float(d["epsilon"]),
            target_recall=float(d["target_recall"]),
            n_calibration=int(c["n_queries"]),
            risk_bound=float(c["risk_bound"]),
            calibration_recall=float(c["recall"]),
            calibration_rows_read=float(c["rows_read_mean"]),
            calibration_scan_fraction=float(c["scan_stop_fraction"]),
            index=dict(d["index"]),
        )
        if not (
            p.band in BANDS
            and 1 <= p.k <= p.max_candidates
            and math.isfinite(p.epsilon)
            and p.epsilon >= 0
            and 0 < p.target_recall < 1
            and p.risk_bound <= 1 - p.target_recall + 1e-12
        ):
            raise ValueError("adaptive policy fields are inconsistent")
        return p


def calibrate(
    index,
    queries: np.ndarray,
    originals: np.ndarray,
    k: int = 10,
    target_recall: float = 0.99,
    max_candidates: int | None = None,
    band: str = "score",
    block: int = 256,
) -> AdaptivePolicy:
    """Choose the band for ``index`` that certifies ``target_recall`` at ``k``.

    ``queries`` are the calibration queries, exchangeable with the ones the
    policy will serve and not used to build the index's codec (a query used to
    fit is not a calibration query). ``originals`` is the fp32 corpus in index
    order. ``max_candidates`` caps the compressed candidates a query can send
    to exact rescoring (default ``20 * k``); the cap is part of the certified
    procedure. ``band`` is ``"score"`` (the adaptive margin) or ``"rank"``
    (a fixed depth, the baseline).
    """
    if band not in BANDS:
        raise ValueError(f"band must be one of {BANDS}")
    if not 0 < target_recall < 1:
        raise ValueError("target_recall must lie strictly between 0 and 1")
    metric = check_metric(index.metric)
    q = np.asarray(queries, dtype=np.float32)
    x = np.asarray(originals)
    n, size = len(q), int(index.size)
    if len(x) != size:
        raise ValueError(f"originals has {len(x)} rows, the index {size}")
    if not 1 <= k <= size:
        raise ValueError(f"k must be in 1..{size}")
    cap = min(int(max_candidates or 20 * k), size)
    if cap < k:
        raise ValueError("max_candidates must be at least k")
    alpha = 1.0 - target_recall
    need = math.ceil(1.0 / alpha - 1.0 - 1e-9)
    if n < need:
        raise InfeasibleTarget(
            f"certifying recall {target_recall} needs at least {need} calibration "
            f"queries (the finite-sample term 1/(n+1) alone exceeds {alpha:g}); "
            f"got {n}"
        )
    idx, sc = _scan(index, q, cap)
    entry = _entry(sc, k, _scale(q, metric), band)
    hits = np.zeros_like(entry, dtype=bool)
    xf = np.asarray(x, dtype=np.float32)
    for s in range(0, n, block):
        e = min(s + block, n)
        ex = exact_scores(q[s:e], xf, metric)
        top = np.argpartition(-ex, k - 1, axis=1)[:, :k]
        for i in range(e - s):
            hits[s + i] = np.isin(idx[s + i], top[i])
    pooled = np.sort(entry[hits])  # every true neighbour's entry epsilon

    def bound(found: int) -> float:  # (n R_n + 1) / (n + 1) given hits found
        return (n * (1.0 - found / (n * k)) + 1.0) / (n + 1.0)

    # R_n steps down only at pooled values; the first one meeting the bound wins.
    found_at = np.arange(1, len(pooled) + 1, dtype=np.float64)
    ok = np.nonzero((n * (1.0 - found_at / (n * k)) + 1.0) / (n + 1.0) <= alpha)[0]
    if len(ok) == 0:
        raise InfeasibleTarget(
            f"recall {target_recall} is out of reach with {cap} candidates: even "
            f"rescoring all of them certifies only "
            f"{1.0 - bound(len(pooled)):.6f} (raise max_candidates, or the codec "
            "loses true neighbours before the cap)"
        )
    j = int(ok[0])
    # Ties: every true neighbour with the same entry value is admitted with it.
    eps = float(pooled[j])
    found = int(np.searchsorted(pooled, eps, side="right"))
    width = (entry <= eps).sum(axis=1)
    return AdaptivePolicy(
        k=k,
        max_candidates=cap,
        band=band,
        epsilon=eps,
        target_recall=float(target_recall),
        n_calibration=n,
        risk_bound=bound(found),
        calibration_recall=found / (n * k),
        calibration_rows_read=float(np.where(width > k, width, 0).mean()),
        calibration_scan_fraction=float((width == k).mean()),
        index=index_fingerprint(index),
    )


# --------------------------------------------------------------------------- #
# Search                                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AdaptiveReport:
    """What each query cost. ``rows_read[i]`` original rows were rescored (zero
    when the scan was decisive); ``truncated[i]`` means the band reached the
    candidate cap, which the certified procedure allows but a caller may want
    to see."""

    stage: np.ndarray  # "scan" or "rerank" per query
    rows_read: np.ndarray
    truncated: np.ndarray
    row_bytes: int

    @property
    def stage_fractions(self) -> dict:
        n = max(len(self.stage), 1)
        return {s: float((self.stage == s).sum() / n) for s in ("scan", "rerank")}

    @property
    def mean_rows_read(self) -> float:
        return float(self.rows_read.mean()) if len(self.rows_read) else 0.0

    @property
    def bytes_read(self) -> np.ndarray:
        """Original bytes each query read (the compressed scan is the same for
        every query and every policy, so it is not counted)."""
        return self.rows_read * self.row_bytes


def search(index, queries: np.ndarray, originals: np.ndarray, policy: AdaptivePolicy):
    """Top-``policy.k`` ids per query under the calibrated band, and a report.

    Queries whose band holds only ``k`` rows return the compressed top ``k``
    in compressed order; the others return the exact order of their band.
    """
    have = index_fingerprint(index)
    if have != policy.index:
        raise ValueError(
            "this policy was calibrated on another index "
            f"({policy.index.get('fingerprint_sha256', '?')[:12]}, this one is "
            f"{have['fingerprint_sha256'][:12]}); its guarantee does not transfer"
        )
    metric = check_metric(index.metric)
    q = np.asarray(queries, dtype=np.float32)
    k = policy.k
    idx, sc = _scan(index, q, policy.max_candidates)
    entry = _entry(sc, k, _scale(q, metric), policy.band)
    width = (entry <= policy.epsilon).sum(axis=1)
    out = idx[:, :k].copy()
    rows = np.zeros(len(q), dtype=np.int64)
    for i in np.nonzero(width > k)[0]:
        cand = idx[i, : width[i]]
        s = exact_scores(q[i : i + 1], np.asarray(originals[cand], np.float32), metric)
        out[i] = cand[np.argsort(-s[0], kind="stable")[:k]]
        rows[i] = width[i]
    x = np.asarray(originals[:1])
    return out, AdaptiveReport(
        stage=np.where(width > k, "rerank", "scan"),
        rows_read=rows,
        truncated=(width >= policy.max_candidates)
        & (policy.max_candidates < index.size),
        row_bytes=int(x.shape[1] * x.dtype.itemsize),
    )

"""Exact-versus-index retrieval oracle for query-only fuzzing.

The corpus and compressed index stay immutable during an MVP campaign.  Query
mutators therefore invalidate exact nearest-neighbour truth on every candidate;
this module deliberately accepts vectors, not a precomputed truth file.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np

CLASSIFICATIONS = frozenset(
    {"certified_violation", "consumer_regression", "stress_discovery"}
)


class SearchIndex(Protocol):
    """The immutable compressed-index surface used by the oracle."""

    def search(
        self, queries: np.ndarray, k: int = 10, **kwargs: object
    ) -> tuple[np.ndarray, np.ndarray]: ...


def _validated_vectors(
    corpus: np.ndarray,
    queries: np.ndarray,
    corpus_ids: np.ndarray | None,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.ascontiguousarray(corpus, dtype=np.float64)
    q = np.ascontiguousarray(queries, dtype=np.float64)
    if base.ndim != 2 or q.ndim != 2 or base.shape[1] != q.shape[1]:
        raise ValueError(
            "corpus and queries must be finite 2-D arrays of matching width"
        )
    if not len(q) or not np.isfinite(base).all() or not np.isfinite(q).all():
        raise ValueError("corpus and queries must be non-empty and finite")
    if not 1 <= k <= len(base):
        raise ValueError(f"k must be in [1, {len(base)}], got {k}")
    ids = (
        np.arange(len(base), dtype=np.int64)
        if corpus_ids is None
        else np.asarray(corpus_ids, dtype=np.int64)
    )
    if ids.shape != (len(base),) or len(np.unique(ids)) != len(ids):
        raise ValueError("corpus_ids must contain one unique integer ID per corpus row")
    return base, q, ids


def exact_top_k(
    corpus: np.ndarray,
    queries: np.ndarray,
    *,
    k: int,
    metric: str,
    corpus_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute deterministic exact top-k IDs and scores for every query.

    Ties are ordered by ascending external ID, avoiding platform-dependent
    partition ordering.  Scores match ``TQEIndex.search``: cosine similarity or
    negative squared L2 distance.
    """
    if metric not in {"cosine", "l2"}:
        raise ValueError("metric must be 'cosine' or 'l2'")
    base, q, ids = _validated_vectors(corpus, queries, corpus_ids, k)
    if metric == "cosine":
        base = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-30)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-30)
        scores = q @ base.T
    else:
        base_norm = np.einsum("ij,ij->i", base, base)
        query_norm = np.einsum("ij,ij->i", q, q)
        scores = -(query_norm[:, None] + base_norm[None, :] - 2.0 * (q @ base.T))
    order = np.empty((len(q), k), dtype=np.int64)
    for row, row_scores in enumerate(scores):
        order[row] = np.lexsort((ids, -row_scores))[:k]
    return ids[order], np.take_along_axis(scores, order, axis=1)


def _validate_observed(ids: np.ndarray, n_queries: int, k: int) -> np.ndarray:
    values = np.asarray(ids, dtype=np.int64)
    if values.shape != (n_queries, k):
        raise ValueError(
            "compressed index returned IDs with a shape different from (n_queries, k)"
        )
    if np.any(values < 0):
        raise ValueError("compressed index returned fewer than k live results")
    return values


def _rank_disagreement(exact: np.ndarray, observed: np.ndarray) -> float:
    """Mean normalized top-k rank displacement with a fixed missing-item rank."""
    k = exact.shape[1]
    total = 0.0
    for wanted, actual in zip(exact, observed):
        actual_ranks = {int(value): rank for rank, value in enumerate(actual)}
        total += sum(
            abs(rank - actual_ranks.get(int(value), k))
            for rank, value in enumerate(wanted)
        ) / (k * max(k, 1))
    return total / len(exact)


def exact_vs_tqp(
    corpus: np.ndarray,
    queries: np.ndarray,
    index: SearchIndex,
    *,
    k: int,
    metric: str,
    corpus_ids: np.ndarray | None = None,
    certificate_contradicted: bool = False,
    search_kwargs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Compare freshly recomputed exact truth with the actual index path.

    ``certificate_contradicted`` is intentionally explicit.  Ordinary retrieval
    disagreement is a consumer regression, never a formal certificate claim.
    """
    exact_ids, exact_scores = exact_top_k(
        corpus, queries, k=k, metric=metric, corpus_ids=corpus_ids
    )
    kwargs = dict(search_kwargs or {})
    observed_ids, observed_scores = index.search(queries, k=k, **kwargs)
    observed = _validate_observed(observed_ids, len(exact_ids), k)
    observed_scores = np.asarray(observed_scores)
    if (
        observed_scores.shape != observed.shape
        or not np.isfinite(observed_scores).all()
    ):
        raise ValueError("compressed index returned invalid top-k scores")
    recall_per_query = np.array(
        [
            len(set(wanted).intersection(actual)) / k
            for wanted, actual in zip(exact_ids, observed)
        ],
        dtype=np.float64,
    )
    first_incorrect = [
        next(
            (
                rank + 1
                for rank, pair in enumerate(zip(wanted, actual))
                if pair[0] != pair[1]
            ),
            k + 1,
        )
        for wanted, actual in zip(exact_ids, observed)
    ]
    recall = float(recall_per_query.mean())
    if certificate_contradicted:
        classification = "certified_violation"
    elif recall < 1.0:
        classification = "consumer_regression"
    else:
        classification = "stress_discovery"
    return {
        "classification": classification,
        "exact_top_k": exact_ids.tolist(),
        "observed_tqp_top_k": observed.tolist(),
        "metrics": {
            "recall_at_k": recall,
            "rank_disagreement": _rank_disagreement(exact_ids, observed),
            "first_incorrect_rank": int(min(first_incorrect)),
            "queries_with_membership_change": int(
                np.count_nonzero(recall_per_query < 1.0)
            ),
        },
        "exact_scores": exact_scores.tolist(),
        "observed_tqp_scores": observed_scores.astype(np.float64).tolist(),
        "metric": metric,
        "k": int(k),
    }

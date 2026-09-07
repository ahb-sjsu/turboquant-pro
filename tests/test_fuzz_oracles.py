"""Fresh exact-truth and compressed-index oracle regressions."""

from __future__ import annotations

import numpy as np

from turboquant_pro.fuzz.oracles import CLASSIFICATIONS, exact_top_k, exact_vs_tqp


class _StaticIndex:
    def __init__(self, ids: np.ndarray) -> None:
        self.ids = ids
        self.calls = 0

    def search(self, queries, k=10, **kwargs):
        self.calls += 1
        assert len(queries) == len(self.ids)
        assert k == self.ids.shape[1]
        return self.ids, np.ones_like(self.ids, dtype=np.float32)


def test_exact_top_k_is_tie_stable_by_external_id():
    corpus = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    ids, _ = exact_top_k(
        corpus,
        np.array([[0.0, 0.0]]),
        k=3,
        metric="l2",
        corpus_ids=np.array([30, 10, 20]),
    )

    np.testing.assert_array_equal(ids, [[10, 20, 30]])


def test_oracle_recomputes_mutated_query_truth_and_reports_regression():
    corpus = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 2.0]])
    first = exact_top_k(corpus, np.array([[0.1, 0.0]]), k=2, metric="l2")[0]
    mutated = np.array([[3.9, 0.0]])
    fresh = exact_top_k(corpus, mutated, k=2, metric="l2")[0]
    assert not np.array_equal(first, fresh)
    index = _StaticIndex(np.array([[0, 2]], dtype=np.int64))

    result = exact_vs_tqp(corpus, mutated, index, k=2, metric="l2")

    assert index.calls == 1
    assert result["exact_top_k"] == fresh.tolist()
    assert result["classification"] == "consumer_regression"
    assert result["metrics"]["recall_at_k"] == 0.5
    assert result["metrics"]["first_incorrect_rank"] == 1
    assert result["classification"] in CLASSIFICATIONS


def test_oracle_reserves_certificate_violation_for_explicit_contradiction():
    corpus = np.eye(2)
    queries = np.eye(2)
    exact, _ = exact_top_k(corpus, queries, k=1, metric="cosine")
    result = exact_vs_tqp(
        corpus,
        queries,
        _StaticIndex(exact),
        k=1,
        metric="cosine",
        certificate_contradicted=True,
    )

    assert result["classification"] == "certified_violation"

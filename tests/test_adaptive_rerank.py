"""Adaptive certified refinement (turboquant_pro.adaptive_rerank, issue #175):
the calibrated band keeps its declared recall on fresh queries, calibration and
search run the same procedure, and a policy refuses what it cannot certify."""

from __future__ import annotations

import json

import numpy as np
import pytest

from turboquant_pro import ADCIndex, PCAMatryoshka
from turboquant_pro import adaptive_rerank as AR
from turboquant_pro.metrics import exact_scores
from turboquant_pro.schemas import load_schema

D, N, K = 48, 4000, 10


def _draw(rng, n):
    return (rng.standard_normal((n, D)) * np.geomspace(2.0, 0.2, D)).astype(np.float32)


def _world(seed=0, bits=2, out=24, metric="cosine"):
    rng = np.random.default_rng(seed)
    X = _draw(rng, N)
    pca = PCAMatryoshka(input_dim=D, output_dim=out)
    pca.fit(X[:2000])
    index = ADCIndex(pca.with_quantizer(bits=bits), metric=metric).add(X)
    return index, X, rng


def _recall(index, Q, X, ids, k=K):
    ex = exact_scores(Q, X, index.metric)
    top = np.argpartition(-ex, k - 1, axis=1)[:, :k]
    return np.array([len(set(top[i]) & set(ids[i])) / k for i in range(len(Q))])


def test_declared_recall_holds_on_fresh_queries():
    index, X, rng = _world()
    policy = AR.calibrate(index, _draw(rng, 300), X, k=K, target_recall=0.95)
    Q = _draw(rng, 600)  # exchangeable with the calibration queries
    ids, rep = AR.search(index, Q, X, policy)
    rec = _recall(index, Q, X, ids)
    # The guarantee is in expectation; 600 queries put the mean within ~0.01.
    assert rec.mean() >= 0.95 - 0.02, rec.mean()
    plain = _recall(index, Q, X, index.search(Q, k=K)[0])
    assert plain.mean() < 0.95  # the codec alone does not reach it
    assert rep.mean_rows_read < policy.max_candidates
    assert set(rep.stage_fractions) == {"scan", "rerank"}
    assert AR.AdaptivePolicy.from_dict(policy.as_dict()).band == "score"


def test_calibration_and_search_are_the_same_procedure():
    """Pipeline, not components: searching the calibration queries reproduces
    the recall and rows read that calibration recorded, exactly."""
    index, X, rng = _world(seed=1)
    C = _draw(rng, 200)
    policy = AR.calibrate(index, C, X, k=K, target_recall=0.9)
    ids, rep = AR.search(index, C, X, policy)
    assert _recall(index, C, X, ids).mean() == pytest.approx(
        policy.calibration_recall, abs=1e-12
    )
    assert rep.mean_rows_read == pytest.approx(policy.calibration_rows_read)
    assert rep.stage_fractions["scan"] == pytest.approx(
        policy.calibration_scan_fraction
    )
    assert policy.risk_bound <= 1 - 0.9


def test_a_higher_target_widens_the_band():
    index, X, rng = _world(seed=2)
    C = _draw(rng, 300)
    lo = AR.calibrate(index, C, X, k=K, target_recall=0.8)
    hi = AR.calibrate(index, C, X, k=K, target_recall=0.95)
    assert hi.epsilon >= lo.epsilon
    assert hi.calibration_rows_read >= lo.calibration_rows_read
    assert hi.calibration_recall >= lo.calibration_recall


def test_a_decisive_scan_reads_nothing_and_keeps_the_compressed_set():
    index, X, rng = _world(seed=3, bits=4, out=D)
    Q = _draw(rng, 200)
    policy = AR.calibrate(index, Q, X, k=K, target_recall=0.9)
    ids, rep = AR.search(index, Q, X, policy)
    stop = rep.stage == "scan"
    assert stop.any()
    assert (rep.rows_read[stop] == 0).all() and (rep.bytes_read[stop] == 0).all()
    np.testing.assert_array_equal(ids[stop], index.search(Q, k=K)[0][stop])
    assert (rep.rows_read[~stop] > K).all()
    assert rep.row_bytes == D * 4


def test_query_scale_does_not_change_an_inner_product_decision():
    """Metamorphic: the band is in units of ||q||, so scaling the queries by a
    positive constant changes no stage, no count and no id."""
    index, X, rng = _world(seed=4, metric="inner_product")
    C, Q = _draw(rng, 200), _draw(rng, 100)
    policy = AR.calibrate(index, C, X, k=K, target_recall=0.9)
    a, ra = AR.search(index, Q, X, policy)
    b, rb = AR.search(index, 8.0 * Q, X, policy)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ra.rows_read, rb.rows_read)


def test_too_few_calibration_queries_are_refused():
    index, X, rng = _world(seed=5)
    with pytest.raises(AR.InfeasibleTarget, match="at least 999"):
        AR.calibrate(index, _draw(rng, 500), X, k=K, target_recall=0.999)


def test_a_target_the_cap_cannot_reach_is_refused():
    index, X, rng = _world(seed=6, bits=1, out=8)
    with pytest.raises(AR.InfeasibleTarget, match="out of reach"):
        AR.calibrate(
            index, _draw(rng, 150), X, k=K, target_recall=0.99, max_candidates=K
        )


def test_a_policy_does_not_transfer_to_another_index():
    index, X, rng = _world(seed=7)
    policy = AR.calibrate(index, _draw(rng, 100), X, k=K, target_recall=0.9)
    other, Y, _ = _world(seed=8)  # same shape, other rows
    assert AR.index_fingerprint(other)["rows"] == policy.index["rows"]
    with pytest.raises(ValueError, match="another index"):
        AR.search(other, _draw(rng, 5), Y, policy)


def test_policy_round_trips_and_matches_the_schema():
    jsonschema = pytest.importorskip("jsonschema")
    index, X, rng = _world(seed=9)
    policy = AR.calibrate(index, _draw(rng, 100), X, k=K, target_recall=0.9)
    doc = json.loads(json.dumps(policy.as_dict()))
    jsonschema.validate(doc, load_schema("adaptive_policy.schema.json"))
    assert AR.AdaptivePolicy.from_dict(doc) == policy
    doc["calibration"]["risk_bound"] = 0.5  # claims more risk than the target allows
    with pytest.raises(ValueError, match="inconsistent"):
        AR.AdaptivePolicy.from_dict(doc)


def test_the_rank_band_is_a_certified_fixed_depth():
    """``band="rank"`` reranks the same number of rows for every query: the
    smallest fixed depth that certifies the target, and it keeps it too."""
    index, X, rng = _world(seed=10)
    policy = AR.calibrate(
        index, _draw(rng, 300), X, k=K, target_recall=0.95, band="rank"
    )
    assert policy.epsilon == int(policy.epsilon)
    Q = _draw(rng, 600)
    ids, rep = AR.search(index, Q, X, policy)
    assert set(rep.rows_read.tolist()) == {K + int(policy.epsilon)}
    assert _recall(index, Q, X, ids).mean() >= 0.95 - 0.02
    with pytest.raises(ValueError, match="band"):
        AR.calibrate(index, _draw(rng, 100), X, k=K, band="radius")

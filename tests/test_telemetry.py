"""Console Phase 0: the telemetry contract (turboquant_pro.telemetry)."""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro import ADCIndex, PCAMatryoshka, telemetry
from turboquant_pro.schemas import load_schema
from turboquant_pro.telemetry import metrics as M
from turboquant_pro.telemetry import trace as T

jsonschema = pytest.importorskip("jsonschema")


@pytest.fixture
def corpus():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1500, 64)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    pca = PCAMatryoshka(input_dim=64, output_dim=32)
    pca.fit(X[:800])
    return X, ADCIndex(pca.with_quantizer(bits=3)).add(X)


@pytest.fixture(autouse=True)
def _off():
    telemetry.disable()
    yield
    telemetry.disable()


def test_off_by_default_and_begin_is_none():
    assert telemetry.active() is None
    assert T.begin("x", np.zeros((1, 4))) is None


def test_tracing_never_changes_what_a_search_returns(corpus):
    """Metamorphic: the same calls with tracing on return identical ids and scores."""
    X, index = corpus
    q = X[:7]
    plain = index.search(q, k=5)
    plain_rr = index.search(q, k=5, rerank=4, originals=X)
    telemetry.enable()
    traced = index.search(q, k=5)
    traced_rr = index.search(q, k=5, rerank=4, originals=X)
    np.testing.assert_array_equal(plain[0], traced[0])
    np.testing.assert_array_equal(plain[1], traced[1])
    np.testing.assert_array_equal(plain_rr, traced_rr)


def test_trace_documents_validate_and_carry_the_stages(corpus):
    X, index = corpus
    tr = telemetry.enable()
    index.search(X[:3], k=5)
    out = index.search(X[:3], k=5, rerank=4, originals=X)
    schema = load_schema("query_trace.schema.json")
    docs = tr.traces()
    assert len(docs) == 2
    for d in docs:
        jsonschema.validate(d, schema)
        assert d["scan_path"] in T.SCAN_PATHS
        assert d["input"]["n_queries"] == 3 and "vectors" not in d["input"]
    assert [s["name"] for s in docs[0]["stages"]] == ["encode", "scan"]
    assert [s["name"] for s in docs[1]["stages"]] == ["encode", "scan", "rerank"]
    res = docs[1]["results"]
    assert [r["id"] for r in res["final"]] == [int(i) for i in out[0] if i >= 0]
    # exact scores descend, and rank movement is approximate position minus final
    ex = [r["exact_score"] for r in res["final"]]
    assert ex == sorted(ex, reverse=True)
    for pos, r in enumerate(res["final"]):
        if r["approx_rank"] is not None:
            assert r["rank_movement"] == r["approx_rank"] - pos
    assert 0.0 <= res["rerank_agreement"] <= 1.0


def test_readings_carry_their_spec_and_unavailable_says_why(corpus):
    X, index = corpus
    tr = telemetry.enable()
    schema = load_schema("metric_reading.schema.json")
    for r in tr.snapshot():  # nothing traced yet: latencies are unavailable, qps is 0
        jsonschema.validate(r, schema)
        if r["value"] is None:
            assert r["unavailable_reason"]
    for _ in range(3):
        index.search(X[:4], k=5, rerank=2, originals=X)
    snap = {r["name"]: r for r in tr.snapshot()}
    for r in snap.values():
        jsonschema.validate(r, schema)
    assert snap["search.latency_ms.p50"]["value"] > 0
    assert snap["search.latency_ms.p50"]["unit"] == "ms"
    assert snap["search.qps"]["n"] == 12
    assert snap["search.rerank_agreement"]["reference"]
    with pytest.raises(ValueError, match="must say why"):
        M.reading("search.qps", None)


def test_sampling_is_bounded_marked_and_seeded(corpus):
    X, index = corpus
    tr = telemetry.enable(rate=0.5, seed=3, capacity=5)
    for _ in range(40):
        index.search(X[:1], k=3)
    docs = tr.traces()
    assert 0 < len(docs) <= 5 and all(d["sampled"] for d in docs)
    assert {r["kind"] for r in tr.snapshot() if r["name"].startswith("search.lat")} == {
        "sampled"
    }
    with pytest.raises(ValueError):
        telemetry.enable(rate=0.0)


def test_capture_is_opt_in_and_a_broken_listener_cannot_break_a_search(corpus):
    X, index = corpus
    tr = telemetry.enable(capture_vectors=True)
    tr.subscribe(lambda d: 1 / 0)
    got = []
    tr.subscribe(got.append)
    index.search(X[:2], k=3)
    assert got and len(got[0]["input"]["vectors"]) == 2
    assert tr.get(got[0]["id"]) is got[0]


def test_capabilities_name_every_registered_metric():
    caps = telemetry.capabilities()
    assert caps["metrics"] == sorted(M.REGISTRY)
    assert caps["schemas"]["turboquant-pro/query-trace"] == T.SCHEMA_VERSION
    assert caps["features"]["operator_actions"] is False


def test_scopes_bind_a_tracer_carry_context_and_collect_their_traces(corpus):
    X, index = corpus
    t = telemetry.Tracer()
    assert telemetry.active() is None
    with telemetry.scope(t, row=7) as sc:
        assert telemetry.active() is t
        index.search(X[:1], k=3)
        index.search(X[1:2], k=3)
    assert telemetry.active() is None  # the scope ended
    assert len(sc.captured) == 2 and sc.last is sc.captured[-1]
    assert all(d["params"]["row"] == 7 and d["params"]["k"] == 3 for d in sc.captured)
    assert t.traces() == sc.captured


def test_scopes_nest_force_beats_sampling_and_the_default_is_a_fallback(corpus):
    X, index = corpus
    default = telemetry.enable()
    outer, inner = telemetry.Tracer(), telemetry.Tracer(rate=0.01, seed=0)
    with telemetry.scope(outer):
        with telemetry.scope(inner, force=True) as sc:
            for _ in range(5):
                index.search(X[:1], k=3)
        index.search(X[:1], k=3)
    index.search(X[:1], k=3)
    assert len(sc.captured) == 5 and len(inner.traces()) == 5  # force: none sampled out
    assert len(outer.traces()) == 1 and len(default.traces()) == 1


def test_a_scope_is_invisible_to_other_threads(corpus):
    import threading

    X, index = corpus
    t = telemetry.Tracer()
    seen = {}

    def other():
        seen["active"] = telemetry.active()
        index.search(X[:1], k=3)

    with telemetry.scope(t):
        th = threading.Thread(target=other)
        th.start()
        th.join()
    assert seen["active"] is None and t.traces() == []

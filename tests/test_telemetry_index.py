"""Tracing the persisted indexes (TQEIndex, ShardedIndex) and the console workload
that drives them.

Before this, only ``ADCIndex.search`` was instrumented: a TQE index opened
memory-mapped (as ``tqp console --index`` opens it) took the blocked scan and
recorded nothing, and the console passed ``originals=`` to indexes that take no
such argument, so every reranked query raised.
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro import telemetry
from turboquant_pro.index import TQEIndex
from turboquant_pro.rerank_tier import NpyOriginalStore
from turboquant_pro.schemas import load_schema
from turboquant_pro.sharded_index import ShardedIndex

jsonschema = pytest.importorskip("jsonschema")


def _corpus(n=1200, d=48, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.fixture(autouse=True)
def _off():
    telemetry.disable()
    yield
    telemetry.disable()


@pytest.fixture
def tqe(tmp_path):
    x = _corpus()
    path = str(tmp_path / "c.tqe")
    TQEIndex.create(x, output_dim=24, bits=4).save(path)
    lean = str(tmp_path / "lean.tqe")
    TQEIndex.create(x, output_dim=24, bits=4, keep_originals=False).save(lean)
    return x, path, lean


def _valid(doc):
    jsonschema.validate(doc, load_schema("query_trace.schema.json"))


def _stage(doc, name):
    return next(s for s in doc["stages"] if s["name"] == name)


@pytest.mark.parametrize("mmap", [False, True])
def test_a_tqe_search_is_one_trace_and_tracing_changes_nothing(tqe, mmap):
    x, path, _ = tqe
    idx = TQEIndex.open(path, mmap=mmap)
    q = x[:5]
    plain = idx.search(q, k=10, rerank=4)
    tr = telemetry.enable()
    traced = idx.search(q, k=10, rerank=4)
    np.testing.assert_array_equal(plain[0], traced[0])
    np.testing.assert_array_equal(plain[1], traced[1])
    docs = tr.traces()
    assert [d["component"] for d in docs] == ["TQEIndex.search"]  # no inner ADC one
    d = docs[0]
    _valid(d)
    # Memory-mapped: the blocked float scan. In RAM: the kernel when it is built.
    assert d["scan_path"] in (("numpy",) if mmap else ("kernel", "numpy"))
    assert _stage(d, "scan")["blocked"] is mmap
    assert _stage(d, "rerank")["basis"] == "originals"
    res = d["results"]
    assert [r["id"] for r in res["final"]] == [int(i) for i in traced[0][0]]
    assert 0.0 <= res["rerank_agreement"] <= 1.0


def test_a_rerank_of_reconstructions_claims_no_exact_comparison(tqe):
    x, _, lean = tqe
    idx = TQEIndex.open(lean, mmap=True)
    tr = telemetry.enable()
    ids, _ = idx.search(x[:2], k=5, rerank=4)
    d = tr.traces()[-1]
    _valid(d)
    assert _stage(d, "rerank")["basis"] == "reconstruction"
    assert "final" not in d["results"]
    assert "rerank_agreement" not in d["results"]
    assert [r["id"] for r in d["results"]["approximate"]] == [int(i) for i in ids[0]]


def test_a_sharded_search_is_one_trace_not_one_per_shard(tmp_path):
    x = _corpus(1500)
    ShardedIndex.create(x, str(tmp_path / "s"), shard_size=400, output_dim=24, bits=4)
    sh = ShardedIndex.open(str(tmp_path / "s" / "manifest.json"))
    plain = sh.search(x[:4], k=10)
    tr = telemetry.enable()
    traced = sh.search(x[:4], k=10)
    np.testing.assert_array_equal(plain[0], traced[0])
    docs = tr.traces()
    assert [d["component"] for d in docs] == ["ShardedIndex.search"]
    _valid(docs[0])
    assert _stage(docs[0], "scan")["shards"] == 4
    assert [s["name"] for s in docs[0]["stages"]] == ["scan", "merge"]


def test_a_tiered_ivf_search_compares_the_shortlist_with_the_cold_tier(tmp_path):
    x = _corpus(1500)
    sh = ShardedIndex.create(
        x,
        str(tmp_path / "s"),
        shard_size=500,
        output_dim=24,
        bits=4,
        keep_originals=False,
    ).build_ivf(nlist=16)
    store = NpyOriginalStore.write(str(tmp_path / "orig.npy"), x)
    plain = sh.search(x[:3], k=5, nprobe=8, rerank=4, rerank_store=store)
    tr = telemetry.enable()
    traced = sh.search(x[:3], k=5, nprobe=8, rerank=4, rerank_store=store)
    np.testing.assert_array_equal(plain[0], traced[0])
    d = tr.traces()[-1]
    _valid(d)
    assert [s["name"] for s in d["stages"]] == ["scan", "rerank"]
    assert [r["id"] for r in d["results"]["final"]] == [int(i) for i in traced[0][0]]


def test_the_console_workload_reranks_a_tqe_index_without_raising(tqe):
    from turboquant_pro.console.server import Workload

    x, path, lean = tqe
    for p, basis in ((path, "stored originals"), (lean, "reconstruction")):
        idx = TQEIndex.open(p, mmap=True)
        wl = Workload(idx, x[:8], k=5, rerank=4, originals=x, tracer=telemetry.Tracer())
        doc = wl.run_one(0)
        assert doc is not None and doc["component"] == "TQEIndex.search"
        assert wl.errors == 0
        assert wl.state()["rerank_basis"] == basis
    assert wl.state()["mode"] == "rerank on reconstruction (not exact)"


# ---- the other search paths --------------------------------------------------


def _one_trace(tr, component):
    docs = tr.traces()
    assert [d["component"] for d in docs] == [component]
    _valid(docs[0])
    return docs[0]


@pytest.mark.parametrize("nprobe", [4, None])
def test_an_ivf_search_is_one_trace_with_its_probe_statistics(nprobe):
    from turboquant_pro.ivf import IVFIndex

    x = _corpus(1500)
    idx = IVFIndex.create(x, output_dim=24, bits=4, nlist=16)
    plain = idx.search(x[:3], k=5, nprobe=nprobe, rerank=4)
    tr = telemetry.enable()
    traced = idx.search(x[:3], k=5, nprobe=nprobe, rerank=4)
    np.testing.assert_array_equal(plain[0], traced[0])
    d = _one_trace(tr, "IVFIndex.search")
    assert [s["name"] for s in d["stages"]] == ["encode", "scan", "rerank"]
    scan = _stage(d, "scan")
    assert 1 <= scan["cells_probed"] <= 16 and 0 < scan["scan_fraction"] <= 1
    assert [r["id"] for r in d["results"]["final"]] == [int(i) for i in traced[0][0]]


def test_an_adaptive_search_is_one_trace_that_counts_what_it_read():
    from turboquant_pro import ADCIndex, PCAMatryoshka
    from turboquant_pro import adaptive_rerank as AR

    x = _corpus(1500)
    pca = PCAMatryoshka(input_dim=48, output_dim=24)
    pca.fit(x)
    index = ADCIndex(pca.with_quantizer(bits=4)).add(x)
    policy = AR.calibrate(
        index, _corpus(200, seed=5), x, k=5, target_recall=0.8, max_candidates=400
    )
    q = _corpus(4, seed=9)
    plain, _ = AR.search(index, q, x, policy)
    tr = telemetry.enable()
    traced, report = AR.search(index, q, x, policy)
    np.testing.assert_array_equal(plain, traced)
    d = _one_trace(tr, "adaptive_rerank.search")  # no inner ADCIndex trace
    rr = _stage(d, "rerank")
    assert rr["candidates"] == int(report.rows_read.sum())
    assert rr["queries_reranked"] == int((report.stage == "rerank").sum())


def test_an_hnsw_search_is_one_trace_and_its_rerank_is_not_called_exact():
    from turboquant_pro.hnsw import CompressedHNSW
    from turboquant_pro.pgvector import TurboQuantPGVector

    x = _corpus(200, d=64)
    index = CompressedHNSW(TurboQuantPGVector(dim=64, bits=3, seed=42), M=8)
    for i, v in enumerate(x):
        index.insert(i, v)
    plain = index.search(x[0], k=5)
    tr = telemetry.enable()
    assert index.search(x[0], k=5) == plain
    d = _one_trace(tr, "CompressedHNSW.search")
    assert _stage(d, "rerank")["basis"] == "reconstruction"
    assert "final" not in d["results"]


def test_scatter_gather_is_one_trace_with_per_server_timing(tmp_path):
    from turboquant_pro.distributed import (
        ShardServer,
        partition_manifest,
        scatter_gather,
    )

    x = _corpus(1200)
    ShardedIndex.create(x, str(tmp_path / "s"), shard_size=300, output_dim=24, bits=4)
    subs = partition_manifest(str(tmp_path / "s" / "manifest.json"), 2)
    servers = {p: ShardServer(p) for p in subs}
    tr = telemetry.enable()
    scatter_gather(x[:2], 5, subs, lambda ep, b: servers[ep].handle(b), max_parallel=2)
    d = _one_trace(tr, "scatter_gather")  # the in-process servers add none
    sc = _stage(d, "scatter")["servers"]
    assert len(sc) == 2 and all(s["ms"] >= 0 and s["response_bytes"] > 0 for s in sc)


def test_a_faiss_search_is_one_trace_scored_higher_is_closer():
    pytest.importorskip("faiss")
    from turboquant_pro import PCAMatryoshka
    from turboquant_pro.faiss_index import TurboQuantFAISS

    x = _corpus(500)
    pca = PCAMatryoshka(input_dim=48, output_dim=24)
    pca.fit(x)
    f = TurboQuantFAISS(pca, metric="l2")
    f.add(x)
    tr = telemetry.enable()
    dist, ids = f.search(x[:2], k=5)
    d = _one_trace(tr, "TurboQuantFAISS.search")
    got = d["results"]["approximate"]
    assert [r["id"] for r in got] == [int(i) for i in ids[0]]
    assert [r["score"] for r in got] == pytest.approx([-float(v) for v in dist[0]])

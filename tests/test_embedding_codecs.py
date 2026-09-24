"""Embedding codecs in the planner's registry (issue #172, step 1).

The planner could not plan a retrieval workload because the registry held no
``embedding`` codec. These tests pin what the P0 exit test will rely on:

* the four codecs register for the ``embedding`` target;
* stored bytes per vector equal the RaBitQ campaign's accounting exactly
  (tq: packed codes + one fp32 norm; PQ/OPQ: m; RaBitQ: faiss code_size), with
  shared structures reported separately and never charged per vector;
* a codec lists its configurations and a byte prior, and the planner prunes
  on the prior before compressing anything;
* the retrieval consumer reranks at a declared depth and, given a codec's own
  search, ranks candidates with it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from turboquant_pro import plugins
from turboquant_pro.consumers import TopKRetrievalConsumer
from turboquant_pro.embedding_codecs import (
    EmbeddingContainer,
    TQEmbeddingCodec,
    pq_m_options,
    tq_bytes_per_vector,
)
from turboquant_pro.planner import (
    ABSTAIN,
    Artifact,
    Budget,
    CompressionPlanner,
    QualityFloor,
    WorkloadSpec,
    container_bytes,
)

NAMES = ("tq_embedding", "faiss_pq", "faiss_opq", "faiss_rabitq")


def _unit(n, d, seed=0):
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_the_four_codecs_register_for_embeddings():
    reg = plugins.available_plugins(target="embedding")
    for name in NAMES:
        assert name in reg


# ------------------------------------------------------------------ #
# turboquant-pro                                                      #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("out_dim", [None, 16])
def test_tq_bytes_are_the_campaigns(bits, out_dim):
    d, n = 64, 300
    codec = TQEmbeddingCodec(bits=bits, out_dim=out_dim, seed=1)
    c = codec.compress(_unit(n, d))
    acct = container_bytes(c)
    assert acct["total_bytes"] / n == tq_bytes_per_vector(out_dim or d, bits)
    assert (
        tq_bytes_per_vector(out_dim or d, bits)
        == math.ceil((out_dim or d) * bits / 8) + 4
    )
    assert "shared_bytes" in acct  # the pipeline is reported, not charged


def test_tq_roundtrip_shape_and_quality():
    x = _unit(400, 64)
    codec = TQEmbeddingCodec(bits=4)
    rec = codec.decompress(codec.compress(x))
    assert rec.shape == x.shape
    cos = np.sum(rec * x, 1) / np.linalg.norm(rec, axis=1)
    assert np.median(cos) > 0.95


def test_tq_lists_pca_configs_only_on_wide_arms():
    narrow = TQEmbeddingCodec(dim=100).capabilities()["configs"]
    assert {c["out_dim"] for c in narrow} == {100}
    wide = TQEmbeddingCodec(dim=1024).capabilities()["configs"]
    assert {c["out_dim"] for c in wide} == {1024, 256, 512}
    assert len(wide) == 7  # the campaign: 3 at full dimension, 2 x (3, 4) reduced


def test_a_pinned_design_space_overrides_the_declared_one():
    x = _unit(700, 32)
    pinned = {"tq_embedding": [{"bits": 3, "out_dim": 16}]}
    plan = CompressionPlanner(_spec(candidate_configs=pinned)).plan(
        Artifact(x), queries=_unit(60, 32, 9)
    )
    doc = plan.as_dict()
    assert [c["config"] for c in doc["candidate_results"]] == pinned["tq_embedding"]
    assert doc["workload"]["candidate_configs"] == pinned


def test_shared_is_excluded_from_every_container_shape():
    per = np.zeros((10, 3), np.uint8)
    big = np.zeros(10_000, np.float32)
    for c in (
        EmbeddingContainer(codes=per, shared={"t": big}),
        {"codes": per, "shared": {"t": big}},
    ):
        acct = container_bytes(c)
        assert acct["total_bytes"] == 30
        assert acct["shared_bytes"] == big.nbytes


# ------------------------------------------------------------------ #
# The planner                                                          #
# ------------------------------------------------------------------ #


def _spec(**over):
    base = dict(
        target="embedding",
        consumer="topk_inner_product",
        consumer_config={"k": 10, "n_queries": 60, "rerank": 5},
        floor=QualityFloor(minimum=0.5, confidence=0.95),
        candidates=("tq_embedding",),
        seed=0,
        n_boot=64,
    )
    base.update(over)
    return WorkloadSpec(**base)


def test_planner_enumerates_the_declared_configs():
    x = _unit(700, 32)
    plan = CompressionPlanner(_spec()).plan(Artifact(x), queries=_unit(60, 32, 9))
    configs = [c["config"] for c in plan.as_dict()["candidate_results"]]
    assert sorted(c["bits"] for c in configs) == [2, 3, 4]
    assert all(c["out_dim"] == 32 for c in configs)
    assert plan.selected_codec == "tq_embedding"


def test_a_byte_budget_prunes_on_the_prior_before_compressing():
    x = _unit(700, 32)
    budget = Budget(max_bytes_per_vector=tq_bytes_per_vector(32, 3))
    plan = CompressionPlanner(_spec(budget=budget)).plan(
        Artifact(x), queries=_unit(60, 32, 9)
    )
    by_bits = {c["config"]["bits"]: c for c in plan.as_dict()["candidate_results"]}
    assert by_bits[4]["verdict"] == "infeasible" and by_bits[4]["left_at"] == "prior"
    assert by_bits[4]["quality"] is None  # never compressed
    assert plan.selected_parameters["bits"] in (2, 3)


# ------------------------------------------------------------------ #
# The consumer                                                         #
# ------------------------------------------------------------------ #


def test_rerank_never_loses_recall_and_is_exact_on_a_perfect_codec():
    x = _unit(500, 24)
    q = _unit(40, 24, 3)
    noisy = x + 0.3 * np.random.default_rng(1).standard_normal(x.shape).astype(
        np.float32
    )
    r1 = TopKRetrievalConsumer(k=10, rerank=1).per_item(x, noisy, queries=q)
    r5 = TopKRetrievalConsumer(k=10, rerank=5).per_item(x, noisy, queries=q)
    assert r5.mean() >= r1.mean()
    assert (TopKRetrievalConsumer(k=10, rerank=5).per_item(x, x, queries=q) == 1).all()
    assert TopKRetrievalConsumer(rerank=5).name.endswith("+rerank5")


def test_a_codec_search_replaces_decoded_ranking():
    x = _unit(300, 16)
    q = _unit(20, 16, 4)
    exact = np.argsort(-(q @ x.T), axis=1)
    calls = []

    def perfect(queries, n):
        calls.append(n)
        return exact[:, :n]

    garbage = np.zeros_like(x)  # decoding would score nothing
    hits = TopKRetrievalConsumer(k=10, rerank=5).per_item(
        x, garbage, queries=q, searcher=perfect
    )
    assert calls == [50] and (hits == 1).all()


# ------------------------------------------------------------------ #
# faiss (optional)                                                     #
# ------------------------------------------------------------------ #


def test_faiss_bytes_and_search():
    pytest.importorskip("faiss")
    from turboquant_pro.embedding_codecs import FaissCodec

    x = _unit(3000, 64)
    for kind, kw in (("pq", {"m": 16}), ("opq", {"m": 16})):
        c = FaissCodec(kind, train_rows=3000, **kw).compress(x)
        assert container_bytes(c)["total_bytes"] / len(x) == 16
    rq = FaissCodec("rabitq", bits=4, train_rows=3000)
    c = rq.compress(x)
    per = container_bytes(c)["total_bytes"] / len(x)
    assert per == c.shared["index"].code_size  # codes + correction factors
    assert rq.decompress(c).shape == x.shape
    ids = rq.search(c, _unit(5, 64, 2), 50)
    assert ids.shape == (5, 50) and (ids >= 0).all()


def test_faiss_pq_configs_divide_the_dimension():
    assert pq_m_options(96) == [12, 24, 48]
    assert all(1024 % m == 0 for m in pq_m_options(1024))


def test_planner_runs_every_family_when_faiss_is_present():
    pytest.importorskip("faiss")
    x = _unit(1500, 64)
    spec = _spec(
        candidates=NAMES, consumer_config={"k": 10, "n_queries": 60, "rerank": 5}
    )
    plan = CompressionPlanner(spec).plan(Artifact(x), queries=_unit(60, 64, 9))
    doc = plan.as_dict()
    families = {
        c["codec"] for c in doc["candidate_results"] if c["verdict"] != "unsupported"
    }
    assert families == set(NAMES), [
        (c["codec"], c["reason"]) for c in doc["candidate_results"]
    ]
    assert plan.selected_codec != ABSTAIN

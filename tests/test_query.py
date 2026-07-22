"""`tqp query` — the SQL-ish dialect: parser contract + the full
ANALYZE -> EXPLAIN -> SELECT lifecycle on a real (tiny) TQE index."""

from __future__ import annotations

import json

import numpy as np
import pytest

from turboquant_pro.query import (
    AnalyzeStmt,
    QuerySyntaxError,
    SelectStmt,
    analyze,
    catalog_path,
    execute,
    explain,
    parse,
)

# --------------------------------------------------------------------------- #
# Parser                                                                       #
# --------------------------------------------------------------------------- #


def test_parse_analyze_with_queries_and_options():
    s = parse("ANALYZE INDEX 'x.tqe' USING QUERIES 'q.npy' WITH (SAMPLE=500, K=10)")
    assert isinstance(s, AnalyzeStmt)
    assert s.index == "x.tqe" and s.queries == "q.npy"
    assert s.options == {"SAMPLE": 500, "K": 10}


def test_parse_select_full_form():
    s = parse(
        "SELECT id, score FROM 'x.tqe' ORDER BY COSINE(:q) LIMIT 5 "
        "WITH (RECALL >= 0.95, CERTIFY)"
    )
    assert isinstance(s, SelectStmt) and not s.explain
    assert s.columns == ["id", "score"]
    assert s.simfn == "COSINE" and s.limit == 5
    assert s.options["RECALL"] == 0.95 and s.options["CERTIFY"] is True


def test_parse_explain_wraps_select():
    s = parse("EXPLAIN SELECT id FROM x.tqe LIMIT 3")
    assert isinstance(s, SelectStmt) and s.explain
    assert s.columns == ["id"] and s.limit == 3


def test_parse_keywords_case_insensitive():
    s = parse("select id from 'x.tqe' order by l2(:q) limit 7")
    assert s.simfn == "L2" and s.limit == 7


@pytest.mark.parametrize(
    "bad",
    [
        "SELECT * FROM x.tqe",  # only id, score are selectable
        "SELECT id FROM x.tqe WHERE a = 1",  # WHERE unsupported, clear error
        "SELECT id FROM x.tqe ORDER BY COSINE(q)",  # placeholder must be :q
        "SELECT id FROM x.tqe LIMIT 2.5",  # integer limit
        "DELETE FROM x.tqe",  # unknown verb
        "SELECT id FROM x.tqe LIMIT 3 garbage",  # trailing input
    ],
)
def test_parse_rejects(bad):
    with pytest.raises(QuerySyntaxError):
        parse(bad)


# --------------------------------------------------------------------------- #
# Lifecycle on a real index                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def small_index(tmp_path_factory):
    from turboquant_pro.index import TQEIndex

    rng = np.random.default_rng(0)
    # clustered so recall/geometry are non-trivial
    centres = rng.standard_normal((8, 48)).astype(np.float32)
    x = np.repeat(centres, 75, axis=0) + 0.25 * rng.standard_normal((600, 48)).astype(
        np.float32
    )
    path = str(tmp_path_factory.mktemp("qidx") / "small.tqe")
    TQEIndex.create(x, bits=4, keep_originals=True).save(path)
    q = x[rng.choice(600, size=20, replace=False)] + 0.05 * rng.standard_normal(
        (20, 48)
    ).astype(np.float32)
    return path, x, q


def test_analyze_writes_catalog_with_calibration(small_index):
    path, _, _ = small_index
    doc = analyze(AnalyzeStmt(index=path, options={"SAMPLE": 600}))
    assert doc["schema"] == "turboquant-pro/query-catalog"
    g = doc["geometry"]
    assert g["n"] == 600 and g["dim"] == 48 and g["advisory"] is True
    assert g["eff_rank"] > 1
    calib = doc["calibration"]
    assert calib is not None and calib["k"] == 10
    recalls = [p["recall"] for p in sorted(calib["sweep"], key=lambda p: p["rerank"])]
    assert recalls[-1] >= recalls[0]  # rerank never hurts measured recall
    on_disk = json.load(open(catalog_path(path), encoding="utf-8"))
    assert on_disk["schema_version"] == 1


def test_explain_picks_cheapest_point_meeting_target(small_index):
    path, _, _ = small_index
    catalog = json.load(open(catalog_path(path), encoding="utf-8"))
    stmt = parse(f"EXPLAIN SELECT id FROM '{path}' LIMIT 10 WITH (RECALL >= 0.9)")
    doc = explain(stmt, catalog)
    plan = doc["plan"]
    assert plan["predicted"]["recall"] >= 0.9 or plan.get("target_unreachable")
    # cheapest: no smaller measured rerank also meets the target
    sweep = sorted(catalog["calibration"]["sweep"], key=lambda p: p["rerank"])
    smaller = [p for p in sweep if p["rerank"] < plan["rerank"]]
    assert all(p["recall"] < 0.9 for p in smaller)
    assert "calibration" in plan["chosen_by"]
    assert doc["honesty"]  # the WP5-lite basis line is present


def test_select_executes_plan_and_shapes_columns(small_index):
    path, x, q = small_index
    catalog = json.load(open(catalog_path(path), encoding="utf-8"))
    stmt = parse(
        f"SELECT id, score FROM '{path}' ORDER BY COSINE(:q) LIMIT 5 "
        "WITH (RECALL >= 0.9)"
    )
    doc = execute(stmt, q, catalog)
    assert doc["n_queries"] == 20 and len(doc["results"]) == 20
    first = doc["results"][0]
    assert len(first["ids"]) == 5 and len(first["scores"]) == 5
    assert doc["plan"]["rerank"] >= 0 and "calibration" in doc["plan"]["chosen_by"]


def test_select_recall_without_catalog_raises(small_index, tmp_path):
    path, x, q = small_index
    stmt = parse(f"SELECT id FROM '{path}' LIMIT 5 WITH (RECALL >= 0.9)")
    with pytest.raises(RuntimeError, match="ANALYZE"):
        execute(stmt, q, catalog=None)


def test_select_certify_attaches_certificate(small_index):
    path, _, q = small_index
    stmt = parse(f"SELECT id FROM '{path}' LIMIT 5 WITH (CERTIFY)")
    doc = execute(stmt, q, catalog=None)
    assert "certificate" in doc and "tau_floor" in doc["certificate"]

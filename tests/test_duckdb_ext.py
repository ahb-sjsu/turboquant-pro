"""Tests for the DuckDB-over-compressed-TQE surface (2.0 Pillar 3)."""

import numpy as np
import pytest

duckdb = pytest.importorskip("duckdb")
pa = pytest.importorskip("pyarrow")

from turboquant_pro import duckdb_ext as tq  # noqa: E402
from turboquant_pro.index import TQEIndex  # noqa: E402


@pytest.fixture(scope="module")
def index_path(tmp_path_factory):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((512, 48)).astype(np.float32)
    idx = TQEIndex.create(x, bits=3, keep_originals=True)
    idx.delete(idx._ids[:8])  # tombstone a few rows
    p = tmp_path_factory.mktemp("tqe") / "corpus.tqe"
    idx.save(str(p))
    return str(p), x


def test_scan_streams_batches_and_skips_tombstones(index_path):
    path, x = index_path
    con = duckdb.connect()
    tq.attach(con, "corpus", path, batch_rows=100)  # forces >1 batch
    n, dim = con.sql("SELECT count(*), max(len(vector)) FROM corpus").fetchone()
    assert n == 512 - 8
    assert dim == 48


def test_scan_vectors_match_originals(index_path):
    path, x = index_path
    con = duckdb.connect()
    tq.attach(con, "corpus", path, batch_rows=64)
    rows = con.sql("SELECT id, vector FROM corpus ORDER BY id LIMIT 16").fetchall()
    for rid, vec in rows:
        np.testing.assert_allclose(
            np.array(vec, dtype=np.float32), x[int(rid)], rtol=1e-5, atol=1e-6
        )


def test_scan_without_originals_reconstructs(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((200, 32)).astype(np.float32)
    idx = TQEIndex.create(x, bits=4, keep_originals=False)
    p = tmp_path / "noorig.tqe"
    idx.save(str(p))
    con = duckdb.connect()
    tq.attach(con, "corpus", str(p), batch_rows=64)
    rows = con.sql("SELECT id, vector FROM corpus ORDER BY id LIMIT 20").fetchall()
    assert len(rows) == 20
    for rid, vec in rows:
        v = np.array(vec, dtype=np.float32)
        o = x[int(rid)]
        cos = float(v @ o / (np.linalg.norm(v) * np.linalg.norm(o)))
        assert cos > 0.9  # lossy reconstruction, but the right vector


def test_search_matches_direct_index_search(index_path):
    path, x = index_path
    con = duckdb.connect()
    q = x[100:103] + 0.01
    rel = tq.search(con, path, q, k=5, view="hits")
    got = con.sql(
        "SELECT query, rank, id, score FROM hits ORDER BY query, rank"
    ).fetchall()
    idx = TQEIndex.open(path, mmap=True)
    ids, scores = idx.search(q.astype(np.float32), k=5)
    want = [
        (qi, r, int(ids[qi, r]), pytest.approx(float(scores[qi, r]), rel=1e-5))
        for qi in range(3)
        for r in range(5)
        if ids[qi, r] >= 0
    ]
    assert got == want
    assert rel.columns == ["query", "rank", "id", "score"]


def test_search_joins_with_sql_metadata(index_path):
    path, x = index_path
    con = duckdb.connect()
    con.sql(
        "CREATE TABLE meta AS SELECT range AS id, 'row_' || range AS title "
        "FROM range(512)"
    )
    # Row 42 is live (the fixture tombstones ids 0-7); exact rerank makes the
    # query's own row the deterministic top hit.
    tq.search(con, path, x[42], k=3, rerank=5, view="hits")
    rows = con.sql(
        "SELECT h.rank, m.title FROM hits h JOIN meta m USING (id) ORDER BY h.rank"
    ).fetchall()
    assert len(rows) == 3
    assert rows[0][1] == "row_42"  # the query vector's own row ranks first

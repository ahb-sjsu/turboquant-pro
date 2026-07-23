"""Track A (2.0) tests: COPY ingestion + the in-database calibration catalog.

Runs against a fake psycopg2 connection that records SQL and COPY payloads
and emulates the tiny slice of Postgres behavior these paths touch (a live-PG
lane is a compatibility-matrix item, added when its CI exists). What is
actually asserted: the COPY payload decodes back to the exact bytea the
search path reads; catalog rows carry the full recall-contract key; planning
honours lower confidence bounds, misses on unknown fingerprints, and names
the best achievable bound when a target is infeasible.
"""

import numpy as np
import pytest

from turboquant_pro.pgvector import CompressedEmbedding, TurboQuantPGVector


class FakeCursor:
    def __init__(self, db):
        self.db = db

    def execute(self, sql, params=None):
        self.db.statements.append((sql.strip(), params))
        s = sql.strip().upper()
        if s.startswith("INSERT INTO") and params is not None:
            self.db.rows.append(tuple(params))
        elif s.startswith("SELECT OPERATING_POINT"):
            fp, k, metric = params
            self.db.last_select = [
                (r[7], r[8], r[9], r[11])
                for r in self.db.rows
                if r[0] == fp and r[6] == k and r[2] == metric
            ]

    def copy_expert(self, sql, buf):
        self.db.copies.append((sql, buf.read()))

    def fetchall(self):
        return self.db.last_select

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self):
        self.statements = []
        self.copies = []
        self.rows = []
        self.last_select = []
        self.commits = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1


@pytest.fixture()
def q():
    return TurboQuantPGVector(dim=32, bits=3, seed=7)


def test_copy_payload_roundtrips_to_search_bytea(q):
    conn = FakeConn()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 32)).astype(np.float32)
    n = q.insert_compressed_copy(conn, "emb", ids=[10, 11, 12, 13, 14], embeddings=x)
    assert n == 5 and conn.commits == 1
    sql, payload = conn.copies[0]
    assert "COPY emb" in sql and "FROM STDIN" in sql
    lines = payload.strip().split("\n")
    assert len(lines) == 5
    rid, hexa, dim, bits = lines[0].split("\t")
    assert (rid, dim, bits) == ("10", "32", "3")
    assert hexa.startswith("\\\\x")
    # The COPY bytea must equal what insert_compressed would store — i.e.
    # exactly what search_compressed reads back.
    blob = bytes.fromhex(hexa[3:])
    want = q.compress_embedding(x[0]).to_pgbytea()
    assert blob == want
    ce = CompressedEmbedding.from_pgbytea(blob, 32, 3)
    rec = q.decompress_embedding(ce)
    cos = float(rec @ x[0] / (np.linalg.norm(rec) * np.linalg.norm(x[0])))
    assert cos > 0.9


def test_copy_chunking_commits_per_chunk(q):
    conn = FakeConn()
    x = np.random.default_rng(1).standard_normal((7, 32)).astype(np.float32)
    q.insert_compressed_copy(conn, "emb", ids=list(range(7)), embeddings=x, chunk=3)
    assert len(conn.copies) == 3  # 3 + 3 + 1
    assert conn.commits == 3


def _seed_catalog(q, conn, fp, sweep=None):
    q.create_calibration_table(conn, "emb")
    sweep = sweep or [
        {
            "operating_point": {"oversample": 1},
            "recall": 0.83,
            "ci_low": 0.80,
            "latency_ms": 1.0,
        },
        {
            "operating_point": {"oversample": 5},
            "recall": 0.97,
            "ci_low": 0.95,
            "latency_ms": 3.0,
        },
        {
            "operating_point": {"oversample": 20},
            "recall": 0.999,
            "ci_low": 0.99,
            "latency_ms": 9.0,
        },
    ]
    n = q.store_calibration(
        conn,
        "emb",
        index_fingerprint=fp,
        k=10,
        sweep=sweep,
        query_population="heldout-1k",
        ground_truth="exact cosine vs originals",
        sample_size=1000,
    )
    return n


def test_catalog_rows_carry_the_contract_key(q):
    conn = FakeConn()
    fp = q.quantizer_fingerprint("corpus-v1")
    assert _seed_catalog(q, conn, fp) == 3
    ddl = conn.statements[0][0]
    for col in (
        "index_fingerprint",
        "query_population",
        "ground_truth",
        "op_family",
        "software_version",
        "recall_ci_low",
    ):
        assert col in ddl
    row = conn.rows[0]
    assert row[0] == fp
    assert row[3] == "exact cosine vs originals"


def test_planner_uses_ci_low_and_picks_cheapest_feasible(q):
    conn = FakeConn()
    fp = q.quantizer_fingerprint("corpus-v1")
    _seed_catalog(q, conn, fp)
    plan = q.plan_operating_point(
        conn, "emb", index_fingerprint=fp, k=10, min_recall=0.95
    )
    # oversample=5 has ci_low exactly 0.95 — feasible and cheaper than 20.
    assert plan["operating_point"] == {"oversample": 5}
    # Point estimate 0.97 would also pass 0.96, but the BOUND (0.95) must not.
    plan99 = q.plan_operating_point(
        conn, "emb", index_fingerprint=fp, k=10, min_recall=0.96
    )
    assert plan99["operating_point"] == {"oversample": 20}


def test_planner_misses_on_unknown_fingerprint_and_infeasible_target(q):
    conn = FakeConn()
    fp = q.quantizer_fingerprint("corpus-v1")
    _seed_catalog(q, conn, fp)
    with pytest.raises(LookupError, match="no calibration"):
        q.plan_operating_point(
            conn, "emb", index_fingerprint="stale" + fp[5:], k=10, min_recall=0.9
        )
    with pytest.raises(ValueError, match="best achievable bound is 0.9900"):
        q.plan_operating_point(
            conn, "emb", index_fingerprint=fp, k=10, min_recall=0.995
        )


def test_fingerprint_changes_with_config_and_corpus(q):
    a = q.quantizer_fingerprint("corpus-v1")
    assert a == q.quantizer_fingerprint("corpus-v1")
    assert a != q.quantizer_fingerprint("corpus-v2")
    q2 = TurboQuantPGVector(dim=32, bits=2, seed=7)
    assert a != q2.quantizer_fingerprint("corpus-v1")


def test_planned_search_returns_plan_with_results(q, monkeypatch):
    conn = FakeConn()
    fp = q.quantizer_fingerprint("corpus-v1")
    _seed_catalog(q, conn, fp)
    calls = {}

    def fake_search(conn_, table, query, top_k):
        calls["top_k"] = top_k
        return [(i, 1.0 - i * 0.01) for i in range(top_k)]

    monkeypatch.setattr(q, "search_compressed", fake_search)
    results, plan = q.search_compressed_planned(
        conn,
        "emb",
        np.zeros(32, dtype=np.float32),
        k=10,
        min_recall=0.95,
        index_fingerprint=fp,
    )
    assert plan["operating_point"] == {"oversample": 5}
    assert calls["top_k"] == 50  # k * oversample widened the candidate pool
    assert len(results) == 10

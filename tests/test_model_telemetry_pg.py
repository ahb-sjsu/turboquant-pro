"""model_telemetry_pg against a real PostgreSQL. Runs only when ``TQP_TEST_PG_DSN``
names a scratch database that may hold the ``vector`` and ``tqvector`` extensions; the
test creates and drops its own tables."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

DSN = os.environ.get("TQP_TEST_PG_DSN")
pytestmark = pytest.mark.skipif(
    not DSN, reason="set TQP_TEST_PG_DSN to a scratch database"
)

psycopg2 = pytest.importorskip("psycopg2")

from turboquant_pro.model_telemetry_pg import PgReadSink  # noqa: E402
from turboquant_pro.telemetry.metrics import reading  # noqa: E402


def _readings(rng, window, d=16, heads=3, status=None):
    out = []
    for h in range(heads):
        prof = rng.dirichlet(np.ones(d))
        lab = {"model": "tiny", "window": window, "layer": 0, "kv_head": h,
               "consumer": "softmax"}  # fmt: skip
        r = reading("model.read.trace", 1.0 + h, as_of=time.time())
        r.update(labels=lab, read_profile=list(prof))
        out.append(r)
        r = reading("model.plan.overlap", 0.9, as_of=time.time())
        r.update(labels=lab, status=status or "VALID", status_reason="test")
        out.append(r)
    return out


@pytest.fixture
def conn():
    c = psycopg2.connect(DSN)
    yield c
    c.rollback()
    with c.cursor() as cur:
        for t in ("t_vec", "t_tq"):
            cur.execute(f"DROP VIEW IF EXISTS {t}_latest, {t}_status")
            cur.execute(f"DROP TABLE IF EXISTS {t}")
    c.commit()
    c.close()


@pytest.mark.parametrize("vtype,table", [("vector", "t_vec"), ("tqvector", "t_tq")])
def test_write_search_and_status(conn, vtype, table):
    rng = np.random.default_rng(0)
    sink = PgReadSink(conn, table=table, vector_type=vtype, run="r1")
    sink.ensure_schema()
    sink.ensure_schema()  # idempotent
    first = _readings(rng, 0)
    later = _readings(rng, 1, status="STALE")
    assert sink.write(first) == len(first)
    sink.write(later)
    probe = first[2]["read_profile"]  # head 1's profile in window 0
    hits = sink.nearest(probe, k=3, model="tiny")
    assert (hits[0]["window_index"], hits[0]["kv_head"]) == (0, 1)
    if vtype == "vector":
        assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-6)
    assert [h["distance"] for h in hits] == sorted(h["distance"] for h in hits)
    st = sink.status("tiny")
    assert [s["kv_head"] for s in st] == [0, 1, 2]
    assert all(s["status"] == "STALE" and s["window_index"] == 1 for s in st)
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table}_latest")
        assert cur.fetchone()[0] == 6  # 3 heads x 2 metrics


def test_identifiers_are_checked():
    with pytest.raises(ValueError):
        PgReadSink(object(), table="x; DROP TABLE y")
    with pytest.raises(ValueError):
        PgReadSink(object(), vector_type="float8[]")

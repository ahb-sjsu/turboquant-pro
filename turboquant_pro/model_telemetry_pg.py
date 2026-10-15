# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""A PostgreSQL store for model telemetry, queryable in SQL
(``docs/DESIGN_inference_telemetry.md``, section 3).

One row per reading (:meth:`turboquant_pro.model_telemetry.ModelMonitor.readings`): the
metric, its value, unit and kind, its labels (model, run, window, layer, key head,
consumer), and a plan's status where there was one. The ``model.read.trace`` rows also
carry the head's **read profile**, the diagonal of its read operator over its trace (how
strongly each native key channel is read), as a vector column. That column is there for
search, not space: :meth:`PgReadSink.nearest` finds the stored windows whose consumers
look most like a live one, so a calibrated plan can be found by its consumer.

Retrieval may use any distance; acceptance never does. A plan found this way is accepted
only by :class:`turboquant_pro.model_telemetry.ReadPlan`'s own comparison (overlap,
``kappa``, regret), as everywhere else in the library.

The vector column is pgvector's ``vector`` (exact) or this repository's ``tqvector``
(``pgext/``, compressed with ``tq_compress``). Both are searched by cosine distance
(``<=>``), which every ``tqvector`` build has (the L2 functions came later) and which
suits a profile that is non-negative and normalized by its trace. Requires ``psycopg2``
(the ``pgvector`` extra).

Views: ``<table>_latest`` (the newest reading per metric and head) and
``<table>_status`` (the newest plan status per head).
"""

from __future__ import annotations

import datetime as _dt
import re

__all__ = ["PgReadSink"]

_IDENT = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
_LABELS = ("model", "window", "layer", "kv_head", "consumer")


class PgReadSink:
    """Write readings to PostgreSQL and search them by read profile.

    ``conn`` is an open psycopg2 connection (or pass ``dsn``). ``vector_type`` is
    ``"vector"`` or ``"tqvector"``; ``tq_bits`` the width ``tq_compress`` codes at."""

    def __init__(
        self,
        conn=None,
        *,
        dsn: str | None = None,
        table: str = "tqp_model_reads",
        vector_type: str = "vector",
        tq_bits: int = 4,
        run: str | None = None,
    ):
        if not _IDENT.match(table):
            raise ValueError(
                f"table name {table!r} is not a plain lower-case identifier"
            )
        if vector_type not in ("vector", "tqvector"):
            raise ValueError("vector_type must be 'vector' or 'tqvector'")
        if conn is None:
            if dsn is None:
                raise ValueError("pass conn or dsn")
            import psycopg2

            conn = psycopg2.connect(dsn)
        self.conn = conn
        self.table = table
        self.vector_type = vector_type
        self.tq_bits = int(tq_bits)
        self.run = run

    # -- schema ------------------------------------------------------------- #
    def ensure_schema(self) -> None:
        t, vt = self.table, self.vector_type
        ext = "vector" if vt == "vector" else "tqvector"
        with self.conn.cursor() as cur:
            cur.execute(f"CREATE EXTENSION IF NOT EXISTS {ext}")
            cur.execute(f"""CREATE TABLE IF NOT EXISTS {t} (
                    id bigserial PRIMARY KEY,
                    as_of timestamptz NOT NULL,
                    run text,
                    model text,
                    window_index integer,
                    layer integer,
                    kv_head integer,
                    consumer text,
                    metric text NOT NULL,
                    value double precision,
                    unit text NOT NULL,
                    kind text NOT NULL,
                    status text,
                    status_reason text,
                    unavailable_reason text,
                    read_profile {vt}
                )""")
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {t}_head_idx "
                f"ON {t} (model, layer, kv_head, metric, as_of)"
            )
            cur.execute(f"""CREATE OR REPLACE VIEW {t}_latest AS
                    SELECT DISTINCT ON (model, layer, kv_head, metric) *
                    FROM {t}
                    ORDER BY model, layer, kv_head, metric, as_of DESC, id DESC""")
            cur.execute(f"""CREATE OR REPLACE VIEW {t}_status AS
                    SELECT DISTINCT ON (model, layer, kv_head)
                           model, run, window_index, layer, kv_head, status,
                           status_reason, as_of
                    FROM {t} WHERE status IS NOT NULL
                    ORDER BY model, layer, kv_head, as_of DESC, id DESC""")
        self.conn.commit()

    # -- write -------------------------------------------------------------- #
    def _vec_sql(self) -> str:
        if self.vector_type == "vector":
            return "%s::vector"
        return f"tq_compress(%s::float4[], {self.tq_bits})"

    def _vec_param(self, profile):
        vals = [float(v) for v in profile]
        if self.vector_type == "vector":
            return "[" + ",".join(repr(v) for v in vals) + "]"
        return vals

    def write(self, readings) -> int:
        """Insert readings; returns the number of rows written."""
        cols = (
            "as_of, run, model, window_index, layer, kv_head, consumer, metric, value, "
            "unit, kind, status, status_reason, unavailable_reason, read_profile"
        )
        n = 0
        with self.conn.cursor() as cur:
            for r in readings:
                lab = r.get("labels", {})
                prof = r.get("read_profile")
                vec_sql = self._vec_sql() if prof is not None else "NULL"
                params = [
                    _dt.datetime.fromtimestamp(r["as_of"], _dt.timezone.utc),
                    self.run,
                    lab.get("model"),
                    lab.get("window"),
                    lab.get("layer"),
                    lab.get("kv_head"),
                    lab.get("consumer"),
                    r["name"],
                    r["value"],
                    r["unit"],
                    r["kind"],
                    r.get("status"),
                    r.get("status_reason"),
                    r.get("unavailable_reason"),
                ]
                if prof is not None:
                    params.append(self._vec_param(prof))
                cur.execute(
                    f"INSERT INTO {self.table} ({cols}) VALUES "
                    f"({', '.join(['%s'] * 14)}, {vec_sql})",
                    params,
                )
                n += 1
        self.conn.commit()
        return n

    # -- search ------------------------------------------------------------- #
    def nearest(self, profile, *, k: int = 5, model=None, layer=None, kv_head=None):
        """The ``k`` stored windows whose read profile is nearest to ``profile``
        (cosine), optionally within one model, layer or key head: dicts of the labels,
        run and distance, nearest first."""
        where = ["metric = 'model.read.trace'", "read_profile IS NOT NULL"]
        params: list = [self._vec_param(profile)]
        for col, val in (("model", model), ("layer", layer), ("kv_head", kv_head)):
            if val is not None:
                where.append(f"{col} = %s")
                params.append(val)
        params.append(int(k))
        sql = (
            f"SELECT run, model, window_index, layer, kv_head, consumer, as_of, "
            f"read_profile <=> {self._vec_sql()} AS distance FROM {self.table} "
            f"WHERE {' AND '.join(where)} ORDER BY distance LIMIT %s"
        )
        # the probe vector is the first placeholder in the SELECT list
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            names = [d[0] for d in cur.description]
            return [dict(zip(names, row)) for row in cur.fetchall()]

    def status(self, model=None) -> list[dict]:
        """The newest plan status per head (the ``_status`` view)."""
        sql = f"SELECT * FROM {self.table}_status"
        params: list = []
        if model is not None:
            sql += " WHERE model = %s"
            params.append(model)
        with self.conn.cursor() as cur:
            cur.execute(sql + " ORDER BY layer, kv_head", params)
            names = [d[0] for d in cur.description]
            return [dict(zip(names, row)) for row in cur.fetchall()]

# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""DuckDB over compressed TQE indexes (2.0 Pillar 3 — ``docs/ROADMAP_2.0.md``).

Standard SQL over a turboquant-pro index **without decompressing the corpus
into host RAM first**: the index stays memory-mapped and compressed on disk;
scans stream **compressed blocks → reconstructed Arrow batches** one batch at
a time (RAM bounded by the batch size, not the corpus), and searches run the
index's own compressed-domain blocked ADC scan and hand DuckDB only the top-k
rows.

    import duckdb
    from turboquant_pro import duckdb_ext as tq

    con = duckdb.connect()
    hits = tq.search(con, "corpus.tqe", query_vec, k=10)     # a DuckDB relation
    con.sql("SELECT h.id, h.score, m.title FROM hits h JOIN meta m USING (id)")

    tq.attach(con, "corpus", "corpus.tqe")     # streaming scan view (one-shot)
    con.sql("SELECT count(*) FROM corpus")

Acceptance coherence: ``search`` exposes ``rerank`` (exact two-stage rescoring)
and returns scores in the index's own metric — pair it with the 1.9.1
``tqp query`` calibration catalog when you need a *declared-recall* operating
point; raw ADC top-k alone is a candidate list, not a certified answer.

Requires the optional extras: ``pip install turboquant-pro[duckdb]``
(``duckdb`` + ``pyarrow``). All imports are lazy so the core package stays
dependency-light.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "scan_reader",
    "search_table",
    "search",
    "attach",
    "attach_strata",
    "hub_census",
    "transit_by_area",
    "strata_gate",
]

_BATCH_ROWS = 8192


def _require_arrow():
    try:
        import pyarrow as pa
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "duckdb_ext needs pyarrow: pip install 'turboquant-pro[duckdb]'"
        ) from e
    return pa


def _open_index(index_path: str, mmap: bool):
    from .index import TQEIndex

    return TQEIndex.open(index_path, mmap=mmap)


# ------------------------------------------------------------------- scan
def scan_reader(
    index_path: str,
    *,
    batch_rows: int = _BATCH_ROWS,
    mmap: bool = True,
    with_vectors: bool = True,
) -> Any:
    """A streaming ``pyarrow.RecordBatchReader`` over the live index rows.

    Schema: ``id: int64`` (+ ``vector: fixed_size_list<float32>[dim]`` when
    ``with_vectors``). Vectors are reconstructed from the compressed codes
    **one batch at a time** — via stored originals when the index kept them,
    else code-level reconstruction — so peak memory is ``O(batch_rows · dim)``
    regardless of corpus size. Tombstoned rows are skipped.

    The reader is one-shot (Arrow streaming semantics): register it, run one
    query. Use :func:`attach` per query, or materialize a small projection.
    """
    pa = _require_arrow()
    idx = _open_index(index_path, mmap)
    ids = np.asarray(idx._ids)
    tomb = np.asarray(idx._tomb).astype(bool)
    live = np.flatnonzero(~tomb)
    originals = idx._originals
    dim = int(idx._pca.input_dim)

    fields = [pa.field("id", pa.int64())]
    if with_vectors:
        fields.append(pa.field("vector", pa.list_(pa.float32(), dim)))
    schema = pa.schema(fields)

    def _batches():
        for s in range(0, len(live), batch_rows):
            pos = live[s : s + batch_rows]
            cols: list[Any] = [pa.array(ids[pos].astype(np.int64))]
            if with_vectors:
                if originals is not None:
                    vecs = np.ascontiguousarray(
                        np.asarray(originals[pos]), dtype=np.float32
                    )
                else:
                    vecs = idx._reconstruct_rows(pos)
                flat = pa.array(vecs.reshape(-1), type=pa.float32())
                cols.append(pa.FixedSizeListArray.from_arrays(flat, dim))
            yield pa.RecordBatch.from_arrays(cols, schema=schema)

    return pa.RecordBatchReader.from_batches(schema, _batches())


# ------------------------------------------------------------------ search
def search_table(
    index_path: str,
    queries: np.ndarray,
    *,
    k: int = 10,
    rerank: int = 0,
    mmap: bool = True,
    block: int | None = None,
) -> Any:
    """Top-``k`` results as a ``pyarrow.Table``.

    Columns: ``query int32, rank int32, id int64, score float32``. The search
    itself is the index's compressed-domain blocked ADC scan (memory-mapped
    when ``mmap``), so the corpus is never decompressed wholesale; with
    ``rerank > 0`` only the candidate rows are reconstructed for exact
    rescoring.
    """
    pa = _require_arrow()
    idx = _open_index(index_path, mmap)
    q = np.asarray(queries, dtype=np.float32)
    if q.ndim == 1:
        q = q[None]
    ids, scores = idx.search(q, k=k, rerank=rerank, block=block)
    nq, kk = ids.shape
    qcol = np.repeat(np.arange(nq, dtype=np.int32), kk)
    rcol = np.tile(np.arange(kk, dtype=np.int32), nq)
    keep = ids.reshape(-1) >= 0  # drop unfilled slots
    return pa.table(
        {
            "query": pa.array(qcol[keep]),
            "rank": pa.array(rcol[keep]),
            "id": pa.array(ids.reshape(-1)[keep]),
            "score": pa.array(scores.reshape(-1)[keep].astype(np.float32)),
        }
    )


def search(
    con: Any,
    index_path: str,
    queries: np.ndarray,
    *,
    k: int = 10,
    rerank: int = 0,
    view: str = "hits",
    mmap: bool = True,
    block: int | None = None,
) -> Any:
    """Run a top-``k`` search and register it with ``con`` as ``view``.

    Returns the DuckDB relation, immediately composable with SQL::

        tq.search(con, "corpus.tqe", qvec, k=10)
        con.sql("SELECT h.id, m.title FROM hits h JOIN meta m USING (id)")
    """
    tbl = search_table(index_path, queries, k=k, rerank=rerank, mmap=mmap, block=block)
    con.register(view, tbl)
    return con.table(view)


def attach(
    con: Any,
    view: str,
    index_path: str,
    *,
    batch_rows: int = _BATCH_ROWS,
    mmap: bool = True,
    with_vectors: bool = True,
) -> Any:
    """Register a streaming scan of the index as DuckDB view ``view``.

    One-shot semantics (the underlying Arrow stream is consumed by the first
    query that scans it) — call :func:`attach` again for the next query, or
    ``CREATE TABLE t AS SELECT ...`` a projection you will reuse.
    """
    reader = scan_reader(
        index_path, batch_rows=batch_rows, mmap=mmap, with_vectors=with_vectors
    )
    con.register(view, reader)
    return con.table(view)


# --------------------------------------------------------------------------
# STRATA relational surface (docs/STRATA_RFC.md): hubness as queryable
# relations that carry their contracts. Once the kNN graph is an edge table,
# N_k is a GROUP BY and τ is a JOIN — the commodity part. The house part:
# every relation carries its area_map_digest, and helpers REFUSE to combine
# relations computed under different maps (the replication predicate enforced
# in the schema, not the docs).
# --------------------------------------------------------------------------


def _digest_guard(con: Any, tables: list[str]) -> str:
    """Refuse, not warn: one distinct area_map_digest across the relations."""
    digests: set[str] = set()
    for t in tables:
        rows = con.sql(f"SELECT DISTINCT area_map_digest FROM {t}").fetchall()
        digests.update(r[0] for r in rows)
    if len(digests) != 1:
        raise ValueError(
            "area_map_mismatch: relations carry different area-map digests "
            f"({sorted(d[:12] for d in digests)}); artifacts computed under "
            "different maps MUST NOT be compared, merged, or gated together"
        )
    return next(iter(digests))


def attach_strata(
    con: Any,
    knn_idx: np.ndarray,
    area_map: Any,
    *,
    query_labels: list[str] | None = None,
    edges_view: str = "knn_edges",
    strata_view: str = "strata",
) -> None:
    """Register ``knn_edges(query_id, neighbor_id, rank)`` + ``strata``.

    Both relations carry ``area_map_digest``. ``knn_idx`` is an (n_q, k)
    neighbour-id array; ``query_labels`` defaults to the corpus labels
    (corpus->corpus battery).
    """
    pa = _require_arrow()
    idx = np.asarray(knn_idx)
    n_q, k = idx.shape
    digest = area_map.digest
    edges = pa.table(
        {
            "query_id": pa.array(
                np.repeat(np.arange(n_q, dtype=np.int64), k), type=pa.int64()
            ),
            "neighbor_id": pa.array(idx.ravel().astype(np.int64), type=pa.int64()),
            "rank": pa.array(
                np.tile(np.arange(k, dtype=np.int64), n_q), type=pa.int64()
            ),
            "area_map_digest": pa.array([digest] * (n_q * k), type=pa.string()),
        }
    )
    labels = list(area_map.labels)
    strata = pa.table(
        {
            "id": pa.array(np.arange(len(labels), dtype=np.int64), type=pa.int64()),
            "area": pa.array(labels, type=pa.string()),
            "area_map_digest": pa.array([digest] * len(labels), type=pa.string()),
        }
    )
    con.register(edges_view, edges)
    con.register(strata_view, strata)
    if query_labels is not None:
        qs = pa.table(
            {
                "query_id": pa.array(
                    np.arange(len(query_labels), dtype=np.int64), type=pa.int64()
                ),
                "area": pa.array(list(query_labels), type=pa.string()),
                "area_map_digest": pa.array(
                    [digest] * len(query_labels), type=pa.string()
                ),
            }
        )
        con.register("query_strata", qs)


def hub_census(con: Any, *, edges: str = "knn_edges") -> Any:
    """N_k is just in-degree: the hub census as a relation."""
    _digest_guard(con, [edges])
    return con.sql(
        f"SELECT neighbor_id, COUNT(*) AS n_k FROM {edges} "
        "GROUP BY neighbor_id ORDER BY n_k DESC"
    )


def transit_by_area(
    con: Any, *, edges: str = "knn_edges", strata: str = "strata"
) -> Any:
    """Mean transit fraction per query area (τ in SQL) — digest-guarded."""
    _digest_guard(con, [edges, strata])
    q_strata = "query_strata" if _has_table(con, "query_strata") else strata
    join_col = "query_id" if q_strata == "query_strata" else "id"
    return con.sql(
        f"SELECT q.area, AVG(CAST(q.area <> n.area AS DOUBLE)) AS transit, "
        f"COUNT(*) AS n_edges FROM {edges} e "
        f"JOIN {q_strata} q ON e.query_id = q.{join_col} "
        f"JOIN {strata} n ON e.neighbor_id = n.id "
        "GROUP BY q.area ORDER BY q.area"
    )


def strata_gate(con: Any, report: dict, *, min_anti_recall: float) -> Any:
    """The gate as a query: rows returned ⇒ the gate fails (exit 1).

    Registers the report's areas as ``strata_report`` (digest-carrying,
    guarded against already-attached strata relations) and returns the
    failing rows. ABSTAIN strata are excluded from gating — ABSTAIN is not
    a pass, and it is not a silent fail either; count them separately.
    """
    pa = _require_arrow()
    digest = report["provenance"]["area_map_digest"]
    areas = report["areas"]
    tbl = pa.table(
        {
            "area_id": pa.array([a["id"] for a in areas], type=pa.string()),
            "verdict": pa.array([a["verdict"] for a in areas], type=pa.string()),
            "anti_hub_recall": pa.array(
                [a.get("anti_hub_recall") for a in areas], type=pa.float64()
            ),
            "area_map_digest": pa.array([digest] * len(areas), type=pa.string()),
        }
    )
    con.register("strata_report", tbl)
    guarded = ["strata_report"]
    if _has_table(con, "strata"):
        guarded.append("strata")
    _digest_guard(con, guarded)
    return con.sql(
        "SELECT area_id, anti_hub_recall FROM strata_report "
        "WHERE verdict <> 'ABSTAIN' AND anti_hub_recall IS NOT NULL "
        f"AND anti_hub_recall < {float(min_anti_recall)!r}"
    )


def _has_table(con: Any, name: str) -> bool:
    try:
        con.table(name)
        return True
    except Exception:
        return False

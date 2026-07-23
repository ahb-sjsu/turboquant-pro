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

__all__ = ["scan_reader", "search_table", "search", "attach"]

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

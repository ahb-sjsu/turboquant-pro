# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Tiered rerank — break the compressed-domain (ADC) recall ceiling (experimental).

IVF + ADC search is sublinear and its shortlist recall against the *exact ADC ranking*
is high, but ADC itself is a lossy score: its ranking against the *true* nearest
neighbours (exact inner product on the original fp32 vectors) has a ceiling. The fix is
a **rerank tier**: keep only compressed codes on the hot tier (the ``--no-originals``
index — ~30 B/row, the 1B/1T substrate), and hold the original vectors on a **cold
tier** (S3, CephFS, a separate PVC). Search runs the cheap IVF/ADC pass to get a *wide*
shortlist of candidate ids, then fetches **only those candidates'** originals from the
cold tier and re-scores them exactly — so the expensive, high-fidelity read is bounded
to ``k · rerank`` rows per query, not the corpus.

This module is deliberately storage-agnostic (mirroring :mod:`distributed`): the rerank
stage takes a ``fetch(ids) -> (len(ids), dim) fp32`` callable, so the cold tier can be
anything. :class:`NpyOriginalStore` is a reference implementation over a memory-mapped
``.npy``; an S3/object-store fetch is a drop-in callable. In a distributed index the
rerank runs **once at the coordinator** over the merged shortlist — the cold tier is hit
once per query batch, not once per shard.

Acceptance stays honest: rerank raises recall against the *true* neighbours; it is not a
reconstruction-cosine metric.
"""

from __future__ import annotations

import numpy as np


class NpyOriginalStore:
    """Cold-tier original vectors as a memory-mapped ``(N, dim)`` ``.npy``, addressed by
    global id. ``fetch(ids)`` faults in only those rows (the shortlist), never the whole
    array. By default row ``i`` is global id ``i`` (contiguous ids from
    :meth:`ShardedIndex.create`); pass ``ids`` to map arbitrary global ids -> rows."""

    def __init__(self, path: str, ids: np.ndarray | None = None):
        self._arr = np.load(path, mmap_mode="r")
        self._row = None if ids is None else {int(x): i for i, x in enumerate(ids)}

    def fetch(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids)
        if self._row is None:
            return np.asarray(self._arr[ids])
        rows = np.fromiter(
            (self._row[int(i)] for i in ids), dtype=np.int64, count=len(ids)
        )
        return np.asarray(self._arr[rows])

    @classmethod
    def write(cls, path: str, originals: np.ndarray) -> NpyOriginalStore:
        """Persist ``originals`` (row == global id) as the cold tier and open it."""
        np.save(path, np.asarray(originals, dtype=np.float32))
        return cls(path)


def _as_fetch(store):
    """Accept a store object (``.fetch``) or a plain ``fetch(ids)`` callable."""
    return store.fetch if hasattr(store, "fetch") else store


def rerank_candidates(
    queries: np.ndarray,
    cand_ids: np.ndarray,
    k: int,
    store,
    *,
    metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray]:
    """Exact-rescore the shortlist against cold-tier originals; return the top-``k``.

    ``cand_ids`` is ``(nq, m)`` global ids (``-1`` = padding) — the wide ADC shortlist.
    ``store`` is a :class:`NpyOriginalStore` (or any ``fetch(ids)`` callable). It uses
    the index's metric: cosine (normalize both, ``orig·q``) or ``l2`` (negative sqdist),
    matching :meth:`TQEIndex.search`'s rerank. The whole batch's *unique* candidate ids
    are fetched in **one** cold-tier read, so overlapping shortlists share the fetch —
    the read is ``O(unique candidates)``, not ``O(nq · m)``."""
    fetch = _as_fetch(store)
    q = np.asarray(queries, dtype=np.float32)
    if q.ndim == 1:
        q = q[None]
    nq = len(q)
    out_ids = np.full((nq, k), -1, dtype=np.int64)
    out_sc = np.full((nq, k), np.nan, dtype=np.float32)
    cand_ids = np.asarray(cand_ids)
    if cand_ids.ndim == 1:
        cand_ids = cand_ids[None]

    valid = cand_ids >= 0
    uids = np.unique(cand_ids[valid])
    if len(uids) == 0:
        return out_ids, out_sc
    orig = np.asarray(fetch(uids)).astype(
        np.float32
    )  # one cold-tier read for the batch
    pos = {int(x): i for i, x in enumerate(uids.tolist())}
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-30)
    on = None
    if metric != "l2":
        on = orig / np.maximum(np.linalg.norm(orig, axis=1, keepdims=True), 1e-30)

    for r in range(nq):
        ids_r = cand_ids[r][valid[r]]
        if len(ids_r) == 0:
            continue
        rows = np.fromiter(
            (pos[int(i)] for i in ids_r.tolist()), dtype=np.int64, count=len(ids_r)
        )
        if metric == "l2":
            exact = -((orig[rows] - q[r]) ** 2).sum(axis=1)
        else:
            exact = on[rows] @ qn[r]
        order = np.argsort(-exact)[:k]
        sel = ids_r[order]
        out_ids[r, : len(sel)] = sel
        out_sc[r, : len(sel)] = exact[order].astype(np.float32)
    return out_ids, out_sc

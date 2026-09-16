"""Dataset access for the RaBitQ public comparison.

Every dataset exposes the same interface: a corpus that can be streamed in normalized
blocks and gathered by id, a normalized query matrix, and exact top-100 ground truth in
corpus positions. Corpora are never fully materialized for the 10M arm: rows are read
from memory-mapped float32 ``.npy`` parts, so peak memory is the index plus one block.

Datasets (the registered arms, see docs/PREREG_rabitq_public.md section 1):

- ann-benchmarks HDF5 (``glove-100-angular``, ``nytimes-256-angular``,
  ``deep-image-96-angular``): corpus = ``train``, queries = the first ``NQ`` rows of
  ``test``, ground truth = the provided ``neighbors``.
- ``wiki1024-10m``: rows 0..9,999,999 of CohereLabs/wikipedia-2023-11-embed-multilingual-v3
  (English config) as ``part_000..009.npy``. ``NQ`` query rows are drawn from those 10M
  with a fixed seed and removed from the corpus; ground truth is computed by ``gt.py``.
- ``dbpedia-ada002-1m`` / ``dbpedia-3large-1536-1m``: the Qdrant DBpedia entity embedding
  sets (990,000 corpus rows, 1,000 held-out rows in ``queries.npy`` written by the download
  script); ground truth by ``gt.py``.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

QUERY_SEED = 20260914
BLOCK = 250_000
SWEEP_BLOCK = 100_000
# Sweep when the wanted rows are denser than one in SWEEP_RATIO of the span they cover.
# Measured on the campaign's CephFS volume (io_probe.py, 2026-09-15): ~100 MiB/s sequential,
# which is 25,600 rows/s at d=1024, against 13-685 rows/s scattered depending on how cold the
# part is. The crossover therefore sits between one row in 40 and one in 2,000; 400 is inside
# that range at the pessimistic end, and it is what makes a 200k-row training sample (one row
# in 50 of its span) a sweep instead of the scattered read that stalled the wiki PQ cells.
SWEEP_RATIO = 400
TRAIN_MAX = 655_360  # 40 x the largest nlist (16,384)


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)


def normalize_inplace(x: np.ndarray, block: int = 500_000) -> np.ndarray:
    """Same values as ``normalize`` (per-row norms do not depend on the chunking), without
    the full-size quotient and x*x temporaries. A 10M x 96 corpus otherwise needs twice its
    size and was OOM-killed at 6 GiB."""
    for s in range(0, len(x), block):
        v = x[s : s + block]
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-30)
    return x


@dataclass(frozen=True)
class Spec:
    name: str
    kind: str  # "hdf5" | "npy"
    path: str  # file (hdf5) or directory (npy), relative to the data root
    nq: int
    parts: int = 0  # npy: number of part files that form the corpus pool
    holdout_from_pool: bool = (
        False  # npy: queries drawn from the pool (else queries.npy)
    )


SPECS = {
    s.name: s
    for s in (
        Spec("glove-100-angular", "hdf5", "ann/glove-100-angular.hdf5", 2000),
        Spec("nytimes-256-angular", "hdf5", "ann/nytimes-256-angular.hdf5", 2000),
        Spec("deep-image-96-angular", "hdf5", "ann/deep-image-96-angular.hdf5", 2000),
        Spec("wiki1024-10m", "npy", "wiki1024", 1000, parts=10, holdout_from_pool=True),
        Spec("dbpedia-ada002-1m", "npy", "dbpedia1536", 1000, parts=1),
        Spec("dbpedia-3large-1536-1m", "npy", "dbpedia3072", 1000, parts=1),
        # smoke-test fixtures only; never part of the registered grid
        Spec("smoke-hdf5", "hdf5", "ann/smoke-32-angular.hdf5", 50),
        Spec("smoke-npy", "npy", "smoke", 50, parts=2, holdout_from_pool=True),
    )
}


class Dataset:
    """Corpus rows in corpus positions 0..n-1, queries and top-100 ground truth."""

    def __init__(self, name: str, root: str):
        self.spec = SPECS[name]
        self.root = root
        sp = self.spec
        if sp.kind == "hdf5":
            import h5py

            with h5py.File(os.path.join(root, sp.path), "r") as f:
                self._mem = normalize_inplace(np.asarray(f["train"][:], np.float32))
                self.queries = normalize(f["test"][: sp.nq])
                self.gt = np.asarray(f["neighbors"][: sp.nq, :100], dtype=np.int64)
            self.n, self.dim = self._mem.shape
            self._keep = None
            return
        d = os.path.join(root, sp.path)
        self._parts = [
            np.load(os.path.join(d, f"part_{i:03d}.npy"), mmap_mode="r")
            for i in range(sp.parts)
        ]
        self._offsets = np.cumsum([0] + [len(p) for p in self._parts])
        pool = int(self._offsets[-1])
        self.dim = self._parts[0].shape[1]
        self._mem = None
        if sp.holdout_from_pool:
            qrows = np.sort(
                np.random.default_rng(QUERY_SEED).choice(pool, sp.nq, replace=False)
            )
            self.queries = normalize(self._gather_pool(qrows))
            mask = np.ones(pool, bool)
            mask[qrows] = False
            self._keep = np.flatnonzero(mask)  # corpus position -> pool row
            self.query_pool_rows = qrows
        else:
            self.queries = normalize(np.load(os.path.join(d, "queries.npy"))[: sp.nq])
            self._keep = None
        self.n = pool - (sp.nq if sp.holdout_from_pool else 0)
        gt_path = os.path.join(root, "gt", f"{name}.npy")
        self.gt = np.load(gt_path) if os.path.exists(gt_path) else None

    # -- pool access (npy) --------------------------------------------------
    def _gather_pool(self, rows: np.ndarray) -> np.ndarray:
        """Rows from the memory-mapped parts, by scattered reads or one sequential sweep.

        Measured on the campaign's CephFS volume (io_probe.py, 2026-09-15): scattered rows
        arrive at 400-700 a second, while a sequential stream runs at ~100 MiB/s. A cell that
        wants 655,360 training rows out of 10M therefore waits about twenty minutes for the
        scattered read and about one for a sweep, at a few percent of one CPU either way.
        The choice is made per part on the density of the wanted rows within their own span,
        not within the whole part, so a contiguous range sweeps exactly itself and a sparse
        set keeps the scattered read. The rows returned are identical either way.
        """
        out = np.empty((len(rows), self.dim), np.float32)
        part = np.searchsorted(self._offsets, rows, side="right") - 1
        for p in np.unique(part):
            sel = np.flatnonzero(part == p)
            local = rows[sel] - self._offsets[p]
            order = np.argsort(local)
            src = self._parts[p]
            want = local[order]
            span = int(want[-1]) - int(want[0]) + 1
            if len(want) * SWEEP_RATIO >= span:
                self._sweep(src, want, out, sel[order])
            else:
                out[sel[order]] = src[want]
        return out

    def _sweep(self, src, local_sorted, out, dest):
        """One pass from the first wanted row to the last, copying rows as they go by.

        Only the span is read, never the whole part: ``blocks`` asks for contiguous ranges,
        and sweeping the part for each would read it four times over.
        """
        pos = 0
        first, last = int(local_sorted[0]), int(local_sorted[-1])
        for s in range(first, last + 1, SWEEP_BLOCK):
            e = min(last + 1, s + SWEEP_BLOCK)
            hi = pos + int(np.searchsorted(local_sorted[pos:], e))
            if hi > pos:
                blk = np.array(src[s:e], np.float32, copy=True)
                out[dest[pos:hi]] = blk[local_sorted[pos:hi] - s]
                pos = hi
            if pos >= len(local_sorted):
                break

    def _pool_rows(self, pos: np.ndarray) -> np.ndarray:
        return pos if self._keep is None else self._keep[pos]

    # -- public interface -----------------------------------------------------
    def take(self, pos: np.ndarray) -> np.ndarray:
        """Normalized corpus rows at corpus positions ``pos`` (any order)."""
        pos = np.asarray(pos, dtype=np.int64)
        if self._mem is not None:
            return self._mem[pos]
        return normalize(self._gather_pool(self._pool_rows(pos)))

    def blocks(self, block: int = BLOCK):
        """Yield (start, normalized block) over the corpus in position order.

        A memory-mapped corpus is read one block ahead on a worker thread, so the volume is
        being read while the caller encodes the previous block. Reading 38 GiB at ~100 MiB/s
        is six minutes during which a cell would otherwise sit at nearly no CPU at all, which
        is both slow and, on a shared cluster, a utilization violation.
        """
        if self._mem is not None:
            for s in range(0, self.n, block):
                yield s, self._mem[s : min(self.n, s + block)]
            return
        starts = list(range(0, self.n, block))
        with ThreadPoolExecutor(max_workers=1) as pool:
            ahead = pool.submit(
                self.take, np.arange(starts[0], min(self.n, starts[0] + block))
            )
            for i, s in enumerate(starts):
                cur = ahead
                if i + 1 < len(starts):
                    nxt = starts[i + 1]
                    ahead = pool.submit(
                        self.take, np.arange(nxt, min(self.n, nxt + block))
                    )
                yield s, cur.result()

    def train_sample(self, seed: int, size: int) -> np.ndarray:
        """Seeded training rows: the first ``size`` of one fixed random order per seed.

        The order is a draw of ``TRAIN_MAX`` positions (or all rows) in random order, so
        every prefix is itself a uniform sample and methods needing different training
        sizes see nested samples of the same draw.
        """
        rng = np.random.default_rng(1_000_003 * (seed + 1))
        order = rng.choice(self.n, min(self.n, TRAIN_MAX), replace=False)
        return self.take(order[: min(size, len(order))])

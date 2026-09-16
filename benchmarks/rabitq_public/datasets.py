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
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

QUERY_SEED = 20260914
BLOCK = 250_000
GATHER_WORKERS = 8  # concurrent readers for a scattered gather
CACHE_KEEP_FREE = (
    8 * 2**30
)  # never take the volume below this when caching a training sample
SWEEP_BLOCK = 100_000
# Reading is chosen by cost, from rates measured on the campaign's CephFS volume
# (io_probe.py, 2026-09-15): a sequential stream runs at ~100 MiB/s, and a scattered row costs
# ~76 ms of round trip, which GATHER_WORKERS readers hide in parallel. Counting bytes alone
# would mislead: 200k training rows are 800 MiB against a 38 GiB sweep, yet at one round trip
# each they would take half an hour where the sweep takes six minutes.
SEQ_BYTES_S = 100 * 2**20
ROW_LATENCY_S = 0.076
TRAIN_MAX = 655_360  # 40 x the largest nlist (16,384)


def _sweep_is_cheaper(n_rows, n_runs, want, dim):
    """Whether streaming the span beats fetching the rows, at the measured rates."""
    row_bytes = dim * 4
    span = int(want[-1]) - int(want[0]) + 1
    sweep_s = span * row_bytes / SEQ_BYTES_S
    scatter_s = (
        n_runs * ROW_LATENCY_S / GATHER_WORKERS + n_rows * row_bytes / SEQ_BYTES_S
    )
    return sweep_s < scatter_s


def _write_cache(path, rows):
    """Write the training rows for reuse, atomically, if the volume can spare the space.

    The campaign volume is shared with the corpora and the results, so a cache that fills it
    would cost far more than the reads it saves. Anything left over is never fatal: a cell
    that cannot cache simply gathers, as it did before.
    """
    try:
        d = os.path.dirname(path)
        os.makedirs(d, exist_ok=True)
        free = shutil.disk_usage(d).free
        if free - rows.nbytes < CACHE_KEEP_FREE:
            return
        tmp = f"{path}.{os.getpid()}.tmp"
        try:
            with open(
                tmp, "wb"
            ) as fh:  # np.save(name) would append .npy to the temp name
                np.save(fh, rows)
            os.replace(tmp, path)
        except OSError:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
    except OSError:
        pass


def _npy_layout(path):
    """(path, data offset, dim) for a C-order float32 .npy, or None if it is anything else.

    Lets a gather read rows with os.pread instead of through the memory map, which is what
    makes concurrent reads possible at all.
    """
    try:
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            else:
                return None
            if fortran or dtype != np.dtype(np.float32) or len(shape) != 2:
                return None
            return path, f.tell(), int(shape[1])
    except (OSError, ValueError):
        return None


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
        self._layout = [
            _npy_layout(os.path.join(d, f"part_{i:03d}.npy")) for i in range(sp.parts)
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
            runs = int(np.count_nonzero(np.diff(want) != 1)) + 1
            if self._layout[p] is None:
                out[sel[order]] = src[want]
            elif _sweep_is_cheaper(len(want), runs, want, self.dim):
                self._sweep(src, want, out, sel[order])
            else:
                self._pread(self._layout[p], want, out, sel[order])
        return out

    def _pread(self, layout, local_sorted, out, dest):
        """Read exactly the wanted rows, several at a time.

        A scattered row costs one CephFS round trip, measured at 13-685 rows a second from a
        single thread, so a cell gathering 200k training rows waited minutes at no CPU. The
        wait is latency, not bandwidth, so concurrent reads hide it; each worker keeps its own
        file handle, and reading releases the GIL.
        """
        path, offset, dim = layout
        row_bytes = dim * 4
        breaks = np.flatnonzero(np.diff(local_sorted) != 1) + 1
        runs = np.split(np.arange(len(local_sorted)), breaks)
        groups = [runs[i::GATHER_WORKERS] for i in range(GATHER_WORKERS)]

        def work(group):
            with open(path, "rb", buffering=0) as fh:  # one handle per worker
                for idx in group:
                    if not len(idx):
                        continue
                    first = int(local_sorted[idx[0]])
                    fh.seek(offset + first * row_bytes)
                    buf = fh.read(len(idx) * row_bytes)
                    out[dest[idx]] = np.frombuffer(buf, np.float32).reshape(-1, dim)

        if len(runs) == 1:
            work(runs)
            return
        with ThreadPoolExecutor(max_workers=GATHER_WORKERS) as pool:
            list(pool.map(work, [g for g in groups if g]))

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

        The draw is spread over the whole corpus, so collecting it from the memory-mapped
        parts means streaming all of them: 38 GiB to obtain 2.7 GiB of rows, six minutes at
        almost no CPU, repeated by every cell of the arm. The rows are therefore kept once per
        (dataset, seed) and afterwards read as a prefix of that file. The cache holds as many
        rows as the largest request so far and never more, so it costs a cell no memory it was
        not going to use anyway, and a larger request rebuilds it. The cached rows are exactly
        what ``take`` returned, so a cell sees the same numbers either way.
        """
        rng = np.random.default_rng(1_000_003 * (seed + 1))
        order = rng.choice(self.n, min(self.n, TRAIN_MAX), replace=False)
        size = min(size, len(order))
        if self._mem is not None:  # corpus already in RAM; a cache would only cost disk
            return self.take(order[:size])
        path = os.path.join(self.root, "trainsample", f"{self.spec.name}-s{seed}.npy")
        if os.path.exists(path):
            try:
                cached = np.load(path, mmap_mode="r")
                enough = len(cached) >= size
                rows = (
                    np.array(cached[:size], np.float32, copy=True) if enough else None
                )
                del cached  # release the map: an open one blocks the rebuild's replace
                if enough:
                    return rows
            except (OSError, ValueError):
                pass  # unreadable cache: fall through and rebuild it
        # Only ever hold what this cell asked for. Materializing the whole TRAIN_MAX draw to
        # fill the cache cost 3.8 GiB on a 1536-dim arm and OOM-killed a cell needing 1.2.
        rows = self.take(order[:size])
        _write_cache(path, rows)
        return rows

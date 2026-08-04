# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Shared constants + the seeded corpus for the NRP multi-node fleet run.

The corpus is defined *by seed*, per global shard: ``gen_block(g)`` is byte-
reproducible anywhere, so the distributed build moves **zero** corpus bytes —
each build job regenerates exactly its own row-range. Every artifact of the
run (bootstrap basis, coarse quantizer, per-server manifests, results) lives
under ``/shared/fleet`` on the RWX CephFS PVC; each server's index lives on
its own RWO Linstor (block) PVC at ``/idx`` per the storage law.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

N_SERVERS = 4
# 50 -> 1B (the completed run); 500 -> 10B (same recipe, same seeds scheme).
SHARDS_PER_SERVER = int(os.environ.get("TQP_SHARDS_PER_SERVER", "50"))
SHARD_ROWS = 5_000_000
N_TOTAL = N_SERVERS * SHARDS_PER_SERVER * SHARD_ROWS
DIM, OUT_DIM, BITS, NLIST = 32, 24, 4, 2048

SHARED = "/shared/fleet"
BOOT = f"{SHARED}/bootstrap"
RESULTS = f"{SHARED}/results"

# Query sample: rows from shards spread across the servers' ranges. Both are
# env-overridable so reference jobs and coordinators of a given run always
# derive the identical seeded query set (10B: 0,501,1002,1503 x 125 = nq 500).
QUERY_SHARDS = tuple(
    int(x) for x in os.environ.get("TQP_QUERY_SHARDS", "0,67,133,199").split(",")
)
QUERIES_PER_SHARD = int(os.environ.get("TQP_QUERIES_PER_SHARD", "25"))


def drop_page_cache(path: str) -> None:
    """Flush and evict ``path``'s page cache (no-op where unsupported).

    Inside a 2Gi cgroup the page cache is not free: cgroup-v2
    ``memory.current`` charges it, and a measured build pod sat at anon 292Mi /
    file 1526Mi — the spill file and freshly written shards — with the OOM
    killer firing whenever a dirty-page spike beat reclaim. Data written
    through here is never re-read (spill is consumed once, shards are read
    back mmap'd in a later phase), so evicting is pure win. fsync first:
    DONTNEED silently skips dirty pages.
    """
    if not hasattr(os, "posix_fadvise"):  # e.g. Windows dev box
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def gen_block_bands(gshard: int, rows: int = SHARD_ROWS, dim: int = DIM):
    """Yield global shard ``gshard``'s rows as consecutive float32 row bands.

    The band-at-a-time core of :func:`gen_block` — same seeds, same draw order,
    same arithmetic, so ``concat(bands) == gen_block(g)`` byte for byte (see
    the identity notes on :func:`gen_block`; asserted by ``verify_identical``).
    Feeding the bands straight into ``ShardedIndex.write_shard_streaming``
    means the full block never exists in memory at all: resident peak is a few
    band buffers, ~0.1 GiB, versus ~0.7 GiB for the materialized block — which
    is what lets a build pod fit NRP's enforcement-exempt cpu<=1/mem<=2Gi
    envelope end to end.
    """
    rng = np.random.default_rng(777_000 + gshard)
    rank = dim // 2
    basis = rng.standard_normal((rank, dim))
    scale = np.linspace(1.0, 0.3, rank)
    band = max(1, rows // 32)
    spill_dir = os.environ.get("TQP_SPILL_DIR", tempfile.gettempdir())

    fd, spill = tempfile.mkstemp(dir=spill_dir, suffix=f".coeffs{gshard}")
    os.close(fd)
    try:
        buf = np.empty((band, rank), dtype=np.float64)
        with open(spill, "wb") as fh:
            for a in range(0, rows, band):
                k = min(a + band, rows) - a
                v = buf[:k]
                rng.standard_normal(out=v)
                fh.write(v.tobytes())
        # 640MB of freshly written spill would otherwise sit in the cgroup's
        # page cache through the whole read-back (measured: file 1526Mi vs
        # anon 292Mi at OOM).
        drop_page_cache(spill)

        nz = np.empty((band, dim), dtype=np.float64)
        with open(spill, "rb") as fh:
            fdno = fh.fileno()
            for a in range(0, rows, band):
                k = min(a + band, rows) - a
                pos = fh.tell()
                c = (
                    np.frombuffer(fh.read(k * rank * 8), dtype=np.float64).reshape(
                        k, rank
                    )
                    * scale
                )
                # Evict each consumed chunk; it is never read again.
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(fdno, pos, k * rank * 8, os.POSIX_FADV_DONTNEED)
                blk = nz[:k]
                rng.standard_normal(out=blk)
                tmp = c @ basis
                tmp += 0.05 * blk
                # Same float64 -> float32 rounding as gen_block's slice
                # assignment into its float32 output buffer.
                yield tmp.astype(np.float32)
    finally:
        try:
            os.remove(spill)
        except OSError:
            pass


def gen_block(gshard: int, rows: int = SHARD_ROWS, dim: int = DIM) -> np.ndarray:
    """Global shard ``gshard``'s rows — same low-rank recipe as
    ``bench_ivf_sharded``, but seeded per shard so any worker can regenerate
    any range independently.

    Generated in row bands, with the coefficient draw spilled to disk, so no
    full-size float64 array is ever resident. The naive form materialises four
    of them — coeffs 0.60 GiB, coeffs@basis 1.19, noise 1.19, sum 1.19 — and
    peaks near 4.2 GiB, measured at 4042Mi. NRP deletes any pod asking for more
    than 2 GiB, so that peak has to come down rather than be requested.

    Two properties make this safe, and both are load-bearing:

    * The generator fills its output buffer sequentially, so drawing into
      consecutive slices consumes exactly the stream one whole-array call
      would.
    * Every coefficient is drawn before any noise, preserving the original
      draw order. Interleaving the two per band would reorder the stream and
      silently produce a different corpus, invalidating comparison with the
      published 1B, 10B and 100B points.

    The coefficients are needed after the noise draws but must be drawn before
    them, so they are written to a spill file and streamed back a band at a
    time. Ordinary file IO rather than mmap, because page cache from read() is
    reclaimable while mapped pages are not — an earlier memmap attempt still
    OOMed for exactly that reason.

    Resident peak is the float32 output plus a few band buffers, about 0.7 GiB.
    Byte-identity is asserted by ``verify_identical`` against the original
    expression, not assumed.

    The draws themselves live in :func:`gen_block_bands`; this is that stream
    accumulated into one array, kept for the callers (queries, verification,
    the originals cold store) that genuinely need the whole block.
    """
    out = np.empty((rows, dim), dtype=np.float32)
    a = 0
    for band in gen_block_bands(gshard, rows, dim):
        out[a : a + len(band)] = band
        a += len(band)
    return out


def queries() -> np.ndarray:
    return np.concatenate([gen_block(g)[:QUERIES_PER_SHARD] for g in QUERY_SHARDS])


# --------------------------------------------------------------------------- #
# Cold store — ONE .npy PER GLOBAL SHARD, never a single shared file. Writers  #
# of one 128 GB file share boundary pages (slices start at header+k*640 MB,    #
# not page-aligned), and concurrent read-modify-write of a shared page across  #
# CephFS clients corrupted the header twice. Per-shard files have exactly one  #
# writer each; readers mmap read-only after all writers closed.                #
# --------------------------------------------------------------------------- #

ORIG_DIR = f"{SHARED}/orig"


def orig_path(gshard: int) -> str:
    return f"{ORIG_DIR}/g_{gshard:05d}.npy"


def write_original(gshard: int, block: np.ndarray) -> None:
    """Write global shard ``gshard``'s fp32 originals (atomic tmp+rename)."""
    import os

    tmp = orig_path(gshard) + ".tmp.npy"
    np.save(tmp, np.ascontiguousarray(block, dtype=np.float32))
    os.replace(tmp, orig_path(gshard))


class ShardedNpyStore:
    """Cold-tier ``fetch(ids)`` over the per-shard files (global id -> shard
    file + row). Duck-typed for :func:`turboquant_pro.rerank_tier
    .rerank_candidates`.

    Cold CephFS random reads cost ~0.5 s per touched location regardless of
    size, and the kernel client largely *serializes reads within one file*
    (per-inode caps) — measured: a 200-row fetch over 4 shards runs ~50 s no
    matter the thread count, while wide shortlists parallelize across their
    distinct shard files. Reads are therefore issued row-parallel on a wide
    pool: effective speedup scales with the number of shards the shortlist
    touches (~200 at 1B → expect ~10-20x over the 1186 s mmap-loop
    measurement). The real fix at production scale is a random-read-capable
    cold tier (object-store ranged GETs / RBD / NVMe)."""

    def __init__(self, dim: int = DIM, max_threads: int = 64):
        self._dim = dim
        self._threads = max_threads
        self._fds: dict[int, int] = {}
        self._base: dict[int, int] = {}

    def _handle(self, g: int) -> tuple[int, int]:
        import os

        if g not in self._fds:
            self._fds[g] = os.open(orig_path(g), os.O_RDONLY)
            self._base[g] = os.path.getsize(orig_path(g)) - SHARD_ROWS * self._dim * 4
        return self._fds[g], self._base[g]

    def fetch(self, ids: np.ndarray) -> np.ndarray:
        import os
        from concurrent.futures import ThreadPoolExecutor

        ids = np.asarray(ids, dtype=np.int64)
        rb = self._dim * 4
        out = np.empty((len(ids), self._dim), dtype=np.float32)
        for g in np.unique(ids // SHARD_ROWS):
            self._handle(int(g))  # open serially; reads go wide

        def read_one(i: int) -> None:
            g = int(ids[i]) // SHARD_ROWS
            fd, base = self._fds[g], self._base[g]
            row = int(ids[i]) - g * SHARD_ROWS
            out[i] = np.frombuffer(os.pread(fd, rb, base + row * rb), np.float32)

        with ThreadPoolExecutor(max_workers=min(self._threads, max(len(ids), 1))) as ex:
            list(ex.map(read_one, range(len(ids))))
        return out

    def close(self) -> None:
        import os

        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

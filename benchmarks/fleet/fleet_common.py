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


def gen_block(gshard: int, rows: int = SHARD_ROWS, dim: int = DIM) -> np.ndarray:
    """Global shard ``gshard``'s rows — same low-rank recipe as
    ``bench_ivf_sharded``, but seeded per shard so any worker can regenerate
    any range independently.

    Generated in row bands so the float64 intermediates never exist at full
    size. The naive form materialises four of them — coeffs 0.60 GiB,
    coeffs@basis 1.19, noise 1.19 and the sum temporary 1.19 — and peaks near
    4.2 GiB, measured at 4042Mi. That does not fit NRP's 2 GiB ignored range,
    and a peak-to-trough swing that wide has no legal request at all.

    Banding is safe here for a reason worth stating: the generator fills its
    output buffer sequentially, so drawing into consecutive slices consumes
    exactly the stream one whole-array call would. All of ``coeffs`` is drawn
    before any noise, preserving the original order. What is NOT safe is
    interleaving the two draws per band, which would reorder the stream and
    silently change the corpus. Byte-identity against the original is asserted
    by ``verify_identical`` rather than assumed.

    Peak is now coeffs (0.60 GiB) plus the float32 output (0.60 GiB) plus two
    small band buffers, about 1.35 GiB.
    """
    rng = np.random.default_rng(777_000 + gshard)
    rank = dim // 2
    basis = rng.standard_normal((rank, dim))

    band = max(1, rows // 32)
    coeffs = np.empty((rows, rank), dtype=np.float64)
    for a in range(0, rows, band):
        rng.standard_normal(out=coeffs[a : min(a + band, rows)])
    coeffs *= np.linspace(1.0, 0.3, rank)

    out = np.empty((rows, dim), dtype=np.float32)
    nz = np.empty((band, dim), dtype=np.float64)
    for a in range(0, rows, band):
        b = min(a + band, rows)
        blk = nz[: b - a]
        rng.standard_normal(out=blk)
        tmp = coeffs[a:b] @ basis
        tmp += 0.05 * blk
        out[a:b] = tmp
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

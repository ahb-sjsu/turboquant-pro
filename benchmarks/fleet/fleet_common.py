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

import numpy as np

N_SERVERS = 4
SHARDS_PER_SERVER = 50
SHARD_ROWS = 5_000_000
N_TOTAL = N_SERVERS * SHARDS_PER_SERVER * SHARD_ROWS  # 1B
DIM, OUT_DIM, BITS, NLIST = 32, 24, 4, 2048

SHARED = "/shared/fleet"
BOOT = f"{SHARED}/bootstrap"
RESULTS = f"{SHARED}/results"

# Query sample: a few rows from shards spread across the servers' ranges.
QUERY_SHARDS = (0, 67, 133, 199)
QUERIES_PER_SHARD = 25


def gen_block(gshard: int, rows: int = SHARD_ROWS, dim: int = DIM) -> np.ndarray:
    """Global shard ``gshard``'s rows — same low-rank recipe as
    ``bench_ivf_sharded``, but seeded per shard so any worker can regenerate
    any range independently."""
    rng = np.random.default_rng(777_000 + gshard)
    rank = dim // 2
    basis = rng.standard_normal((rank, dim))
    coeffs = rng.standard_normal((rows, rank)) * np.linspace(1.0, 0.3, rank)
    return (coeffs @ basis + 0.05 * rng.standard_normal((rows, dim))).astype(np.float32)


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
    .rerank_candidates`; opens (mmaps) only the shard files the shortlist
    touches."""

    def __init__(self, dim: int = DIM):
        self._dim = dim
        self._open: dict[int, np.ndarray] = {}

    def _shard(self, g: int) -> np.ndarray:
        if g not in self._open:
            self._open[g] = np.load(orig_path(g), mmap_mode="r")
        return self._open[g]

    def fetch(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        out = np.empty((len(ids), self._dim), dtype=np.float32)
        g_of = ids // SHARD_ROWS
        for g in np.unique(g_of):
            m = g_of == g
            out[m] = self._shard(int(g))[ids[m] - int(g) * SHARD_ROWS]
        return out

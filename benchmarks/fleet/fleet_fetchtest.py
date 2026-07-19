# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Validate the threaded-pread cold-store fetch: bit-exact vs the seeded
corpus, and timed (the mmap-gather it replaces measured ~119 ms/row cold)."""

import time

import numpy as np
from fleet_common import SHARD_ROWS, ShardedNpyStore, gen_block

rng = np.random.default_rng(5)
GS = (3, 77, 141, 198)
PER = 50
ids = np.concatenate(
    [rng.integers(0, SHARD_ROWS, PER) + g * SHARD_ROWS for g in GS]
)

t0 = time.time()
got = ShardedNpyStore().fetch(ids)
dt = time.time() - t0

for i, g in enumerate(GS):
    block = gen_block(g)
    sel = ids[i * PER : (i + 1) * PER] - g * SHARD_ROWS
    if not np.array_equal(got[i * PER : (i + 1) * PER], block[sel]):
        raise SystemExit(f"MISMATCH in shard g={g}")
print(f"FETCH_OK rows={len(ids)} wall_s={dt:.2f} ms_per_row={1000 * dt / len(ids):.1f}")

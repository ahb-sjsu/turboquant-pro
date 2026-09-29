# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""Regenerate the float rows the rerank bound needs, one slice of the shards per job.

The corpus is seeded per shard (``gen_block_bands``), so any row can be regenerated from
its global id without the index or a cold store. The draw stream of a shard is all
coefficients then the noise band by band, so producing row r costs every coefficient draw
plus the noise bands up to r: half a shard on average, a few seconds at one CPU, with the
band generator's own spill file keeping the peak near 0.1 GiB.

Slice ``TQP_RGEN_SLICE`` of ``TQP_RGEN_N`` takes every ``N``-th shard of the sorted unique
shard list from ``rerank{TAG}_rows.npy`` (round robin, so the slices are balanced) and
writes ``rerank{TAG}_vec_<slice>.npz`` with ``ids`` and ``vecs`` (float32, the same bytes
``gen_block`` would give). Idempotent: exits at once if its file exists.
"""

import os
import resource
import time

import numpy as np
from fleet_common import DIM, RESULTS, SHARD_ROWS, gen_block_bands

TAG = os.environ.get("TQP_RUN_TAG", "1t")
SLICE = int(os.environ["TQP_RGEN_SLICE"])
N = int(os.environ.get("TQP_RGEN_N", "100"))
OUT = f"{RESULTS}/rerank{TAG}_vec_{SLICE:03d}.npz"

if os.path.exists(OUT):
    print(f"slice {SLICE}: exists, skipping", flush=True)
    print("RGEN_DONE", flush=True)
    raise SystemExit(0)

rows = np.load(f"{RESULTS}/rerank{TAG}_rows.npy").astype(np.int64)
shards = np.unique(rows // SHARD_ROWS)
mine = shards[SLICE::N]
print(
    f"slice {SLICE}/{N}: {len(mine)} of {len(shards)} shards, {len(rows)} rows total",
    flush=True,
)


def rows_of(g):
    lo, hi = g * SHARD_ROWS, (g + 1) * SHARD_ROWS
    return rows[(rows >= lo) & (rows < hi)] - lo


out_ids, out_vecs = [], []
t0 = time.time()
for n, g in enumerate(mine):
    want = rows_of(int(g))
    last = int(want.max())
    got = np.empty((len(want), DIM), dtype=np.float32)
    a = 0
    bands = gen_block_bands(int(g))
    for band in bands:
        k = len(band)
        sel = (want >= a) & (want < a + k)
        if sel.any():
            got[sel] = band[want[sel] - a]
        a += k
        if a > last:
            break
    bands.close()  # removes the spill file
    out_ids.append(want + int(g) * SHARD_ROWS)
    out_vecs.append(got)
    if (n + 1) % 20 == 0 or n + 1 == len(mine):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        print(
            f"  {n + 1}/{len(mine)} shards, {time.time() - t0:.0f}s, rss={rss}MB",
            flush=True,
        )

ids = np.concatenate(out_ids) if out_ids else np.zeros(0, np.int64)
vecs = np.concatenate(out_vecs) if out_vecs else np.zeros((0, DIM), np.float32)
tmp = OUT + ".tmp.npz"
np.savez(tmp, ids=ids, vecs=vecs)
os.replace(tmp, OUT)
print(f"slice {SLICE}: {len(ids)} rows in {time.time() - t0:.1f}s", flush=True)
print("RGEN_DONE", flush=True)

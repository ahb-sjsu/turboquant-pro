# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Partial fp32 ground truth over one server's shard-range (scatter the GT).

A single GT pass reads 128 GB of cold store; on an unlucky CephFS node that
crawls. Four of these jobs each cover 50 shards (32 GB) on independent node
draws and write a per-range partial top-K; the coordinator's merge is exact
(a global top-K row is in its range's top-K). Idempotent per range.
"""

import os

import numpy as np
from fleet_common import (
    RESULTS,
    SHARD_ROWS,
    SHARDS_PER_SERVER,
    gen_block,
    orig_path,
    queries,
)

K = 10
SID = int(os.environ["TQP_SERVER_ID"])
out = f"{RESULTS}/gt_part_{SID}.npz"
if os.path.exists(out):
    print("partial exists, skipping", flush=True)
    print("GT_PART_DONE", flush=True)
    raise SystemExit(0)

q = queries()
qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-30)
best_sc = np.full((len(q), K), -np.inf, dtype=np.float32)
best_id = np.full((len(q), K), -1, dtype=np.int64)
SLAB = 1_000_000
for j in range(SHARDS_PER_SERVER):
    g = SID * SHARDS_PER_SERVER + j
    # Full sequential read, NOT mmap: CephFS mmap page-faults read ~10x
    # slower than a buffered streaming read (no readahead on faults).
    arr = np.load(orig_path(g))
    for s in range(0, len(arr), SLAB):
        chunk = np.asarray(arr[s : s + SLAB])
        cn = chunk / np.maximum(np.linalg.norm(chunk, axis=1, keepdims=True), 1e-30)
        sc = (qn @ cn.T).astype(np.float32)
        part = np.argpartition(-sc, K - 1, axis=1)[:, :K]
        msc = np.concatenate([best_sc, np.take_along_axis(sc, part, axis=1)], axis=1)
        mid = np.concatenate([best_id, part + g * SHARD_ROWS + s], axis=1)
        order = np.argsort(-msc, axis=1)[:, :K]
        best_sc = np.take_along_axis(msc, order, axis=1)
        best_id = np.take_along_axis(mid, order, axis=1)
    print(f"gt {j + 1}/{SHARDS_PER_SERVER} (g={g})", flush=True)

tmp = out + ".tmp.npz"
np.savez(tmp, ids=best_id, scores=best_sc)
os.replace(tmp, out)
print("GT_PART_DONE", flush=True)

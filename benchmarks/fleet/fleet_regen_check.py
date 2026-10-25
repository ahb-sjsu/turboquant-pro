# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""Score a rebuilt shard against the released exact top-10s, under one search setting.

The cell files of a rebuilt shard can hash equal to the released ones while its codes differ:
shard 2 written as one block and streamed has identical cell files and differs in 1,315,472
bytes of codes and norms (Atlas, 2026-10-25). The ``.tqe`` hash cannot settle it, because the
container stamps its creation time to the microsecond. What the release does record of the
codes is the exact reference of two earlier query sets (``ref1t_part_<sid>.npz`` and
``ref1tnm_part_<sid>.npz``): ids and float32 scores of each server's top 10. Every released
neighbour that lives in this shard must come back from the rebuilt shard with the same score,
bit for bit.

The score of a row depends on the search's arithmetic as well as on the codes: the OpenBLAS
kernel, its thread count and the scan block all change the low bits (Atlas, 2026-10-25). The
reference jobs set none of them; the kernel and the thread count were whatever the pod's node
gave, and the block was the default (``sh.search(q, k)``) before 8f25bc4 and 65536 after it.
So the caller runs this script under one candidate setting at a time and keeps the one under
which every released neighbour matches. On server 0, shard 0 (244 released neighbours), the
block-written shard matches all 244 under Haswell, 6 threads, default block (the 6 CPU pod of
the first reference run) and no other setting tried; the streamed shard matches at most 217.

The setting is passed as the environment (``OPENBLAS_CORETYPE``, ``OPENBLAS_NUM_THREADS``)
and the block as an argument, because the kernel can only be chosen before numpy loads.

Usage: python fleet_regen_check.py <shard dir> <global shard> <block|none> <set>=<queries>=<ref>...
Prints ``CHECK <set> <equal> <neighbours>`` per set and ``CHECK_DONE``.
"""

import os
import sys

import numpy as np
from fleet_common import SHARD_ROWS

from turboquant_pro import ShardedIndex

K = 10
d, g = sys.argv[1], int(sys.argv[2])
block = None if sys.argv[3] == "none" else int(sys.argv[3])
sh = ShardedIndex.open(os.path.join(d, "manifest.json"), mmap=True, max_open_shards=1)
for spec in sys.argv[4:]:
    name, qpath, rpath = spec.split("=")
    with np.load(rpath) as z:
        rid, rsc = z["ids"], z["scores"].astype(np.float32)
    here = (rid >= g * SHARD_ROWS) & (rid < (g + 1) * SHARD_ROWS)
    if not here.any():
        print(f"CHECK {name} 0 0", flush=True)
        continue
    # The whole query set in one call, as the reference ran it: the batch changes the bits too.
    ids, sc = sh.search(np.load(qpath).astype(np.float32), k=K, block=block)
    sc = sc.astype(np.float32)
    equal = 0
    for qi, ki in zip(*np.nonzero(here)):
        hit = np.flatnonzero(ids[qi] == rid[qi, ki])
        equal += bool(len(hit)) and sc[qi, hit[0]].tobytes() == rsc[qi, ki].tobytes()
    print(f"CHECK {name} {equal} {int(here.sum())}", flush=True)
print("CHECK_DONE", flush=True)

# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Scattered exact ADC full-scan reference — one server's partial top-K.

The full-scan reference is the expensive half of a fleet measurement; served
through exempt-class pods it would take hours. As a CPU-pegged batch job per
shard-range (this script), it costs minutes per server and the coordinator's
top-K merge over the partials is exact (shared basis -> comparable scores).
Idempotent per range.
"""

import os
import time

import numpy as np
from fleet_common import RESULTS, queries

from turboquant_pro import ShardedIndex

K = 10
SID = int(os.environ["TQP_SERVER_ID"])
TAG = os.environ.get("TQP_RUN_TAG", "10b")
out = f"{RESULTS}/ref{TAG}_part_{SID}.npz"
if os.path.exists(out):
    print("partial exists, skipping", flush=True)
    print("REF_PART_DONE", flush=True)
    raise SystemExit(0)

q = queries()
sh = ShardedIndex.open("/idx/manifest.json", mmap=True)
print(f"server {SID}: full-scan reference, {sh.n_rows} rows, nq={len(q)}", flush=True)
t0 = time.time()
ids, sc = sh.search(q, k=K)
wall = time.time() - t0

tmp = out + ".tmp.npz"
np.savez(tmp, ids=ids, scores=sc, wall_s=np.float64(wall))
os.replace(tmp, out)
print(f"wall_s={wall:.1f}", flush=True)
print("REF_PART_DONE", flush=True)

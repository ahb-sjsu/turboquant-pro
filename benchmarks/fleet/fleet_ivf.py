# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Routed-IVF recall over one server's shard-range, as a batch job.

The scientific question at 10B — does the IVF shortlist still reproduce the
exact ADC ranking? — is a property of the *index*, not of the wire. Measuring
it through the serving window turned out to be dominated by per-request shard
opening (>900 s for a single 1-query request at 250 shards/server; see
README), so this measures it the same way the reference was measured: a
CPU-pegged batch job per shard-range, whose partial top-K the coordinator
merges exactly (shared basis -> comparable scores).

Emits one .npz per (server, nprobe) with ids/scores/wall_s. Idempotent.

Runs in NRP's exempt class (1 CPU, 2 GiB) like the reference scan, so
``TQP_IVF_OPEN_SHARDS`` (default 8) bounds the shards held open, each of
which materializes 45 MB of id and tombstone arrays even when memory-mapped,
and ``TQP_WORKERS`` defaults to 2 since the pod has one CPU.
"""

import os
import time

import numpy as np
from fleet_common import RESULTS, queries

from turboquant_pro import ShardedIndex

K = 10
SID = int(os.environ["TQP_SERVER_ID"])
TAG = os.environ.get("TQP_RUN_TAG", "10b")
NPROBES = [int(x) for x in os.environ.get("TQP_NPROBES", "32,128").split(",")]
WORKERS = int(os.environ.get("TQP_WORKERS", "2"))
OPEN_SHARDS = int(os.environ.get("TQP_IVF_OPEN_SHARDS", "8"))


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


qcache = f"{RESULTS}/{os.environ.get('TQP_QCACHE_NAME', f'queries{TAG}.npy')}"
q = np.load(qcache) if os.path.exists(qcache) else queries()
sh = ShardedIndex.open("/idx/manifest.json", mmap=True, max_open_shards=OPEN_SHARDS)
print(
    f"server {SID}: {sh.n_rows} rows, {sh.n_shards} shards, nq={len(q)}, "
    f"max {OPEN_SHARDS} open, workers {WORKERS}, rss={rss_mb():.0f}MB",
    flush=True,
)

for nprobe in NPROBES:
    out = f"{RESULTS}/ivf{TAG}_p{nprobe}_part_{SID}.npz"
    if os.path.exists(out):
        print(f"nprobe={nprobe} exists, skipping", flush=True)
        continue
    t0 = time.time()
    ids, sc = sh.search(q, k=K, nprobe=nprobe, workers=WORKERS)
    wall = time.time() - t0
    tmp = out + ".tmp.npz"
    np.savez(tmp, ids=ids, scores=sc, wall_s=np.float64(wall))
    os.replace(tmp, out)
    print(f"nprobe={nprobe} wall_s={wall:.1f} rss={rss_mb():.0f}MB", flush=True)
print("IVF_PART_DONE", flush=True)

# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Scattered exact ADC full-scan reference — one server's partial top-K.

The full-scan reference is the expensive half of a fleet measurement. At the
1T run's 100 queries the scan is bound by the volume's read rate, not by CPU
(metered at 8% of 6 CPU and 3% of 8 GiB on 2026-09-25, an NRP utilization
violation), so the job runs in NRP's exempt class, 1 CPU and 2 GiB, where the
utilization rule does not apply. Two knobs keep the process inside 2 GiB.
``TQP_REF_BLOCK`` (default 65536 rows) bounds the per-block scoring
temporaries to tens of MB. ``TQP_REF_OPEN_SHARDS`` (default 2) bounds how
many shards stay open: a memory-mapped shard still materializes its id and
tombstone arrays in RAM, 45 MB per 5M-row shard, and the default of 128 open
shards is 5.8 GB, which is the growth that OOM-killed the first exempt-class
pilot at 2 GiB on 2026-09-25 and the 7.5 GB peak the right-sizer saw. The
mapped code pages above that are clean and reclaimable. The query set is read
from the cache the qcache phase wrote (``TQP_QCACHE_NAME`` under the results
volume) rather than regenerated: ``queries()`` generates four whole 5M-row
blocks, about 640 MB each plus temporaries, and that spike OOM-killed the
second exempt-class pilot 85 s after start. The
coordinator's top-K merge over the partials is exact (shared basis ->
comparable scores). Idempotent per range.
"""

import os
import time

import numpy as np
from fleet_common import RESULTS, queries


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


from turboquant_pro import ShardedIndex

K = 10
BLOCK = int(os.environ.get("TQP_REF_BLOCK", "65536"))
OPEN_SHARDS = int(os.environ.get("TQP_REF_OPEN_SHARDS", "2"))
SID = int(os.environ["TQP_SERVER_ID"])
TAG = os.environ.get("TQP_RUN_TAG", "10b")
out = f"{RESULTS}/ref{TAG}_part_{SID}.npz"
if os.path.exists(out):
    print("partial exists, skipping", flush=True)
    print("REF_PART_DONE", flush=True)
    raise SystemExit(0)

qcache = f"{RESULTS}/{os.environ.get('TQP_QCACHE_NAME', '')}"
if os.environ.get("TQP_QCACHE_NAME") and os.path.exists(qcache):
    q = np.load(qcache)
    print(
        f"queries from cache {qcache} shape={q.shape} rss={rss_mb():.0f}MB", flush=True
    )
else:
    q = queries()
    print(f"queries generated shape={q.shape} rss={rss_mb():.0f}MB", flush=True)
sh = ShardedIndex.open("/idx/manifest.json", mmap=True, max_open_shards=OPEN_SHARDS)
print(
    f"index open, {len(sh._shards)} shards, max {OPEN_SHARDS} open, block {BLOCK}, rss={rss_mb():.0f}MB",
    flush=True,
)
print(f"server {SID}: full-scan reference, {sh.n_rows} rows, nq={len(q)}", flush=True)
t0 = time.time()
ids, sc = sh.search(q, k=K, block=BLOCK)
wall = time.time() - t0
print(f"scan done rss={rss_mb():.0f}MB", flush=True)

tmp = out + ".tmp.npz"
np.savez(tmp, ids=ids, scores=sc, wall_s=np.float64(wall))
os.replace(tmp, out)
print(f"wall_s={wall:.1f}", flush=True)

if os.environ.get("TQP_REF_HASH") == "1":
    # Content fingerprint of this server's index, so a rebuild from the seeds can be checked
    # byte for byte: sha256 and size of every file the manifest names and of every sidecar
    # beside it. A second pass over the volume, read-bound, a few minutes after the scan.
    import hashlib
    import json

    hout = f"{RESULTS}/hash{TAG}_part_{SID}.json"
    if not os.path.exists(hout):
        t0 = time.time()
        with open("/idx/manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        files = ["manifest.json"]
        for s in manifest["shards"]:
            stem = os.path.splitext(s["path"])[0]
            files.append(s["path"])
            for side in (".ivf.off.npy", ".ivf.memb.npy"):
                if os.path.exists(os.path.join("/idx", stem + side)):
                    files.append(stem + side)
        digest = {}
        for rel in files:
            h = hashlib.sha256()
            with open(os.path.join("/idx", rel), "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
            digest[rel] = {
                "sha256": h.hexdigest(),
                "bytes": os.path.getsize(os.path.join("/idx", rel)),
            }
        hwall = time.time() - t0
        with open(hout + ".tmp", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "server": SID,
                    "files": len(digest),
                    "bytes": sum(v["bytes"] for v in digest.values()),
                    "wall_s": round(hwall, 1),
                    "digest": digest,
                },
                f,
            )
        os.replace(hout + ".tmp", hout)
        print(f"hash: {len(digest)} files, wall_s={hwall:.1f}", flush=True)
print("REF_PART_DONE", flush=True)

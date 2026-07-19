# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""One fleet build job: this server's 250M-row shard-range, built in place.

Regenerates its seeded row-range (zero corpus movement), writes 50 v3 shards
onto the server's own Linstor block PVC via ``write_shard(basis_from=…)``
(shared basis -> comparable scores), fills its slice of the CephFS cold store,
assigns cells against the **global** coarse quantizer, then adopts the
bootstrap's radius file so the entire fleet probes identical cells. Exports
the tiny coordinator artifacts (manifest + ``.ivf.off.npy`` sidecars) to the
shared volume.
"""

import glob
import json
import os
import shutil

import numpy as np
from fleet_common import (
    BOOT,
    SHARD_ROWS,
    SHARDS_PER_SERVER,
    SHARED,
    gen_block,
    write_original,
)

from turboquant_pro import ShardedIndex

SID = int(os.environ["TQP_SERVER_ID"])
IDX = "/idx"
BASIS = f"{BOOT}/shard_00000.tqe"
# 10B runs skip the cold store (1.28 TB of fp32 originals is beyond quota;
# rerank/true-recall are already demonstrated at 1B) and export to their own
# shared dirs so the 1B artifacts stay intact.
WRITE_ORIGINALS = os.environ.get("TQP_WRITE_ORIGINALS", "1") == "1"
EXPORT_PREFIX = os.environ.get("TQP_EXPORT_PREFIX", "server_")

metas = []
for j in range(SHARDS_PER_SERVER):
    g = SID * SHARDS_PER_SERVER + j
    block = gen_block(g)
    metas.append(
        ShardedIndex.write_shard(
            IDX,
            block,
            j,
            ids=np.arange(g * SHARD_ROWS, (g + 1) * SHARD_ROWS, dtype=np.int64),
            basis_from=BASIS,
            keep_originals=False,
        )
    )
    if WRITE_ORIGINALS:
        write_original(g, block)
    print(f"shard {j + 1}/{SHARDS_PER_SERVER} (g={g}) written", flush=True)

sh = ShardedIndex.finalize_manifest(IDX, metas)
print("assigning cells against the global coarse quantizer", flush=True)
sh.build_ivf(centroids=np.load(f"{BOOT}/coarse_centroids.npy"))
# Fleet-wide consistent probing: every server (and the router) uses the
# bootstrap's radius, not the locally-estimated one.
shutil.copy(f"{BOOT}/coarse_radius.npy", f"{IDX}/coarse_radius.npy")

out = f"{SHARED}/{EXPORT_PREFIX}{SID:03d}"
os.makedirs(out, exist_ok=True)
shutil.copy(f"{IDX}/manifest.json", out)
for p in glob.glob(f"{IDX}/*.ivf.off.npy"):
    shutil.copy(p, out)

n_rows = SHARDS_PER_SERVER * SHARD_ROWS
nbytes = sum(os.path.getsize(p) for p in glob.glob(f"{IDX}/*"))
print(
    json.dumps(
        {
            "server": SID,
            "rows": n_rows,
            "index_bytes": nbytes,
            "bytes_per_row": round(nbytes / n_rows, 2),
        }
    ),
    flush=True,
)
print("BUILD_DONE", flush=True)

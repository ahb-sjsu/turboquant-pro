# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""Per-server cell census: list sizes from the IVF offset sidecars, and the cell of every
merged reference neighbour that lives on this server.

No distance is computed. The 400 ``.ivf.off.npy`` sidecars (2049 offsets each) give the size
of every cell on every shard; summed they are this server's row in the global cell histogram.
For each reference top-10 row on this server (``analysis1t_ref_top10.npz`` from
``fleet_partials_analysis.py``), the shard and position follow from the id layout, the id is
checked against the shard's own id array, and the cell is found in that shard's inverted lists.
With the query's probe order, the merge (``fleet_cellmerge.py``) then knows at which probe width
every reference neighbour becomes reachable, which is the exact recall the routing gives at any
width.

Writes ``cellhist{TAG}_part_{SID}.npz``. Idempotent. Exempt class (1 CPU, 2 GiB).
"""

import json
import os
import time

import numpy as np
from fleet_common import RESULTS, SHARD_ROWS, SHARDS_PER_SERVER

from turboquant_pro import ShardedIndex

SID = int(os.environ["TQP_SERVER_ID"])
TAG = os.environ.get("TQP_RUN_TAG", "1t")
out = f"{RESULTS}/cellhist{TAG}_part_{SID}.npz"
if os.path.exists(out):
    print("partial exists, skipping", flush=True)
    print("CELLHIST_PART_DONE", flush=True)
    raise SystemExit(0)

t0 = time.time()
with open("/idx/manifest.json", encoding="utf-8") as f:
    manifest = json.load(f)
nlist = int(manifest["ivf"]["nlist"])
shards = manifest["shards"]
assert len(shards) == SHARDS_PER_SERVER, (len(shards), SHARDS_PER_SERVER)

# Cell sizes: sum of per-shard inverted-list lengths.
counts = np.zeros(nlist, dtype=np.int64)
per_shard_nonempty = np.zeros(len(shards), dtype=np.int32)


def sidecar(s):
    return os.path.join("/idx", os.path.splitext(s["path"])[0] + ".ivf")


for i, s in enumerate(shards):
    off = np.load(sidecar(s) + ".off.npy")
    sizes = np.diff(off)
    assert sizes.shape[0] == nlist and sizes.sum() == SHARD_ROWS, (
        sizes.shape,
        int(sizes.sum()),
    )
    counts += sizes
    per_shard_nonempty[i] = int((sizes > 0).sum())

# Cells of the reference neighbours that live here.
a = np.load(f"{RESULTS}/analysis{TAG}_ref_top10.npz")
ref_ids = a["ref_ids"]  # (nq, k) global ids
id_min = np.array([int(s["id_min"]) for s in shards])
id_max = np.array([int(s["id_max"]) for s in shards])
lo, hi = int(id_min.min()), int(id_max.max())
mine = [
    (qi, ki, int(ref_ids[qi, ki]))
    for qi in range(ref_ids.shape[0])
    for ki in range(ref_ids.shape[1])
    if lo <= int(ref_ids[qi, ki]) <= hi
]
cells = []
if mine:
    sh = ShardedIndex.open("/idx/manifest.json", mmap=True, max_open_shards=2)
    by_shard = {}
    for qi, ki, g in mine:
        hits = np.nonzero((id_min <= g) & (g <= id_max))[0]
        assert len(hits) == 1, ("id in no or several shards", g, hits)
        by_shard.setdefault(int(hits[0]), []).append((qi, ki, g))
    for local_shard, rows in sorted(by_shard.items()):
        idx = sh._get_shard(local_shard)
        off = np.load(sidecar(shards[local_shard]) + ".off.npy")
        memb = np.load(sidecar(shards[local_shard]) + ".memb.npy", mmap_mode="r")
        for qi, ki, g in rows:
            pos = g - int(shards[local_shard]["id_min"])
            assert int(idx._ids[pos]) == g, (g, int(idx._ids[pos]))
            cell = -1
            for c in range(nlist):
                s, e = int(off[c]), int(off[c + 1])
                if e > s:
                    seg = memb[s:e]
                    j = int(np.searchsorted(seg, pos))
                    if j < e - s and int(seg[j]) == pos:
                        cell = c
                        break
            assert cell >= 0, ("row not in any cell", g)
            cells.append((qi, ki, g, cell))
cells_arr = np.array(cells, dtype=np.int64).reshape(-1, 4)
wall = time.time() - t0
tmp = out + ".tmp.npz"
np.savez(
    tmp,
    counts=counts,
    per_shard_nonempty=per_shard_nonempty,
    ref_cells=cells_arr,
    wall_s=np.float64(wall),
)
os.replace(tmp, out)
print(
    f"server {SID}: {int(counts.sum())} rows in {nlist} cells, {len(cells)} reference neighbours here, wall_s={wall:.1f}",
    flush=True,
)
print("CELLHIST_PART_DONE", flush=True)

# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""Score the regenerated-index run (tag 1trg): merge 500 server partials, recall per query set.

Reads the 500 ``regen1trg_part_*.npz`` from ``fleet_regen_score.py`` and the query metadata
``queries1trg.json`` from ``fleet_qcache_regen.py``. Merges the reference top-10 and the routed
top-10 at each width exactly, as ``fleet_score10.py`` merges, and reports for the non-member half
(rows 0..99) and the corpus-row half (rows 100..199) separately:

  recall at ten          per width, mean, per-query distribution, 95 percent bootstrap interval
                         over queries (seed 20260929, 100000 resamples, as fig_data.py)
  reachability check     the share of reference neighbours whose cell ranks below p in the
                         query's probe order, against the measured recall at p
  placement              servers holding the reference neighbours, largest share on one server
  self-match             for corpus rows, whether the query's own row is in its reference top-10
  regeneration           shards whose two cell files hash equal to the released index, and the
                         build variant (kernel, write, library) that reproduced each
  work                   rows scored per query at each width from the regenerated cell counts

Prints one ANALYSIS_JSON line and writes ``analysis1trg.json`` beside the partials. Reads only
the partials, the queries and the bootstrap.
"""

import glob
import json
import os
import re

import numpy as np
from fleet_common import BOOT, RESULTS, SHARD_ROWS

from turboquant_pro import ShardedIndex
from turboquant_pro.sharded_index import _normalize

K = 10
TAG = os.environ.get("TQP_RUN_TAG", "1trg")
N_SRV = int(os.environ.get("TQP_N_SERVERS", "500"))
SHARDS_PER_SERVER = 400
RADIUS_SCALE = 0.5
SEED, RESAMPLES = 20260929, 100_000
BOOT = os.environ.get("TQP_BOOT", BOOT)
RESULTS = os.environ.get("TQP_RESULTS", RESULTS)
QDIR = os.environ.get("TQP_QUERIES_DIR", RESULTS)


def sid_of(path):
    return int(re.search(r"_part_(\d+)\.npz$", path).group(1))


def merge(parts, key):
    ids = np.concatenate([p[f"{key}_ids"] for p in parts], axis=1)
    sc = np.concatenate([p[f"{key}_sc"] for p in parts], axis=1)
    sc = np.where(np.isfinite(sc), sc, -np.inf)
    order = np.argsort(-sc, axis=1, kind="stable")[:, :K]
    return order, np.take_along_axis(ids, order, axis=1)


def recall(got, ref):
    return np.array([len(set(a) & set(b)) / K for a, b in zip(got, ref)])


paths = sorted(glob.glob(f"{RESULTS}/regen{TAG}_part_*.npz"), key=sid_of)
assert len(paths) == N_SRV, len(paths)
parts = [dict(np.load(p)) for p in paths]
for p in parts:
    assert int(p["shards_done"]) == SHARDS_PER_SERVER, int(p["shards_done"])
nprobes = [int(x) for x in parts[0]["nprobes"]]
meta = json.load(open(f"{QDIR}/queries{TAG}.json", encoding="utf-8"))["rows"]
q = np.load(f"{QDIR}/queries{TAG}.npy").astype(np.float32)
nq = len(q)

order, ref_ids = merge(parts, "ref")
ref_cells = np.take_along_axis(
    np.concatenate([p["ref_cells"] for p in parts], axis=1), order, axis=1
)
ref_src = np.take_along_axis(
    np.concatenate(
        [np.full(p["ref_ids"].shape, i) for i, p in enumerate(parts)], axis=1
    ),
    order,
    axis=1,
)
routed = {p: merge(parts, f"p{p}")[1] for p in nprobes}

boot = ShardedIndex.open(f"{BOOT}/manifest.json", mmap=True, max_open_shards=1)
cent, radius = boot._load_ivf()
q_rot, _ = boot._get_shard(0)._adc._query_terms(q)
theta = np.arccos(np.clip(_normalize(q_rot) @ cent.T, -1.0, 1.0))
probe_order = np.argsort(
    -np.cos(np.maximum(0.0, theta - RADIUS_SCALE * radius[None, :])), axis=1
)
rank_of = np.argsort(probe_order, axis=1)
cell_rank = np.take_along_axis(rank_of, np.clip(ref_cells, 0, None), axis=1)
counts = np.sum([p["counts"] for p in parts], axis=0)

rng = np.random.default_rng(SEED)
out = {"tag": TAG, "n_servers": N_SRV, "nq": int(nq), "sets": {}}
for name, rows in (("non_members", range(0, 100)), ("corpus_rows", range(100, 200))):
    rows = np.array(list(rows))
    s = {"nq": int(len(rows)), "widths": {}}
    for p in nprobes:
        v = recall(routed[p][rows], ref_ids[rows])
        means = v[rng.integers(0, len(v), (RESAMPLES, len(v)))].mean(1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        s["widths"][str(p)] = {
            "recall_mean": round(float(v.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "min": round(float(v.min()), 3),
            "queries_at_1": int((v >= 1 - 1e-12).sum()),
            "queries_below_0.9": int((v < 0.9).sum()),
            "reachability_share": round(float((cell_rank[rows] < p).mean()), 4),
            "rows_scored_median": float(
                np.median(counts[probe_order[rows, :p]].sum(axis=1))
            ),
        }
    src = ref_src[rows].ravel()
    c = np.bincount(src, minlength=N_SRV)
    s["placement"] = {
        "servers_holding_any": int((c > 0).sum()),
        "max_share_one_server": round(float(c.max() / c.sum()), 4),
        "servers_per_query_median": float(
            np.median([len(set(r)) for r in ref_src[rows]])
        ),
    }
    s["cell_rank_of_reference_neighbours"] = {
        "median": int(np.median(cell_rank[rows])),
        "p90": int(np.percentile(cell_rank[rows], 90)),
        "p99": int(np.percentile(cell_rank[rows], 99)),
        "max": int(cell_rank[rows].max()),
    }
    if name == "corpus_rows":
        own = [m["shard"] * SHARD_ROWS for m in meta[100:200]]
        s["self_in_reference_top10"] = int(
            sum(o in set(r) for o, r in zip(own, ref_ids[rows]))
        )
    out["sets"][name] = s

h = np.concatenate([p["hash_ok"] for p in parts])
# The .tqe container stamps its creation time, so its hash cannot match a regeneration; the
# two cell files carry no timestamp, so their hashes are the content check. The released
# index's own cell census (cellhist1t) bounds the rest: the per-server L1 distance between
# released and regenerated cell counts is twice a lower bound on the rows whose cell changed.
released = []
for i in range(N_SRV):
    path = f"{RESULTS}/cellhist1t_part_{i}.npz"
    released.append(np.load(path)["counts"] if os.path.exists(path) else None)
l1 = [
    int(np.abs(parts[i]["counts"] - r).sum())
    for i, r in enumerate(released)
    if r is not None
]
kern = np.concatenate([p["kernel"] for p in parts])
ks, kn = np.unique(kern, return_counts=True)
out["regeneration"] = {
    "shards": int(len(h)),
    "cell_files_hash_equal": int((h[:, 1:] == 1).all(axis=1).sum()),
    "variant_that_reproduced": {str(k) or "none": int(n) for k, n in zip(ks, kn)},
    "builds_per_shard_mean": round(
        float(np.concatenate([p["trials"] for p in parts]).mean()), 3
    ),
    "servers_with_released_census": len(l1),
    "servers_with_identical_cell_counts": int(sum(x == 0 for x in l1)),
    "rows_moved_between_cells_lower_bound": int(sum(l1) // 2),
}
out["rows"] = int(counts.sum())
out["regen_and_search_wall_s_per_server"] = [
    round(float(p["wall_s"]), 1) for p in parts
]
with open(f"{RESULTS}/analysis{TAG}.json.tmp", "w", encoding="utf-8") as f:
    json.dump(out, f)
os.replace(f"{RESULTS}/analysis{TAG}.json.tmp", f"{RESULTS}/analysis{TAG}.json")
print("ANALYSIS_JSON " + json.dumps(out), flush=True)

# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""Everything the 1T partials already say, without a new scan.

Reads the 500 reference and routed partials on the shared volume and reports

  per-query recall      at each probe width, the distribution over queries rather than the
                        mean the score phase prints (how many queries at 1.0, min, p10, p50);
  server-drop loss      the exact share of the merged reference top-10 that lives on each
                        server, so the recall lost by losing N servers is a sum, not a scan;
  neighbour occurrence  how often each global row appears across the 100 queries' reference
                        top-10, the hubness statistic the corpus note reports, here on 1000
                        slots only, which the JSON says;
  probe order           for each query, the order in which the shared coarse quantizer would
                        probe the 2048 cells, computed the way ``ShardedIndex._ivf_search`` does
                        it from the bootstrap on the shared volume, written with the merged
                        reference top-10 ids to ``analysis1t_ref_top10.npz`` for the per-server
                        cell job (``fleet_cellhist.py``), which turns them into the exact recall
                        the routing must give at every probe width.

Writes ``analysis1t_partials.json`` and ``analysis1t_ref_top10.npz`` under RESULTS. Idempotent.
"""

import glob
import json
import os
import re

import numpy as np
from fleet_common import BOOT, RESULTS

from turboquant_pro import ShardedIndex
from turboquant_pro.sharded_index import _normalize

K = 10
TAG = os.environ.get("TQP_RUN_TAG", "1t")
N_SRV = int(os.environ.get("TQP_N_SERVERS", "500"))
RADIUS_SCALE = 0.5  # fleet_ivf.py calls search() with its defaults
NPROBES = [int(x) for x in os.environ.get("TQP_NPROBES", "32,128").split(",")]


def sid_of(path):
    return int(re.search(r"_part_(\d+)\.npz$", path).group(1))


def merge(paths):
    parts = [np.load(p) for p in paths]
    ids = np.concatenate([p["ids"] for p in parts], axis=1)
    scs = np.concatenate([p["scores"] for p in parts], axis=1)
    src = np.concatenate(
        [np.full(p["ids"].shape, sid_of(path)) for p, path in zip(parts, paths)], axis=1
    )
    scs = np.where(np.isfinite(scs), scs, -np.inf)
    order = np.argsort(-scs, axis=1)[:, :K]
    return np.take_along_axis(ids, order, axis=1), np.take_along_axis(
        src, order, axis=1
    )


def per_query_recall(got, ref):
    return np.array([len(set(a[:K]) & set(b[:K])) / K for a, b in zip(got, ref)])


def dist(v):
    v = np.asarray(v, float)
    return {
        "mean": round(float(v.mean()), 4),
        "min": round(float(v.min()), 3),
        "p10": round(float(np.percentile(v, 10)), 3),
        "p50": round(float(np.percentile(v, 50)), 3),
        "queries_at_1": int((v >= 1.0 - 1e-12).sum()),
        "queries_below_0.9": int((v < 0.9).sum()),
        "n": int(len(v)),
    }


ref_paths = sorted(glob.glob(f"{RESULTS}/ref{TAG}_part_*.npz"), key=sid_of)
assert len(ref_paths) == N_SRV, (len(ref_paths), N_SRV)
ref_ids, ref_src = merge(ref_paths)
nq = ref_ids.shape[0]
out = {"n_servers": N_SRV, "nq": int(nq), "k": K, "per_query_recall": {}}

for p in NPROBES:
    paths = sorted(glob.glob(f"{RESULTS}/ivf{TAG}_p{p}_part_*.npz"), key=sid_of)
    if len(paths) != N_SRV:
        out["per_query_recall"][str(p)] = {"partials": len(paths), "skipped": True}
        continue
    got, _ = merge(paths)
    out["per_query_recall"][str(p)] = dist(per_query_recall(got, ref_ids))

# Server-drop loss: share of the merged reference top-10 on each server, exact.
counts = np.bincount(ref_src.ravel(), minlength=N_SRV)
share = counts / counts.sum()
srt = np.sort(share)[::-1]
out["server_drop"] = {
    "reference_slots": int(counts.sum()),
    "servers_holding_any": int((counts > 0).sum()),
    "max_share_one_server": round(float(srt[0]), 4),
    "expected_recall_after_losing_n_random_servers": {
        str(n): round(float(1 - n / N_SRV), 4) for n in (1, 5, 10, 50)
    },
    "worst_case_recall_after_losing_n_servers": {
        str(n): round(float(1 - srt[:n].sum()), 4) for n in (1, 5, 10, 50)
    },
}

# Neighbour occurrence over the reference top-10 (1000 slots at nq=100).
ids_flat = ref_ids.ravel()
uniq, occ = np.unique(ids_flat, return_counts=True)
occ_sorted = np.sort(occ)[::-1]
n_slots = len(ids_flat)
top1pct = max(1, int(np.ceil(0.01 * len(uniq))))
gini = (
    float(np.abs(occ[:, None] - occ[None, :]).sum() / (2 * len(occ) ** 2 * occ.mean()))
    if len(occ) > 1
    else 0.0
)
out["neighbour_occurrence"] = {
    "slots": int(n_slots),
    "distinct_rows": int(len(uniq)),
    "max_occurrence": int(occ_sorted[0]),
    "rows_seen_more_than_once": int((occ > 1).sum()),
    "share_of_slots_in_top_1pct_rows": round(
        float(occ_sorted[:top1pct].sum() / n_slots), 4
    ),
    "gini": round(gini, 4),
    "note": "occurrence over 100 queries x top-10; not comparable to the corpus note's all-points hubness",
}

# Probe order per query from the bootstrap's shared coarse quantizer, as _ivf_search orders cells.
boot = ShardedIndex.open(f"{BOOT}/manifest.json", mmap=True, max_open_shards=1)
q = np.load(f"{RESULTS}/queries{TAG}.npy").astype(np.float32)
assert q.shape[0] == nq
cent, radius = boot._load_ivf()
q_rot, _ = boot._get_shard(0)._adc._query_terms(q)
qdir = _normalize(q_rot)
theta = np.arccos(np.clip(qdir @ cent.T, -1.0, 1.0))
ub = np.cos(np.maximum(0.0, theta - RADIUS_SCALE * radius[None, :]))
probe_order = np.argsort(-ub, axis=1).astype(np.int32)  # (nq, nlist) best cell first
out["probe_order"] = {
    "nlist": int(cent.shape[0]),
    "radius_scale": RADIUS_SCALE,
    "bound": "weighted",
}

np.savez(
    f"{RESULTS}/analysis{TAG}_ref_top10.npz.tmp.npz",
    ref_ids=ref_ids,
    ref_src=ref_src,
    probe_order=probe_order,
)
os.replace(
    f"{RESULTS}/analysis{TAG}_ref_top10.npz.tmp.npz",
    f"{RESULTS}/analysis{TAG}_ref_top10.npz",
)
with open(f"{RESULTS}/analysis{TAG}_partials.json.tmp", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
os.replace(
    f"{RESULTS}/analysis{TAG}_partials.json.tmp",
    f"{RESULTS}/analysis{TAG}_partials.json",
)
print("ANALYSIS_JSON " + json.dumps(out), flush=True)
print("ANALYSIS_DONE", flush=True)

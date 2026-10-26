# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""Work per query of routed search at 10^12 rows: rows scored, bytes touched, servers touched.

No distance is computed and no index is opened. The 500 per-server cell histograms
(``cellhist1t_part_*.npz`` from ``fleet_cellhist.py``) give how many rows every server holds in
every cell. A query's probe order is a function of the query, the bootstrap's basis, the global
coarse centroids and the global radius only, computed here exactly as ``ShardedIndex._ivf_search``
computes it (weighted order, radius_scale 0.5) and checked against the probe order the first
run recorded (``analysis1t_ref_top10.npz``). At width p the routed scan scores every row of the
p cells first in that order, on every server, so

  rows scored        sum over the p cells of the global cell size
  bytes touched      rows scored x 24 (12 bytes of 4-bit codes, 8 of norms, 4 of member id),
                     the payload the scan reads, not a device-level measurement
  servers touched    servers holding at least one row in those cells

for each query of the corpus-row set (``queries1t.npy``) and the non-member set
(``queries1tnm.npy``), at the widths the paper reports, beside the balanced-cell figure
p / 2048 x 10^12.

Usage: python fleet_work_analysis.py <shared dir with bootstrap/ and results/> <queries1tnm.npy>
Prints one ANALYSIS_JSON line. Reads only.
"""

import glob
import json
import os
import re
import sys

import numpy as np

from turboquant_pro.index import TQEIndex

N_SRV = 500
ROWS = 10**12
WIDTHS = (16, 32, 64, 128, 256)
BYTES_PER_ROW_SCORED = 24  # codes 12 + norms 8 + member id 4
RADIUS_SCALE = 0.5  # ShardedIndex.search default, used by every routed pass


def sid_of(path):
    return int(re.search(r"_part_(\d+)\.npz$", path).group(1))


def probe_order(q, boot):
    """The full cell order of ``ShardedIndex._ivf_search`` (flat IVF, weighted bound)."""
    adc = TQEIndex.open(os.path.join(boot, "shard_00000.tqe"), mmap=True)._adc
    cent = np.load(os.path.join(boot, "coarse_centroids.npy"))
    radius = np.load(os.path.join(boot, "coarse_radius.npy"))
    q_rot, _ = adc._query_terms(np.asarray(q, dtype=np.float32))
    qdir = q_rot / np.maximum(np.linalg.norm(q_rot, axis=1, keepdims=True), 1e-30)
    theta = np.arccos(np.clip(qdir @ cent.T, -1.0, 1.0))
    ub = np.cos(np.maximum(0.0, theta - RADIUS_SCALE * radius[None, :]))
    return np.argsort(-ub, axis=1)


def summary(v):
    v = np.asarray(v, float)
    return {
        "mean": float(v.mean()),
        "min": float(v.min()),
        "p10": float(np.percentile(v, 10)),
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "max": float(v.max()),
    }


shared, nm_queries = sys.argv[1], sys.argv[2]
boot, results = os.path.join(shared, "bootstrap"), os.path.join(shared, "results")
paths = sorted(glob.glob(os.path.join(results, "cellhist1t_part_*.npz")), key=sid_of)
assert len(paths) == N_SRV, len(paths)
C = np.stack([np.load(p)["counts"] for p in paths])  # (500, 2048)
assert int(C.sum()) == ROWS, int(C.sum())
cell_size = C.sum(axis=0)

out = {"rows": ROWS, "nlist": int(C.shape[1]), "bytes_per_row_scored": 24, "sets": {}}
for name, qpath in (
    ("corpus_rows", os.path.join(results, "queries1t.npy")),
    ("non_members", nm_queries),
):
    q = np.load(qpath)
    order = probe_order(q, boot)
    if name == "corpus_rows":
        rec = os.path.join(results, "analysis1t_ref_top10.npz")
        if os.path.exists(rec):
            # The work at width p depends only on the set of the first p cells. Orders
            # recomputed on another host swap a few near-equal cells deep in the order
            # (measured: 99.97 percent of positions equal), so the sets are what is checked.
            recorded = np.load(rec)["probe_order"]
            for p in WIDTHS:
                assert all(set(a[:p]) == set(b[:p]) for a, b in zip(order, recorded)), p
            out["probed_cells_match_record_at_widths"] = list(WIDTHS)
            out["probe_order_positions_equal"] = round(
                float((order == recorded).mean()), 5
            )
    s = {"nq": int(len(q)), "widths": {}}
    for p in WIDTHS:
        cells = order[:, :p]
        rows = cell_size[cells].sum(axis=1)
        servers = np.array([(C[:, c].sum(axis=1) > 0).sum() for c in cells])
        s["widths"][str(p)] = {
            "balanced_rows": p / C.shape[1] * ROWS,
            "rows_scored": summary(rows),
            "share_of_index": summary(rows / ROWS),
            "bytes_touched": summary(rows * BYTES_PER_ROW_SCORED),
            "servers_touched": summary(servers),
        }
    out["sets"][name] = s
print("ANALYSIS_JSON " + json.dumps(out), flush=True)

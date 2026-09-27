# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""Merge the per-server cell census into the global cell histogram and the exact recall the
routing gives at every probe width.

Cell histogram: 500 x 2048 counts summed, with the imbalance the coarse quantizer produced at
10^12 rows (max/mean, Gini, share of rows in the largest 1 percent of cells, empty cells).

Reachability: every merged reference neighbour has a cell (from ``fleet_cellhist.py``) and its
query has a probe order (from ``fleet_partials_analysis.py``). The rank of the neighbour's cell
in that order is the smallest probe width at which the routing can return it. Because scoring
inside a probed cell is the same asymmetric distance the reference used, recall at width p is
exactly the share of reference neighbours with cell rank below p. That curve is a prediction for
any width, including the 16, 64 and 256 the probe sweep measures, and the sweep's own numbers at
32 and 128 are checked against it here.

Writes ``analysis{TAG}_cells.json``. Idempotent.
"""

import glob
import json
import os
import re

import numpy as np
from fleet_common import RESULTS

TAG = os.environ.get("TQP_RUN_TAG", "1t")
N_SRV = int(os.environ.get("TQP_N_SERVERS", "500"))


def sid_of(path):
    return int(re.search(r"_part_(\d+)\.npz$", path).group(1))


paths = sorted(glob.glob(f"{RESULTS}/cellhist{TAG}_part_*.npz"), key=sid_of)
assert len(paths) == N_SRV, (len(paths), N_SRV)
parts = [np.load(p) for p in paths]
counts = np.sum([p["counts"] for p in parts], axis=0)
nlist = counts.shape[0]
srt = np.sort(counts)[::-1]
top1 = max(1, int(np.ceil(0.01 * nlist)))
gini = float(
    np.abs(counts[:, None] - counts[None, :]).sum() / (2 * nlist**2 * counts.mean())
)
out = {
    "n_servers": N_SRV,
    "nlist": int(nlist),
    "rows": int(counts.sum()),
    "cell_sizes": {
        "mean": float(counts.mean()),
        "median": float(np.median(counts)),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "max_over_mean": round(float(counts.max() / counts.mean()), 3),
        "gini": round(gini, 4),
        "share_of_rows_in_largest_1pct_cells": round(
            float(srt[:top1].sum() / counts.sum()), 4
        ),
        "empty_cells": int((counts == 0).sum()),
    },
    "per_server_wall_s": {
        "median": round(float(np.median([float(p["wall_s"]) for p in parts])), 1),
        "max": round(float(max(float(p["wall_s"]) for p in parts)), 1),
    },
}

a = np.load(f"{RESULTS}/analysis{TAG}_ref_top10.npz")
ref_ids, probe_order = a["ref_ids"], a["probe_order"]
nq, k = ref_ids.shape
cells = np.concatenate(
    [p["ref_cells"] for p in parts], axis=0
)  # (n, 4): qi, ki, id, cell
assert cells.shape[0] == nq * k, (cells.shape, nq * k)
rank_of = np.argsort(
    probe_order, axis=1
)  # cell -> its rank in each query's probe order
ranks = np.array([int(rank_of[qi, cell]) for qi, _ki, _g, cell in cells])
widths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
curve = {str(p): round(float((ranks < p).mean()), 4) for p in widths}
per_query = np.zeros((nq, len(widths)))
for j, p in enumerate(widths):
    for qi, _ki, _g, cell in cells:
        per_query[qi, j] += float(rank_of[qi, cell] < p) / k
out["reachability"] = {
    "predicted_recall_at_probe_width": curve,
    "queries_at_1_at_width": {
        str(p): int((per_query[:, j] >= 1 - 1e-12).sum()) for j, p in enumerate(widths)
    },
    "cell_rank_of_reference_neighbours": {
        "median": int(np.median(ranks)),
        "p90": int(np.percentile(ranks, 90)),
        "p99": int(np.percentile(ranks, 99)),
        "max": int(ranks.max()),
        "share_in_first_probed_cell": round(float((ranks == 0).mean()), 4),
    },
}

# Check against the measured routed recall where partials exist.
checks = {}
for p in (16, 32, 64, 128, 256):
    ps = sorted(glob.glob(f"{RESULTS}/ivf{TAG}_p{p}_part_*.npz"), key=sid_of)
    if len(ps) != N_SRV:
        continue
    ids = np.concatenate([np.load(x)["ids"] for x in ps], axis=1)
    scs = np.concatenate([np.load(x)["scores"] for x in ps], axis=1)
    scs = np.where(np.isfinite(scs), scs, -np.inf)
    order = np.argsort(-scs, axis=1)[:, :k]
    got = np.take_along_axis(ids, order, axis=1)
    measured = float(
        np.mean([len(set(x[:k]) & set(y[:k])) / k for x, y in zip(got, ref_ids)])
    )
    checks[str(p)] = {
        "measured": round(measured, 4),
        "predicted": curve[str(p)],
        "difference": round(measured - curve[str(p)], 4),
    }
out["measured_vs_predicted"] = checks

with open(f"{RESULTS}/analysis{TAG}_cells.json.tmp", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
os.replace(
    f"{RESULTS}/analysis{TAG}_cells.json.tmp", f"{RESULTS}/analysis{TAG}_cells.json"
)
print("CELLS_JSON " + json.dumps(out), flush=True)
print("CELLMERGE_DONE", flush=True)

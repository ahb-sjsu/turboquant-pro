# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""Control for the router's cell order at 10^12 rows: plain centroid order against the radius
order the fleet used, at matched rows scored, with no distance scan.

The fleet's router ranks cells by cos(max(0, theta - beta * r)), theta the angle from the query
to the cell centroid and r the cell's angular radius on the bootstrap, with beta = 0.5. Plain
centroid order is beta = 0, and beta = 1 is the full-radius order. Inside a probed cell every row
is scored as the exact scan scores it, so a reference neighbour is returned exactly when its cell
is probed (Section 4.3 of the paper, and the audit of the partials for the floating-point
caveat). Recall of any order at any budget therefore follows from three archived things: the
cell of every reference neighbour (``ref_cells`` in ``cellhist1t_part_*.npz``), the global cell
sizes (``counts`` in the same files), and the query's order, recomputed here from the bootstrap.

Budgets are matched per query. For each query and each width p, the budget is the rows the
fleet's order (beta = 0.5) scored at width p. Every order then probes cells in its own order
while the running total of rows stays within that budget, and its recall is the share of the
query's reference neighbours whose cell is among them. The fleet order at its own budget is the
measured recall, which this script asserts.

Usage: python fleet_order_control.py <shared dir with bootstrap/ and results/>
Writes ``order_control1t.json`` in the current directory and prints RESULT_JSON. Reads only.
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
K = 10
WIDTHS = (8, 16, 32, 64, 128, 256)
BETAS = (0.0, 0.25, 0.5, 1.0)
FLEET_BETA = 0.5
MEASURED = {16: 0.952, 32: 0.989, 64: 0.997, 128: 0.999, 256: 1.0}


def sid_of(path):
    return int(re.search(r"_part_(\d+)\.npz$", path).group(1))


shared = sys.argv[1]
boot, results = os.path.join(shared, "bootstrap"), os.path.join(shared, "results")
paths = sorted(glob.glob(os.path.join(results, "cellhist1t_part_*.npz")), key=sid_of)
assert len(paths) == N_SRV, len(paths)
parts = [np.load(p) for p in paths]
cell_size = np.sum([p["counts"] for p in parts], axis=0).astype(np.int64)
assert int(cell_size.sum()) == ROWS
ref_cells = np.concatenate([p["ref_cells"] for p in parts], axis=0)  # qi, ki, id, cell

q = np.load(os.path.join(results, "queries1t.npy"))
nq = len(q)
assert ref_cells.shape[0] == nq * K
nb_cells = np.full((nq, K), -1, np.int64)
for qi, ki, _g, cell in ref_cells:
    nb_cells[qi, ki] = cell
assert (nb_cells >= 0).all()

adc = TQEIndex.open(os.path.join(boot, "shard_00000.tqe"), mmap=True)._adc
cent = np.load(os.path.join(boot, "coarse_centroids.npy"))
radius = np.load(os.path.join(boot, "coarse_radius.npy"))
q_rot, _ = adc._query_terms(np.asarray(q, dtype=np.float32))
qdir = q_rot / np.maximum(np.linalg.norm(q_rot, axis=1, keepdims=True), 1e-30)
theta = np.arccos(np.clip(qdir @ cent.T, -1.0, 1.0))


def order_for(beta):
    return np.argsort(-np.cos(np.maximum(0.0, theta - beta * radius[None, :])), axis=1)


orders = {b: order_for(b) for b in BETAS}
fleet = orders[FLEET_BETA]
recorded = np.load(os.path.join(results, "analysis1t_ref_top10.npz"))["probe_order"]
for p in WIDTHS:
    assert all(set(a[:p]) == set(b[:p]) for a, b in zip(fleet, recorded)), p


def recall_within(order, budget):
    """Per query: probe cells in order while cumulative rows <= budget; share of neighbours."""
    out = np.empty(nq)
    cells_used = np.empty(nq, np.int64)
    rows_used = np.empty(nq, np.int64)
    for i in range(nq):
        cum = np.cumsum(cell_size[order[i]])
        n = int(np.searchsorted(cum, budget[i], side="right"))
        probed = set(order[i, :n].tolist())
        out[i] = np.mean([c in probed for c in nb_cells[i]])
        cells_used[i] = n
        rows_used[i] = int(cum[n - 1]) if n else 0
    return out, cells_used, rows_used


res = {
    "rows": ROWS,
    "nq": nq,
    "fleet_beta": FLEET_BETA,
    "betas": list(BETAS),
    "probed_cells_match_record_at_widths": list(WIDTHS),
    "widths": {},
}
for p in WIDTHS:
    budget = cell_size[fleet[:, :p]].sum(axis=1)
    entry = {
        "budget_rows_median": float(np.median(budget)),
        "budget_share_median": float(np.median(budget) / ROWS),
        "orders": {},
    }
    for b in BETAS:
        r, n, used = recall_within(orders[b], budget)
        entry["orders"][str(b)] = {
            "recall": round(float(r.mean()), 4),
            "queries_at_1": int((r == 1.0).sum()),
            "cells_probed_median": float(np.median(n)),
            "rows_scored_median": float(np.median(used)),
        }
        if b == FLEET_BETA and p in MEASURED:
            assert round(float(r.mean()), 3) == MEASURED[p], (p, r.mean())
    # Paired comparison against plain centroid order, bootstrap over queries.
    r0, _, _ = recall_within(orders[0.0], budget)
    rf, _, _ = recall_within(fleet, budget)
    d = rf - r0
    rng = np.random.default_rng(7)
    boots = [d[rng.integers(0, nq, nq)].mean() for _ in range(2000)]
    entry["fleet_minus_centroid"] = {
        "mean": round(float(d.mean()), 4),
        "ci95": [
            round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4),
        ],
        "queries_better": int((d > 0).sum()),
        "queries_worse": int((d < 0).sum()),
    }
    res["widths"][str(p)] = entry

# Centroid order at its own width p (not budget-matched), for the rows it would score.
res["centroid_order_at_width"] = {}
for p in WIDTHS:
    cells = orders[0.0][:, :p]
    probed = [set(c.tolist()) for c in cells]
    r = np.array([np.mean([c in probed[i] for c in nb_cells[i]]) for i in range(nq)])
    res["centroid_order_at_width"][str(p)] = {
        "recall": round(float(r.mean()), 4),
        "rows_scored_median": float(np.median(cell_size[cells].sum(axis=1))),
    }

with open("order_control1t.json.tmp", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
os.replace("order_control1t.json.tmp", "order_control1t.json")
print("RESULT_JSON " + json.dumps(res), flush=True)

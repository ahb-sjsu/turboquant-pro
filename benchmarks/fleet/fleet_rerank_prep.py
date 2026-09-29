# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""Wide shortlists for the rerank bound, from the partials already on the volume.

For the reference scan and for each routed width, merge the per-server top-10 partials
(500 servers, so 5000 candidates a query) and keep the ``WIDE`` best by asymmetric-distance
score. That is the shortlist the fleet actually returns: each server's ten best, cut to the
best hundred overall, the same width the 1B run reranked (``K * 10``). It is not the exact
top-100 of the compressed scan, because one server may hold more than ten of those; the
score job says so in its JSON.

Writes under RESULTS

  rerank{TAG}_shortlists.npz   ids_<set> (nq, WIDE) int64, descending by ADC score, and
                               adc_<set> (nq, WIDE) float32, for set in ref, 16, 32, ...
  rerank{TAG}_rows.npy         the sorted unique global ids across every set, which the
                               regeneration jobs (``fleet_rerank_gen.py``) turn into floats.

Idempotent: rewrites both files from the partials every time (seconds).
"""

import glob
import json
import os
import re

import numpy as np
from fleet_common import RESULTS

TAG = os.environ.get("TQP_RUN_TAG", "1t")
N_SRV = int(os.environ.get("TQP_N_SERVERS", "500"))
WIDE = int(os.environ.get("TQP_RERANK_WIDE", "100"))
SETS = os.environ.get("TQP_RERANK_SETS", "ref,16,32,64,128,256").split(",")


def sid_of(path):
    return int(re.search(r"_part_(\d+)\.npz$", path).group(1))


def merge_wide(paths, wide):
    parts = [np.load(p) for p in paths]
    ids = np.concatenate([p["ids"] for p in parts], axis=1).astype(np.int64)
    scs = np.concatenate([p["scores"] for p in parts], axis=1).astype(np.float32)
    scs = np.where(np.isfinite(scs) & (ids >= 0), scs, -np.inf)
    order = np.argsort(-scs, axis=1, kind="stable")[:, :wide]
    return np.take_along_axis(ids, order, axis=1), np.take_along_axis(
        scs, order, axis=1
    )


out = {}
summary = {"n_servers": N_SRV, "wide": WIDE, "sets": {}}
for s in SETS:
    pat = (
        f"{RESULTS}/ref{TAG}_part_*.npz"
        if s == "ref"
        else f"{RESULTS}/ivf{TAG}_p{int(s)}_part_*.npz"
    )
    paths = sorted(glob.glob(pat), key=sid_of)
    assert len(paths) == N_SRV, (s, len(paths), N_SRV)
    ids, adc = merge_wide(paths, WIDE)
    assert (ids >= 0).all(), s
    out[f"ids_{s}"] = ids
    out[f"adc_{s}"] = adc
    summary["sets"][s] = {"nq": int(ids.shape[0]), "wide": int(ids.shape[1])}
    print(f"set {s}: {ids.shape[0]} queries x {ids.shape[1]} wide", flush=True)

rows = np.unique(
    np.concatenate([v.ravel() for k, v in out.items() if k.startswith("ids_")])
)
summary["unique_rows"] = int(len(rows))
summary["unique_shards"] = int(len(np.unique(rows // 5_000_000)))

tmp = f"{RESULTS}/rerank{TAG}_shortlists.tmp.npz"
np.savez(tmp, **out)
os.replace(tmp, f"{RESULTS}/rerank{TAG}_shortlists.npz")
tmp = f"{RESULTS}/rerank{TAG}_rows.tmp.npy"
np.save(tmp, rows)
os.replace(tmp, f"{RESULTS}/rerank{TAG}_rows.npy")
print("PREP_JSON " + json.dumps(summary), flush=True)
print("PREP_DONE", flush=True)

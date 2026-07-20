# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Score the 10B run: merge the per-server partials and report recall.

Both merges are exact (shared basis -> comparable scores; a global top-K row
is in its range's top-K), so IVF recall here is measured against the same
exact ADC full-scan the fleet coordinator would have used.
"""

import glob
import json
import os

import numpy as np
from fleet_common import RESULTS

K = 10
N_SRV = int(os.environ.get("TQP_N_SERVERS", "8"))
TAG = os.environ.get("TQP_RUN_TAG", "10b")
N_ROWS = int(os.environ.get("TQP_N_ROWS", str(10_000_000_000)))


def merge(pattern: str):
    parts = [np.load(p) for p in sorted(glob.glob(pattern))]
    ids = np.concatenate([p["ids"] for p in parts], axis=1)
    scs = np.concatenate([p["scores"] for p in parts], axis=1)
    scs = np.where(np.isfinite(scs), scs, -np.inf)
    order = np.argsort(-scs, axis=1)[:, :K]
    walls = [float(p["wall_s"]) for p in parts]
    return np.take_along_axis(ids, order, axis=1), walls


def recall(got, ref):
    return float(np.mean([len(set(a[:K]) & set(b[:K])) / K for a, b in zip(got, ref)]))


ref_ids, ref_walls = merge(f"{RESULTS}/ref{TAG}_part_*.npz")
res = {
    "n_rows": N_ROWS,
    "n_servers": N_SRV,
    "nq": int(ref_ids.shape[0]),
    "reference_wall_s_per_server": [round(w, 1) for w in ref_walls],
    "ivf": {},
}
for nprobe in (32, 128):
    hits = sorted(glob.glob(f"{RESULTS}/ivf{TAG}_p{nprobe}_part_*.npz"))
    if len(hits) < N_SRV:
        print(f"nprobe={nprobe}: {len(hits)}/{N_SRV} partials, skipping", flush=True)
        continue
    ids, walls = merge(f"{RESULTS}/ivf{TAG}_p{nprobe}_part_*.npz")
    res["ivf"][str(nprobe)] = {
        "recall_vs_adc_fullscan": recall(ids, ref_ids),
        "wall_s_per_server": [round(w, 1) for w in walls],
        "speedup_vs_fullscan": round(
            float(np.mean(ref_walls)) / max(float(np.mean(walls)), 1e-9), 1
        ),
    }
    print(json.dumps({str(nprobe): res["ivf"][str(nprobe)]}), flush=True)

with open(f"{RESULTS}/fleet_run_{TAG.upper()}.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print("RESULT_JSON " + json.dumps(res), flush=True)
print("SCORE_DONE", flush=True)

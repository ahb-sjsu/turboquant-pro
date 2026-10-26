# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""Per-query recall and neighbour placement for the non-member query run (tag 1tnm).

The score phase of run 1tnm printed only the mean recall at 32 and 128 probes. This reads the
500 reference and routed partials the run wrote (copied off the shared volume before the
volumes were released, under scale_transfer/1tnm-all on Atlas) and reports what
``fleet_partials_analysis.py`` reports for the member queries, without the probe-order part,
which needs the bootstrap volume:

  per-query recall   at 32 and 128 probes, with the score phase's mean checked to the digit
  bootstrap interval 95 percent percentile interval of mean recall, resampling queries, the
                     method and seed ``paper/pvldb1t/figdata/fig_data.py`` uses
  server placement   how many servers hold any of the merged reference top-10, and the largest
                     share held by one server

Usage: python fleet_nonmember_analysis.py <dir with ref1tnm_part_*.npz and ivf1tnm_*.npz>
Prints one ANALYSIS_JSON line. Reads only.
"""

import glob
import json
import os
import re
import sys

import numpy as np

K = 10
TAG = "1tnm"
N_SRV = 500
NPROBES = (32, 128)
SCORED_MEAN = {32: 0.969, 128: 0.999}  # score_1Tnm.log RESULT_JSON
SEED, RESAMPLES = 20260929, 100_000


# sid_of, merge, per_query_recall and dist are those of fleet_partials_analysis.py, which
# runs on import and so cannot be imported.
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


d = sys.argv[1]
ref_paths = sorted(glob.glob(os.path.join(d, f"ref{TAG}_part_*.npz")), key=sid_of)
assert len(ref_paths) == N_SRV, len(ref_paths)
ref_ids, ref_src = merge(ref_paths)
nq = ref_ids.shape[0]
out = {
    "tag": TAG,
    "n_servers": N_SRV,
    "nq": int(nq),
    "per_query_recall": {},
    "recall_ci95": {},
}

rng = np.random.default_rng(SEED)
for p in NPROBES:
    paths = sorted(glob.glob(os.path.join(d, f"ivf{TAG}_p{p}_part_*.npz")), key=sid_of)
    assert len(paths) == N_SRV, (p, len(paths))
    got, _ = merge(paths)
    v = per_query_recall(got, ref_ids)
    assert abs(v.mean() - SCORED_MEAN[p]) < 1e-9, (p, v.mean())
    out["per_query_recall"][str(p)] = dist(v)
    means = v[rng.integers(0, len(v), (RESAMPLES, len(v)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    out["recall_ci95"][str(p)] = {
        "mean": round(float(v.mean()), 4),
        "lo": round(float(lo), 4),
        "hi": round(float(hi), 4),
        "method": f"percentile bootstrap over queries, {RESAMPLES} resamples, seed {SEED}",
    }

counts = np.bincount(ref_src.ravel(), minlength=N_SRV)
share = np.sort(counts / counts.sum())[::-1]
out["server_placement"] = {
    "reference_slots": int(counts.sum()),
    "servers_holding_any": int((counts > 0).sum()),
    "max_share_one_server": round(float(share[0]), 4),
    "servers_per_query_median": float(np.median([len(set(r)) for r in ref_src])),
}
print("ANALYSIS_JSON " + json.dumps(out), flush=True)

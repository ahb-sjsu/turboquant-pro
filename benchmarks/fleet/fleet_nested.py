# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""Recall against corpus size inside one index, from the partials, with no new scan.

The per-server partials hold the exact top-10 of the reference scan and of every routed
pass for each query on each server, with scores on the shared basis. The exact top-10 over
any subset of servers is therefore the merge of that subset's partials, for the reference
and for every width alike, and recall of routing against the exact scan of the subset is
exact. A subset of k servers is a corpus of 2k billion rows built from the same law, with
the same basis, quantizer, router and queries, so the only thing that changes along k is N.

For each size k the script reports the prefix subset (servers 0 to k-1) and ``TQP_NESTED_R``
random subsets of k servers (seeded), and for every subset the mean recall at ten at each
width and its minimum over queries. It also counts how many of the query home servers the
subset holds, because a query is a corpus row and its home server carries its exact match
and, through the shared basis, most of its close neighbours.

Writes ``nested{TAG}.json`` and prints RESULT_JSON. Idempotent, seconds.
"""

import glob
import json
import os
import re

import numpy as np
from fleet_common import QUERY_SHARDS, RESULTS, SHARDS_PER_SERVER

K = 10
TAG = os.environ.get("TQP_RUN_TAG", "1t")
N_SRV = int(os.environ.get("TQP_N_SERVERS", "500"))
NPROBES = [int(x) for x in os.environ.get("TQP_NPROBES", "16,32,64,128,256").split(",")]
SIZES = [
    int(x)
    for x in os.environ.get(
        "TQP_NESTED_SIZES", "1,2,3,5,10,20,50,100,200,300,400,500"
    ).split(",")
]
R = int(os.environ.get("TQP_NESTED_R", "5"))
SEED = int(os.environ.get("TQP_NESTED_SEED", "1234"))
SPS = int(os.environ.get("TQP_SHARDS_PER_SERVER", str(SHARDS_PER_SERVER)))
HOME = sorted({g // SPS for g in QUERY_SHARDS})


def sid_of(path):
    return int(re.search(r"_part_(\d+)\.npz$", path).group(1))


def load(pattern):
    paths = sorted(glob.glob(pattern), key=sid_of)
    assert len(paths) == N_SRV, (pattern, len(paths), N_SRV)
    parts = [np.load(p) for p in paths]
    ids = np.stack([p["ids"].astype(np.int64) for p in parts])  # (srv, nq, 10)
    scs = np.stack([p["scores"].astype(np.float32) for p in parts])
    scs = np.where(np.isfinite(scs) & (ids >= 0), scs, -np.inf)
    return ids, scs


def top10(ids, scs, servers):
    i = np.concatenate([ids[s] for s in servers], axis=1)
    c = np.concatenate([scs[s] for s in servers], axis=1)
    order = np.argsort(-c, axis=1, kind="stable")[:, :K]
    return np.take_along_axis(i, order, axis=1)


def recall(got, ref):
    return np.array([len(set(a) & set(b)) / K for a, b in zip(got, ref)])


ref_ids, ref_scs = load(f"{RESULTS}/ref{TAG}_part_*.npz")
routed = {p: load(f"{RESULTS}/ivf{TAG}_p{p}_part_*.npz") for p in NPROBES}
nq = ref_ids.shape[1]
rng = np.random.default_rng(SEED)


def measure(servers):
    ref = top10(ref_ids, ref_scs, servers)
    out = {
        "servers": int(len(servers)),
        "n_rows": int(len(servers)) * SPS * 5_000_000,
        "home_servers_held": int(sum(1 for h in HOME if h in set(servers))),
        "recall": {},
        "recall_min": {},
    }
    for p in NPROBES:
        r = recall(top10(*routed[p], servers), ref)
        out["recall"][str(p)] = round(float(r.mean()), 4)
        out["recall_min"][str(p)] = round(float(r.min()), 3)
    return out


res = {
    "n_servers": N_SRV,
    "nq": int(nq),
    "k": K,
    "nprobes": NPROBES,
    "home_servers": HOME,
    "random_subsets_per_size": R,
    "seed": SEED,
    "sizes": {},
}
for k in SIZES:
    entry = {"prefix": measure(list(range(k)))}
    if k < N_SRV:
        subs = [
            measure(sorted(rng.choice(N_SRV, k, replace=False).tolist()))
            for _ in range(R)
        ]
        entry["random"] = {
            "mean": {
                str(p): round(float(np.mean([s["recall"][str(p)] for s in subs])), 4)
                for p in NPROBES
            },
            "min": {
                str(p): round(float(np.min([s["recall"][str(p)] for s in subs])), 4)
                for p in NPROBES
            },
            "max": {
                str(p): round(float(np.max([s["recall"][str(p)] for s in subs])), 4)
                for p in NPROBES
            },
            "home_servers_held": [s["home_servers_held"] for s in subs],
        }
    res["sizes"][str(k)] = entry
    print(json.dumps({str(k): entry}), flush=True)

tmp = f"{RESULTS}/nested{TAG}.json.tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
os.replace(tmp, f"{RESULTS}/nested{TAG}.json")
print("RESULT_JSON " + json.dumps(res), flush=True)
print("NESTED_DONE", flush=True)

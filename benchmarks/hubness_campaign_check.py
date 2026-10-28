"""Discrepancy check: does the campaign's rising real hubness come from its file-ordered split?

openvector-bench harness/rc1/r14_real_targets.py measured attractiveness_skew on wiki1024 rising
+0.54/decade (k=10, n 12.5k-50k, rho 4). The density-ratio pilot measured it flat on part_000.
r14 samples rows part by part and splits the pool in file order, so its corpus comes from the
early parts and its queries from later ones. This script reruns r14's protocol exactly (arm
"ordered") and with the same rows shuffled before the split (arm "shuffled"). Nothing else changes.

Usage (Atlas, agi-hpc venv): taskset -c 12-21 python hubness_campaign_check.py OUT.json
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import numpy as np
import torch
from hubness_density_ratio import attractiveness_skew, topk_ids

REAL_DIR = "/archive/tqp_real/wiki1024"
NS = [12_500, 25_000, 50_000]
K = 10
RHO = 4.0
SEEDS = list(range(6))


def sealed(i: int) -> bool:
    return hashlib.blake2b(str(i).encode(), digest_size=1).digest()[0] % 4 == 3


def load_pool(n_rows: int, rng):
    """Verbatim sampling logic of r14_real_targets.load_pool, plus the source part of each row."""
    parts = sorted(glob.glob(os.path.join(REAL_DIR, "part_*.npy")))
    per = max(1, n_rows // len(parts))
    out, src, taken = [], [], 0
    for j, p in enumerate(parts):
        a = np.load(p, mmap_mode="r")
        idx = np.sort(rng.choice(len(a), size=min(per * 2, len(a)), replace=False))
        keep = np.array([i for i in idx if not sealed(int(i))], dtype=np.int64)
        if len(keep):
            out.append(np.asarray(a[keep], dtype=np.float32))
            src.append(np.full(len(keep), j))
            taken += len(keep)
        if taken >= n_rows:
            break
    return np.concatenate(out)[:n_rows], np.concatenate(src)[:n_rows]


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main(out_path):
    n_max = max(NS)
    pool, src = load_pool(int(n_max * 1.6), np.random.default_rng(20260807))
    perm = np.random.default_rng(99).permutation(len(pool))
    out = {"protocol": __doc__, "cells": [], "parts": {}}
    for arm, order in (("ordered", np.arange(len(pool))), ("shuffled", perm)):
        base, qp = pool[order[:n_max]], pool[order[n_max:]]
        out["parts"][arm] = {
            "base": [int(src[order[:n_max]].min()), int(src[order[:n_max]].max())],
            "query": [int(src[order[n_max:]].min()), int(src[order[n_max:]].max())],
        }
        print(arm, out["parts"][arm], flush=True)
        for n in NS:
            nq = int(round(RHO * n / K))
            for sd in SEEDS:
                r = np.random.default_rng(10_000 * sd + n + K)
                b = torch.from_numpy(
                    normalize(base[r.choice(len(base), n, replace=False)])
                )
                q = torch.from_numpy(
                    normalize(qp[r.choice(len(qp), nq, replace=False)])
                )
                c = np.bincount(topk_ids(q, b, K).ravel().numpy(), minlength=n)
                cell = {
                    "arm": arm,
                    "n": n,
                    "nq": nq,
                    "seed": sd,
                    "attractiveness_skew": attractiveness_skew(c),
                }
                out["cells"].append(cell)
                print(json.dumps(cell), flush=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

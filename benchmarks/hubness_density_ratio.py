"""Pilot: hubness as a query/corpus density ratio at the k-NN scale.

Registered predictions: docs/PREREG_hubness_density_ratio.md (committed before this ran).
Real Cohere Embed-V3 wiki1024, cosine, exact brute-force top-k (torch CPU).

Usage (Atlas, agi-hpc venv):
    taskset -c 12-21 python hubness_density_ratio.py OUT.json
"""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import torch

PART = "/archive/tqp_real/wiki1024/part_000.npy"
QUERIES = "/archive/tqp_real/wiki1024/queries.npy"
N_POOL_C, N_QB1, N_QB2 = 250_000, 100_000, 50_000
GRID_N = [12_500, 25_000, 50_000, 100_000, 200_000]
GRID_K = [5, 10, 20, 40, 80]
RHO = 2.5
SEEDS = [0, 1, 2]
BLOCK = 2048
torch.set_num_threads(10)


def attractiveness_skew(counts: np.ndarray) -> float:
    """Poisson moment deconvolution, verbatim logic of openvector_bench.hubness.attractiveness_skew."""
    c = np.asarray(counts, dtype=np.float64)
    rho = float(c.mean())
    if rho <= 0:
        return float("nan")
    var_w = (float(c.var()) - rho) / rho**2
    if var_w <= 1e-9:
        return float("nan")
    mu3_w = (float(((c - rho) ** 3).mean()) - rho - 3 * rho**2 * var_w) / rho**3
    return float(mu3_w / var_w**1.5)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation with average ranks for ties (counts tie heavily)."""
    from scipy.stats import rankdata

    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / math.sqrt((ra * ra).sum() * (rb * rb).sum()))


def topk_ids(q: torch.Tensor, c: torch.Tensor, k: int) -> torch.Tensor:
    """Exact top-k corpus ids for each query (queries are never corpus rows here)."""
    out = torch.empty((q.shape[0], k), dtype=torch.int64)
    for s in range(0, q.shape[0], BLOCK):
        out[s : s + BLOCK] = torch.topk(q[s : s + BLOCK] @ c.T, k, dim=1).indices
    return out


def kth_sim_self(c: torch.Tensor, k: int) -> torch.Tensor:
    """Similarity of each corpus point to its k-th nearest other corpus point."""
    out = torch.empty(c.shape[0])
    for s in range(0, c.shape[0], BLOCK):
        sims = c[s : s + BLOCK] @ c.T
        out[s : s + sims.shape[0]] = torch.topk(sims, k + 1, dim=1).values[:, k]
    return out


def count_above(a: torch.Tensor, c: torch.Tensor, t: float) -> torch.Tensor:
    """For each row of c: number of rows of a with cosine >= t."""
    cnt = torch.zeros(c.shape[0], dtype=torch.int64)
    for s in range(0, a.shape[0], BLOCK):
        cnt += (a[s : s + BLOCK] @ c.T >= t).sum(dim=0)
    return cnt


def load():
    t0 = time.time()
    x = np.load(PART, mmap_mode="r")
    rng = np.random.default_rng(12345)
    pick = np.sort(rng.choice(x.shape[0], N_POOL_C + N_QB1 + N_QB2, replace=False))
    data = np.asarray(x[pick], dtype=np.float32)
    perm = rng.permutation(data.shape[0])
    data = data[perm]
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    q = np.asarray(np.load(QUERIES), dtype=np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    qp = rng.permutation(q.shape[0])
    half = q.shape[0] // 2
    print(
        f"loaded part shape {x.shape}, queries {q.shape}, {time.time() - t0:.0f}s",
        flush=True,
    )
    return (
        torch.from_numpy(data[:N_POOL_C].copy()),
        torch.from_numpy(data[N_POOL_C : N_POOL_C + N_QB1].copy()),
        torch.from_numpy(data[N_POOL_C + N_QB1 :].copy()),
        torch.from_numpy(q[qp[:half]].copy()),
        torch.from_numpy(q[qp[half:]].copy()),
    )


def predictor_cell(name, corpus, q_count, q_pred, k):
    n = corpus.shape[0]
    ids = topk_ids(q_count, corpus, k)
    counts = np.bincount(ids.ravel().numpy(), minlength=n).astype(np.float64)
    kth = kth_sim_self(corpus, k)
    t = float(kth.median())
    qmass = count_above(q_pred, corpus, t).numpy().astype(np.float64)
    cmass = (
        (count_above(corpus, corpus, t) - 1).numpy().astype(np.float64)
    )  # minus self
    ratio = (qmass + 0.5) / (cmass + 0.5)
    mean = corpus.mean(dim=0)
    central = (corpus @ (mean / mean.norm())).numpy().astype(np.float64)
    density = kth.numpy().astype(np.float64)
    res = {
        "cell": name,
        "n": n,
        "k": k,
        "nq_count": int(q_count.shape[0]),
        "nq_pred": int(q_pred.shape[0]),
        "rho": q_count.shape[0] * k / n,
        "threshold_t": t,
        "attractiveness_skew": attractiveness_skew(counts),
        "raw_skew": float(((counts - counts.mean()) ** 3).mean() / counts.std() ** 3),
        "frac_qmass_zero": float((qmass == 0).mean()),
        "spearman": {
            "ratio": spearman(counts, ratio),
            "qmass": spearman(counts, qmass),
            "cmass": spearman(counts, cmass),
            "centrality": spearman(counts, central),
            "density": spearman(counts, density),
        },
    }
    print(json.dumps(res), flush=True)
    return res


def collapse_cell(pool_c, qb1, n, k, seed):
    rng = np.random.default_rng(1000 * seed + n + k)
    ci = torch.from_numpy(rng.choice(pool_c.shape[0], n, replace=False))
    nq = int(round(RHO * n / k))
    qi = torch.from_numpy(rng.choice(qb1.shape[0], nq, replace=False))
    corpus, q = pool_c[ci], qb1[qi]
    ids = topk_ids(q, corpus, k)
    counts = np.bincount(ids.ravel().numpy(), minlength=n).astype(np.float64)
    return {
        "n": n,
        "k": k,
        "seed": seed,
        "nq": nq,
        "rho": nq * k / n,
        "attractiveness_skew": attractiveness_skew(counts),
        "raw_skew": float(((counts - counts.mean()) ** 3).mean() / counts.std() ** 3),
    }


def main(out_path):
    t0 = time.time()
    pool_c, qb1, qb2, qa1, qa2 = load()
    out = {
        "protocol": "docs/PREREG_hubness_density_ratio.md",
        "rho_grid": RHO,
        "p1": [],
        "p3": [],
        "p2": [],
    }
    rng = np.random.default_rng(7)
    # P1: real queries, n = 12500, k in {20, 40}
    c12 = pool_c[torch.from_numpy(rng.choice(pool_c.shape[0], 12_500, replace=False))]
    for k in (20, 40):
        out["p1"].append(predictor_cell(f"QA n=12500 k={k}", c12, qa1, qa2, k))
    # P3: same-distribution queries, n in {25k, 50k}, k = 10; plus QB at the matched QA cell
    for n in (25_000, 50_000):
        cn = pool_c[torch.from_numpy(rng.choice(pool_c.shape[0], n, replace=False))]
        nq = int(round(RHO * n / 10))
        out["p3"].append(predictor_cell(f"QB n={n} k=10", cn, qb1[:nq], qb2, 10))
    out["p3_matched_qb"] = predictor_cell(
        "QB n=12500 k=40 (matched to QA)", c12, qb1[: qa1.shape[0]], qb2, 40
    )
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    # P2 grid
    for seed in SEEDS:
        for n in GRID_N:
            for k in GRID_K:
                r = collapse_cell(pool_c, qb1, n, k, seed)
                out["p2"].append(r)
                print(json.dumps(r), f"{time.time() - t0:.0f}s", flush=True)
                with open(out_path, "w") as fh:
                    json.dump(out, fh, indent=1)
    print(f"DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

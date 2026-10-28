"""POST HOC (not registered): local-bandwidth basin mass after P1 failed.

P1 used a single global similarity threshold for qmass/cmass. The first-order theory says the basin of x_j is a ball
of x_j's own k-NN radius, so mu_Q of that ball already carries the 1/f_C factor without dividing by a corpus count.
Here: local_qmass_j = number of predictor queries with cos >= (similarity of x_j to its k-th corpus neighbour),
compared with the registered global-threshold qmass on the same cells, same splits, same seeds as the pilot.
Exploratory only; it cannot rescue P1.
"""

from __future__ import annotations

import json
import sys

import hubness_density_ratio as h
import numpy as np
import torch


def local_count(q: torch.Tensor, c: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
    cnt = torch.zeros(c.shape[0], dtype=torch.int64)
    for s in range(0, q.shape[0], h.BLOCK):
        cnt += (q[s : s + h.BLOCK] @ c.T >= thr.unsqueeze(0)).sum(dim=0)
    return cnt


def cell(name, corpus, q_count, q_pred, k):
    n = corpus.shape[0]
    counts = np.bincount(
        h.topk_ids(q_count, corpus, k).ravel().numpy(), minlength=n
    ).astype(np.float64)
    kth = h.kth_sim_self(corpus, k)
    t = float(kth.median())
    g = h.count_above(q_pred, corpus, t).numpy().astype(np.float64)
    loc = local_count(q_pred, corpus, kth).numpy().astype(np.float64)
    res = {
        "cell": name,
        "n": n,
        "k": k,
        "spearman_global_qmass": h.spearman(counts, g),
        "spearman_local_qmass": h.spearman(counts, loc),
        "frac_local_zero": float((loc == 0).mean()),
    }
    print(json.dumps(res), flush=True)
    return res


def main(out_path):
    pool_c, qb1, qb2, qa1, qa2 = h.load()
    rng = np.random.default_rng(7)  # same draws as the pilot
    c12 = pool_c[torch.from_numpy(rng.choice(pool_c.shape[0], 12_500, replace=False))]
    out = [cell(f"QA n=12500 k={k}", c12, qa1, qa2, k) for k in (20, 40)]
    for n in (25_000, 50_000):
        cn = pool_c[torch.from_numpy(rng.choice(pool_c.shape[0], n, replace=False))]
        out.append(
            cell(f"QB n={n} k=10", cn, qb1[: int(round(h.RHO * n / 10))], qb2, 10)
        )
    json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])

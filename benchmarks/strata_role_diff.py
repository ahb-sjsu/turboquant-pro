#!/usr/bin/env python3
"""S1/S2/S3 of PREREG_role_differentiation.md (blob fc91233).

Computes, from a directed kNN id array + area labels ONLY (so the definition is
identical across operating points, which is what P3 requires):

  S1  role bimodality   -- Hartigan dip on per-area tau_a (+ bootstrap p),
                           corroborated by 2-means silhouette on the role vector
                           (tau_a, gini_a, centrality_pct_a), all bounded coords.
  S2  directed asymmetry-- A = ||F - F^T||_F / ||F||_F on the area->area flow
                           matrix of directed kNN edges. A->0 is exchangeable.
  S3  redundancy        -- plug-in I(area_query ; area_neighbour) on F.
                           DECLARED near-monotone with mean tau: context only.

tau replicates turboquant_pro.strata.stratified_anatomy exactly: it is an
IN-DEGREE quantity -- of the queries that retrieved row j, the fraction from a
different area -- with counts==0 rows contributing tau=0 via max(counts,1).

ABSTAIN (n < n_min or n_queries < q_min) areas are excluded from every
statistic, per the inherited protocol; they are listed in the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np


def sha256(path: str, cap: int = 1 << 26) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
            if f.tell() > cap:      # cap for multi-GB vectors: prefix digest
                return h.hexdigest() + f"+prefix{cap}"
    return h.hexdigest()


def per_area_stats(idx, labels, k, n_min, q_min):
    """Replicates strata.stratified_anatomy's tau, plus per-area Gini of in-degree."""
    n = len(labels)
    hit = idx[:, :k]
    counts = np.bincount(hit.ravel(), minlength=n).astype(np.float64)
    same = labels[:, None] == labels[hit]          # self-mode: qlab == labels
    intra = np.zeros(n, dtype=np.float64)
    np.add.at(intra, hit.ravel(), same.ravel().astype(np.float64))
    tau_row = (counts - intra) / np.maximum(counts, 1.0)
    return counts, tau_row


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    if x.sum() <= 0:
        return 0.0
    nn = len(x)
    return float((2.0 * np.arange(1, nn + 1) - nn - 1).dot(x) / (nn * x.sum()))


def flow_matrix(idx, labels, areas, k):
    a_of = {a: i for i, a in enumerate(areas)}
    code = np.full(len(labels), -1, dtype=np.int64)
    for a, i in a_of.items():
        code[labels == a] = i
    hit = idx[:, :k]
    src = np.repeat(code, k)
    dst = code[hit.ravel()]
    ok = (src >= 0) & (dst >= 0)                    # both ends non-ABSTAIN
    m = len(areas)
    F = np.bincount(src[ok] * m + dst[ok], minlength=m * m).astype(np.float64)
    F = F.reshape(m, m)
    return F / max(F.sum(), 1.0)


def analyse(idx_path, labels_path, vec_path, k, n_min, q_min, seed, tag,
            permute_labels=False):
    idx = np.load(idx_path)
    labels = np.array([ln.strip() for ln in open(labels_path, encoding="utf-8")])
    if len(labels) != len(idx):
        raise SystemExit(f"labels {len(labels)} != ids {len(idx)}")
    rng = np.random.default_rng(seed)
    if permute_labels:
        labels = rng.permutation(labels)

    counts, tau_row = per_area_stats(idx, labels, k, n_min, q_min)

    uniq = sorted(set(labels.tolist()))
    keep, abstain = [], []
    for a in uniq:
        n_i = int((labels == a).sum())
        (keep if (n_i >= n_min and n_i >= q_min) else abstain).append(a)

    cent_pct = None
    if vec_path and os.path.exists(vec_path):
        X = np.load(vec_path, mmap_mode="r")
        mu = np.asarray(X[:: max(1, len(X) // 50000)]).mean(0)   # subsampled mean
        cen = np.empty(len(X), dtype=np.float64)
        for s in range(0, len(X), 20000):
            blk = np.asarray(X[s:s + 20000], dtype=np.float32)
            cen[s:s + len(blk)] = -np.linalg.norm(blk - mu, axis=1)
        cent_pct = cen.argsort().argsort() / max(len(cen) - 1, 1)  # [0,1] percentile

    rows = []
    for a in keep:
        m = labels == a
        entry = {
            "id": a,
            "n": int(m.sum()),
            "tau_mean": float(tau_row[m].mean()),
            "gini_indegree": gini(counts[m]),
            "max_Nk": float(counts[m].max()),
        }
        if cent_pct is not None:
            tr = m & (tau_row >= 0.5)               # transit rows, tau threshold 0.5
            entry["transit_centrality_pct"] = (
                float(cent_pct[tr].mean()) if tr.any() else float("nan")
            )
            entry["n_transit_rows"] = int(tr.sum())
        rows.append(entry)

    tau_a = np.array([r["tau_mean"] for r in rows])
    out = {
        "tag": tag,
        "provenance": {
            "prereg_blob": "fc91233e3bdba70f42e4d9a98ce1f8b500ccfe0d",
            "ids": os.path.basename(idx_path), "ids_sha256": sha256(idx_path),
            "labels_sha256": sha256(labels_path),
            "vectors": os.path.basename(vec_path) if vec_path else None,
            "k": k, "n_min": n_min, "q_min": q_min, "seed": seed,
            "tau_threshold": 0.5, "labels_permuted": bool(permute_labels),
            "n_rows": int(len(labels)),
        },
        "areas_scored": len(rows), "areas_abstain": abstain,
        "per_area": rows,
        "tau_mean_overall": float(tau_row.mean()),
    }

    # ---- S1 primary: Hartigan dip on tau_a -------------------------------
    import diptest
    if len(tau_a) >= 4:
        dip, p = diptest.diptest(tau_a)
        out["S1_dip_statistic"] = float(dip)
        out["S1_dip_p"] = float(p)
    else:
        out["S1_dip_statistic"] = None
        out["S1_dip_p"] = None
        out["S1_note"] = "fewer than 4 scored areas"

    # ---- S1 corroborating: 2-means silhouette on the role vector ---------
    cols = ["tau_mean", "gini_indegree"] + (
        ["transit_centrality_pct"] if cent_pct is not None else []
    )
    V = np.array([[r[c] for c in cols] for r in rows], dtype=np.float64)
    if len(V) >= 4 and np.isfinite(V).all():
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(V)
        out["S1_silhouette_2means"] = float(silhouette_score(V, km.labels_))
        out["S1_silhouette_cols"] = cols
        out["S1_cluster_sizes"] = np.bincount(km.labels_, minlength=2).tolist()
    else:
        out["S1_silhouette_2means"] = None

    # ---- S1 exploratory (NOT the registered statistic) -------------------
    sub = tau_row[rng.choice(len(tau_row), size=min(20000, len(tau_row)),
                             replace=False)]
    d_r, p_r = diptest.diptest(sub)
    out["exploratory_dip_per_row_tau"] = {"dip": float(d_r), "p": float(p_r),
                                          "n_sampled": int(len(sub)),
                                          "label": "EXPLORATORY, not registered"}

    # ---- S2 directed asymmetry ------------------------------------------
    F = flow_matrix(idx, labels, keep, k)
    num = np.linalg.norm(F - F.T)
    out["S2_directed_asymmetry"] = float(num / max(np.linalg.norm(F), 1e-12))

    # ---- S3 plug-in MI (declared context only) --------------------------
    pj = F / max(F.sum(), 1e-12)
    pr, pc = pj.sum(1, keepdims=True), pj.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = pj * np.log2(pj / (pr * pc))
    out["S3_mi_bits_declared_context_only"] = float(np.nansum(t))
    out["S3_H_area_bits"] = float(-np.nansum(pr * np.log2(pr)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--vectors", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-min", type=int, default=2000, dest="n_min")
    ap.add_argument("--q-min", type=int, default=500, dest="q_min")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--permute-labels", action="store_true", dest="permute")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = analyse(a.ids, a.labels, a.vectors, a.k, a.n_min, a.q_min, a.seed,
                  a.tag, a.permute)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"[{a.tag}] areas={res['areas_scored']} abstain={len(res['areas_abstain'])} "
          f"tau_bar={res['tau_mean_overall']:.4f}")
    print(f"  S1 dip={res['S1_dip_statistic']} p={res['S1_dip_p']} "
          f"silhouette={res['S1_silhouette_2means']}")
    print(f"  S2 A={res['S2_directed_asymmetry']:.4f}   "
          f"S3 MI={res['S3_mi_bits_declared_context_only']:.4f} bits "
          f"(H={res['S3_H_area_bits']:.4f})")
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Corrected role-differentiation statistics — PREREG Amendment 1, blob 29211dc.

E    excess transit  e_a = tau_a - mean over R label permutations of tau_a.
                     Permutation preserves every area's size, so the null
                     absorbs the size confound exactly (the defect that fired
                     P4: raw tau_a tends to 1 - n_a/N under permutation).
S1'  PRIMARY  T = 1 - SS_within(2-means on {e_a}) / SS_total, p from the same
                     R replicates. Exact under the permutation null at ANY
                     number of areas -- P3's non-resolution at 7 areas was a
                     power failure of an asymptotic test, not a null problem.
S1'b SECONDARY Hartigan dip on {e_a}: descriptive only, never a gate.
S2'  A = ||F - F^T||_F / ||F||_F with a permutation p-value.
S3   plug-in MI, declared context only (near-monotone with tau_bar).

ABSTAINs with registered cause `too_few_areas` below --min-areas eligible
areas. n_min/q_min are inherited protocol and are not loosened here.

Permuting the label vector is implemented as permuting the integer area code
(including ineligible areas), which is the same operation and O(n) per
replicate rather than O(areas x n).
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np


def sha256(path: str, cap: int = 1 << 26) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
            if f.tell() > cap:
                return h.hexdigest() + f"+prefix{cap}"
    return h.hexdigest()


def two_means_T(e: np.ndarray) -> float:
    """Exact 1-D 2-means separation in [0,1]: every sorted split point is tried,
    which is optimal in one dimension, so there is no initialisation to botch."""
    x = np.sort(np.asarray(e, dtype=np.float64))
    m = len(x)
    ss_tot = float(((x - x.mean()) ** 2).sum())
    if m < 4 or ss_tot <= 0.0:
        return 0.0
    c1, c2 = np.cumsum(x), np.cumsum(x * x)
    i = np.arange(1, m, dtype=np.float64)
    s1, q1 = c1[:-1], c2[:-1]
    s2, q2 = c1[-1] - s1, c2[-1] - q1
    n2 = m - i
    ss = (q1 - s1 * s1 / i) + (q2 - s2 * s2 / n2)
    return float(1.0 - ss.min() / ss_tot)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-min", type=int, default=2000, dest="n_min")
    ap.add_argument("--min-areas", type=int, default=5, dest="min_areas")
    ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--permute-labels", action="store_true", dest="permute")
    a = ap.parse_args()

    idx = np.load(a.ids)[:, : a.k]
    lab = np.array([ln.strip() for ln in open(a.labels, encoding="utf-8")])
    if len(lab) != len(idx):
        raise SystemExit(f"labels {len(lab)} != ids {len(idx)}")
    rng = np.random.default_rng(a.seed)
    if a.permute:
        lab = rng.permutation(lab)

    n = len(lab)
    all_areas, full = np.unique(lab, return_inverse=True)   # full: (n,) int codes
    nA = len(all_areas)
    area_n = np.bincount(full, minlength=nA)
    elig = np.where(area_n >= a.n_min)[0]
    rank_of = np.full(nA, -1, dtype=np.int64)
    rank_of[elig] = np.arange(len(elig))
    m = len(elig)

    out: dict = {
        "tag": a.tag,
        "provenance": {
            "prereg_blob": "29211dc95659f80a18a4117339ba7f1806612c45",
            "amendment": 1,
            "ids": a.ids.split("/")[-1], "ids_sha256": sha256(a.ids),
            "labels": a.labels.split("/")[-1], "labels_sha256": sha256(a.labels),
            "k": a.k, "n_min": a.n_min, "R": a.R, "seed": a.seed,
            "labels_permuted": bool(a.permute), "n_rows": int(n),
        },
        "areas_scored": int(m),
        "areas": [str(all_areas[i]) for i in elig],
        "area_n": [int(area_n[i]) for i in elig],
        "areas_abstain": [str(all_areas[i]) for i in range(nA) if rank_of[i] < 0],
    }
    if m < a.min_areas:
        out["verdict"] = "ABSTAIN"
        out["cause"] = "too_few_areas"
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"[{a.tag}] ABSTAIN too_few_areas ({m} < {a.min_areas})")
        return

    src = np.repeat(np.arange(n, dtype=np.int64), a.k)
    dst = idx.ravel().astype(np.int64)
    counts = np.bincount(dst, minlength=n).astype(np.float64)
    denom = np.maximum(counts, 1.0)

    def stats(fc: np.ndarray) -> tuple[np.ndarray, float]:
        """(per-area mean transit, flow asymmetry) for one area-code labelling."""
        same = (fc[src] == fc[dst]).astype(np.float64)
        intra = np.bincount(dst, weights=same, minlength=n)
        tau = (counts - intra) / denom
        cd = rank_of[fc]
        ok = cd >= 0
        s = np.bincount(cd[ok], weights=tau[ok], minlength=m)
        c = np.bincount(cd[ok], minlength=m).astype(np.float64)
        eok = (cd[src] >= 0) & (cd[dst] >= 0)
        F = np.bincount(cd[src[eok]] * m + cd[dst[eok]],
                        minlength=m * m).astype(np.float64).reshape(m, m)
        F /= max(F.sum(), 1.0)
        A = float(np.linalg.norm(F - F.T) / max(np.linalg.norm(F), 1e-12))
        return s / np.maximum(c, 1.0), A

    tau_obs, A_obs = stats(full)

    prng = np.random.default_rng(a.seed + 1000)
    taus = np.empty((a.R, m), dtype=np.float64)
    A_null = np.empty(a.R, dtype=np.float64)
    for r in range(a.R):
        taus[r], A_null[r] = stats(prng.permutation(full))
        if (r + 1) % 50 == 0:
            print(f"  replicate {r + 1}/{a.R}", flush=True)

    S = taus.sum(0)
    e_obs = tau_obs - taus.mean(0)
    T_obs = two_means_T(e_obs)
    T_null = np.array([two_means_T(taus[r] - (S - taus[r]) / (a.R - 1))
                       for r in range(a.R)])
    p_T = (1.0 + int((T_null >= T_obs).sum())) / (a.R + 1.0)
    p_A = (1.0 + int((A_null >= A_obs).sum())) / (a.R + 1.0)

    import diptest
    dip, p_dip = diptest.diptest(e_obs) if m >= 4 else (float("nan"), float("nan"))

    cd = rank_of[full]
    eok = (cd[src] >= 0) & (cd[dst] >= 0)
    F = np.bincount(cd[src[eok]] * m + cd[dst[eok]],
                    minlength=m * m).astype(np.float64).reshape(m, m)
    pj = F / max(F.sum(), 1.0)
    pr, pc = pj.sum(1, keepdims=True), pj.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = float(np.nansum(pj * np.log2(pj / (pr * pc))))

    out.update({
        "verdict": "scored",
        "tau_area_observed": [round(float(v), 6) for v in tau_obs],
        "tau_area_null_mean": [round(float(v), 6) for v in taus.mean(0)],
        "excess_e_a": [round(float(v), 6) for v in e_obs],
        "mean_abs_excess": float(np.abs(e_obs).mean()),
        "mean_excess": float(e_obs.mean()),
        "S1p_T": T_obs, "S1p_p": p_T, "S1p_T_null_mean": float(T_null.mean()),
        "S1pb_dip": float(dip), "S1pb_dip_p": float(p_dip),
        "S2p_A": A_obs, "S2p_p": p_A, "S2p_A_null_mean": float(A_null.mean()),
        "S3_mi_bits_context_only": mi,
    })
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"[{a.tag}] areas={m}  mean|e_a|={out['mean_abs_excess']:.5f}  "
          f"mean e_a={out['mean_excess']:+.5f}")
    print(f"  S1' T={T_obs:.4f} p={p_T:.4f} (null mean {T_null.mean():.4f})   "
          f"dip={dip:.4f} p={p_dip:.3f}")
    print(f"  S2' A={A_obs:.4f} p={p_A:.4f} (null mean {A_null.mean():.4f})   "
          f"S3 MI={mi:.4f} bits")
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Directed-asymmetry chronology tests — PREREG blob 0b4de32 (frozen 2026-07-29).

Lambda  mass-weighted share of net flow running later -> earlier, over UNORDERED
        pairs of eligible strata with differing bins (unordered on both sides so
        Lambda lands in [-1, 1] as the prereg states).
D2      Kendall tau between sign(N_ab) and sign(bin_a - bin_b).
Null    CHRONOLOGY permutation: F held fixed, bins permuted AMONG strata
        (R=2000). Isolates the ordering from the partition -- the question the
        row-permutation null cannot ask.

F is row-normalized per the prereg: F_ab = share of area a's OUTGOING kNN edges
landing in b, so a small stratum's retrieval behaviour counts equally with a
large one's. N_ab = F_ab - F_ba.

  dirasym.py --ids I --labels L --tag T --out J [--order alpha]

--order alpha replaces the parsed century bin with an alphabetical rank of the
label, which is the T3 negative control: an arbitrary ordering carrying no
chronology.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np


def sha256(p: str, cap: int = 1 << 26) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(1 << 20):
            h.update(c)
            if f.tell() > cap:
                return h.hexdigest() + f"+prefix{cap}"
    return h.hexdigest()


def flow_N(code: np.ndarray, src: np.ndarray, dst: np.ndarray, m: int) -> np.ndarray:
    """Row-normalized flow, antisymmetrized: N = F - F.T."""
    ok = (code[src] >= 0) & (code[dst] >= 0)
    F = (
        np.bincount(code[src[ok]] * m + code[dst[ok]], minlength=m * m)
        .astype(np.float64)
        .reshape(m, m)
    )
    F /= np.maximum(F.sum(1, keepdims=True), 1e-12)  # row-normalize
    return F - F.T


def lam_tau(N: np.ndarray, bins: np.ndarray) -> tuple[float, float, int]:
    """Lambda and Kendall tau over unordered pairs with differing bins."""
    m = len(bins)
    iu, ju = np.triu_indices(m, k=1)
    d = bins[iu] - bins[ju]
    sel = d != 0
    iu, ju, d = iu[sel], ju[sel], d[sel]
    if len(iu) == 0:
        return float("nan"), float("nan"), 0
    n = N[iu, ju]  # net flow from iu -> ju
    # orient each pair so the FIRST element is the later stratum
    later_to_earlier = np.where(d > 0, n, -n)
    denom = np.abs(n).sum()
    lam = float(later_to_earlier.sum() / denom) if denom > 0 else float("nan")
    from scipy.stats import kendalltau

    kt = kendalltau(np.sign(n), np.sign(d))
    corr = kt.correlation if kt.correlation is not None else np.nan
    return lam, float(corr), int(len(iu))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-min", type=int, default=2000, dest="n_min")
    ap.add_argument("--R-chrono", type=int, default=2000, dest="Rc")
    ap.add_argument("--order", choices=["chronology", "alpha"], default="chronology")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    idx = np.load(a.ids)[:, : a.k]
    lab = np.array([ln.strip() for ln in open(a.labels, encoding="utf-8")])
    n = len(lab)
    areas, full = np.unique(lab, return_inverse=True)
    cnt = np.bincount(full, minlength=len(areas))
    elig = np.where(cnt >= a.n_min)[0]
    rank_of = np.full(len(areas), -1, dtype=np.int64)
    rank_of[elig] = np.arange(len(elig))
    m = len(elig)
    names = [str(areas[i]) for i in elig]

    if a.order == "chronology":
        bins = np.array([int(s.rsplit("|", 1)[1]) for s in names], dtype=np.float64)
    else:
        bins = np.argsort(np.argsort(names)).astype(np.float64)  # alphabetical rank

    src = np.repeat(np.arange(n, dtype=np.int64), a.k)
    dst = idx.ravel().astype(np.int64)
    code = rank_of[full]
    N = flow_N(code, src, dst, m)
    lam, kt, npairs = lam_tau(N, bins)

    # ---- chronology-permutation null: F fixed, bins permuted among strata ----
    rng = np.random.default_rng(a.seed + 99)
    lam_null = np.empty(a.Rc)
    kt_null = np.empty(a.Rc)
    for r in range(a.Rc):
        lam_null[r], kt_null[r], _ = lam_tau(N, rng.permutation(bins))
    finite = np.isfinite(lam_null)
    p_lam_two = (1.0 + int((np.abs(lam_null[finite]) >= abs(lam)).sum())) / (
        finite.sum() + 1.0
    )
    p_lam_one = (1.0 + int((lam_null[finite] >= lam).sum())) / (finite.sum() + 1.0)

    out = {
        "tag": a.tag,
        "provenance": {
            "prereg_blob": "0b4de32b7bdf23b9e4d347d67ead195f4aa609d3",
            "ids": a.ids.split("/")[-1],
            "ids_sha256": sha256(a.ids),
            "labels": a.labels.split("/")[-1],
            "labels_sha256": sha256(a.labels),
            "k": a.k,
            "n_min": a.n_min,
            "R_chrono": a.Rc,
            "seed": a.seed,
            "order": a.order,
            "F_normalization": "row",
        },
        "areas": names,
        "bins": bins.tolist(),
        "n_pairs_differing_bin": npairs,
        "Lambda": lam,
        "kendall_tau": kt,
        "Lambda_null_mean": float(np.nanmean(lam_null)),
        "Lambda_null_sd": float(np.nanstd(lam_null)),
        "p_Lambda_one_sided_positive": p_lam_one,
        "p_Lambda_two_sided": p_lam_two,
        "N_matrix": [[round(float(v), 8) for v in row] for row in N],
    }
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"[{a.tag}] areas={m} pairs={npairs} order={a.order}")
    print(
        f"  Lambda={lam:+.4f}  (null mean {out['Lambda_null_mean']:+.4f} "
        f"sd {out['Lambda_null_sd']:.4f})"
    )
    print(f"  p(one-sided, Lambda>0)={p_lam_one:.4f}   p(two-sided)={p_lam_two:.4f}")
    print(f"  Kendall tau(sign N, sign dbin)={kt:+.4f}")
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()

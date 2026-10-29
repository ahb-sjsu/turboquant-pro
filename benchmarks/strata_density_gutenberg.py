"""Gutenberg replication of the density-direction test for STRATA retrieval flow.

Registered in docs/PREREG_density_asymmetry_gutenberg.md (frozen before any statistic below was
computed). Statistics reuse strata_density_asym.py.

Two separate invocations, in this order:
  gate   s(x) permuted over rows (true labels and flow kept), P1 recomputed with 2,000 orderings,
         200 replicates per arm. Writes rejection counts only.
  score  Refuses to run unless the gate file exists and stayed quiet.

Usage (Atlas, agi-hpc venv, cores 12-23):
  python strata_density_gutenberg.py gate  OUT_GATE.json
  python strata_density_gutenberg.py score OUT_GATE.json OUT_SCORE.json
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys

import numpy as np
from scipy.stats import spearmanr
from strata_density_asym import (
    border_mask,
    holm,
    lam_order,
    r2_continuous,
    raw_counts,
    row_perm_lambda,
    sha256,
    strat_median,
)

ROOT = "/archive/tqp_strata/gutenberg_pair"
CATALOG = "/archive/gutenberg/pg_catalog.csv"
CACHE = "/archive/experiments/hubness-density-ratio"
ARMS = {
    "G1": ("gut_bge_m3.npy", "exact_ids_bge.npy"),
    "G2": ("gut_labse.npy", "exact_ids_labse.npy"),
}
K, N_MIN = 10, 2000
R_ORDER = 20_000  # orderings for scoring
R_ORDER_GATE = 2_000  # orderings inside each gate replicate
R_ROW = 200  # gate replicates and P2 reference partitions
FIRE_COUNT = 17
SEED_ORDER, SEED_REF, SEED_GATE = 20261101, 20261102, 20261103
BIN_YEARS = 50


def eligible_code(lab):
    areas, full = np.unique(lab, return_inverse=True)
    cnt = np.bincount(full, minlength=len(areas))
    elig = np.where(cnt >= N_MIN)[0]
    rank_of = np.full(len(areas), -1, dtype=np.int64)
    rank_of[elig] = np.arange(len(elig))
    return rank_of[full], [str(areas[i]) for i in elig], cnt[elig]


def load_language():
    lab = np.array([ln.strip() for ln in open(os.path.join(ROOT, "labels.txt"))])
    return eligible_code(lab)


def load_books():
    books = []
    with open(os.path.join(ROOT, "passages.jsonl"), encoding="utf-8") as f:
        for ln in f:
            books.append(json.loads(ln)["book"])
    return np.array(books)


def birth_year(authors):
    """Birth year of the first author with life dates; BCE years negative. None if absent."""
    m = re.search(r"(\d{1,4})\??( BCE)?\s*-", authors)
    if not m:
        return None
    y = int(m.group(1))
    return -y if m.group(2) else y


def load_date_map(books):
    cat = {}
    with open(CATALOG, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cat[r["Text#"].strip()] = r.get("Authors", "")
    lab = []
    for b in books:
        y = birth_year(cat.get(re.sub(r"\D", "", b.split(".")[0]), ""))
        lab.append("none" if y is None else str((y // BIN_YEARS) * BIN_YEARS))
    code, names, cnt = eligible_code(np.array(lab))
    keep = [i for i, s in enumerate(names) if s != "none"]
    remap = np.full(len(names), -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    code = np.where(code >= 0, remap[np.maximum(code, 0)], -1)
    return code, [names[i] for i in keep], cnt[keep]


def edges(arm):
    idx = np.load(os.path.join(ROOT, ARMS[arm][1]))[:, :K]
    src = np.repeat(np.arange(len(idx), dtype=np.int64), K)
    return idx, src, idx.ravel().astype(np.int64)


def cosine_to(X, cols):
    s = np.full(len(cols), np.nan)
    ok = np.where(cols >= 0)[0]
    for i in range(0, len(ok), 20000):
        r = ok[i : i + 20000]
        a = np.asarray(X[r], dtype=np.float32)
        b = np.asarray(X[cols[r]], dtype=np.float32)
        s[r] = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return s


def density(arm, idx, books=None):
    """s(x) to the last stored neighbour; with books, the last one outside the row's book."""
    tag = "" if books is None else "_xbook"
    path = os.path.join(CACHE, f"sdens_gut_{arm}{tag}.npy")
    if os.path.exists(path):
        return np.load(path)
    if books is None:
        target = idx[:, K - 1].astype(np.int64)
    else:
        bid = np.unique(books, return_inverse=True)[1]
        other = bid[idx] != bid[:, None]
        last = K - 1 - np.argmax(other[:, ::-1], axis=1)
        target = np.where(other.any(1), idx[np.arange(len(idx)), last], -1)
    s = cosine_to(np.load(os.path.join(ROOT, ARMS[arm][0]), mmap_mode="r"), target)
    np.save(path, s)
    return s


def nan_median(s, code, m, mask=None):
    keep = (code >= 0) & np.isfinite(s)
    if mask is not None:
        keep &= mask
    return np.array(
        [
            np.median(s[keep & (code == a)]) if (keep & (code == a)).any() else np.nan
            for a in range(m)
        ]
    )


def ordering_test(C, vals, R, rng):
    """Lambda on the strata with finite vals; one-sided upper and lower p from R orderings."""
    ok = np.where(np.isfinite(vals))[0]
    C, vals = C[np.ix_(ok, ok)], vals[ok]
    obs = lam_order(C, vals[None])[0]
    perms = np.array([rng.permutation(len(vals)) for _ in range(R)])
    null = lam_order(C, vals[perms])
    return {
        "Lambda": float(obs),
        "p_upper": float((1 + (null >= obs - 1e-12).sum()) / (R + 1)),
        "p_lower": float((1 + (null <= obs + 1e-12).sum()) / (R + 1)),
        "strata_used": int(len(ok)),
    }


def gate(out_path):
    rng = np.random.default_rng(SEED_GATE)
    code, names, _ = load_language()
    out = {"protocol": __doc__, "calibration": "gutenberg-gate", "G": {}, "inputs": {}}
    for arm in ARMS:
        out["inputs"][arm] = {f: sha256(os.path.join(ROOT, f)) for f in ARMS[arm]}
        idx, src, dst = edges(arm)
        s = density(arm, idx)
        C = raw_counts(code, src, dst, len(names))
        rej = 0
        for _ in range(R_ROW):
            rho = strat_median(s[rng.permutation(len(s))], code, len(names))
            rej += ordering_test(C, rho, R_ORDER_GATE, rng)["p_upper"] < 0.05
        out["G"][arm] = {"P1_rejections": int(rej), "of": R_ROW}
        print(arm, "gate", out["G"][arm], flush=True)
    out["fired"] = any(v["P1_rejections"] > FIRE_COUNT for v in out["G"].values())
    json.dump(out, open(out_path, "w"), indent=1)
    print("fired", out["fired"], "DONE", flush=True)


def score(gate_path, out_path):
    g = json.load(open(gate_path))
    if g.get("calibration") != "gutenberg-gate" or g.get("fired", True):
        sys.exit(
            "Gate missing, wrong type, or fired: scoring is blocked (prereg section 4)."
        )
    r_ord = np.random.default_rng(SEED_ORDER)
    r_ref = np.random.default_rng(SEED_REF)
    code, names, size = load_language()
    books = load_books()
    dcode, dnames, dsize = load_date_map(books)
    transit = json.load(open(os.path.join(ROOT, "p2_score.json")))["arms"]
    out = {"gate": gate_path, "arms": {}}
    for arm in ARMS:
        idx, src, dst = edges(arm)
        s = density(arm, idx)
        C = raw_counts(code, src, dst, len(names))
        rho = strat_median(s, code, len(names))
        p1 = ordering_test(C, rho, R_ORDER, r_ord)
        ref = row_perm_lambda(code, s, src, dst, len(names), R_ROW, r_ref)
        p2 = float((1 + (ref >= p1["Lambda"]).sum()) / (R_ROW + 1))
        rho_b = nan_median(s, code, len(names), border_mask(code, src, dst, len(code)))
        sx = density(arm, idx, books)
        rho_x = nan_median(sx, code, len(names))
        bid = np.unique(books, return_inverse=True)[1]
        same = (bid[idx] == bid[:, None]).mean(1)
        Cd = raw_counts(dcode, src, dst, len(dnames))
        rho_d = strat_median(s, dcode, len(dnames))
        dbin = np.array([float(x) for x in dnames])
        key = {"G1": "bge_m3", "G2": "labse"}[arm]
        tm = transit.get(key, {}).get("transit_mass_by_language", {})
        out["arms"][arm] = {
            "areas": names,
            "size": size.tolist(),
            "rho": rho.tolist(),
            "C": C.tolist(),
            "P1": p1,
            "P2": {
                "p": p2,
                "ref_mean": float(ref.mean()),
                "ref_sd": float(ref.std()),
            },
            "Lambda_n": ordering_test(C, size.astype(np.float64), R_ORDER, r_ord),
            "spearman_rho_size": float(spearmanr(rho, size).correlation),
            "R2": r2_continuous(C, rho),
            "border": {
                "rho": rho_b.tolist(),
                "dropped": [names[i] for i in np.where(~np.isfinite(rho_b))[0]],
                **ordering_test(C, rho_b, R_ORDER, r_ord),
            },
            "same_book_share": [
                float(same[code == a].mean()) for a in range(len(names))
            ],
            "xbook": {
                "rho": rho_x.tolist(),
                "rows_without_xbook_neighbour": int((~np.isfinite(sx)).sum()),
                **ordering_test(C, rho_x, R_ORDER, r_ord),
            },
            "date_map": {
                "bins": dnames,
                "size": dsize.tolist(),
                "rho": rho_d.tolist(),
                "Lambda_rho": ordering_test(Cd, rho_d, R_ORDER, r_ord),
                "Lambda_chron_later_to_earlier": ordering_test(
                    Cd, -dbin, R_ORDER, r_ord
                ),
            },
            "transit_mass": [tm.get(n) for n in names],
        }
        print(arm, json.dumps({"P1": p1, "P2": p2}), flush=True)
    pp1 = max(out["arms"][a]["P1"]["p_upper"] for a in ARMS)
    pp2 = max(out["arms"][a]["P2"]["p"] for a in ARMS)
    adj = holm([pp1, pp2])
    out["family"] = {
        "p": {"P1": pp1, "P2": pp2},
        "holm": {"P1": adj[0], "P2": adj[1]},
        "P1_pass": all(out["arms"][a]["P1"]["Lambda"] > 0 for a in ARMS)
        and adj[0] < 0.05,
        "P2_pass": adj[1] < 0.05,
    }
    json.dump(out, open(out_path, "w"), indent=1)
    print(json.dumps(out["family"], indent=1), "\nDONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "gate":
        gate(sys.argv[2])
    elif sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)

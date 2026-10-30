"""Do the two Gutenberg encoders disagree about which language is denser, beyond book noise?

Registered in docs/PREREG_density_reversal_gutenberg.md (frozen before any resampled statistic
below was computed). Densities and loaders are the frozen Gutenberg harness's
(strata_density_gutenberg.py); this file adds the book bootstrap and the clear-reversal rule.

Two separate invocations, in this order:
  gate   Same-channel null: per encoder, 20 random half-splits of each language's books, the
         halves compared as if they were two encoders. Writes counts of splits with a clear
         reversal only.
  score  Refuses to run unless the gate file exists and stayed quiet.

Usage (Atlas, agi-hpc venv, cores 12-23):
  python strata_density_reversal.py gate  OUT_GATE.json
  python strata_density_reversal.py score OUT_GATE.json OUT_SCORE.json
"""

from __future__ import annotations

import json
import os
import sys
from itertools import combinations

import numpy as np
from scipy.stats import kendalltau, norm
from strata_density_asym import border_mask, raw_counts, sha256
from strata_density_gutenberg import (
    ARMS,
    CACHE,
    density,
    edges,
    load_books,
    load_language,
    nan_median,
)

B = 2_000  # bootstrap draws
N_SPLIT = 20  # gate half-splits per encoder
FIRE_SPLITS = 3  # gate fires at this many splits with a clear reversal
Z_STAR = float(norm.ppf(1 - 0.05 / 156))  # two-sided Bonferroni over 78 pairs
N2_MIN = 10
N3_MAX = 0.5
CHUNK = 100
SEED_SCORE = 20261104
SEED_GATE = 20261105


def lang_data(s, code, bid, m, mask=None):
    """Per language: finite values sorted, their book (local id), and the language's book count."""
    out = []
    for a in range(m):
        sel = code == a
        books_a, local = np.unique(bid[sel], return_inverse=True)
        keep = np.isfinite(s[sel])
        if mask is not None:
            keep &= mask[sel]
        v, b = s[sel][keep], local[keep]
        o = np.argsort(v, kind="stable")
        out.append((v[o], b[o], len(books_a)))
    return out


def weighted_median(v, b, cnt):
    """np.median of the multiset where row i appears cnt[:, b[i]] times; one value per draw."""
    w = cnt[:, b]
    cum = w.cumsum(1)
    t = cum[:, -1]
    k1, k2 = (t - 1) // 2, t // 2
    i1 = (cum >= (k1 + 1)[:, None]).argmax(1)
    i2 = (cum >= (k2 + 1)[:, None]).argmax(1)
    return 0.5 * (v[i1] + v[i2])


def draw_counts(rng, nb, size):
    return rng.multinomial(nb, np.full(nb, 1.0 / nb), size=size).astype(np.int32)


def boot_medians(datas, nbooks, rng, nboot):
    """Medians per draw for each dataset in datas (paired: one book draw per language serves all)."""
    m = len(nbooks)
    res = [np.empty((nboot, m)) for _ in datas]
    for c0 in range(0, nboot, CHUNK):
        n = min(CHUNK, nboot - c0)
        for a in range(m):
            cnt = draw_counts(rng, nbooks[a], n)
            for r, d in zip(res, datas):
                v, b, _ = d[a]
                r[c0 : c0 + n, a] = weighted_median(v, b, cnt) if len(v) else np.nan
    return res


def pair_table(rho1, rho2, bm1, bm2):
    """Per pair: differences, SEs, clear signs, clear reversal."""
    rows = []
    for a, b in combinations(range(len(rho1)), 2):
        d1, d2 = rho1[a] - rho1[b], rho2[a] - rho2[b]
        se1 = float(np.nanstd(bm1[:, a] - bm1[:, b], ddof=1))
        se2 = float(np.nanstd(bm2[:, a] - bm2[:, b], ddof=1))
        c1 = bool(np.isfinite(d1) and abs(d1) > Z_STAR * se1)
        c2 = bool(np.isfinite(d2) and abs(d2) > Z_STAR * se2)
        rows.append(
            {
                "a": a,
                "b": b,
                "d": [float(d1), float(d2)],
                "se": [se1, se2],
                "z": [float(d1 / se1), float(d2 / se2)],
                "clear": [c1, c2],
                "clear_reversal": bool(c1 and c2 and np.sign(d1) != np.sign(d2)),
            }
        )
    return rows


def tau_ci(rho1, rho2, bm1, bm2):
    ok = np.isfinite(rho1) & np.isfinite(rho2)
    taus = np.array([kendalltau(x[ok], y[ok])[0] for x, y in zip(bm1, bm2)])
    return {
        "tau": float(kendalltau(rho1[ok], rho2[ok])[0]),
        "ci95": [
            float(np.nanpercentile(taus, 2.5)),
            float(np.nanpercentile(taus, 97.5)),
        ],
    }


def load_all():
    code, names, size = load_language()
    books = load_books()
    bid = np.unique(books, return_inverse=True)[1]
    arms = {}
    for arm in ARMS:
        idx, src, dst = edges(arm)
        arms[arm] = {
            "s": density(arm, idx),
            "sx": density(arm, idx, books),
            "border": border_mask(code, src, dst, len(code)),
            "C": raw_counts(code, src, dst, len(names)),
        }
    return code, names, size, bid, arms


def inputs_hash():
    return {
        f: sha256(os.path.join(CACHE, f))
        for f in [f"sdens_gut_{a}{t}.npy" for a in ARMS for t in ("", "_xbook")]
    }


def gate(out_path):
    rng = np.random.default_rng(SEED_GATE)
    code, names, _, bid, arms = load_all()
    m = len(names)
    out = {
        "protocol": __doc__,
        "calibration": "reversal-gate",
        "inputs": inputs_hash(),
        "G": {},
    }
    for arm in ARMS:
        s = arms[arm]["s"]
        hits, counts = 0, []
        for _ in range(N_SPLIT):
            half = np.zeros(len(code), dtype=bool)
            for a in range(m):
                ub = np.unique(bid[code == a])
                pick = rng.permutation(ub)[: len(ub) // 2]
                half |= (code == a) & np.isin(bid, pick)
            rho, bms = [], []
            for h in (half, ~half & (code >= 0)):
                c = np.where(h, code, -1)
                d = lang_data(s, c, bid, m)
                rho.append(nan_median(s, c, m))
                bms.append(boot_medians([d], [x[2] for x in d], rng, B)[0])
            n_rev = sum(r["clear_reversal"] for r in pair_table(rho[0], rho[1], *bms))
            counts.append(int(n_rev))
            hits += n_rev > 0
            print(arm, "split", len(counts), "clear reversals", n_rev, flush=True)
        out["G"][arm] = {
            "splits_with_clear_reversal": int(hits),
            "of": N_SPLIT,
            "counts": counts,
        }
    out["fired"] = any(
        v["splits_with_clear_reversal"] >= FIRE_SPLITS for v in out["G"].values()
    )
    json.dump(out, open(out_path, "w"), indent=1)
    print("fired", out["fired"], "DONE", flush=True)


def variant(code, bid, m, arms, key, rng):
    """rho, bootstrap medians and pair table for one density variant (paired over encoders)."""
    s1, s2 = (arms[a]["sx" if key == "xbook" else "s"] for a in ARMS)
    k1, k2 = ((arms[a]["border"] if key == "border" else None) for a in ARMS)
    d1, d2 = lang_data(s1, code, bid, m, k1), lang_data(s2, code, bid, m, k2)
    rho1, rho2 = nan_median(s1, code, m, k1), nan_median(s2, code, m, k2)
    bm1, bm2 = boot_medians([d1, d2], [x[2] for x in d1], rng, B)
    return rho1, rho2, bm1, bm2, pair_table(rho1, rho2, bm1, bm2)


def score(gate_path, out_path):
    g = json.load(open(gate_path))
    if g.get("calibration") != "reversal-gate" or g.get("fired", True):
        sys.exit(
            "Gate missing, wrong type, or fired: scoring is blocked (prereg section 4)."
        )
    rng = np.random.default_rng(SEED_SCORE)
    code, names, size, bid, arms = load_all()
    m = len(names)
    out = {"gate": gate_path, "inputs": inputs_hash(), "areas": names, "z_star": Z_STAR}
    rho1, rho2, bm1, bm2, pt = variant(code, bid, m, arms, "rho", rng)
    nrev = sum(r["clear_reversal"] for r in pt)
    tau = tau_ci(rho1, rho2, bm1, bm2)
    C1, C2 = (arms[a]["C"] for a in ARMS)
    flow = []
    for r in pt:
        if not r["clear_reversal"]:
            continue
        a, b = r["a"], r["b"]
        net = [float(C[a, b] - C[b, a]) for C in (C1, C2)]
        toward = [bool(np.sign(n) == np.sign(-d)) for n, d in zip(net, r["d"])]
        flow.append(
            {"pair": [names[a], names[b]], "net_a_to_b": net, "toward_denser": toward}
        )
    anomaly = []
    for r in pt:
        a, b = r["a"], r["b"]
        if np.sign(r["d"][0]) * np.sign(r["d"][1]) >= 0:
            continue
        net = float(C1[a, b] - C1[b, a])
        if np.sign(net) != np.sign(-r["d"][0]):
            anomaly.append({"pair": [names[a], names[b]], "G1_net": net, "z": r["z"]})
    out["registered"] = {
        "pairs": [{**r, "pair": [names[r["a"]], names[r["b"]]]} for r in pt],
        "rho": [rho1.tolist(), rho2.tolist()],
        "se_language": [bm1.std(0, ddof=1).tolist(), bm2.std(0, ddof=1).tolist()],
        "clear_reversals": int(nrev),
        "tau": tau,
        "N1_pass": bool(nrev >= 1),
        "N2_pass": bool(nrev >= N2_MIN),
        "N3_pass": bool(tau["ci95"][1] < N3_MAX),
    }
    print(
        "registered",
        json.dumps(
            {
                k: v
                for k, v in out["registered"].items()
                if k not in ("pairs", "rho", "se_language")
            }
        ),
        flush=True,
    )
    out["descriptive"] = {
        "flow_on_clear_reversals": flow,
        "G1_against_own_density_on_reversed": anomaly,
    }
    for key in ("xbook", "border"):
        r1, r2, b1, b2, p = variant(code, bid, m, arms, key, rng)
        out["descriptive"][key] = {
            "clear_reversals": int(sum(r["clear_reversal"] for r in p)),
            "point_reversals": int(
                sum(np.sign(r["d"][0]) * np.sign(r["d"][1]) < 0 for r in p)
            ),
            "tau": tau_ci(r1, r2, b1, b2),
        }
        print(key, json.dumps(out["descriptive"][key]), flush=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "gate":
        gate(sys.argv[2])
    elif sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)

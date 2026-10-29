"""Joint density / chronology scoring of STRATA directed asymmetry.

Registered in docs/PREREG_density_asymmetry.md (frozen before any statistic below was computed).
The chronology tests T1-T3 reuse the frozen harness strata_dirasym.py unchanged.

Two separate invocations, in this order:
  gate   G1 = the chronology prereg's T3 (language map, alphabetical ordering).
         G2 = calibration of every new test on row-permuted area codes.
         Writes no statistic computed on the true labels except T3's own.
  score  Refuses to run unless the gate file exists and G2 stayed quiet. Skips T1/T2 if G1 fired.

Usage (Atlas, agi-hpc venv, cores 12-23):
  python strata_density_asym.py gate    OUT_GATE.json   (original gate; fired, see Amendment 1)
  python strata_density_asym.py gate-a1 OUT_GATE.json   (Amendment 1 gate)
  python strata_density_asym.py score   OUT_GATE.json OUT_SCORE.json
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from strata_dirasym import flow_N, lam_tau

ROOT = "/archive/tqp_strata"
CACHE = "/archive/experiments/hubness-density-ratio"
ARMS = {
    "D1": ("sample_emb.npy", "exact_ids.npy"),
    "D2": ("ethics_labse.npy", "exact_ids_labse_ethics.npy"),
}
MAPS = {"language": "labels.txt", "tradcent": "labels_tradcent.txt"}
K, N_MIN = 10, 2000
R_ROW = 200  # row-permutation replicates (G2 and the T2 null)
R_CHRONO = 2000  # frozen harness value for T1 and T3
R_J = 5000  # ordering permutations for the joint regression
R_J_GATE = 1000  # same, inside each G2 replicate
FIRE_COUNT = 17  # G2 fires if more than 17 of 200 replicates reject at 0.05
SEED = 20261028


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(1 << 22):
            h.update(c)
    return h.hexdigest()


def load_map(name):
    lab = np.array(
        [ln.strip() for ln in open(os.path.join(ROOT, MAPS[name]), encoding="utf-8")]
    )
    areas, full = np.unique(lab, return_inverse=True)
    cnt = np.bincount(full, minlength=len(areas))
    elig = np.where(cnt >= N_MIN)[0]
    rank_of = np.full(len(areas), -1, dtype=np.int64)
    rank_of[elig] = np.arange(len(elig))
    return rank_of[full], [str(areas[i]) for i in elig], cnt[elig]


def density(arm):
    """s(x) = cosine of row x to its last stored neighbour (column K-1). Cached per arm."""
    path = os.path.join(CACHE, f"sdens_{arm}.npy")
    if os.path.exists(path):
        return np.load(path)
    X = np.load(os.path.join(ROOT, ARMS[arm][0]), mmap_mode="r")
    last = np.load(os.path.join(ROOT, ARMS[arm][1]))[:, K - 1]
    s = np.empty(len(last), dtype=np.float64)
    for i in range(0, len(last), 20000):
        a = np.asarray(X[i : i + 20000], dtype=np.float32)
        b = np.asarray(X[np.sort(last[i : i + 20000])], dtype=np.float32)
        b = b[np.argsort(np.argsort(last[i : i + 20000]))]
        num = (a * b).sum(1)
        s[i : i + 20000] = num / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    np.save(path, s)
    return s


def edges(arm):
    idx = np.load(os.path.join(ROOT, ARMS[arm][1]))[:, :K]
    src = np.repeat(np.arange(len(idx), dtype=np.int64), K)
    return src, idx.ravel().astype(np.int64)


def raw_counts(code, src, dst, m):
    ok = (code[src] >= 0) & (code[dst] >= 0)
    return (
        np.bincount(code[src[ok]] * m + code[dst[ok]], minlength=m * m)
        .reshape(m, m)
        .astype(np.float64)
    )


def strat_median(s, code, m, mask=None):
    keep = code >= 0 if mask is None else (code >= 0) & mask
    return np.array([np.median(s[keep & (code == a)]) for a in range(m)])


def border_mask(code, src, dst, n):
    ok = (code[src] >= 0) & (code[dst] >= 0) & (code[src] != code[dst])
    return np.bincount(src[ok], minlength=n) > 0


def pairs(m):
    return np.triu_indices(m, k=1)


def lam_order(C, vals_perms):
    """Lambda for each row of vals_perms (orderings x m). Flow into the higher value counts positive."""
    iu, ju = pairs(C.shape[0])
    net = C[iu, ju] - C[ju, iu]
    sg = np.sign(vals_perms[:, ju] - vals_perms[:, iu])
    return (sg * net).sum(1) / np.abs(net).sum()


def exact_lambda(C, vals):
    perms = np.array(list(itertools.permutations(range(len(vals)))))
    null = lam_order(C, vals[perms])
    obs = lam_order(C, vals[None])[0]
    return float(obs), float((null >= obs - 1e-12).mean()), len(perms)


def sampled_lambda(C, vals, R, rng):
    perms = np.array([rng.permutation(len(vals)) for _ in range(R)])
    null = lam_order(C, vals[perms])
    obs = lam_order(C, vals[None])[0]
    return float(obs), float((1 + (null >= obs - 1e-12).sum()) / (R + 1))


def r2_continuous(C, rho):
    iu, ju = pairs(C.shape[0])
    w = C[iu, ju] + C[ju, iu]
    y = (C[iu, ju] - C[ju, iu]) / np.maximum(w, 1e-12)
    x = rho[ju] - rho[iu]
    beta = (w * x * y).sum() / (w * x * x).sum()
    return float(beta), float(1 - (w * (y - beta * x) ** 2).sum() / (w * y * y).sum())


def joint_fit(C, bins_perms, rho_perms):
    """Weighted LS through the origin of m_ab on [sgn(c_a - c_b), sgn(rho_b - rho_a)], vectorized."""
    iu, ju = pairs(C.shape[0])
    w = C[iu, ju] + C[ju, iu]
    y = (C[iu, ju] - C[ju, iu]) / np.maximum(w, 1e-12)
    x1 = np.sign(bins_perms[:, iu] - bins_perms[:, ju])
    x2 = np.sign(rho_perms[:, ju] - rho_perms[:, iu])
    a11, a22, a12 = (w * x1 * x1).sum(1), (w * x2 * x2).sum(1), (w * x1 * x2).sum(1)
    b1, b2 = (w * x1 * y).sum(1), (w * x2 * y).sum(1)
    det = a11 * a22 - a12 * a12
    return (a22 * b1 - a12 * b2) / det, (a11 * b2 - a12 * b1) / det


def joint_test(C, bins, rho, R, rng):
    m = len(bins)
    bc, br = joint_fit(C, bins[None], rho[None])
    pb = np.array([rng.permutation(m) for _ in range(R)])
    pr = np.array([rng.permutation(m) for _ in range(R)])
    nb, _ = joint_fit(C, bins[pb], np.repeat(rho[None], R, 0))
    _, nr = joint_fit(C, np.repeat(bins[None], R, 0), rho[pr])
    p_c = (1 + (nb >= bc[0] - 1e-12).sum()) / (R + 1)
    p_r = (1 + (nr >= br[0] - 1e-12).sum()) / (R + 1)
    return float(bc[0]), float(p_c), float(br[0]), float(p_r)


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        adj[j] = run
    return adj.tolist()


def tradcent_bins(names):
    return np.array([int(s.rsplit("|", 1)[1]) for s in names], dtype=np.float64)


def gate(out_path):
    rng = np.random.default_rng(SEED)
    out = {"protocol": __doc__, "G1_T3": {}, "G2": {}, "inputs": {}}
    lcode, lnames, _ = load_map("language")
    tcode, tnames, _ = load_map("tradcent")
    alpha = np.argsort(np.argsort(lnames)).astype(np.float64)
    tb = tradcent_bins(tnames)
    for arm in ARMS:
        out["inputs"][arm] = {f: sha256(os.path.join(ROOT, f)) for f in ARMS[arm][1:]}
        src, dst = edges(arm)
        s = density(arm)
        out["inputs"][arm]["sdens_sha256"] = sha256(
            os.path.join(CACHE, f"sdens_{arm}.npy")
        )
        # G1: the frozen T3, harness functions and harness null (R=2000, seed 7 + 99).
        N = flow_N(lcode, src, dst, len(lnames))
        lam, _, npairs = lam_tau(N, alpha)
        r3 = np.random.default_rng(7 + 99)
        null = np.array([lam_tau(N, r3.permutation(alpha))[0] for _ in range(R_CHRONO)])
        p_two = (1 + (np.abs(null) >= abs(lam)).sum()) / (R_CHRONO + 1)
        out["G1_T3"][arm] = {
            "Lambda_alpha": lam,
            "p_two_sided": float(p_two),
            "pairs": npairs,
        }
        print(arm, "G1", out["G1_T3"][arm], flush=True)
        # G2: every new test on row-permuted codes, both maps.
        rej = {"R_language_Lambda_rho": 0, "J_beta_chron": 0, "J_beta_rho": 0}
        for _ in range(R_ROW):
            pl = lcode[rng.permutation(len(lcode))]
            C = raw_counts(pl, src, dst, len(lnames))
            _, p, _ = exact_lambda(C, strat_median(s, pl, len(lnames)))
            rej["R_language_Lambda_rho"] += p < 0.05
            pt = tcode[rng.permutation(len(tcode))]
            C = raw_counts(pt, src, dst, len(tnames))
            _, pc, _, pr = joint_test(
                C, tb, strat_median(s, pt, len(tnames)), R_J_GATE, rng
            )
            rej["J_beta_chron"] += pc < 0.05
            rej["J_beta_rho"] += pr < 0.05
        out["G2"][arm] = {k: int(v) for k, v in rej.items()}
        print(arm, "G2 rejections of", R_ROW, out["G2"][arm], flush=True)
    out["G1_fired"] = any(v["p_two_sided"] < 0.05 for v in out["G1_T3"].values())
    out["G2_fired"] = any(
        c > FIRE_COUNT for v in out["G2"].values() for c in v.values()
    )
    json.dump(out, open(out_path, "w"), indent=1)
    print("G1_fired", out["G1_fired"], "G2_fired", out["G2_fired"], "DONE", flush=True)


def gate_a1(out_path):
    """Amendment 1 gate. G1 unchanged. G2 replaced by a null that breaks the row-density link.

    Density tests: s(x) is permuted over rows, true labels kept, so the flow is real and the
    densities carry no information about it. Chronology term: area codes permuted over rows,
    as before, since a random partition carries no chronology.
    """
    rng = np.random.default_rng(SEED + 2)
    out = {"protocol": gate_a1.__doc__, "calibration": "amendment-1"}
    out.update({"G1_T3": {}, "G2": {}, "inputs": {}})
    lcode, lnames, _ = load_map("language")
    tcode, tnames, _ = load_map("tradcent")
    alpha = np.argsort(np.argsort(lnames)).astype(np.float64)
    tb = tradcent_bins(tnames)
    for arm in ARMS:
        out["inputs"][arm] = {f: sha256(os.path.join(ROOT, f)) for f in ARMS[arm][1:]}
        src, dst = edges(arm)
        s = density(arm)
        N = flow_N(lcode, src, dst, len(lnames))
        lam, _, npairs = lam_tau(N, alpha)
        r3 = np.random.default_rng(7 + 99)
        null = np.array([lam_tau(N, r3.permutation(alpha))[0] for _ in range(R_CHRONO)])
        p_two = (1 + (np.abs(null) >= abs(lam)).sum()) / (R_CHRONO + 1)
        out["G1_T3"][arm] = {
            "Lambda_alpha": lam,
            "p_two_sided": float(p_two),
            "pairs": npairs,
        }
        print(arm, "G1", out["G1_T3"][arm], flush=True)
        Cl = raw_counts(lcode, src, dst, len(lnames))
        Ct = raw_counts(tcode, src, dst, len(tnames))
        rej = {"R_language_Lambda_rho": 0, "J_beta_chron": 0, "J_beta_rho": 0}
        for _ in range(R_ROW):
            sp = s[rng.permutation(len(s))]
            _, p, _ = exact_lambda(Cl, strat_median(sp, lcode, len(lnames)))
            rej["R_language_Lambda_rho"] += p < 0.05
            _, _, _, pr = joint_test(
                Ct, tb, strat_median(sp, tcode, len(tnames)), R_J_GATE, rng
            )
            rej["J_beta_rho"] += pr < 0.05
            pt = tcode[rng.permutation(len(tcode))]
            C = raw_counts(pt, src, dst, len(tnames))
            _, pc, _, _ = joint_test(
                C, tb, strat_median(s, pt, len(tnames)), R_J_GATE, rng
            )
            rej["J_beta_chron"] += pc < 0.05
        out["G2"][arm] = {k: int(v) for k, v in rej.items()}
        print(arm, "G2 (amendment 1) rejections of", R_ROW, out["G2"][arm], flush=True)
    out["G1_fired"] = any(v["p_two_sided"] < 0.05 for v in out["G1_T3"].values())
    out["G2_fired"] = any(
        c > FIRE_COUNT for v in out["G2"].values() for c in v.values()
    )
    json.dump(out, open(out_path, "w"), indent=1)
    print("G1_fired", out["G1_fired"], "G2_fired", out["G2_fired"], "DONE", flush=True)


def row_perm_lambda(code, s, src, dst, m, R, rng):
    """Lambda_rho on R random partitions with the true sizes: what row-level density alone gives."""
    vals = []
    for _ in range(R):
        pc = code[rng.permutation(len(code))]
        C = raw_counts(pc, src, dst, m)
        vals.append(lam_order(C, strat_median(s, pc, m)[None])[0])
    return np.array(vals)


def score(gate_path, out_path):
    g = json.load(open(gate_path))
    if g.get("calibration") != "amendment-1":
        sys.exit("Gate file is not an Amendment-1 gate (prereg Amendment 1).")
    if g.get("G2_fired", True):
        sys.exit(
            "G2 fired or gate file incomplete: R1/R2 and J are blocked (prereg section 5)."
        )
    rng = np.random.default_rng(SEED + 1)
    out = {"gate": gate_path, "arms": {}}
    lcode, lnames, lsize = load_map("language")
    tcode, tnames, tsize = load_map("tradcent")
    tb = tradcent_bins(tnames)
    iu, ju = pairs(len(tnames))
    dbin = tb[iu] != tb[ju]
    trad = np.array([s.rsplit("|", 1)[0] for s in tnames])
    Ns = {}
    for arm in ARMS:
        src, dst = edges(arm)
        s = density(arm)
        a = {}
        # ---- density, language map (R1/R2) ----
        C = raw_counts(lcode, src, dst, len(lnames))
        rho = strat_median(s, lcode, len(lnames))
        rho_b = strat_median(
            s, lcode, len(lnames), border_mask(lcode, src, dst, len(lcode))
        )
        lam, p, nperm = exact_lambda(C, rho)
        lam_n, p_n, _ = exact_lambda(C, lsize.astype(np.float64))
        lam_b, p_b, _ = exact_lambda(C, rho_b)
        beta, r2 = r2_continuous(C, rho)
        a["language"] = {
            "areas": lnames,
            "size": lsize.tolist(),
            "rho": rho.tolist(),
            "rho_border": rho_b.tolist(),
            "C": C.tolist(),
            "Lambda_rho": lam,
            "p_one_sided_exact": p,
            "orderings": nperm,
            "Lambda_n": lam_n,
            "p_Lambda_n": p_n,
            "Lambda_rho_border": lam_b,
            "p_Lambda_rho_border": p_b,
            "D_rho2_beta": beta,
            "D_rho2_R2": r2,
            "spearman_rho_size": float(spearmanr(rho, lsize).correlation),
        }
        rp = row_perm_lambda(lcode, s, src, dst, len(lnames), R_ROW, rng)
        a["language"]["row_level_Lambda_rho"] = {
            "mean": float(rp.mean()),
            "sd": float(rp.std()),
            "share_at_or_above_observed": float((rp >= lam).mean()),
        }
        # ---- chronology, tradcent map: T1 exactly as the frozen harness ----
        if not g["G1_fired"]:
            N = flow_N(tcode, src, dst, len(tnames))
            Ns[arm] = N
            lt, kt, npairs = lam_tau(N, tb)
            r1 = np.random.default_rng(7 + 99)
            null = np.array(
                [lam_tau(N, r1.permutation(tb))[0] for _ in range(R_CHRONO)]
            )
            p1 = (1 + (null >= lt).sum()) / (R_CHRONO + 1)
            within = (trad[iu] == trad[ju]) & dbin
            nw = N[iu, ju][within]
            lam_within = float(
                np.where((tb[iu] - tb[ju])[within] > 0, nw, -nw).sum()
                / np.abs(nw).sum()
            )
            a["T1"] = {
                "Lambda": lt,
                "kendall_tau": kt,
                "pairs": npairs,
                "p_one_sided": float(p1),
                "Lambda_within_tradition": lam_within,
                "within_pairs": int(within.sum()),
            }
        # ---- joint, tradcent map, raw counts ----
        Ct = raw_counts(tcode, src, dst, len(tnames))
        rho_t = strat_median(s, tcode, len(tnames))
        bc, pc, br, pr = joint_test(Ct, tb, rho_t, R_J, rng)
        net = Ct[iu, ju] - Ct[ju, iu]
        sel = dbin
        lam_chron_raw = float(
            np.where((tb[iu] - tb[ju])[sel] > 0, net[sel], -net[sel]).sum()
            / np.abs(net[sel]).sum()
        )
        lam_rt, p_rt = sampled_lambda(Ct, rho_t, R_J, rng)
        lam_nt, p_nt = sampled_lambda(Ct, tsize.astype(np.float64), R_J, rng)
        a["tradcent"] = {
            "areas": tnames,
            "bins": tb.tolist(),
            "size": tsize.tolist(),
            "rho": rho_t.tolist(),
            "C": Ct.tolist(),
            "J_beta_chron": bc,
            "J_p_chron": pc,
            "J_beta_rho": br,
            "J_p_rho": pr,
            "Lambda_chron_raw": lam_chron_raw,
            "Lambda_rho": lam_rt,
            "p_Lambda_rho": p_rt,
            "Lambda_n": lam_nt,
            "p_Lambda_n": p_nt,
            "spearman_rho_bin": float(spearmanr(rho_t, tb).correlation),
        }
        out["arms"][arm] = a
        print(
            arm,
            json.dumps({k: v for k, v in a.items() if k != "language"}, default=str)[
                :2000
            ],
            flush=True,
        )
    # ---- T2: cross-encoder correlation of N, row-permutation null with shared permutations ----
    if not g["G1_fired"]:
        d3 = float(np.corrcoef(Ns["D1"][iu, ju][dbin], Ns["D2"][iu, ju][dbin])[0, 1])
        e1, e2 = edges("D1"), edges("D2")
        null = []
        for _ in range(R_ROW):
            pt = tcode[rng.permutation(len(tcode))]
            n1 = flow_N(pt, *e1, len(tnames))[iu, ju][dbin]
            n2 = flow_N(pt, *e2, len(tnames))[iu, ju][dbin]
            null.append(np.corrcoef(n1, n2)[0, 1])
        null = np.array(null)
        out["T2"] = {
            "D3": d3,
            "p": float((1 + (null >= d3).sum()) / (R_ROW + 1)),
            "null_mean": float(null.mean()),
        }
        p_t1 = max(out["arms"][x]["T1"]["p_one_sided"] for x in ARMS)
        p_t3 = min(v["p_two_sided"] for v in g["G1_T3"].values())
        adj = holm([p_t1, out["T2"]["p"], p_t3])
        out["T_family"] = {
            "p": {"T1": p_t1, "T2": out["T2"]["p"], "T3": p_t3},
            "holm": {"T1": adj[0], "T2": adj[1], "T3": adj[2]},
            "T1_pass": all(out["arms"][x]["T1"]["Lambda"] > 0 for x in ARMS)
            and adj[0] < 0.05,
            "T2_pass": d3 >= 0.5 and adj[1] < 0.05,
        }
    pr1, pr2 = (out["arms"][x]["language"]["p_one_sided_exact"] for x in ARMS)
    adj = holm([pr1, pr2])
    out["R_family"] = {
        "holm": {"R1": adj[0], "R2": adj[1]},
        "R1_pass": out["arms"]["D1"]["language"]["Lambda_rho"] > 0 and adj[0] < 0.05,
        "R2_pass": out["arms"]["D2"]["language"]["Lambda_rho"] > 0 and adj[1] < 0.05,
    }
    pjc = max(out["arms"][x]["tradcent"]["J_p_chron"] for x in ARMS)
    pjr = max(out["arms"][x]["tradcent"]["J_p_rho"] for x in ARMS)
    adj = holm([pjc, pjr])
    out["J_family"] = {
        "p": {"J_chron": pjc, "J_rho": pjr},
        "holm": {"J_chron": adj[0], "J_rho": adj[1]},
        "J_chron_pass": all(
            out["arms"][x]["tradcent"]["J_beta_chron"] > 0 for x in ARMS
        )
        and adj[0] < 0.05,
        "J_rho_pass": all(out["arms"][x]["tradcent"]["J_beta_rho"] > 0 for x in ARMS)
        and adj[1] < 0.05,
    }
    json.dump(out, open(out_path, "w"), indent=1)
    print(
        json.dumps(
            {k: out[k] for k in out if k.endswith("family") or k == "T2"}, indent=1
        ),
        flush=True,
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "gate":
        gate(sys.argv[2])
    elif sys.argv[1] == "gate-a1":
        gate_a1(sys.argv[2])
    elif sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)

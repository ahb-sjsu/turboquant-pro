# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""Scale transfer: predict routed recall at 10^12 rows from subsets of at most 20 servers.

The preregistration is docs/PREREG_scale_transfer.md; this script is the procedure it names.

Recall of routing against the exact scan of any subset of servers is exact from the per-server
partials (see fleet_nested.py). A subset of m servers is a corpus of 2m billion rows from the same
generator with the same basis, coarse quantizer, router and queries. The calibration set is 20
servers drawn once with CAL_SEED. Five seeded orderings of it give nested subsets of 1, 2, 3, 5,
10 and 20 servers, whose mean recall at ten per width is the calibration curve R_cal(N) for N
from 2e9 to 4e10 rows. Each model below maps that curve to a prediction at N* = 1e12.

  M0  no change: R_cal at the largest calibration N
  M1  power law in the deficit: log(1 - R) linear in log N, least squares on the six points
  M2  recall linear in log N, least squares, capped at 1

Modes
  --dev      member-query run (tag 1t, all 500 servers): every model's prediction beside the
             measured value, per width. Writes scale_transfer_dev.json. This is the development
             set on which the registered model was chosen.
  --predict  the registered run (tag 1tnm): reads the partials of the 20 calibration servers and
             nothing else, writes scale_transfer_predict_{tag}.json with the sha256 of every file
             read and of this script. Run and commit before the run's score is read.
  --grade    the registered run, after the prediction is committed: reads all 500 servers,
             measures R(1e12), and grades each model by the paired bootstrap rule of the
             preregistration. Writes scale_transfer_grade_{tag}.json.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import time

import numpy as np

K = 10
N_SERVERS = 500
ROWS_PER_SERVER = 2_000_000_000
N_STAR = N_SERVERS * ROWS_PER_SERVER
CAL_SEED = 20260929
N_CAL = 20
CAL_SIZES = (1, 2, 3, 5, 10, 20)
ORDERINGS = 5
BOOT = 2000
BOOT_SEED = 7
MODELS = ("M0", "M1", "M2")


def calibration_servers() -> list[int]:
    return sorted(
        int(s)
        for s in np.random.default_rng(CAL_SEED).choice(N_SERVERS, N_CAL, replace=False)
    )


def orderings() -> list[list[int]]:
    cal = calibration_servers()
    rng = np.random.default_rng(CAL_SEED + 1)
    return [list(rng.permutation(cal)) for _ in range(ORDERINGS)]


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load(results: str, pattern: str, servers):
    """ids, scores (servers x queries x 10) for the listed servers only, and file hashes."""
    ids, scs, hashes = {}, {}, {}
    for s in servers:
        path = os.path.join(results, pattern.format(s=s))
        z = np.load(path)
        i = z["ids"].astype(np.int64)
        c = z["scores"].astype(np.float32)
        ids[s] = i
        scs[s] = np.where(np.isfinite(c) & (i >= 0), c, -np.inf)
        hashes[os.path.basename(path)] = sha(path)
    return ids, scs, hashes


def top10(ids, scs, servers):
    i = np.concatenate([ids[s] for s in servers], axis=1)
    c = np.concatenate([scs[s] for s in servers], axis=1)
    order = np.argsort(-c, axis=1, kind="stable")[:, :K]
    return np.take_along_axis(i, order, axis=1)


def per_query_recall(got, ref) -> np.ndarray:
    return np.array([len(set(a) & set(b)) / K for a, b in zip(got, ref)])


def calibration_matrix(ref, routed) -> np.ndarray:
    """(len(CAL_SIZES), nq) per-query recall averaged over the orderings."""
    out = []
    for m in CAL_SIZES:
        acc = 0.0
        for order in orderings():
            sub = order[:m]
            acc = acc + per_query_recall(top10(*routed, sub), top10(*ref, sub))
        out.append(acc / ORDERINGS)
    return np.array(out)


def predict(curve: np.ndarray) -> dict:
    """Every model's prediction at N_STAR from the calibration curve (one value per size)."""
    x = np.log(np.array(CAL_SIZES, float) * ROWS_PER_SERVER)
    xs = np.log(float(N_STAR))
    out = {"M0": float(curve[-1])}
    d = np.clip(1.0 - curve, 1e-6, None)
    b, a = np.polyfit(x, np.log(d), 1)
    out["M1"] = float(1.0 - np.exp(a + b * xs))
    b2, a2 = np.polyfit(x, curve, 1)
    out["M2"] = float(min(1.0, a2 + b2 * xs))
    return out


def widths(results: str, tag: str) -> list[int]:
    return sorted(
        {
            int(m.group(1))
            for p in glob.glob(os.path.join(results, f"ivf{tag}_p*_part_0.npz"))
            for m in [re.search(r"_p(\d+)_part_", p)]
            if m
        }
    )


def run_dev(results: str, out: str) -> dict:
    tag = "1t"
    all_srv = list(range(N_SERVERS))
    ref = load(results, f"ref{tag}_part_{{s}}.npz", all_srv)[:2]
    rec = {"tag": tag, "calibration_servers": calibration_servers(), "widths": {}}
    for p in widths(results, tag):
        routed = load(results, f"ivf{tag}_p{p}_part_{{s}}.npz", all_srv)[:2]
        cal = calibration_matrix(ref, routed).mean(axis=1)
        measured = float(
            per_query_recall(top10(*routed, all_srv), top10(*ref, all_srv)).mean()
        )
        pred = predict(cal)
        rec["widths"][str(p)] = {
            "calibration_curve": dict(zip(map(str, CAL_SIZES), map(float, cal))),
            "prediction": pred,
            "measured_1e12": measured,
            "error": {m: pred[m] - measured for m in MODELS},
        }
    json.dump(rec, open(out, "w"), indent=1)
    return rec


def run_predict(results: str, tag: str, out: str, registered: dict) -> dict:
    cal = calibration_servers()
    ref_ids, ref_scs, h = load(results, f"ref{tag}_part_{{s}}.npz", cal)
    rec = {
        "tag": tag,
        "calibration_servers": cal,
        "registered_models": registered,
        "script_sha256": sha(__file__),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files_read": dict(h),
        "widths": {},
    }
    for p in sorted(int(w) for w in registered):
        r_ids, r_scs, hr = load(results, f"ivf{tag}_p{p}_part_{{s}}.npz", cal)
        rec["files_read"].update(hr)
        curve = calibration_matrix((ref_ids, ref_scs), (r_ids, r_scs)).mean(axis=1)
        pred = predict(curve)
        rec["widths"][str(p)] = {
            "calibration_curve": dict(zip(map(str, CAL_SIZES), map(float, curve))),
            "prediction": pred,
            "registered_prediction": pred[registered[str(p)]],
        }
    json.dump(rec, open(out, "w"), indent=1)
    return rec


def run_grade(results: str, tag: str, pred_path: str, out: str) -> dict:
    pred_rec = json.load(open(pred_path))
    all_srv = list(range(N_SERVERS))
    ref = load(results, f"ref{tag}_part_{{s}}.npz", all_srv)[:2]
    rng = np.random.default_rng(BOOT_SEED)
    rec = {"tag": tag, "prediction_file_sha256": sha(pred_path), "widths": {}}
    for p, w in pred_rec["widths"].items():
        routed = load(results, f"ivf{tag}_p{p}_part_{{s}}.npz", all_srv)[:2]
        cal_q = calibration_matrix(ref, routed)  # (sizes, nq)
        meas_q = per_query_recall(top10(*routed, all_srv), top10(*ref, all_srv))
        nq = len(meas_q)
        measured = float(meas_q.mean())
        # Paired bootstrap over queries: the calibration curve and the measurement share the
        # queries, so the difference is resampled as one quantity.
        diffs = {m: [] for m in MODELS}
        for _ in range(BOOT):
            idx = rng.integers(0, nq, nq)
            pb = predict(cal_q[:, idx].mean(axis=1))
            mb = meas_q[idx].mean()
            for m in MODELS:
                diffs[m].append(pb[m] - mb)
        graded = {}
        for m in MODELS:
            err = w["prediction"][m] - measured
            se = float(np.std(diffs[m], ddof=1))
            graded[m] = {
                "prediction": w["prediction"][m],
                "error": err,
                "se_of_difference": se,
                "pass": bool(abs(err) <= 2 * se) if se > 0 else bool(abs(err) < 1e-12),
            }
        reg = pred_rec["registered_models"][p]
        rec["widths"][p] = {
            "measured_1e12": measured,
            "models": graded,
            "registered_model": reg,
            "H1_pass": graded[reg]["pass"],
            "H2_beats_no_change": abs(graded[reg]["error"])
            < abs(graded["M0"]["error"]),
        }
    json.dump(rec, open(out, "w"), indent=1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dev", action="store_true")
    g.add_argument("--predict", action="store_true")
    g.add_argument("--grade", action="store_true")
    ap.add_argument("--results", required=True, help="directory holding the partials")
    ap.add_argument("--tag", default="1tnm")
    ap.add_argument(
        "--registered", help='JSON map width -> model, e.g. {"32":"M1","128":"M0"}'
    )
    ap.add_argument("--prediction", help="--grade: the committed prediction JSON")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.dev:
        rec = run_dev(a.results, a.out)
    elif a.predict:
        rec = run_predict(a.results, a.tag, a.out, json.loads(a.registered))
    else:
        rec = run_grade(a.results, a.tag, a.prediction, a.out)
    print("SCALE_TRANSFER_JSON " + json.dumps(rec)[:4000], flush=True)


if __name__ == "__main__":
    main()

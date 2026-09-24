"""Registered scorer of the planner's P0 exit test (docs/PREREG_planner_exit.md).

    python -m planner_exit.score --records benchmarks/planner_exit/records \\
        --grid /data/results --out benchmarks/planner_exit/scored.json

Refuses to run unless every record listed in ``records/MANIFEST.sha256`` is
present and matches its hash: the planner's choices are committed before the
grid is read (section 5). Grid values come from the campaign's own loader and
paired rule (``rabitq_public.score.load``, ``paired_ci``, ``verdict``).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

from rabitq_public.score import PRIMARY, load, paired_ci, verdict

from .run import METHOD_CODEC

REACHABLE = tuple(METHOD_CODEC)
LOSE_R_PASS, LOSE_R_FAIL = 0.05, 0.10
FLOOR_SLACK = 0.005
BYTE_WINDOW = 1.05
COST_PASS, COST_FAIL = 0.10, 0.25
ALPHA = 0.05


def verified_records(root: str) -> list:
    manifest = os.path.join(root, "MANIFEST.sha256")
    if not os.path.exists(manifest):
        raise SystemExit("no MANIFEST.sha256: commit the records before scoring")
    out = []
    for line in open(manifest, encoding="utf-8"):
        digest, rel = line.split()
        path = os.path.join(root, rel)
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if got != digest:
            raise SystemExit(f"{rel}: hash {got[:12]} does not match the manifest")
        out.append(json.load(open(path, encoding="utf-8")))
    listed = {os.path.relpath(p, root) for p in glob.glob(f"{root}/*/*.json")}
    if len(listed) != len(out):
        raise SystemExit("records present that the manifest does not list")
    return out


def grid_cost(results: str, arm: str) -> float:
    """Core-seconds of the arm's reachable cells: mean cores x wall, per cell."""
    total = 0.0
    for p in glob.glob(os.path.join(results, f"{arm}-*.json")):
        r = json.load(open(p, encoding="utf-8"))
        if r["cell"]["method"] not in REACHABLE:
            continue
        u = r.get("usage") or {}
        total += float(u.get("mean_cpu_cores") or 0) * float(u.get("wall_s") or 0)
    return total


def binomial_cap(n: int, p: float = ALPHA) -> int:
    """Smallest x with P(Binomial(n, p) > x) < 0.05."""
    from math import comb

    cdf = 0.0
    for x in range(n + 1):
        cdf += comb(n, x) * p**x * (1 - p) ** (n - x)
        if 1 - cdf < 0.05:
            return x
    return n


def score(records: list, grid_root: str) -> dict:
    configs, incomplete = load(grid_root)
    rows = []
    for rec in records:
        arm, run = rec["arm"], rec["run"]
        mine = {k: c for (ds, k), c in configs.items() if ds == arm}
        reach = {k: c for k, c in mine.items() if c["method"] in REACHABLE}
        ch = rec.get("choice")
        row = {
            "arm": arm,
            "run": run,
            "abstained": ch is None,
            "cpu_seconds": rec["cpu_seconds"],
            "choice": ch and ch["grid_key"],
        }
        p = reach.get(ch["grid_key"]) if ch else None
        if ch and p is None:
            row["error"] = "choice not in the reachable grid"
        reg = rec["registered"]
        if reg["objective"] == "max_quality":
            B = reg["max_bytes"]
            for label, pool in (("reach", reach), ("full", mine)):
                feas = [c for c in pool.values() if c["bytes"] <= B]
                best = (
                    max(feas, key=lambda c: c["hits"][PRIMARY].mean()) if feas else None
                )
                ent = {"best": best and best["key"]}
                if best is not None and p is not None:
                    m, lo, hi = paired_ci(p["hits"][PRIMARY], best["hits"][PRIMARY])
                    ent.update(
                        regret=-m,
                        verdict=verdict(m, lo, hi),
                        lo=lo,
                        hi=hi,
                        same=best["key"] == p["key"],
                    )
                row[label] = ent
            r = row["reach"]
            row["no_regret"] = bool(
                p is not None
                and (r.get("same") or r.get("verdict") in ("TIES", "BEATS"))
            )
            row["loses_by"] = (
                (r["regret"] if r.get("verdict") == "LOSES" else 0.0)
                if p is not None
                else None
            )
        else:
            F = reg["floor"]
            ok = [c for c in reach.values() if c["hits"][PRIMARY].mean() >= F]
            cheapest = min(ok, key=lambda c: c["bytes"]) if ok else None
            row["cheapest"] = cheapest and cheapest["key"]
            if ch is None:
                row["no_regret"] = cheapest is None  # a correct abstention
            elif p is None:
                row["no_regret"] = False
            else:
                meets = p["hits"][PRIMARY].mean() >= F - FLOOR_SLACK
                row["meets_floor"] = bool(meets)
                row["excess_bytes"] = (
                    p["bytes"] / cheapest["bytes"] if cheapest else None
                )
                row["no_regret"] = bool(
                    meets
                    and cheapest is not None
                    and p["bytes"] <= BYTE_WINDOW * cheapest["bytes"]
                )
        if ch and p is not None:
            L = ch["holdout_bound"]
            row["calib_same"] = bool(rec["fresh"]["mean"] < L)
            row["calib_full"] = bool(p["hits"][PRIMARY].mean() < L)
        rows.append(row)

    n = len(rows)
    nr = sum(r["no_regret"] for r in rows)
    worst = max([r.get("loses_by") or 0.0 for r in rows] + [0.0])
    R = (
        "PASS"
        if nr >= 0.8 * n and worst <= LOSE_R_PASS
        else "FAIL" if nr < 0.6 * n or worst > LOSE_R_FAIL else "INCONCLUSIVE"
    )

    arms = sorted({r["arm"] for r in rows})
    ratios = {}
    for arm in arms:
        plan = sum(r["cpu_seconds"] for r in rows if r["arm"] == arm)
        g = grid_cost(grid_root, arm)
        ratios[arm] = plan / g if g > 0 else None
    good = sum(1 for v in ratios.values() if v is not None and v <= COST_PASS)
    bad = sum(1 for v in ratios.values() if v is not None and v > COST_FAIL)
    K = "PASS" if good >= 5 else "FAIL" if bad >= 3 else "INCONCLUSIVE"

    cal = [r["calib_same"] for r in rows if "calib_same" in r]
    viol, cap = sum(cal), binomial_cap(len(cal))
    C = "PASS" if viol <= cap else "FAIL" if viol > 2 * cap else "INCONCLUSIVE"

    return {
        "runs": rows,
        "R": {"verdict": R, "no_regret": nr, "of": n, "worst_loss": worst},
        "K": {"verdict": K, "ratio_by_arm": ratios},
        "C": {
            "verdict": C,
            "violations": viol,
            "of": len(cal),
            "cap": cap,
            "full_scale_violations": sum(r.get("calib_full", False) for r in rows),
        },
        "P0": "PASS" if (R, K, C) == ("PASS",) * 3 else "NOT PASSED",
        "incomplete_grid_configs": [list(x) for x in incomplete],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = score(verified_records(a.records), a.grid)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    for k in ("R", "K", "C", "P0"):
        print(k, json.dumps(res[k], default=float))


if __name__ == "__main__":
    main()

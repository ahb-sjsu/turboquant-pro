"""Apply the registered K1/K2 verdicts to bench_pruned_scan.py output (docs/PREREG_pruned_scan.md s4).

    python score_pruned_scan.py pruned_scan_1m.json pruned_scan_1m_b.json pruned_scan_5m.json

Per configuration and k, over the evaluation seeds:
  K1 (recall) HOLDS if mean recall@k vs v2 >= 0.995 and no seed is below 0.99;
  K2 (speed)  HOLDS if mean speedup >= 1.3x where d' >= 1024, >= 1.0x elsewhere.
Ship as opt-in only if K1 holds everywhere and K2 holds on every d' >= 1024 configuration.
A configuration with no admissible calibration point fails both and is reported as such.
"""

from __future__ import annotations

import json
import sys

import numpy as np

K1_MEAN, K1_MIN = 0.995, 0.99
K2_WIDE, K2_OTHER, WIDE = 1.3, 1.0, 1024
SEEDS = 3


def verdicts(results):
    rows = []
    for rec in results:
        for k, cal in sorted(rec["calibration"].items(), key=lambda kv: int(kv[0])):
            ev = rec["evaluation"].get(k, [])
            wide = rec["dim_out"] >= WIDE
            row = dict(
                arm=rec["arm"],
                dim_out=rec["dim_out"],
                bits=rec["bits"],
                k=int(k),
                chosen=cal["chosen"],
                seeds=len(ev),
            )
            if cal["chosen"] is None or len(ev) < SEEDS:
                row.update(
                    K1=False,
                    K2=False,
                    note=(
                        "no admissible point" if cal["chosen"] is None else "incomplete"
                    ),
                )
            else:
                rec_ = np.array([e["recall"] for e in ev])
                sp = np.array([e["speedup"] for e in ev])
                row.update(
                    recall_mean=float(rec_.mean()),
                    recall_min=float(rec_.min()),
                    survivors=float(np.mean([e["survivor_fraction"] for e in ev])),
                    speedup_mean=float(sp.mean()),
                    K1=bool(rec_.mean() >= K1_MEAN and rec_.min() >= K1_MIN),
                    K2=bool(sp.mean() >= (K2_WIDE if wide else K2_OTHER)),
                    note="",
                )
            rows.append(row)
    ship = all(r["K1"] for r in rows) and all(
        r["K2"] for r in rows if r["dim_out"] >= WIDE
    )
    return rows, ship


def main():
    # Several files because the run was split across pods and resumed (docs/PREREG_pruned_scan.md
    # amendment 2); the configurations are disjoint, which is checked here rather than assumed.
    results, envs = [], []
    for path in sys.argv[1:]:
        with open(path) as f:
            data = json.load(f)
        results.extend(data["results"])
        envs.append(data.get("env"))
    keys = [(r["arm"], r["dim_out"], r["bits"]) for r in results]
    assert len(keys) == len(set(keys)), f"a configuration appears twice: {keys}"
    data = dict(env=envs if len(envs) > 1 else envs[0], results=results)
    rows, ship = verdicts(data["results"])
    print(
        "| arm | d' | bits | k | prefix | z | recall mean | recall min | survivors | speedup | K1 | K2 |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for r in rows:
        c = r["chosen"] or {}
        if "recall_mean" in r:
            vals = f"{r['recall_mean']:.4f} | {r['recall_min']:.4f} | {100 * r['survivors']:.2f}% | {r['speedup_mean']:.2f}x"
        else:
            vals = f"{r['note']} | | | "
        print(
            f"| {r['arm']} | {r['dim_out']} | {r['bits']} | {r['k']} | {c.get('prefix', '')} | {c.get('z', '')} | "
            f"{vals} | {'HOLDS' if r['K1'] else 'FAILS'} | {'HOLDS' if r['K2'] else 'FAILS'} |"
        )
    print(f"\nenv: {data.get('env')}")
    print("SHIP AS OPT-IN" if ship else "STAYS AN EXPERIMENT")


if __name__ == "__main__":
    main()

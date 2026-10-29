"""Score the hubness density-ratio pilot against docs/PREREG_hubness_density_ratio.md.

Usage: python hubness_density_ratio_analyze.py hdr_results.json
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from collections import defaultdict

import numpy as np


def main(path):
    r = json.load(open(path))
    print(
        "=== P1 (real queries): Spearman(N_k, ratio) - Spearman(N_k, qmass) >= 0.05 in each cell"
    )
    p1 = []
    for c in r["p1"]:
        d = c["spearman"]["ratio"] - c["spearman"]["qmass"]
        p1.append(d >= 0.05)
        print(
            f"  {c['cell']}: ratio {c['spearman']['ratio']:.3f} qmass {c['spearman']['qmass']:.3f} diff {d:+.3f}"
            f" rho {c['rho']:.2f} -> {'pass' if d >= 0.05 else 'FAIL'}"
        )
    print(f"  P1 {'HOLDS' if all(p1) else 'FAILS'}")

    print("=== P3 (same-distribution queries)")
    p3 = []
    for c in r["p3"]:
        s = c["spearman"]
        a = abs(s["ratio"]) < 0.2
        b = max(abs(s["centrality"]), abs(s["density"])) > abs(s["ratio"]) + 0.1
        p3 += [a, b]
        print(
            f"  {c['cell']}: |ratio| {abs(s['ratio']):.3f} (<0.2: {a}); max(|central| {abs(s['centrality']):.3f},"
            f" |density| {abs(s['density']):.3f}) > |ratio|+0.1: {b}"
        )
    qa = [c for c in r["p1"] if c["k"] == 40][0]["attractiveness_skew"]
    qb = r["p3_matched_qb"]["attractiveness_skew"]
    p3.append(qa > qb)
    print(f"  matched n=12500 k=40: skew QA {qa:.3f} > QB {qb:.3f}: {qa > qb}")
    print(f"  P3 {'HOLDS' if all(p3) else 'FAILS'}")

    print(f"=== P2 (collapse in n/k), rho = {r['rho_grid']:.1f}")
    cells = defaultdict(list)
    for c in r["p2"]:
        cells[(c["n"], c["k"])].append(c["attractiveness_skew"])
    stat = {}
    for key, v in sorted(cells.items()):
        v = np.array([x for x in v if x == x])
        stat[key] = (
            v.mean(),
            v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else float("nan"),
            len(v),
        )
    print("  cell (n, k): mean skew +- SE (seeds)")
    for key, (m, se, ns) in stat.items():
        print(
            f"    n={key[0]:>6} k={key[1]:>2} n/k={key[0] // key[1]:>6}: {m:.3f} +- {se:.3f} ({ns})"
        )

    def agree(a, b):
        (m1, s1, _), (m2, s2, _) = stat[a], stat[b]
        tol = max(2 * math.sqrt(s1**2 + s2**2), 0.10 * (m1 + m2) / 2)
        return abs(m1 - m2) <= tol, abs(m1 - m2), tol

    same_ratio = [
        (a, b) for a, b in itertools.combinations(stat, 2) if a[0] * b[1] == b[0] * a[1]
    ]
    same_n = [(a, b) for a, b in itertools.combinations(stat, 2) if a[0] == b[0]]
    res_sr = [agree(a, b) for a, b in same_ratio]
    res_sn = [agree(a, b) for a, b in same_n]
    rate_sr = np.mean([x[0] for x in res_sr])
    rate_sn = np.mean([x[0] for x in res_sn])
    by_ratio = defaultdict(list)
    for (n, k), (m, _, _) in stat.items():
        by_ratio[n / k].append(m)
    avg = {q: np.mean(v) for q, v in by_ratio.items()}
    rng = max(avg.values()) - min(avg.values())
    med_se = float(
        np.median([math.sqrt(stat[a][1] ** 2 + stat[b][1] ** 2) for a, b in same_ratio])
    )
    print(
        f"  same-n/k pairs: {len(same_ratio)}, agreement rate {rate_sr:.2f} (need >= 0.80)"
    )
    for (a, b), (ok, d, tol) in zip(same_ratio, res_sr):
        print(
            f"    {a} vs {b}: |diff| {d:.3f} tol {tol:.3f} {'ok' if ok else 'DISAGREE'}"
        )
    print(
        f"  range of n/k-averaged skews {rng:.3f} vs 3 x median SE_pair {3 * med_se:.3f}: "
        f"{'non-vacuous' if rng > 3 * med_se else 'VACUOUS'}"
    )
    print(f"  control (same n, different k) agreement rate {rate_sn:.2f}")
    p2 = rate_sr >= 0.80 and rng > 3 * med_se
    print(
        f"  P2 {'HOLDS' if p2 else 'FAILS'} (informative only if control rate is clearly lower: "
        f"{rate_sn:.2f} vs {rate_sr:.2f})"
    )
    print("  n/k-averaged skew:", {int(q): round(v, 3) for q, v in sorted(avg.items())})


if __name__ == "__main__":
    main(sys.argv[1])

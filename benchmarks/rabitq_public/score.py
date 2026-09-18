"""Apply the registered verdict rules (docs/PREREG_rabitq_public.md section 4) to cell results.

    python -m rabitq_public.score --results /data/results [--markdown out.md] [--json out.json]

Rules, restated from the preregistration:

- A *config* is a cell without its seed. It is scored only when all three seeds finished;
  per-query hits are averaged over the seeds.
- Families: TQ = ``tq``; RABITQ = ``rabitq_flat``, ``rabitq_ivf``, ``rabitqlib_ivf``,
  ``pca_rabitq_ivf``; OPQ = ``opq``; PQ = ``pq``.
- Matching is two-sided and byte-windowed. Anchored at each TQ config (stored bytes B), the
  matched baseline is the family config with the highest mean primary recall among those
  storing between 0.80 B and 1.05 B per vector. Anchored at each baseline config (bytes B'),
  the matched TQ config is the one with the highest mean primary recall between 0.80 B' and
  1.05 B'. Duplicate pairs count once. An anchor with nothing in its window is ``NO-CONFIG``
  (reported as a byte gap; it is neither a win nor a loss).
- Paired per-query difference (TQ minus baseline, recall@10 units), percentile bootstrap
  over queries, 10,000 resamples, seed 0, 95% interval [lo, hi], mean m:
  BEATS if lo > 0 and m >= 0.005; LOSES if hi < 0 and m <= -0.005;
  TIES if lo >= -0.01 and hi <= 0.01; otherwise INCONCLUSIVE.
- Primary endpoint: +rerank x5 recall@10. Single-pass and +rerank x2 are reported with the
  same rules and do not enter the claim verdicts.
- Claim C1 (beats RaBitQ): HOLDS if BEATS in >= 2/3 of the scored comparisons and LOSES in
  none; REFUTED if LOSES in >= 1/3 or BEATS in none; otherwise MIXED.
  Claim C2 (ties OPQ): HOLDS if TIES or BEATS in >= 2/3 and LOSES in <= 1/6; REFUTED if
  LOSES in >= 1/3; otherwise MIXED. INCONCLUSIVE comparisons stay in the denominators.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from .grid import SEEDS

FAMILY = dict(
    tq="TQ",
    tqfix="TQFIX",
    tq_ivf="TQIVF",
    rabitq_flat="RABITQ",
    rabitq_ivf="RABITQ",
    rabitqlib_ivf="RABITQ",
    pca_rabitq_ivf="RABITQ",
    opq="OPQ",
    pq="PQ",
)
ENDPOINTS = ("rr5", "single", "rr2")
PRIMARY = "rr5"
WINDOW = (0.80, 1.05)
N_BOOT = 10_000


def load(results: str):
    configs = defaultdict(dict)
    for p in glob.glob(os.path.join(results, "*.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        c = r["cell"]
        key = r["cell"]["cell_id"].rsplit("-s", 1)[0]
        configs[(c["dataset"], key)][c["seed"]] = r
    out, incomplete = {}, []
    for (ds, key), by_seed in configs.items():
        if set(by_seed) != set(SEEDS):
            incomplete.append((ds, key, sorted(by_seed)))
            continue
        rs = [by_seed[s] for s in SEEDS]
        out[(ds, key)] = dict(
            dataset=ds,
            key=key,
            method=rs[0]["cell"]["method"],
            family=FAMILY[rs[0]["cell"]["method"]],
            bytes=max(float(r["stored_bytes_per_vec"]) for r in rs),
            hits={
                e: np.mean([np.asarray(r[f"hits_{e}"], float) for r in rs], axis=0) / 10
                for e in ENDPOINTS
            },
            seed_means={
                e: [float(np.mean(r[f"hits_{e}"])) / 10 for r in rs] for e in ENDPOINTS
            },
            build_s=float(np.median([r["build_s"] for r in rs])),
            search_s=float(np.median([r["search_s"] for r in rs])),
        )
    return out, incomplete


def paired_ci(a: np.ndarray, b: np.ndarray):
    d = a - b
    rng = np.random.default_rng(0)
    means = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def verdict(m, lo, hi):
    if lo > 0 and m >= 0.005:
        return "BEATS"
    if hi < 0 and m <= -0.005:
        return "LOSES"
    if lo >= -0.01 and hi <= 0.01:
        return "TIES"
    return "INCONCLUSIVE"


def _best(cands):
    return max(cands, key=lambda c: c["hits"][PRIMARY].mean())


def _window(configs, ds, family, b):
    return [
        c
        for (d2, _), c in configs.items()
        if d2 == ds
        and c["family"] == family
        and WINDOW[0] * b <= c["bytes"] <= WINDOW[1] * b
    ]


def _pair_row(tq, base, family, anchor):
    row = dict(
        dataset=tq["dataset"],
        tq=tq["key"],
        tq_bytes=tq["bytes"],
        family=family,
        baseline=base["key"],
        baseline_bytes=base["bytes"],
        anchor=anchor,
        verdict={},
        diff={},
    )
    for e in ENDPOINTS:
        m, lo, hi = paired_ci(tq["hits"][e], base["hits"][e])
        row["diff"][e] = dict(
            tq=float(tq["hits"][e].mean()),
            base=float(base["hits"][e].mean()),
            mean=round(m, 5),
            lo=round(lo, 5),
            hi=round(hi, 5),
        )
        row["verdict"][e] = verdict(m, lo, hi)
    return row


def compare(configs, tq_family="TQ"):
    rows, seen = [], set()
    for fam in ("RABITQ", "OPQ", "PQ"):
        for (ds, key), c in sorted(configs.items()):
            if c["family"] == tq_family:
                cands = _window(configs, ds, fam, c["bytes"])
                if not cands:
                    rows.append(
                        dict(
                            dataset=ds,
                            tq=key,
                            tq_bytes=c["bytes"],
                            family=fam,
                            anchor="tq",
                            verdict={e: "NO-CONFIG" for e in ENDPOINTS},
                        )
                    )
                    continue
                tq, base, anchor = c, _best(cands), "tq"
            elif c["family"] == fam:
                cands = _window(configs, ds, tq_family, c["bytes"])
                if not cands:
                    rows.append(
                        dict(
                            dataset=ds,
                            baseline=key,
                            baseline_bytes=c["bytes"],
                            family=fam,
                            anchor="baseline",
                            verdict={e: "NO-CONFIG" for e in ENDPOINTS},
                        )
                    )
                    continue
                tq, base, anchor = _best(cands), c, "baseline"
            else:
                continue
            if (tq["key"], base["key"]) in seen:
                continue
            seen.add((tq["key"], base["key"]))
            rows.append(_pair_row(tq, base, fam, anchor))
    return rows


def claim(rows, family, rule):
    scored = [
        r
        for r in rows
        if r["family"] == family and r["verdict"][PRIMARY] != "NO-CONFIG"
    ]
    n = len(scored)
    count = defaultdict(int)
    for r in scored:
        count[r["verdict"][PRIMARY]] += 1
    no_config = sum(
        1
        for r in rows
        if r["family"] == family and r["verdict"][PRIMARY] == "NO-CONFIG"
    )
    if n == 0:
        return dict(verdict="UNSCORED", n=0, counts={}, no_config=no_config)
    beats, loses, ties = count["BEATS"], count["LOSES"], count["TIES"]
    if rule == "C1":
        if beats >= 2 * n / 3 and loses == 0:
            v = "HOLDS"
        elif loses >= n / 3 or beats == 0:
            v = "REFUTED"
        else:
            v = "MIXED"
    else:
        if beats + ties >= 2 * n / 3 and loses <= n / 6:
            v = "HOLDS"
        elif loses >= n / 3:
            v = "REFUTED"
        else:
            v = "MIXED"
    return dict(verdict=v, n=n, counts=dict(count), no_config=no_config)


def markdown(configs, rows, claims, incomplete) -> str:
    out = ["## Claim verdicts (primary endpoint: +rerank x5 recall@10)", ""]
    out += [
        f"- **{k}**: {v['verdict']} ({v['counts']}, n={v['n']}, no-config={v['no_config']})"
        for k, v in claims.items()
    ]
    out += ["", "## Matched-byte comparisons", ""]
    out.append(
        "| dataset | tq-pro config | B | family | matched baseline | B | endpoint | tq | baseline | diff [95% CI] | verdict |"
    )
    out.append("|---|---|---:|---|---|---:|---|---:|---:|---|---|")
    for r in rows:
        if r["verdict"][PRIMARY] == "NO-CONFIG":
            who = r.get("tq") or r.get("baseline")
            b = r.get("tq_bytes") or r.get("baseline_bytes")
            out.append(
                f"| {r['dataset']} | {who} ({r['anchor']} anchor) | {b:g} | {r['family']} | nothing in [0.80, 1.05] x B | - | - | - | - | - | NO-CONFIG |"
            )
            continue
        for e in ENDPOINTS:
            d = r["diff"][e]
            out.append(
                f"| {r['dataset']} | {r['tq']} | {r['tq_bytes']:g} | {r['family']} | {r['baseline']} | {r['baseline_bytes']:g} | {e} | "
                f"{d['tq']:.4f} | {d['base']:.4f} | {d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] | {r['verdict'][e]} |"
            )
    out += [
        "",
        "## Every scored config (mean over 3 seeds; seed spread in brackets)",
        "",
    ]
    out.append(
        "| dataset | config | family | B/vec | single | rr2 | rr5 | build s | search s |"
    )
    out.append("|---|---|---|---:|---|---|---|---:|---:|")
    for (ds, key), c in sorted(
        configs.items(), key=lambda kv: (kv[0][0], kv[1]["family"], kv[1]["bytes"])
    ):
        cells = []
        for e in ("single", "rr2", "rr5"):
            sm = c["seed_means"][e]
            cells.append(f"{np.mean(sm):.4f} [{min(sm):.4f}, {max(sm):.4f}]")
        out.append(
            f"| {ds} | {key} | {c['family']} | {c['bytes']:g} | "
            + " | ".join(cells)
            + f" | {c['build_s']:.1f} | {c['search_s']:.2f} |"
        )
    if incomplete:
        out += ["", "## Incomplete configs (excluded)", ""]
        out += [f"- {ds} {key}: seeds {seeds}" for ds, key, seeds in sorted(incomplete)]
    return "\n".join(out) + "\n"


def supplementary_family(configs, method: str = "tqfix"):
    """A supplementary tq-pro family: each tq config is replaced by its ``method``
    twin when one exists (Amendment 2: ``tqfix``; Amendment 3: ``tq_ivf``).

    Configs without a twin stand as they are (for ``tqfix`` those are the kernels
    that could not wrap); configs with a twin are superseded by it.
    """
    family = FAMILY[method]
    out = {}
    for (ds, key), c in configs.items():
        if c["family"] == "TQ":
            twin = (ds, key.replace("-tq-", f"-{method}-", 1))
            if twin in configs:
                continue
            c = dict(c, family=family)
        out[(ds, key)] = c
    for (ds, key), c in configs.items():
        if c["family"] == family:
            out[(ds, key)] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--markdown")
    ap.add_argument("--json")
    ap.add_argument(
        "--supplementary",
        nargs="?",
        const="tqfix",
        default=None,
        choices=["tqfix", "tq_ivf"],
        help="score a supplementary tq-pro family in place of the registered one: "
        "Amendment 2 (tqfix, the v2 kernel where the v1 kernel could wrap; the "
        "default when the flag is bare) or Amendment 3 (tq_ivf, residual-coded IVF)",
    )
    a = ap.parse_args()
    configs, incomplete = load(a.results)
    tq_family = "TQ"
    if a.supplementary:
        configs = supplementary_family(configs, a.supplementary)
        tq_family = FAMILY[a.supplementary]
    rows = compare(configs, tq_family)
    claims = dict(
        C1_beats_rabitq=claim(rows, "RABITQ", "C1"),
        C2_ties_opq=claim(rows, "OPQ", "C2"),
    )
    md = markdown(configs, rows, claims, incomplete)
    if a.markdown:
        with open(a.markdown, "w", encoding="utf-8") as f:
            f.write(md)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(
                dict(claims=claims, comparisons=rows, incomplete=incomplete),
                f,
                indent=1,
            )
    print(md)


if __name__ == "__main__":
    main()

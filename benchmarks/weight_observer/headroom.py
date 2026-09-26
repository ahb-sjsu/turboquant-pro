"""EXPLORATORY: how much a better per-matrix predictor could still buy a weight plan.

    python -m weight_observer.headroom --model qwen2.5-1.5b [--out headroom.json]

Reads only committed files: the registered run (``results/<model>/``), the exploratory
output fetched from NRP (``results/<model>/explore/``: the single-matrix sensitivity sweep
and the measured KL of every plan) and the plans (``planned/<model>.json``). Reports:

- **calibration**: measured single-matrix KL over the table's prediction, per (matrix,
  bits), as a log ratio; its spread, by matrix type and layer, and the share of 4-bit damage
  in matrices the predictor misjudges by more than 2x;
- **plans**: each plan's KL per token against the diagonal-Fisher plan at the same rate,
  paired over the evaluation sequences, 95% percentile bootstrap (10,000 resamples, seed 0);
- **additivity**: the measured KL of a whole plan over the sum of its matrices' single KLs,
  which bounds how far any additive prediction can be trusted;
- **oracle**: the plan the measured single KLs choose (``plans make`` with the sweep
  present), its additive prediction, and its measured KL once ``plans eval`` has run it.

Nothing here is a verdict: the models, the predictor tables and the sweep are the ones Part
III selected on (``docs/PREREG_observer_advantage_weights.md``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def per_seq(seqs) -> np.ndarray:
    return np.array([s["kl_sum"] / s["tokens"] for s in seqs], dtype=np.float64)


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def boot(a, b, rng, n=10_000):
    d = a - b
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def analyse(model: str) -> dict:
    res = os.path.join(HERE, "results", model)
    ex = os.path.join(res, "explore")
    sens = {
        (r["matrix"], r["bits"]): per_seq(r["seqs"])
        for r in jsonl(f"{ex}/sensitivity.jsonl")
    }
    tbl = json.load(open(f"{res}/tables.json", encoding="utf-8"))
    out: dict = {"model": model, "calibration": {}, "plans": {}}

    for pred in ("fisher", "observer"):
        rows = [
            (m, b, float(np.log(v.mean() / tbl[m][str(b)][pred])))
            for (m, b), v in sens.items()
            if v.mean() > 0 and tbl[m][str(b)][pred] > 0
        ]
        lr = np.array([x for _, _, x in rows])
        med = float(np.median(lr))
        kind, layer = defaultdict(list), defaultdict(list)
        for m, _, x in rows:
            kind[m.split(".")[-1]].append(x - med)
            layer[int(re.search(r"layers\.(\d+)", m).group(1))].append(x - med)
        four = [(m, x) for m, b, x in rows if b == 4]
        tot = sum(sens[(m, 4)].mean() for m, _ in four)
        off = sum(sens[(m, 4)].mean() for m, x in four if abs(x - med) > np.log(2))
        out["calibration"][pred] = {
            "log_ratio_sd": float(lr.std()),
            "log_ratio_iqr": float(np.percentile(lr, 75) - np.percentile(lr, 25)),
            "by_type": {
                k: round(float(np.mean(v)), 3) for k, v in sorted(kind.items())
            },
            "by_layer": {
                k: round(float(np.mean(v)), 3) for k, v in sorted(layer.items())
            },
            "largest": [
                [m, b, round(x - med, 2)]
                for m, b, x in sorted(rows, key=lambda r: -abs(r[2] - med))[:8]
            ],
            "damage_4bit_misjudged_2x": float(off / tot),
        }

    meas = {r["plan"]: per_seq(r["seqs"]) for r in jsonl(f"{ex}/plans_results.jsonl")}
    plans = json.load(
        open(os.path.join(HERE, "planned", f"{model}.json"), encoding="utf-8")
    )["plans"]
    rng = np.random.default_rng(0)

    def additive(bits):
        return float(sum(sens[(m, b)].mean() for m, b in bits.items() if b != 8))

    ratios = {p: float(meas[p].mean() / additive(plans[p])) for p in meas if p in plans}
    out["additivity"] = {
        "measured_over_additive": {p: round(r, 3) for p, r in sorted(ratios.items())},
        "median": float(np.median(list(ratios.values()))),
    }
    for rate in sorted({p.split("-")[0] for p in plans}):
        ref = meas[f"{rate}-fisher"]
        row = {"fisher": {"kl": float(ref.mean())}}
        for p in sorted(
            x for x in plans if x.startswith(rate + "-") and x != f"{rate}-fisher"
        ):
            name = p.split("-", 1)[1]
            row[name] = {"additive": additive(plans[p])}
            if p in meas:
                d, lo, hi = boot(meas[p], ref, rng)
                row[name].update(
                    kl=float(meas[p].mean()),
                    delta=d,
                    ci=[lo, hi],
                    judgement="worse" if lo > 0 else "better" if hi < 0 else "tie",
                )
        row["fisher"]["additive"] = additive(plans[f"{rate}-fisher"])
        row["oracle_headroom_additive"] = (
            1 - row["oracle"]["additive"] / row["fisher"]["additive"]
        )
        row["oracle_reassigned"] = sum(
            plans[f"{rate}-fisher"][m] != plans[f"{rate}-oracle"][m]
            for m in plans[f"{rate}-fisher"]
        )
        out["plans"][rate] = row

    # the oracle check: the oracle plans measured in one pod beside the Fisher plan and
    # its nearest rivals (nrp oracle), compared within that pod, and every re-measured
    # plan against its first measurement from another pod
    oc = os.path.join(ex, "oracle_check", "plans_results.jsonl")
    if os.path.exists(oc):
        again = {r["plan"]: per_seq(r["seqs"]) for r in jsonl(oc)}
        out["oracle_check"] = {
            "reproduced": {
                p: float(np.abs(again[p] - meas[p]).max()) for p in again if p in meas
            },
            "oracle_vs_fisher": {},
        }
        for rate in sorted({p.split("-")[0] for p in again}):
            o, f = again.get(f"{rate}-oracle"), again.get(f"{rate}-fisher")
            if o is None or f is None:
                continue
            d, lo, hi = boot(o, f, rng)
            out["oracle_check"]["oracle_vs_fisher"][rate] = {
                "oracle": float(o.mean()),
                "fisher": float(f.mean()),
                "delta": d,
                "ci": [lo, hi],
                "relative": d / float(f.mean()),
                "judgement": "worse" if lo > 0 else "better" if hi < 0 else "tie",
            }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    r = analyse(a.model)
    for pred, c in r["calibration"].items():
        print(
            f"{pred}: log(measured/predicted) sd {c['log_ratio_sd']:.2f}; 4-bit damage in "
            f"matrices misjudged >2x {c['damage_4bit_misjudged_2x']:.1%}"
        )
    print(f"additivity: median {r['additivity']['median']:.2f}")
    for rate, row in r["plans"].items():
        o = row["oracle"]
        print(
            f"{rate}: fisher {row['fisher']['kl']:.4f}; oracle additive headroom "
            f"{row['oracle_headroom_additive']:.1%} ({row['oracle_reassigned']} matrices moved)"
            + (
                f"; oracle measured {o['kl']:.4f} {o['judgement']} "
                f"[{o['ci'][0]:+.4f}, {o['ci'][1]:+.4f}]"
                if "kl" in o
                else "; oracle not measured"
            )
        )
        for name, v in row.items():
            if isinstance(v, dict) and "judgement" in v and name != "oracle":
                print(
                    f"    {name:12s} {v['kl']:.4f} {v['delta']:+.4f} {v['judgement']}"
                )
    chk = r.get("oracle_check")
    if chk:
        worst = max(chk["reproduced"].values()) if chk["reproduced"] else float("nan")
        print(
            f"oracle check: {len(chk['reproduced'])} plans re-measured in another pod, "
            f"largest per-sequence change {worst:.2e} nats/token"
        )
        for rate, v in chk["oracle_vs_fisher"].items():
            print(
                f"  {rate}: oracle {v['oracle']:.4f} fisher {v['fisher']:.4f} "
                f"delta {v['delta']:+.4f} ({v['relative']:+.1%}) "
                f"[{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] {v['judgement']}"
            )
    if a.out:
        json.dump(r, open(a.out, "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""EXPLORATORY diagnostics of why the exact forms do not beat the diagonal Fisher. Not a verdict.

    python -m weight_observer.explore_analysis --run <runs/model> --explore <explore/model>

1. Reliability: each exact form is recomputed on two disjoint halves of the calibration
   sequences; the within-rate Spearman between the halves (mean over rates) is how much of
   its ranking is signal. If the cross terms are estimation noise, ``exact_model`` (all cross
   terms between matrices) is the least reliable and ``exact_block`` the most.
2. Where second order fails: within each rate, the residual of log KL after a linear fit on
   log fisher; for each matrix, the Spearman between its bit width and that residual. A
   matrix whose low bit width raises damage beyond what the Fisher predicts shows up with a
   large negative value.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .score import _kl, spearman
from .variants import generate


def exact(c: np.ndarray, li: np.ndarray) -> tuple:
    cs = c[:, np.arange(c.shape[1]), li]
    return float((cs**2).mean(0).sum()), float((cs.sum(1) ** 2).mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--explore", required=True)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args(argv)
    ex = json.load(open(os.path.join(a.explore, "explore.json")))
    c = np.load(os.path.join(a.explore, "c.npy"))
    names, levels = ex["names"], ex["levels"]
    variants = generate(names, ex["numel"])
    rows = [json.loads(x) for x in open(os.path.join(a.run, "results.jsonl"))]
    rng = np.random.default_rng(0)
    perm = rng.permutation(c.shape[0])
    halves = (c[perm[: len(perm) // 2]], c[perm[len(perm) // 2 :]])
    st = {}
    for r in rows:
        v = r["variant"]
        if not v.startswith("r"):
            continue
        s = v.split("-")[0]
        bits = np.array([variants[v][n] for n in names])
        li = np.array([levels.index(b) for b in bits])
        d = st.setdefault(s, {k: [] for k in ("kl", "fisher", "bits", "eb", "em")})
        d["kl"].append(_kl(r["seqs"], 0, 1))
        d["fisher"].append(r["pred"]["fisher"])
        d["bits"].append(bits)
        for h, hc in enumerate(halves):
            eb, em = exact(hc, li)
            d["eb"].append((h, eb))
            d["em"].append((h, em))
    out = {"reliability": {}, "second_order_misses": {}}
    for k in ("eb", "em"):
        rel = []
        for d in st.values():
            a0 = [x for h, x in d[k] if h == 0]
            a1 = [x for h, x in d[k] if h == 1]
            rel.append(spearman(np.array(a0), np.array(a1)))
        out["reliability"][{"eb": "exact_block", "em": "exact_model"}[k]] = round(
            float(np.mean(rel)), 3
        )
    per = np.zeros(len(names))
    for d in st.values():
        y = np.log(np.array(d["kl"]))
        x = np.log(np.array(d["fisher"]))
        res = y - np.polyval(np.polyfit(x, y, 1), x)
        B = np.array(d["bits"])
        per += np.array([spearman(B[:, j], res) for j in range(len(names))])
    per /= len(st)
    order = np.argsort(per)
    out["second_order_misses"] = {
        names[j]: round(float(per[j]), 3) for j in order[: a.top]
    }
    out["null_sd"] = round(
        float(1 / np.sqrt(len(next(iter(st.values()))["kl"]) - 1) / np.sqrt(len(st))), 3
    )
    print(json.dumps(out, indent=1))
    json.dump(
        out, open(os.path.join(a.explore, "explore_analysis.json"), "w"), indent=1
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

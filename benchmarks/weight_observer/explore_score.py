"""EXPLORATORY scoring: new predictors against the Part III measurements. Not a verdict.

    python -m weight_observer.explore_score --run <runs/model> --explore <explore/model>

Reports rho-bar (mean within-rate Spearman with KL) for the registered predictors and for the
exploratory ones computed from ``explore.py``'s output, plus the wiring check (the registered
``fisher`` table recomputed from the same pass). Whatever wins here must be confirmed on fresh
variants and fresh models under a new registration before it is claimed.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .score import _kl, spearman
from .tables import PREDICTORS
from .variants import generate

EXPLORATORY = ("exact_block", "exact_model", "tok_exact", "fisher_tok")


def predictors(ex: dict, c: np.ndarray, bits: dict) -> dict:
    names, levels = ex["names"], ex["levels"]
    li = [levels.index(bits[n]) for n in names]
    cs = c[:, np.arange(len(names)), li]  # (seq, matrix)
    return {
        "exact_block": float((cs**2).mean(0).sum()),
        "exact_model": float((cs.sum(1) ** 2).mean()),
        "tok_exact": float(
            sum(ex["tok_exact"][n][levels.index(bits[n])] for n in names)
        ),
        "fisher_tok": float(
            sum(ex["fisher_tok"][n][levels.index(bits[n])] for n in names)
        ),
        "fisher_check": float(
            sum(ex["fisher_check"][n][levels.index(bits[n])] for n in names)
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--explore", required=True)
    a = ap.parse_args(argv)
    ex = json.load(open(os.path.join(a.explore, "explore.json")))
    c = np.load(os.path.join(a.explore, "c.npy"))
    variants = generate(ex["names"], ex["numel"])  # the registered, deterministic set
    vp = os.path.join(a.run, "variants.json")
    if os.path.exists(vp) and json.load(open(vp)) != variants:
        raise SystemExit("variants.json differs from the regenerated set")
    rows = [json.loads(line) for line in open(os.path.join(a.run, "results.jsonl"))]
    strata = {}
    for r in rows:
        if not r["variant"].startswith("r"):
            continue
        s = r["variant"].split("-")[0]
        p = dict(r["pred"])
        p.update(predictors(ex, c, variants[r["variant"]]))
        d = strata.setdefault(s, {"kl": []})
        d["kl"].append(_kl(r["seqs"], 0, 1))
        for k, v in p.items():
            d.setdefault(k, []).append(v)
    out = {}
    for k in (*PREDICTORS, *EXPLORATORY):
        per = {
            s: spearman(np.array(d[k]), np.array(d["kl"])) for s, d in strata.items()
        }
        out[k] = {
            "rho_bar": float(np.mean(list(per.values()))),
            "by_rate": {s: round(v, 3) for s, v in per.items()},
        }
    check = max(
        abs(a_ - b_) / max(abs(b_), 1e-30)
        for d in strata.values()
        for a_, b_ in zip(d["fisher_check"], d["fisher"])
    )
    print(json.dumps({"wiring_fisher_max_rel_diff": check}))
    for k, v in sorted(out.items(), key=lambda kv: -kv[1]["rho_bar"]):
        print(f"{k:12s} rho_bar {v['rho_bar']:.3f}  {v['by_rate']}")
    json.dump(out, open(os.path.join(a.explore, "explore_scores.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

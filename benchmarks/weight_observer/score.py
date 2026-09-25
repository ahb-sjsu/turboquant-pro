"""Registered scorer of Part III (docs/PREREG_observer_advantage_weights.md).

    python -m weight_observer.score --runs /data/wo/runs --out results_weights.json

Per model and rate stratum: Spearman rho between each predictor and the measured KL per
token over the stratum's variants; rho-bar is the mean over strata (the fixed-rate
comparison). For the observer against each competitor: the difference in rho-bar, with a
95% percentile bootstrap that resamples variants within strata (10,000 resamples, seed 0).
Split-half reliability of the KL (even against odd evaluation sequences, per stratum) is the
anti-vacuity gate: a stratum whose split-half rho is below 0.8 is excluded and reported.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .tables import PREDICTORS

MODELS = ("qwen2.5-1.5b", "llama3.2-3b")
N_BOOT = 10_000
MIN_SPLIT_HALF = 0.8
COMPETITORS = {"W1": "fisher", "W2": "act", "W3": "raw"}


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return (sums / counts)[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = ranks(np.asarray(a, float)), ranks(np.asarray(b, float))
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def _kl(seqs: list, start: int, step: int) -> float:
    """KL per token over evaluation sequences start, start+step, ..."""
    ss = seqs[start::step]
    return sum(q["kl_sum"] for q in ss) / sum(q["tokens"] for q in ss)


def load(run_dir: str) -> dict:
    """{stratum: {"kl": array, "kl_even": array, "kl_odd": array, pred: array}}."""
    rows = [
        json.loads(line)
        for line in open(os.path.join(run_dir, "results.jsonl"), encoding="utf-8")
    ]
    strata = {}
    for r in rows:
        s = r["variant"].split("-")[0]
        seqs = r["seqs"]

        d = strata.setdefault(
            s, {k: [] for k in ("kl", "kl_even", "kl_odd", *PREDICTORS)}
        )
        d["kl"].append(_kl(seqs, 0, 1))
        d["kl_even"].append(_kl(seqs, 0, 2))
        d["kl_odd"].append(_kl(seqs, 1, 2))
        for p in PREDICTORS:
            d[p].append(r["pred"][p])
    return {
        s: {k: np.asarray(v, float) for k, v in d.items()} for s, d in strata.items()
    }


def rho_bar(strata: dict, pred: str, idx: dict | None = None) -> float:
    vals = []
    for s, d in strata.items():
        i = idx[s] if idx else slice(None)
        vals.append(spearman(d[pred][i], d["kl"][i]))
    return float(np.mean(vals))


def model_report(strata: dict) -> dict:
    gate = {s: spearman(d["kl_even"], d["kl_odd"]) for s, d in strata.items()}
    kept = {s: d for s, d in strata.items() if gate[s] >= MIN_SPLIT_HALF}
    rep = {"split_half": gate, "strata_kept": sorted(kept)}
    if not kept:
        rep["verdicts"] = {}
        return rep
    rep["rho_bar"] = {p: rho_bar(kept, p) for p in PREDICTORS}
    rep["rho_by_stratum"] = {
        p: {s: spearman(d[p], d["kl"]) for s, d in kept.items()} for p in PREDICTORS
    }
    rng = np.random.default_rng(0)
    boots = {p: [] for p in PREDICTORS}
    for _ in range(N_BOOT):
        idx = {s: rng.integers(0, len(d["kl"]), len(d["kl"])) for s, d in kept.items()}
        for p in PREDICTORS:
            boots[p].append(rho_bar(kept, p, idx))
    boots = {p: np.asarray(v) for p, v in boots.items()}
    rep["diff"] = {}
    for comp in PREDICTORS:
        if comp == "observer":
            continue
        diff = boots["observer"] - boots[comp]
        rep["diff"][comp] = {
            "point": rep["rho_bar"]["observer"] - rep["rho_bar"][comp],
            "lo": float(np.nanpercentile(diff, 2.5)),
            "hi": float(np.nanpercentile(diff, 97.5)),
        }
    return rep


def verdicts(reports: dict) -> dict:
    out = {}
    for w, comp in COMPETITORS.items():
        ds = [reports[m]["diff"][comp] for m in MODELS if "diff" in reports.get(m, {})]
        if len(ds) < len(MODELS):
            out[w] = "INCOMPLETE"
        elif all(d["lo"] > 0 for d in ds):
            out[w] = "HOLDS"
        elif any(d["hi"] < 0 for d in ds):
            out[w] = "FAILS (reversed)"
        elif all(d["point"] <= 0 for d in ds):
            out[w] = "FAILS"
        else:
            out[w] = "INCONCLUSIVE"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    reports = {}
    for m in MODELS:
        d = os.path.join(a.runs, m)
        if os.path.exists(os.path.join(d, "results.jsonl")):
            reports[m] = model_report(load(d))
    res = {"models": reports, "verdicts": verdicts(reports)}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["verdicts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Registered scorer of Part III-c (``docs/PREREG_weights_codec_allocation.md``).

    python -m weight_observer.score_codec --results DIR --out results_codec.json

``DIR`` holds one directory per model with the harness's outputs (``arms.json``,
``arms_results.jsonl``, ``hashes.json``) and, optionally, ``arms_repeat.jsonl``: one arm
measured again in another pod (gate G2).

For arm X against arm Y at one budget on one model: the relative KL difference
``mean(KL_X) / mean(KL_Y) - 1`` over the 48 evaluation sequences, with a 95% percentile
bootstrap interval in which each resample draws 48 sequence indices and evaluates both arms
on the same indices (10,000 resamples, seed 0). **Better** = the interval below 0 and the
point estimate at most -5%; **worse** = the mirror.

Gates, each a status per model through the proven machinery of Part II
(``score_keys.decide``, ``tests/test_gate_proof.py``): G1, every arm's stored bits recomputed
from its widths are within its budget; G2, a repeated arm reproduces per sequence to 1e-6
nats per token; samples, the model's ``hashes.json`` equals the registered sample identities
(``registered_samples.json`` beside this file; PENDING until it exists). The verdicts carry
the resulting ``verdict_status``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "kvquant_matrix"))
import score_keys as SK  # noqa: E402  (the proven gate machinery)

from . import quant as Q  # noqa: E402

MODELS = ("qwen2.5-3b", "gemma-2-2b", "llama3.1-8b")
BUDGETS = (3, 4)
THRESHOLD = 0.05
N_BOOT, SEED = 10_000, 0
REGISTERED = os.path.join(HERE, "registered_samples.json")
HYPOTHESES = {
    "C1a": ("gptq_f", "gptq_u"),
    "C1b": ("gptq_f", "awq_u"),
    "C2": ("gptq_f", "rtn_f"),
    "C3": ("gptq_f", "gptq_frtn"),
}


def per_seq(seqs) -> np.ndarray:
    return np.array([s["kl_sum"] / s["tokens"] for s in seqs], dtype=np.float64)


def compare(x: np.ndarray, y: np.ndarray, rng) -> dict:
    """Relative KL of X against Y, paired over sequences, and its judgement."""
    if len(x) != len(y):
        raise ValueError("arms measured on different numbers of sequences")
    rel = float(x.mean() / y.mean() - 1.0)
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    boot = x[idx].mean(axis=1) / y[idx].mean(axis=1) - 1.0
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    if hi < 0 and rel <= -THRESHOLD:
        j = "better"
    elif lo > 0 and rel >= THRESHOLD:
        j = "worse"
    else:
        j = "neither"
    return {"rel": rel, "lo": lo, "hi": hi, "judgement": j}


def verdict(per_model: dict, models=MODELS) -> str:
    """{model: {budget: judgement}}: better at both budgets on >= 2 models and worse on
    none HOLDS; better on none FAILS; worse on >= 2 is reversed; else INCONCLUSIVE."""
    if any(m not in per_model or len(per_model[m]) < len(BUDGETS) for m in models):
        return "INCOMPLETE"
    better = sum(all(per_model[m][b] == "better" for b in BUDGETS) for m in models)
    worse = sum(any(per_model[m][b] == "worse" for b in BUDGETS) for m in models)
    if worse >= 2:
        return "FAILS (reversed)"
    if better >= 2 and worse == 0:
        return "HOLDS"
    if better == 0:
        return "FAILS"
    return "INCONCLUSIVE"


def stored(spec: dict, numel: dict) -> int:
    """Logical stored bits of an arm, recomputed from its widths: codes and per-group
    metadata for every matrix, plus a planned arm's width map (``map_bits`` a matrix).
    """
    bits = sum(Q.stored_bits(numel[n], b) for n, b in spec["bits"].items())
    return bits + int(spec["map_bits"]) * len(spec["bits"])


def g1(arms: dict, numel: dict):
    """True if every arm, recomputed, is within its budget and matches what the plan
    recorded; None when there is nothing to check."""
    if not arms or not numel:
        return None
    for spec in arms.values():
        s = stored(spec, numel)
        if s > spec["budget_bits"] or s != spec["stored_bits"]:
            return False
    return True


def score(results_dir: str) -> dict:
    rng = np.random.default_rng(SEED)
    reg = json.load(open(REGISTERED)) if os.path.exists(REGISTERED) else None
    out = {
        "models": {},
        "comparisons": {},
        "gates": {"G1": {}, "G2": {}, "samples": {}},
    }
    kl = {}
    for m in MODELS:
        d = os.path.join(results_dir, m)
        arms_p = os.path.join(d, "arms.json")
        res_p = os.path.join(d, "arms_results.jsonl")
        arms = json.load(open(arms_p)) if os.path.exists(arms_p) else {}
        res = {}
        if os.path.exists(res_p):
            for line in open(res_p, encoding="utf-8"):
                r = json.loads(line)
                res[r["arm"]] = per_seq(r["seqs"])
        kl[m] = res
        cp = os.path.join(d, "codec_costs.jsonl")
        numel = {}
        if os.path.exists(cp):
            for line in open(cp, encoding="utf-8"):
                r = json.loads(line)
                numel[r["matrix"]] = r["numel"]
        out["gates"]["G1"][m] = SK.gate_status("G1", m, g1(arms, numel), {}, [])
        rep_p = os.path.join(d, "arms_repeat.jsonl")
        g2 = None
        if os.path.exists(rep_p):
            g2 = True
            for line in open(rep_p, encoding="utf-8"):
                r = json.loads(line)
                if r["arm"] not in res:
                    g2 = None
                    break
                g2 &= bool(np.abs(per_seq(r["seqs"]) - res[r["arm"]]).max() <= 1e-6)
        out["gates"]["G2"][m] = SK.gate_status("G2", m, g2, {}, [])
        hp = os.path.join(d, "hashes.json")
        same = None
        if reg is not None and os.path.exists(hp):
            same = json.load(open(hp)) == reg.get(m)
        out["gates"]["samples"][m] = SK.gate_status("samples", m, same, {}, [])
        out["models"][m] = {
            "arms": sorted(res),
            "mean_kl": {a: float(v.mean()) for a, v in res.items()},
        }
    verdicts = {}
    for h, (x, y) in HYPOTHESES.items():
        per_model = {}
        for m in MODELS:
            for b in BUDGETS:
                ax, ay = f"{x}{b}", f"{y}{b}"
                if ax in kl[m] and ay in kl[m]:
                    c = compare(kl[m][ax], kl[m][ay], rng)
                    out["comparisons"][f"{m}|{ax}|{ay}"] = c
                    per_model.setdefault(m, {})[b] = c["judgement"]
        verdicts[h] = verdict(per_model)
    out["verdict_status"] = vs = SK.verdict_status(
        {g: v for g, v in out["gates"].items()}
    )
    out["verdicts"] = (
        {h: "WITHHELD" for h in verdicts} if vs == "WITHHELD" else verdicts
    )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = score(a.results)
    json.dump(r, open(a.out, "w"), indent=1)
    for g, per in r["gates"].items():
        print(g, {m: v["status"] for m, v in per.items()})
    print("VERDICT STATUS", r["verdict_status"])
    for h, v in r["verdicts"].items():
        print(h, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

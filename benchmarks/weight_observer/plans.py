"""EXPLORATORY: exact mixed-precision plans per predictor, and their measured damage.

    # Atlas CPU, with turboquant_pro (feat/weight-plan) importable:
    python -m weight_observer.plans make --run R --explore E --model KEY --out weight_observer/planned/KEY.json
    # GPU (NRP), plans shipped inside the pinned code tar:
    python -m weight_observer.plans eval --model-path M --text T --plans P --out O

``make`` turns each predictor's per-matrix table into a ``tqp.weight_cost_table/1`` and solves
the multiple-choice knapsack exactly (``turboquant_pro.weight_plan``) at every Part III rate, on
the same choices (3, 4, 5, 6, 8 bits), codec and stored-size accounting as the registered
variants. Predictors: the five registered ones, the exploratory ``exact_block`` and
``fisher_tok``, and, when ``sensitivity.jsonl`` exists, ``oracle`` (measured single-matrix KL),
the ceiling for any per-matrix plan. ``eval`` measures each plan's KL from the full-precision
model on the registered evaluation text, exactly as ``run.py`` measures a variant, so plans sit
on the same scale as the 200 random variants and the uniform controls.

Nothing here is a verdict: the predictor tables and the models are the ones Part III selected on.
A planning claim needs fresh models under a new registration.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .variants import GEN_LEVELS, RATES

PLAN_PREDICTORS = ("raw", "act", "fisher", "outdiag", "observer")


def cost_tables(run: str, explore: str, model: str) -> dict:
    """{predictor: tqp.weight_cost_table/1 dict}."""
    from .explore_score import single_kl

    tbl = json.load(open(os.path.join(run, "tables.json")))
    ex = json.load(open(os.path.join(explore, "explore.json")))
    names, levels = ex["names"], ex["levels"]
    mats = {n: {"numel": int(k), "group": 128} for n, k in zip(names, ex["numel"])}
    c = np.load(os.path.join(explore, "c.npy"))
    costs = {
        p: {n: {b: tbl[n][str(b)][p] for b in GEN_LEVELS} for n in names}
        for p in PLAN_PREDICTORS
    }
    costs["exact_block"] = {
        n: {b: float((c[:, j, levels.index(b)] ** 2).mean()) for b in GEN_LEVELS}
        for j, n in enumerate(names)
    }
    costs["fisher_tok"] = {
        n: {b: ex["fisher_tok"][n][levels.index(b)] for b in GEN_LEVELS} for n in names
    }
    sens = single_kl(os.path.join(explore, "sensitivity.jsonl"))
    if sens and all((n, b) in sens for n in names for b in GEN_LEVELS if b != 8):
        costs["oracle"] = {
            n: {b: (0.0 if b == 8 else max(sens[(n, b)], 0.0)) for b in GEN_LEVELS}
            for n in names
        }
    prov = {"run": run, "explore": explore, "levels": list(GEN_LEVELS)}
    return {
        p: {
            "schema": "tqp.weight_cost_table/1",
            "model": model,
            "predictor": p,
            "matrices": mats,
            "costs": {n: {str(b): v for b, v in d.items()} for n, d in cs.items()},
            "provenance": prov,
        }
        for p, cs in costs.items()
    }


def make(a) -> int:
    from turboquant_pro import weight_plan as W

    out = {"model": a.model, "rates": list(RATES), "plans": {}, "records": {}}
    for pred, doc in cost_tables(a.run, a.explore, a.model).items():
        t = W.CostTable.from_dict(doc)
        for r in RATES:
            p = W.solve(t, W.budget_for_rate(t, r))
            pid = f"p{r}-{pred}"
            out["plans"][pid] = p.bits
            rec = p.as_dict()
            rec.pop("bits")
            out["records"][pid] = rec
            print(
                f"{pid:22s} cost {p.cost:.4g} gap/cost {p.gap / p.cost:.1e}", flush=True
            )
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, sort_keys=True)
    return 0


def evaluate(a) -> int:
    import torch

    from . import run as R
    from .measure import apply_variant, kl_per_sequence

    os.makedirs(a.out, exist_ok=True)
    env = R.environment()
    ep = os.path.join(a.out, "env.json")
    if os.path.exists(ep) and json.load(open(ep)) != env:
        raise SystemExit(f"started on {json.load(open(ep))}, now {env}")
    json.dump(env, open(ep, "w"))
    from transformers import AutoTokenizer

    plans = json.load(open(a.plans))["plans"]
    if a.only:
        keep = set(a.only.split(","))
        plans = {p: b for p, b in plans.items() if p.split("-", 1)[1] in keep}
    tok = AutoTokenizer.from_pretrained(a.model_path)
    ref = R.load(a.model_path, a.device)
    var = R.load(a.model_path, a.device)
    text = open(f"{a.text}/test.txt", encoding="utf-8").read()
    evalq = [c[None].to(a.device) for c in R.chunks(tok, text, R.N_EVAL)]
    rp = os.path.join(a.out, "plans_results.jsonl")
    done = set()
    if os.path.exists(rp):
        for line in open(rp, encoding="utf-8"):
            try:
                done.add(json.loads(line)["plan"])
            except (ValueError, KeyError):
                pass
    with open(rp, "a", encoding="utf-8") as fo, torch.no_grad():
        for pid, bits in plans.items():
            if pid in done:
                continue
            t0 = time.time()
            apply_variant(ref, var, bits)
            per = kl_per_sequence(ref, var, evalq)
            fo.write(json.dumps({"plan": pid, "seqs": per}) + "\n")
            fo.flush()
            kl = sum(s["kl_sum"] for s in per) / sum(s["tokens"] for s in per)
            print(f"[plan] {pid} kl/token {kl:.4f} {time.time() - t0:.1f}s", flush=True)
    print("PLANS_DONE", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--run", required=True)
    mk.add_argument("--explore", required=True)
    mk.add_argument("--model", required=True)
    mk.add_argument("--out", required=True)
    ev = sub.add_parser("eval")
    ev.add_argument("--model-path", required=True)
    ev.add_argument("--text", required=True)
    ev.add_argument("--plans", required=True)
    ev.add_argument("--out", required=True)
    ev.add_argument("--device", default="cuda")
    ev.add_argument(
        "--only", default="", help="comma list of predictors to evaluate (default all)"
    )
    a = ap.parse_args(argv)
    return make(a) if a.cmd == "make" else evaluate(a)


if __name__ == "__main__":
    raise SystemExit(main())

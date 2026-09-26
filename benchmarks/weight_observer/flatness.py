"""EXPLORATORY: how flat is the optimum around the Fisher plan?

    python -m weight_observer.flatness make --model qwen2.5-1.5b     # CPU: the perturbed plans
    python -m weight_observer.flatness score --model qwen2.5-1.5b    # after nrp flat + fetch

The headroom exploration found that a far better ranking of random variants (the additive
oracle) buys no better plan. One explanation is that the optimum is broad and flat, so the
matrices a better table would move barely matter. This measures the flatness directly
instead of inferring it.

A perturbation swaps the bit widths of two matrices of the same type (the same shape in
every layer, so the same stored bytes, scales included): the budget is kept exactly, not
approximately. ``k`` disjoint swaps change ``2k`` matrices. For each Part III rate, each
``k`` in ``KS`` and ``DRAWS`` seeded draws, the plan is the Fisher plan with ``k`` swaps;
the Fisher plan itself is re-measured in the same pod as the reference. ``score`` reports,
per rate and ``k``, the measured KL rise over the Fisher plan (paired over sequences), the
share of stored bits moved, and the additive prediction from the single-matrix sweep.

A flat optimum shows as a KL rise that stays small while many bits move. ``KS`` stops at 32
because at some rates most matrices of a type share one width, and 32 disjoint differing
pairs is the most every rate allows (about 5% of stored bits); the far field is the 200
random variants of Part III. Nothing here is a verdict: the models are the ones Part III
selected on.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KS = (1, 2, 4, 8, 16, 32)
DRAWS = 3
SEED = 20260926


def perturb(plan: dict, numel: dict, k: int, rng) -> tuple[dict, float]:
    """``plan`` with ``k`` disjoint same-type width swaps; also the share of stored weight
    bits moved (sum over changed matrices of numel x |bit change|, over total stored bits).
    """
    out = dict(plan)
    free = set(plan)
    for _ in range(k):
        pairs = [
            (a, b)
            for a in sorted(free)
            for b in sorted(free)
            if a < b and a.split(".")[-1] == b.split(".")[-1] and out[a] != out[b]
        ]
        if not pairs:
            raise ValueError(f"no swap left after {_} of {k}")
        a, b = pairs[rng.integers(len(pairs))]
        out[a], out[b] = out[b], out[a]
        free -= {a, b}
    total = sum(numel[m] * plan[m] for m in plan)
    moved = sum(numel[m] * abs(out[m] - plan[m]) for m in plan) / 2
    return out, float(moved / total)


def make(model: str) -> dict:
    ex = json.load(
        open(os.path.join(HERE, "results", model, "explore", "explore.json"), "rb")
    )
    numel = dict(zip(ex["names"], (int(n) for n in ex["numel"])))
    src = json.load(open(os.path.join(HERE, "planned", f"{model}.json"), "rb"))["plans"]
    rng = np.random.default_rng(SEED)
    plans, moved = {}, {}
    for pid in sorted(p for p in src if p.endswith("-fisher")):
        rate = pid.split("-")[0]
        plans[pid] = src[pid]
        for k in KS:
            for d in range(DRAWS):
                q = f"{rate}-k{k:02d}d{d}"
                plans[q], moved[q] = perturb(src[pid], numel, k, rng)
    return {"model": model, "seed": SEED, "plans": plans, "moved": moved}


def score(model: str) -> dict:
    ex = os.path.join(HERE, "results", model, "explore")

    def per_seq(seqs):
        return np.array([s["kl_sum"] / s["tokens"] for s in seqs])

    meas = {}
    with open(os.path.join(ex, "flatness", "plans_results.jsonl"), "rb") as f:
        for line in f:
            r = json.loads(line)
            meas[r["plan"]] = per_seq(r["seqs"])
    sens = {}
    with open(os.path.join(ex, "sensitivity.jsonl"), "rb") as f:
        for line in f:
            r = json.loads(line)
            sens[(r["matrix"], r["bits"])] = per_seq(r["seqs"]).mean()
    spec = json.load(
        open(os.path.join(HERE, "planned", f"{model}.flatness.json"), "rb")
    )

    def additive(bits):
        return sum(sens[(m, b)] for m, b in bits.items() if b != 8)

    rng = np.random.default_rng(0)
    out = {"model": model, "rates": {}}
    for ref in sorted(p for p in spec["plans"] if p.endswith("-fisher")):
        rate, f = ref.split("-")[0], meas[ref]
        fa = additive(spec["plans"][ref])
        rows = {}
        for k in KS:
            ids = [f"{rate}-k{k:02d}d{d}" for d in range(DRAWS)]
            rel = np.array([meas[i].mean() / f.mean() - 1 for i in ids])
            d = np.mean([meas[i] - f for i in ids], axis=0)  # per sequence, over draws
            b = d[rng.integers(0, len(d), (10_000, len(d)))].mean(axis=1)
            rows[k] = {
                "kl_rise_rel": float(rel.mean()),
                "kl_rise_rel_draws": [round(float(x), 4) for x in rel],
                "kl_rise_ci": [
                    float(np.percentile(b, 2.5)),
                    float(np.percentile(b, 97.5)),
                ],
                "bits_moved": float(np.mean([spec["moved"][i] for i in ids])),
                "additive_rise_rel": float(
                    np.mean([additive(spec["plans"][i]) / fa - 1 for i in ids])
                ),
            }
        out["rates"][rate] = {"fisher": float(f.mean()), "k": rows}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("make", "score"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "make":
        doc = make(a.model)
        path = a.out or os.path.join(HERE, "planned", f"{a.model}.flatness.json")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=1, sort_keys=True)
        print(f"{len(doc['plans'])} plans -> {path}")
        return 0
    r = score(a.model)
    for rate, v in r["rates"].items():
        print(f"{rate}: fisher {v['fisher']:.4f}")
        for k, row in v["k"].items():
            print(
                f"   k={k:2d} (bits moved {row['bits_moved']:.1%}): measured "
                f"{row['kl_rise_rel']:+.1%} draws {row['kl_rise_rel_draws']} | additive "
                f"{row['additive_rise_rel']:+.1%}"
            )
    if a.out:
        json.dump(r, open(a.out, "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

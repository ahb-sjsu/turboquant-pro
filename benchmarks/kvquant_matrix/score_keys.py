#!/usr/bin/env python3
"""Registered scorer of observer-advantage Part II (docs/PREREG_observer_advantage_keys.md).

    python score_keys.py --root /data/keys --lbroot ~/LongBench/LongBench \\
        --out results_keys.json

Layout under --root: ``<model_key>/<arm>/`` holding the harness outputs
(``<task>.<shard>.jsonl``, ``config.<shard>.json``) and ``ppl_chunks.jsonl``.

Every cell is verified before it is read: its config sidecars must agree with one
another and with the arm's registered env line (keys_grid.ARMS), and a staged arm
must carry its key-coding record. A cell that fails verification is reported as
such and scores nothing.

Units of comparison are paired: LongBench per-sample score (the official metric,
first-line rule for trec/triviaqa) and WikiText-2 per-chunk NLL per token. For arm
X against reference Y: mean paired difference (positive = X better), a 95%
percentile bootstrap interval over the pairing units (10,000 resamples, seed 0),
and the floor |mean(Y_jit) - mean(Y)|: the reference moved by a one-ulp jitter of
every key, i.e. by fp16 rounding alone (runs are bit-deterministic).
X is MATERIALLY BETTER when the interval's lower end is above 0 and the mean
exceeds twice the floor; MATERIALLY WORSE symmetrically.

Gates carry a machine-readable status per model (Amendment 5): PASS, PENDING (a
cell the gate reads has not run), FAIL_UNEXPLAINED, FAIL_EXPLAINED (explained by
an amendment made before the verdicts were seen) or FAIL_EXPLAINED_POSTHOC
(explained after). An explanation lives in ``gate_dispositions.json``, names its
amendment, and pins the observed numbers it explains, so a rerun that changes them
is unexplained again. The report's ``verdict_status`` follows from the gates: any
FAIL_UNEXPLAINED withholds every verdict (the prereg's "stop scoring"), any PENDING
makes them PROVISIONAL, and a post-hoc explanation stays attached to every verdict.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keys_grid as KG  # noqa: E402

N_BOOT = 10_000
FIRST_LINE = {"trec", "triviaqa", "samsum", "lsht"}
G0_MIN_SAME = 1.0  # share of trec predictions identical to g0_native
G0_PPL_REL = 1e-6  # |ppl / ppl_g0_native - 1|: the residual form is exact

PASS, PENDING = "PASS", "PENDING"
FAIL_UNEXPLAINED, FAIL_EXPLAINED = "FAIL_UNEXPLAINED", "FAIL_EXPLAINED"
FAIL_EXPLAINED_POSTHOC = "FAIL_EXPLAINED_POSTHOC"
DISPOSITIONS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "gate_dispositions.json")


def load_dispositions(path: str = DISPOSITIONS) -> list:
    """The registered explanations of gate failures (empty if none)."""
    if not os.path.exists(path):
        return []
    doc = json.load(open(path, encoding="utf-8"))
    if doc.get("schema") != "tqp-gate-dispositions/1":
        raise ValueError(f"{path}: not a tqp-gate-dispositions/1 file")
    for d in doc["dispositions"]:
        missing = {"gate", "model", "observed", "amendment", "after_verdicts"} - set(d)
        if missing or d["model"] not in KG.MODELS:
            raise ValueError(f"{path}: malformed disposition {d.get('model')}: {missing}")
    return doc["dispositions"]


# A disposition relates to one gate result in exactly one of four ways.
IRRELEVANT = "IRRELEVANT"  # another gate or another model
OTHER_NUMBERS = "OTHER_NUMBERS"  # this gate and model, other observed numbers
EXPLAINS = "EXPLAINS"  # pins these numbers, recorded before the verdicts
EXPLAINS_POSTHOC = "EXPLAINS_POSTHOC"  # pins these numbers, recorded after
KINDS = (IRRELEVANT, OTHER_NUMBERS, EXPLAINS, EXPLAINS_POSTHOC)


def pins(pinned: dict, observed: dict, tol: float = 1e-9) -> bool:
    """The disposition's pinned numbers are these observed numbers: the same keys,
    every value present on both sides and within ``tol``."""
    return set(pinned) == set(observed) and all(
        pinned[k] is not None
        and observed[k] is not None
        and abs(observed[k] - pinned[k]) <= tol
        for k in pinned
    )


def classify(d: dict, gate: str, model: str, observed: dict) -> str:
    """Which of the four ways ``d`` relates to this gate result (the only place a
    disposition's content is read; ``decide`` sees nothing else)."""
    if d["gate"] != gate or d["model"] != model:
        return IRRELEVANT
    if not pins(d["observed"], observed):
        return OTHER_NUMBERS
    return EXPLAINS_POSTHOC if d["after_verdicts"] else EXPLAINS


def decide(passed, kinds: frozenset) -> str:
    """The gate status, a function of the gate's outcome and of WHICH kinds of
    disposition exist: its domain is {None, True, False} x the 16 subsets of KINDS,
    and ``tests/test_score_keys.py`` checks every one of the 48 points against the
    specification, so this function is proven, not sampled. Taking a set makes the
    order and multiplicity of dispositions irrelevant by construction."""
    if not kinds <= frozenset(KINDS):
        raise ValueError(f"unknown disposition kinds {set(kinds) - set(KINDS)}")
    if passed is None:
        return PENDING
    if passed is True:
        return PASS
    if passed is not False:
        raise ValueError(f"passed must be True, False or None, not {passed!r}")
    if EXPLAINS_POSTHOC in kinds:
        return FAIL_EXPLAINED_POSTHOC
    if EXPLAINS in kinds:
        return FAIL_EXPLAINED
    return FAIL_UNEXPLAINED


def gate_status(gate: str, model: str, passed, observed: dict, dispositions: list) -> dict:
    """Status of one gate on one model: ``decide`` over the kinds of the registered
    dispositions, with the amendments behind it named for the report. ``passed`` is
    True, False, or None when a cell the gate reads is missing."""
    kinds = [classify(d, gate, model, observed) for d in dispositions]
    out = {"status": decide(passed, frozenset(kinds))}
    if out["status"] in (FAIL_EXPLAINED, FAIL_EXPLAINED_POSTHOC):
        out["amendment"] = ", ".join(
            d["amendment"]
            for d, k in zip(dispositions, kinds)
            if k in (EXPLAINS, EXPLAINS_POSTHOC)
        )
    elif out["status"] == FAIL_UNEXPLAINED and OTHER_NUMBERS in kinds:
        names = ", ".join(
            d["amendment"] for d, k in zip(dispositions, kinds) if k == OTHER_NUMBERS
        )
        out["reason"] = f"{names} explain other numbers than these"
    return out


# What a gate status lets the verdicts be, as a rank in a total order; the verdict
# status of a report is the join (maximum) over every gate result in it.
VERDICT_RANK = {
    PASS: 0,
    FAIL_EXPLAINED: 0,
    FAIL_EXPLAINED_POSTHOC: 1,
    PENDING: 2,
    FAIL_UNEXPLAINED: 3,
}
VERDICTS = ("FINAL", "FINAL_WITH_POSTHOC_EXPLANATION", "PROVISIONAL", "WITHHELD")


def verdict_status(gates: dict) -> str:
    """What the gates allow the verdicts to be: {gate: {model: {"status": ..}}}.
    A join over a total order, so adding a gate result can only keep or worsen it.
    A report that checked no gate cannot be final, so none is an error."""
    sts = [g["status"] for per in gates.values() for g in per.values()]
    if not sts:
        raise ValueError("no gate result: a report that checked nothing has no status")
    return VERDICTS[max(VERDICT_RANK[s] for s in sts)]


def _env(line: str) -> dict:
    return dict(kv.split("=", 1) for kv in line.split())


def verify(cell: str, arm_env: dict) -> str | None:
    """None if the cell's sidecars match the registered arm, else the reason."""
    cfgs = [json.load(open(f)) for f in sorted(glob.glob(f"{cell}/config.*.json"))]
    if not cfgs:
        return "no config sidecars"
    body = {json.dumps({k: v for k, v in c.items() if k != "shard"}, sort_keys=True)
            for c in cfgs}
    if len(body) > 1:
        return "sidecars disagree across shards"
    c = cfgs[0]
    if int(arm_env.get("NOQUANT", "0")):
        return None if c.get("noquant") else "expected NOQUANT=1"
    want = {"codebook": arm_env["CODEBOOK"], "key_bits": int(arm_env["KEY_BITS"]),
            "sink": int(arm_env["SINK"]), "outlier_frac": float(arm_env["OUTLIER_FRAC"]),
            "hot": int(arm_env["HOT"]), "group": int(arm_env["GROUP"]),
            "prerope": int(arm_env["PREROPE"])}
    for k, v in want.items():
        if c.get(k) != v:
            return f"{k}={c.get(k)!r}, registered {v!r}"
    kc = c.get("key_coding")
    stages = {"KEY_BASIS": ("key_basis", "native"), "BASIS_FIT": ("basis_fit", "prefill"),
              "KEY_ALLOC": ("key_alloc", "uniform"), "BYTE_MATCH": ("byte_match", "0"),
              "KEY_JITTER": ("key_jitter", "0")}
    staged = any(arm_env.get(e, d) != d for e, (_, d) in stages.items())
    if staged != (kc is not None):
        return "key-coding record missing" if staged else "unexpected key-coding record"
    if kc:
        for e, (k, d) in stages.items():
            if str(kc.get(k)) != str(arm_env.get(e, d)):
                return f"{k}={kc.get(k)!r}, registered {arm_env.get(e, d)!r}"
    return None


def task_scores(cell: str, task: str, metrics) -> dict:
    rows = {}
    for f in glob.glob(f"{cell}/{task}.*.jsonl"):
        for line in open(f, encoding="utf-8"):
            o = json.loads(line)
            rows[o["idx"]] = o
    out = {}
    for i, o in rows.items():
        pred = o["pred"].lstrip("\n").split("\n")[0] if task in FIRST_LINE else o["pred"]
        out[i] = max(metrics[task](pred, gt, all_classes=o["all_classes"])
                     for gt in o["answers"]) * 100
    return out


def preds(cell: str, task: str) -> dict:
    rows = {}
    for f in glob.glob(f"{cell}/{task}.*.jsonl"):
        for line in open(f, encoding="utf-8"):
            o = json.loads(line)
            rows[o["idx"]] = o["pred"]
    return rows


def chunk_nll(cell: str) -> dict:
    p = f"{cell}/ppl_chunks.jsonl"
    if not os.path.exists(p):
        return {}
    return {o["chunk"]: o["nll"] / o["tokens"]
            for o in map(json.loads, open(p, encoding="utf-8"))}


def key_bits(cell: str) -> float | None:
    vals = [o["key_bits"]["total"]
            for f in glob.glob(f"{cell}/qasper.*.jsonl")
            for o in map(json.loads, open(f, encoding="utf-8")) if o.get("key_bits")]
    return round(float(np.mean(vals)), 4) if vals else None


def paired(x: dict, y: dict, higher_better: bool = True):
    keys = sorted(set(x) & set(y))
    if not keys:
        return None
    d = np.array([x[k] - y[k] for k in keys], dtype=np.float64)
    if not higher_better:
        d = -d
    rng = np.random.default_rng(0)
    boot = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(1)
    return {"n": len(keys), "mean": float(d.mean()),
            "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5))}


def _mean(x: dict):
    return float(np.mean(list(x.values()))) if x else None


def judge(delta, floor):
    if delta is None or floor is None:
        return "missing"
    bar = 2 * floor
    if delta["lo"] > 0 and delta["mean"] > bar:
        return "better"
    if delta["hi"] < 0 and -delta["mean"] > bar:
        return "worse"
    return "neither"


def k3_verdict(rows: list, n_models: int = len(KG.TIER_A)) -> str:
    """K3: O is better (either endpoint) in >= 2 models against EVERY basis control.
    Amendment 3: undecided until every control is scored on every Tier A model."""
    if any(r["scored"] < n_models for r in rows):
        return "INCOMPLETE"
    return "HOLDS" if all(r["models_better"] >= 2 for r in rows) else "DOES NOT HOLD"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--lbroot", default=os.environ.get("LBROOT", ""))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.lbroot)
    from metrics import classification_score, qa_f1_score  # LongBench's own

    metrics = {"trec": classification_score, "triviaqa": qa_f1_score, "qasper": qa_f1_score}

    report = {"models": {}, "verdicts": {}}
    data = {}
    for mk in KG.MODELS:
        m = report["models"][mk] = {"cells": {}, "g0": {}}
        data[mk] = {}
        for arm, line in {**KG.ARMS, **KG.G0_ARMS}.items():
            cell = f"{a.root}/{mk}/{arm}"
            if not os.path.isdir(cell):
                continue
            why = verify(cell, _env(line))
            if why:
                m["cells"][arm] = {"verified": False, "reason": why}
                continue
            env_path = f"{cell}/env.json"
            ent = {
                "verified": True,
                "key_bits": key_bits(cell),
                "env": json.load(open(env_path)) if os.path.exists(env_path) else None,
            }
            d = {t: task_scores(cell, t, metrics) for t in KG.TASKS}
            d["ppl"] = chunk_nll(cell)
            for t in KG.TASKS:
                ent[t] = _mean(d[t])
            ent["ppl"] = math.exp(_mean(d["ppl"])) if d["ppl"] else None
            m["cells"][arm] = ent
            data[mk][arm] = (d, cell)
        # G0: through the identity codebook every stage reproduces g0_native, the
        # same key path with no stage (fp16 is not the reference: values are still
        # quantized in every quantized arm, identity-key arms included).
        if "g0_native" in data[mk]:
            ref_d, ref_cell = data[mk]["g0_native"]
            for g in KG.G0_ARMS:
                if g == "g0_native" or g not in data[mk]:
                    continue
                d, cell = data[mk][g]
                pr, pf = preds(cell, "trec"), preds(ref_cell, "trec")
                same = np.mean([pr[i] == pf[i] for i in set(pr) & set(pf)]) if pr else None
                pg, pfp = m["cells"][g]["ppl"], m["cells"]["g0_native"]["ppl"]
                rel = abs(pg / pfp - 1) if pg and pfp else None
                ok = same is not None and same >= G0_MIN_SAME and rel is not None and rel <= G0_PPL_REL
                m["g0"][g] = {"same_trec": same, "ppl_rel": rel, "pass": bool(ok)}

    # G1: the shipped arm reproduces the recorded matrix (Tier A)
    rec = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "results_matrix.json")))
    for mk in KG.TIER_A:
        name = mk.replace("-4k", "")
        cell = report["models"][mk]["cells"].get("nf4a", {})
        want_q = rec["models"].get(name, {}).get("nf4a", {}).get("qasper")
        want_p = rec["wikitext2_ppl"].get(name, {}).get("nf4a")
        got_q, got_p = cell.get("qasper"), cell.get("ppl")
        ok = (None not in (want_q, want_p, got_q, got_p)
              and abs(got_q - want_q) <= 1.0 and abs(got_p / want_p - 1) <= 0.01)
        report["models"][mk]["g1"] = {"qasper": [got_q, want_q], "ppl": [got_p, want_p],
                                      "pass": bool(ok)}

    def compare(mk, arm, ref, rep):
        if not all(x in data[mk] for x in (arm, ref, rep)):
            return None
        (dx, _), (dy, _), (dr, _) = data[mk][arm], data[mk][ref], data[mk][rep]
        out = {}
        for ep, hb in (("qasper", True), ("ppl", False), ("trec", True), ("triviaqa", True)):
            delta = paired(dx[ep], dy[ep], hb)
            floor = abs(_mean(dr[ep]) - _mean(dy[ep])) if dr[ep] and dy[ep] else None
            out[ep] = {"delta": delta, "floor": floor, "judgement": judge(delta, floor)}
        return out

    def g0_ok(mk, arm):
        basis = _env(KG.ARMS[arm]).get("KEY_BASIS", "native")
        alloc = _env(KG.ARMS[arm]).get("KEY_ALLOC", "uniform")
        gate = "g0_O_read" if alloc != "uniform" else f"g0_{basis}"
        if gate == "g0_native":
            return "g0_native" in data[mk]  # the reference itself
        return report["models"][mk]["g0"].get(gate, {}).get("pass", False)

    comps = {}
    for mk in KG.MODELS:
        for arm, ref, rep in [c for cs in KG.COMPARISONS.values() for c in cs] + KG.REPORTED:
            r = compare(mk, arm, ref, rep)
            if r is not None:
                envs = {
                    json.dumps(report["models"][mk]["cells"][x]["env"], sort_keys=True)
                    for x in (arm, ref, rep)
                }
                # Amendment 2: arm, reference and floor must share GPU and software.
                r["same_env"] = len(envs) == 1 and "null" not in envs
                r["g0_pass"] = g0_ok(mk, arm) and g0_ok(mk, ref) and r["same_env"]
                comps[f"{mk}|{arm}|{ref}"] = r
    report["comparisons"] = comps

    def tally(arm, ref, eps):
        """Tier-A counts of 'better on every ep' and 'worse on any ep'."""
        better = worse = scored = 0
        for mk in KG.TIER_A:
            r = comps.get(f"{mk}|{arm}|{ref}")
            if r is None or not r["g0_pass"]:
                continue
            scored += 1
            j = [r[e]["judgement"] for e in eps]
            better += all(x == "better" for x in j)
            worse += any(x == "worse" for x in j)
        return better, worse, scored

    def verdict(pairs, eps=("qasper", "ppl")):
        res = []
        for arm, ref, _ in pairs:
            b, w, n = tally(arm, ref, eps)
            if n < len(KG.TIER_A):
                v = "INCOMPLETE"
            elif b >= 2 and w == 0:
                v = "HOLDS"
            elif w >= 2:
                v = "FAILS (reversed)"
            elif b == 0:
                v = "FAILS"
            else:
                v = "INCONCLUSIVE"
            res.append({"arm": arm, "ref": ref, "better": b, "worse": w, "scored": n,
                        "verdict": v})
        return res

    for k, pairs in KG.COMPARISONS.items():
        eps = ("qasper", "ppl")
        if k == "K3":  # observer-specific: O beats every basis control on either endpoint
            rows = []
            for arm, ref, _ in pairs:
                b = n = 0
                for mk in KG.TIER_A:
                    r = comps.get(f"{mk}|{arm}|{ref}")
                    # Amendment 3: a model counts only once both endpoints were measured.
                    if not r or not r["g0_pass"] or any(
                        r[e]["judgement"] == "missing" for e in eps
                    ):
                        continue
                    n += 1
                    b += any(r[e]["judgement"] == "better" for e in eps)
                rows.append({"arm": arm, "ref": ref, "models_better": b, "scored": n})
            report["verdicts"][k] = {"rows": rows, "verdict": k3_verdict(rows)}
        else:
            report["verdicts"][k] = verdict(pairs, eps)
    # Amendment 5: gate statuses, and what they let the verdicts be
    disp = load_dispositions()
    gates = {"G1": {}}
    for mk in KG.TIER_A:
        g1 = report["models"][mk]["g1"]
        got_q, got_p = g1["qasper"][0], g1["ppl"][0]
        missing = None in (got_q, got_p, g1["qasper"][1], g1["ppl"][1])
        gates["G1"][mk] = gate_status(
            "G1", mk, None if missing else g1["pass"],
            {"qasper": got_q, "ppl": got_p}, disp,
        )
    report["gates"] = gates
    report["verdict_status"] = vs = verdict_status(gates)
    if vs == "WITHHELD":
        report["verdicts"] = {k: "WITHHELD" for k in report["verdicts"]}
    json.dump(report, open(a.out, "w"), indent=1, default=float)
    for g, per in gates.items():
        print(g, {m: v["status"] for m, v in per.items()})
    print("VERDICT STATUS", vs)
    for k, v in report["verdicts"].items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Scorer of observer-advantage Part V (docs/PREREG_full_consumer_kv.md).

    python score_part5.py --root /archive/ahb-sjsu/tqp_part5/runs --lbroot ~/LongBench/LongBench \\
        --out results_part5.json

Layout under --root: ``<model_key>/<arm>/`` as the runner writes it (keys_run.py --grid
part5), each LongBench cell beside its WikiText-2 chunks, and a baseline's cell in the
same place under its own name (section 3).

Part II's scorer (score_keys.py) is registered and is not changed; this one imports its
cell readers, statistics, judgement and gate-status machinery, so the two parts read on
one scale, and adds only what Part V registers differently:

  completeness   a cell counts only when every document and every perplexity chunk ran
                 (the runner's ppl.done records the chunk and document limits it ran
                 with); a partial cell is reported, never scored;
  calibration    a calibrated cell must record the registered windows and seed, and the
                 hash of the table it coded with (calib.json); every cell of one model
                 and one weight must have used the same table;
  G0             each arm is gated by the identity codebook through its own allocation
                 path (part5_grid.g0_gate);
  G1             in noise units: on each Tier A model u3 must reproduce Part II's scored
                 u3 within G1_FACTOR times the larger of the repeat difference, the
                 jitter distance and Part II's measured cross-GPU difference;
  verdicts       V1-V4 by Part II's rule; V5 by its own rule against every baseline,
                 scored only when V2 holds, and against ``ea`` alone when AATC's
                 fidelity check failed.

Gate failures are explained, as in Part II, only by a registered disposition
(part5_gate_dispositions.json) pinning the observed numbers.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import part5_grid as KG  # noqa: E402
from score_keys import (  # noqa: E402
    G0_MIN_SAME,
    G0_PPL_REL,
    _env,
    _mean,
    chunk_nll,
    gate_status,
    judge,
    key_bits,
    load_dispositions,
    paired,
    preds,
    task_scores,
    verdict_status,
    verify,
)

DISPOSITIONS = os.path.join(HERE, "part5_gate_dispositions.json")
AATC_FIDELITY = os.path.join(HERE, "part5_aatc_fidelity.json")
DOCS = {"trec": 200, "triviaqa": 200, "qasper": 200}  # LongBench's full tasks
# Endpoints compared per pair; "kl" (per-token KL to fp16 per chunk) is reported only.
EPS = (("qasper", True), ("ppl", False), ("trec", True), ("triviaqa", True), ("kl", False))
BYTES_REL = 0.01  # a baseline's stored bits must match u3's within 1%


def calibrated(env: dict) -> bool:
    return env.get("KEY_ALLOC") in ("head", "fisher") and env.get("BASIS_FIT") == "calib"


def jittered(floor_arm: str) -> str:
    """The arm a jitter arm jitters (u3_jit -> u3)."""
    if not floor_arm.endswith("_jit"):
        raise ValueError(f"{floor_arm}: a floor arm is a jitter arm")
    return floor_arm[: -len("_jit")]


def verify_part5(cell: str, env: dict) -> str | None:
    """None if the cell is the registered Part V arm, complete, and (if calibrated)
    coded with a recorded table at the registered windows; else the reason."""
    why = verify(cell, env)
    if why:
        return why
    tasks = KG.G0_TASKS if env.get("CODEBOOK") == "identity" else KG.TASKS
    for t in tasks:
        n = len(preds(cell, t))
        if n != DOCS[t]:
            return f"incomplete: {t} has {n} of {DOCS[t]} documents"
    done = os.path.join(cell, "ppl.done")
    if not os.path.exists(done):
        return "incomplete: perplexity not finished"
    run = json.load(open(done))
    if run.get("maxchunks", None) != 0 or run.get("docs", None) != 0:
        return f"incomplete: ran with limits {run}"
    if calibrated(env):
        kc = json.load(open(os.path.join(cell, "config.0.json")))["key_coding"]
        want = {"part5_calib_seq": KG.CALIB_SEQ, "part5_calib_n": KG.CALIB_N,
                "fisher_seed": KG.FISHER_SEED if env["KEY_ALLOC"] == "fisher" else 0}
        for k, v in want.items():
            if kc.get(k) != v:
                return f"{k}={kc.get(k)!r}, registered {v!r}"
        if not os.path.exists(os.path.join(cell, "calib.json")):
            return "no calibration record (calib.json)"
    return None


def verify_baseline(cell: str, name: str) -> str | None:
    """A baseline's cell (section 3): its runner's record names the baseline and the
    stored bits per cached element of it and of u3 under the registered accounting."""
    p = os.path.join(cell, "baseline.json")
    if not os.path.exists(p):
        return "no baseline record (baseline.json)"
    rec = json.load(open(p))
    if rec.get("baseline") != name:
        return f"baseline={rec.get('baseline')!r}, expected {name!r}"
    if name in getattr(KG, "MATCHED_BASELINES", KG.BASELINES):
        m = rec.get("matched") or {}
        if m.get("reference") != "u3" or not m.get("reference_bits"):
            return "no matched-bytes record against u3"
        rel = abs(m["arm_bits"] / m["reference_bits"] - 1)
        if rel > BYTES_REL:
            return f"stored bits differ from u3's by {rel:.2%} (> {BYTES_REL:.0%})"
    elif not (rec.get("reported") or {}).get("arm_bits"):  # reported at its own bytes
        return "no stored-bits record"
    for t in KG.TASKS:
        if len(preds(cell, t)) != DOCS[t]:
            return f"incomplete: {t}"
    return None


def chunk_kl(cell: str, nll: dict) -> dict:
    """Per-token KL(fp16 || arm) per chunk (wikitext_ppl.py, KL_REF=1). The fp16 arm is
    its own reference, so its KL is zero by definition; a cell run without the
    reference pass has none."""
    cfg = os.path.join(cell, "config.0.json")
    if os.path.exists(cfg) and json.load(open(cfg)).get("noquant"):
        return {c: 0.0 for c in nll}
    p = os.path.join(cell, "ppl_chunks.jsonl")
    if not os.path.exists(p):
        return {}
    return {o["chunk"]: o["kl"] / o["tokens"]
            for o in map(json.loads, open(p, encoding="utf-8")) if "kl" in o}


def fitting(cell: str) -> dict | None:
    """Reported: seconds spent fitting the key coding per qasper document, and its
    share of the documents' generation time."""
    fit = gen = 0.0
    n = 0
    for f in glob.glob(f"{cell}/qasper.*.jsonl"):
        for o in map(json.loads, open(f, encoding="utf-8")):
            if o.get("timing"):
                fit += o["timing"]["fit_s"]
                gen += o["timing"]["gen_s"]
                n += 1
    if not n:
        return None
    return {"docs": n, "fit_s_per_doc": fit / n, "share_of_generation": fit / gen if gen else None}


def read_cell(cell: str, metrics) -> tuple[dict, dict]:
    d = {t: task_scores(cell, t, metrics) for t in KG.TASKS}
    d["ppl"] = chunk_nll(cell)
    d["kl"] = chunk_kl(cell, d["ppl"])
    ent = {"verified": True, "key_bits": key_bits(cell)}
    for t in KG.TASKS:
        ent[t] = _mean(d[t])
    ent["ppl"] = math.exp(_mean(d["ppl"])) if d["ppl"] else None
    ent["kl"] = _mean(d["kl"])
    ent["fitting"] = fitting(cell)
    for f in ("env.json", "calib.json"):
        p = os.path.join(cell, f)
        ent[f[:-5]] = json.load(open(p)) if os.path.exists(p) else None
    return d, ent


def g1_check(got: dict, rep: dict, jit: dict, rec: dict) -> tuple[bool | None, dict]:
    """G1 on one model (section 2 and section 6). ``got``/``rep``/``jit`` are the A10
    u3, its repeat and u3_jit cells' means; ``rec`` Part II's scored u3. Perplexity is
    compared as a relative difference, qasper in points."""
    out = {}
    passed = True
    for ep in ("qasper", "ppl"):
        vals = (got.get(ep), rep.get(ep), jit.get(ep), rec.get(ep))
        if None in vals:
            return None, {"missing": ep}
        g, r, j, w = vals
        if ep == "ppl":
            diff, repeat, jitter = abs(g / w - 1), abs(r / g - 1), abs(j / g - 1)
        else:
            diff, repeat, jitter = abs(g - w), abs(r - g), abs(j - g)
        tol = KG.G1_FACTOR * max(repeat, jitter, KG.G1_CROSS_GPU[ep])
        out[ep] = {"now": g, "recorded": w, "diff": diff, "repeat": repeat,
                   "jitter": jitter, "tolerance": tol, "pass": diff <= tol}
        passed = passed and diff <= tol
    return passed, out


def rank_agreement(a_path: str, b_path: str) -> dict | None:
    """Reported: per (layer, KV head), Spearman correlation across channels between two
    calibrated weight tables (where the chain changes which channels matter)."""
    try:
        import torch
    except ImportError:
        return None
    a, b = torch.load(a_path, map_location="cpu"), torch.load(b_path, map_location="cpu")
    rho = []
    for li in sorted(set(a) & set(b)):
        wa, wb = a[li][0].numpy(), b[li][0].numpy()
        for h in range(wa.shape[0]):
            ra = np.argsort(np.argsort(wa[h])).astype(float)
            rb = np.argsort(np.argsort(wb[h])).astype(float)
            rho.append(float(np.corrcoef(ra, rb)[0, 1]))
    if not rho:
        return None
    q = np.percentile(rho, [10, 50, 90])
    return {"heads": len(rho), "mean": float(np.mean(rho)),
            "p10": float(q[0]), "median": float(q[1]), "p90": float(q[2])}


def v_rule(rows: list) -> str:
    """V1-V4 (Part II's rule): HOLDS when better on qasper and perplexity in >= 2 Tier A
    models and worse on either in none; FAILS when better in none ("reversed" when
    worse in >= 2); otherwise INCONCLUSIVE. Undecided until every model is scored."""
    if len(rows) < len(KG.TIER_A):
        return "INCOMPLETE"
    better = sum(all(r[e] == "better" for e in ("qasper", "ppl")) for r in rows)
    worse = sum(any(r[e] == "worse" for e in ("qasper", "ppl")) for r in rows)
    if better >= 2 and worse == 0:
        return "HOLDS"
    if worse >= 2:
        return "FAILS (reversed)"
    if better == 0:
        return "FAILS"
    return "INCONCLUSIVE"


def v5_rule(per_baseline: dict) -> str:
    """V5: HOLDS when, against every baseline, better on qasper or perplexity in >= 2
    models, and worse on neither in any model; FAILS when worse on either in >= 2
    models against any baseline; otherwise INCONCLUSIVE. A baseline is judged on its
    endpoints in part5_grid.V5_ENDPOINTS (Expected Attention on qasper alone)."""
    if not per_baseline or any(len(r) < len(KG.TIER_A) for r in per_baseline.values()):
        return "INCOMPLETE"

    def eps(b):
        return getattr(KG, "V5_ENDPOINTS", {}).get(b, ("qasper", "ppl"))

    worse = {b: sum(any(r[e] == "worse" for e in eps(b)) for r in rows)
             for b, rows in per_baseline.items()}
    if any(w >= 2 for w in worse.values()):
        return "FAILS"
    better = {b: sum(any(r[e] == "better" for e in eps(b)) for r in rows)
              for b, rows in per_baseline.items()}
    if all(v >= 2 for v in better.values()) and not any(worse.values()):
        return "HOLDS"
    return "INCONCLUSIVE"


def aatc_fidelity(path: str = AATC_FIDELITY):
    """True / False from the registered fidelity record, None while it has not run."""
    if not os.path.exists(path):
        return None
    rec = json.load(open(path, encoding="utf-8"))
    if rec.get("schema") != "tqp-part5-aatc-fidelity/1" or not isinstance(rec.get("pass"), bool):
        raise ValueError(f"{path}: not a tqp-part5-aatc-fidelity/1 record")
    return rec["pass"]


def score(root: str, metrics, dispositions: list, fidelity, recorded: dict) -> dict:
    report = {"models": {}, "comparisons": {}, "verdicts": {}, "reported": {}}
    data = {}
    for mk in KG.MODELS:
        m = report["models"][mk] = {"cells": {}, "g0": {}}
        data[mk] = {}
        lines = {**KG.ARMS, **KG.G0_ARMS}
        for arm in list(lines) + list(KG.BASELINES):
            cell = os.path.join(root, mk, arm)
            if not os.path.isdir(cell):
                continue
            why = (verify_baseline(cell, arm) if arm in KG.BASELINES
                   else verify_part5(cell, _env(lines[arm])))
            if why:
                m["cells"][arm] = {"verified": False, "reason": why}
                continue
            d, ent = read_cell(cell, metrics)
            m["cells"][arm] = ent
            data[mk][arm] = (d, cell)
        # One table per (model, weight): every calibrated cell must have coded with it.
        tables = {}
        for arm, ent in m["cells"].items():
            if ent.get("verified") and ent.get("calib"):
                tables.setdefault(ent["calib"]["file"], set()).add(ent["calib"]["sha256"])
        for arm, ent in list(m["cells"].items()):
            c = ent.get("calib")
            if ent.get("verified") and c and len(tables[c["file"]]) > 1:
                m["cells"][arm] = {"verified": False,
                                   "reason": f"{c['file']}: cells coded with different tables"}
                data[mk].pop(arm, None)
        # G0: the identity codebook through each path reproduces g0_native exactly.
        if "g0_native" in data[mk]:
            ref = data[mk]["g0_native"][1]
            for g in KG.G0_ARMS:
                if g == "g0_native" or g not in data[mk]:
                    continue
                pr, pf = preds(data[mk][g][1], "trec"), preds(ref, "trec")
                same = float(np.mean([pr[i] == pf[i] for i in pf])) if pf else None
                pg, p0 = m["cells"][g]["ppl"], m["cells"]["g0_native"]["ppl"]
                rel = abs(pg / p0 - 1) if pg and p0 else None
                ok = (same is not None and same >= G0_MIN_SAME and rel is not None
                      and rel <= G0_PPL_REL and set(pr) == set(pf))
                m["g0"][g] = {"same_trec": same, "ppl_rel": rel, "pass": bool(ok)}

    def g0_ok(mk, arm):
        if arm in KG.BASELINES:
            return arm in data[mk]  # no injection path to gate; the record is verified
        gate = KG.g0_gate(arm)
        if gate == "g0_native":
            return "g0_native" in data[mk]
        return report["models"][mk]["g0"].get(gate, {}).get("pass", False)

    def compare(mk, arm, ref, rep):
        # Section 4's floor is |mean(u3_jit) - mean(u3)|: the jitter arm against the
        # arm it jitters, whatever the reference. (Part II's scorer measured the jitter
        # arm against the reference, which for a reference other than u3, such as V1's
        # u3_read or V5's baselines, puts the reference's own effect into the floor.)
        base = jittered(rep)
        if not all(x in data[mk] for x in (arm, ref, rep, base)):
            return None
        (dx, _), (dy, _) = data[mk][arm], data[mk][ref]
        (dr, _), (db, _) = data[mk][rep], data[mk][base]
        out = {}
        for ep, hb in EPS:
            delta = paired(dx[ep], dy[ep], hb)
            floor = abs(_mean(dr[ep]) - _mean(db[ep])) if dr[ep] and db[ep] else None
            out[ep] = {"delta": delta, "floor": floor, "judgement": judge(delta, floor)}
        cells = report["models"][mk]["cells"]
        used = (arm, ref, rep, base)
        envs = {json.dumps(cells[x]["env"], sort_keys=True) for x in used}
        out["same_env"] = len(envs) == 1 and "null" not in envs
        out["g0_pass"] = all(g0_ok(mk, x) for x in used) and out["same_env"]
        return out

    triples = [c for cs in KG.COMPARISONS.values() for c in cs] + KG.REPORTED
    for mk in KG.MODELS:
        for arm, ref, rep in triples:
            r = compare(mk, arm, ref, rep)
            if r is not None:
                report["comparisons"][f"{mk}|{arm}|{ref}"] = r

    def rows(arm, ref):
        out = []
        for mk in KG.TIER_A:
            r = report["comparisons"].get(f"{mk}|{arm}|{ref}")
            if r and r["g0_pass"] and all(r[e]["judgement"] != "missing" for e in ("qasper", "ppl")):
                out.append({e: r[e]["judgement"] for e in ("qasper", "ppl")} | {"model": mk})
        return out

    for k in ("V1", "V2", "V3", "V4"):
        (arm, ref, _), = KG.COMPARISONS[k]
        rr = rows(arm, ref)
        report["verdicts"][k] = {"arm": arm, "ref": ref, "rows": rr, "verdict": v_rule(rr)}
    baselines = [ref for _, ref, _ in KG.COMPARISONS["V5"]]
    note = None
    if fidelity is False:
        baselines = [b for b in baselines if b != "aatc"]
        note = "AATC's reimplementation failed its fidelity check: V5 is scored against ea alone"
    elif fidelity is None:
        note = "AATC's fidelity check has not run"
    per = {b: rows("u3_head", b) for b in baselines}
    v5 = "INCOMPLETE" if fidelity is None else v5_rule(per)
    v2 = report["verdicts"]["V2"]["verdict"]
    if v2 != "HOLDS":
        v5 = "NOT SCORED (V2 does not hold)" if v2 != "INCOMPLETE" else "INCOMPLETE"
    report["verdicts"]["V5"] = {"rows": per, "verdict": v5, "note": note}

    # G1 in noise units, and what the gates let the verdicts be (Part II's Amendment 5).
    gates = {"G1": {}}
    for mk in KG.TIER_A:
        cells = report["models"][mk]["cells"]
        get = {x: cells.get(x, {}) for x in ("u3", "u3_rep", "u3_jit")}
        if not all(c.get("verified") for c in get.values()):
            passed, detail = None, {"missing": [x for x, c in get.items() if not c.get("verified")]}
        else:
            passed, detail = g1_check(get["u3"], get["u3_rep"], get["u3_jit"], recorded[mk])
        report["models"][mk]["g1"] = detail
        obs = {ep: detail[ep]["now"] for ep in ("qasper", "ppl")} if passed is not None else {}
        gates["G1"][mk] = gate_status("G1", mk, passed, obs, dispositions)
    report["gates"] = gates
    report["verdict_status"] = vs = verdict_status(gates)
    if vs == "WITHHELD":
        report["verdicts"] = {k: "WITHHELD" for k in report["verdicts"]}

    for mk in KG.MODELS:  # reported: where the chain changes which channels matter
        a = os.path.join(root, mk, "alloc_calib_head.pt")
        b = os.path.join(root, mk, "alloc_calib_fisher.pt")
        if os.path.exists(a) and os.path.exists(b):
            report["reported"].setdefault("rank_head_vs_fisher", {})[mk] = rank_agreement(a, b)
    return report


# The published comparison AATC's reimplementation must reproduce (section 3): AATC at an
# average of 2 bits against KIVI at 2 bits on Qwen2.5-7B-Instruct, LongBench qasper F1
# (arXiv 2608.14191: 0.386, Table "bit-sweep"; KIVI 0.187, Table 1 / "per-subtask").
PUBLISHED = {"aatc2": 0.386, "kivi2": 0.187, "task": "qasper"}


def fidelity_record(root: str, metrics) -> dict:
    """AATC's fidelity check on FIDELITY_MODEL: PASS when the paired 95% bootstrap
    interval of qasper F1, aatc2 minus kivi2 over all 200 documents, lies above 0 (the
    published direction). Both cells must be complete and registered; trec and triviaqa
    are reported."""
    mk = KG.FIDELITY_MODEL
    d = {}
    for arm in ("kivi2", "aatc2"):
        cell = os.path.join(root, mk, arm)
        why = verify_part5(cell, _env(KG.FIDELITY_ARMS[arm])) if os.path.isdir(cell) else "missing"
        if why:
            raise SystemExit(f"{mk}/{arm}: {why}")
        d[arm] = read_cell(cell, metrics)[0]
    deltas = {t: paired(d["aatc2"][t], d["kivi2"][t], True) for t in KG.TASKS}
    q = deltas["qasper"]
    return {"schema": "tqp-part5-aatc-fidelity/1", "model": mk, "pass": bool(q["lo"] > 0),
            "rule": "qasper F1 aatc2 - kivi2, paired 95% bootstrap interval above 0",
            "deltas": deltas,
            "means": {a: {t: _mean(d[a][t]) for t in KG.TASKS} for a in d},
            "published": PUBLISHED}


def recorded_u3(path: str) -> dict:
    """Part II's scored u3 per Tier A model, from the hash-pinned scorer output."""
    rec = json.load(open(path, encoding="utf-8"))
    canon = json.dumps(rec, sort_keys=True).encode()
    if hashlib.sha256(canon).hexdigest() != KG.G1_SOURCE_SHA256:
        raise SystemExit(f"{path}: not the registered Part II scorer output")
    return {mk: {ep: rec["models"][mk]["cells"]["u3"][ep] for ep in ("qasper", "ppl")}
            for mk in KG.TIER_A}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--lbroot", default=os.environ.get("LBROOT", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--fidelity", action="store_true",
                    help="decide AATC's fidelity check and write part5_aatc_fidelity.json")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.lbroot)
    from metrics import classification_score, qa_f1_score  # LongBench's own

    metrics = {"trec": classification_score, "triviaqa": qa_f1_score, "qasper": qa_f1_score}
    if a.fidelity:
        if os.path.exists(AATC_FIDELITY):
            raise SystemExit(f"{AATC_FIDELITY} exists: the check is decided once")
        rec = fidelity_record(a.root, metrics)
        json.dump(rec, open(a.out, "w"), indent=1)
        json.dump(rec, open(AATC_FIDELITY, "w"), indent=1)
        print("AATC FIDELITY", "PASS" if rec["pass"] else "FAIL", json.dumps(rec["deltas"]["qasper"]))
        return
    report = score(a.root, metrics, load_dispositions(DISPOSITIONS), aatc_fidelity(),
                   recorded_u3(os.path.join(HERE, KG.G1_SOURCE)))
    json.dump(report, open(a.out, "w"), indent=1, default=float)
    for g, per in report["gates"].items():
        print(g, {m: v["status"] for m, v in per.items()})
    print("VERDICT STATUS", report["verdict_status"])
    for k, v in report["verdicts"].items():
        print(k, v if isinstance(v, str) else v["verdict"])


if __name__ == "__main__":
    main()

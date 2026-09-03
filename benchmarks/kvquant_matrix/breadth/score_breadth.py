"""Score a breadth sweep with LongBench's own metrics, verifying the config
sidecar of every cell before trusting its label (the 2026-08-15 errata guard).

    python score_breadth.py <SUFFIX>        # reads out_<SUFFIX>_<arm>/<task>.0.jsonl

Writes results_breadth_<SUFFIX>.json next to the outputs and prints the table.
"""

import glob
import json
import os
import sys

BASE = os.environ.get("BASE", "/archive/c12/breadth")
LBROOT = os.environ.get("LBROOT", os.path.expanduser("~/item4c/LongBench/LongBench"))
sys.path.insert(0, LBROOT)
sys.path.insert(0, "/archive/c12/reval/pylibs")  # rouge / jieba / fuzzywuzzy for metrics.py
from metrics import qa_f1_score, rouge_score  # noqa: E402

SUFFIX = sys.argv[1] if len(sys.argv) > 1 else "run"
TASKS = ["2wikimqa", "hotpotqa", "multifieldqa_en", "narrativeqa", "samsum", "multi_news", "gov_report"]
MAXGEN = {"2wikimqa": 32, "hotpotqa": 32, "multifieldqa_en": 64, "narrativeqa": 128,
          "samsum": 128, "multi_news": 512, "gov_report": 512}
METRIC = {"2wikimqa": qa_f1_score, "hotpotqa": qa_f1_score, "multifieldqa_en": qa_f1_score,
          "narrativeqa": qa_f1_score, "samsum": rouge_score, "multi_news": rouge_score,
          "gov_report": rouge_score}
EXPECT = {"fp16": {"noquant": 1, "codebook": None},
          "nf4a": {"noquant": 0, "codebook": "nf4a"},
          "nf4": {"noquant": 0, "codebook": "nf4"}}
ARMS = ["fp16", "nf4a", "nf4"]


def load(arm, task):
    d = f"{BASE}/out_{SUFFIX}_{arm}"
    if not os.path.isfile(f"{d}/{task}.DONE"):
        return None, None
    cfgs = [json.load(open(p)) for p in sorted(glob.glob(f"{d}/config.*.json"))]
    for c in cfgs:
        for k, v in EXPECT[arm].items():
            if c.get(k) != v:
                raise SystemExit(f"SIDECAR MISMATCH {arm}/{task}: {k}={c.get(k)!r} != {v!r}")
    rows = {}
    for p in sorted(glob.glob(f"{d}/{task}.*.jsonl")):
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                rows[r["idx"]] = r
    return rows, cfgs


def score(rows, idxs, task):
    fn = METRIC[task]
    tot = 0.0
    for i in idxs:
        r = rows[i]
        tot += max(fn(r["pred"], gt, all_classes=r.get("all_classes")) for gt in r["answers"])
    return 100.0 * tot / len(idxs)


res = {"suffix": SUFFIX, "model": "Qwen/Qwen2.5-7B-Instruct", "arms": {}, "tasks": {}}
hdr = f"{'task':<18}{'max_gen':>8}{'n':>5}" + "".join(f"{a:>9}" for a in ARMS) + f"{'gap nf4a':>10}{'gap nf4':>9}"
print("=" * len(hdr))
print(f"BREADTH SWEEP [{SUFFIX}]  Qwen2.5-7B-Instruct, LongBench, scored with LongBench metrics.py")
print("=" * len(hdr))
print(hdr)
print("-" * len(hdr))
means = {a: [] for a in ARMS}
gaps = {"nf4a": [], "nf4": []}
for task in TASKS:
    cells = {a: load(a, task) for a in ARMS}
    have = {a: c for a, (c, _) in cells.items() if c}
    if "fp16" not in have:
        print(f"{task:<18}{MAXGEN[task]:>8}{'':>5}  (fp16 not run)")
        continue
    idxs = sorted(set.intersection(*(set(r) for r in have.values())))
    sc = {a: score(have[a], idxs, task) for a in have}
    for a in ARMS:
        if a in sc:
            means[a].append(sc[a])
    row = {"max_gen": MAXGEN[task], "n": len(idxs), "scores": {a: round(v, 2) for a, v in sc.items()},
           "artifact_sha256": {a: cells[a][1][0]["artifact_sha256"] for a in have}}
    for q in ("nf4a", "nf4"):
        if q in sc:
            row[f"gap_{q}"] = round(sc["fp16"] - sc[q], 2)
            gaps[q].append(sc["fp16"] - sc[q])
    res["tasks"][task] = row
    line = f"{task:<18}{MAXGEN[task]:>8}{len(idxs):>5}"
    line += "".join(f"{sc[a]:>9.2f}" if a in sc else f"{'-':>9}" for a in ARMS)
    line += "".join(f"{row.get(f'gap_{q}', float('nan')):>10.2f}" if f"gap_{q}" in row else f"{'-':>10}" for q in ("nf4a", "nf4"))
    print(line)
print("-" * len(hdr))
mean_line = f"{'mean':<18}{'':>8}{'':>5}"
for a in ARMS:
    m = sum(means[a]) / len(means[a]) if means[a] else None
    res["arms"][a] = {"n_tasks": len(means[a]), "mean": round(m, 2) if m is not None else None}
    mean_line += f"{m:>9.2f}" if m is not None else f"{'-':>9}"
for q in ("nf4a", "nf4"):
    g = sum(gaps[q]) / len(gaps[q]) if gaps[q] else None
    res["arms"].setdefault(q, {})["mean_gap"] = round(g, 2) if g is not None else None
    mean_line += f"{g:>10.2f}" if g is not None else f"{'-':>10}"
print(mean_line)
out = f"{BASE}/results_breadth_{SUFFIX}.json"
json.dump(res, open(out, "w"), indent=1)
print(f"wrote {out}")
print("SCORE_DONE")

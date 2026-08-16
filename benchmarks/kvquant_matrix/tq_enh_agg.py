#!/usr/bin/env python3
"""Aggregate a sharded LongBench run into per-task scores. Handles the full LongBench
English subset (auto-scores whichever datasets have output files for the TAG)."""

import glob
import json
import sys

sys.path.insert(0, "/root/LongBench/LongBench")
from metrics import (  # noqa: E402
    classification_score,
    code_sim_score,
    count_score,
    qa_f1_score,
    retrieval_score,
    rouge_score,
)

TAG = sys.argv[1]

# LongBench official dataset -> metric (English subset + the core 3).
D2M = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "triviaqa": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "samsum": rouge_score,
    "trec": classification_score,
    "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}
# datasets whose first generated line is the answer (few-shot / classification style).
FIRST_LINE = {"trec", "triviaqa", "samsum", "lsht"}


def scorer(dataset, preds, answers, allc):
    tot = 0.0
    for pred, gts in zip(preds, answers):
        if dataset in FIRST_LINE:
            pred = pred.lstrip("\n").split("\n")[0]
        s = 0.0
        for gt in gts:
            s = max(s, D2M[dataset](pred, gt, all_classes=allc))
        tot += s
    return round(100 * tot / len(preds), 2)


# Arm verification (2026-08-15 errata): rows are labeled only by the free-form TAG,
# which once let a wrongly-configured (or mixed) output directory be reported as an
# arm it never ran. Read the config.<shard>.json sidecars the runner now writes,
# require them to agree, and say what was actually scored.
_cfgs = []
for _cf in sorted(glob.glob(f"/root/out_{TAG}/config.*.json")):
    try:
        _cfgs.append(json.load(open(_cf)))
    except Exception:
        pass
_uniq = {
    json.dumps({k: v for k, v in c.items() if k != "shard"}, sort_keys=True)
    for c in _cfgs
}
_hashes = {c.get("artifact_sha256", "unhashed") for c in _cfgs}
if not _cfgs:
    arm = "unknown (no config sidecars; pre-guard run)"
elif len(_uniq) > 1:
    arm = "MIXED CONFIGS ACROSS SHARDS — do not report as a single arm"
else:
    c = _cfgs[0]
    arm = (
        "fp16 (noquant)"
        if c.get("noquant")
        else (
            f"{c.get('codebook')} k{c.get('key_bits')}v{c.get('val_bits')}"
            f" g{c.get('group')} hot{c.get('hot')} sink{c.get('sink')}"
            f" out{c.get('outlier_frac')} prerope{c.get('prerope')}"
        )
    ) + f" model={c.get('model_key')}"
if len(_hashes) > 1:
    arm += "  [ARTIFACT HASHES DISAGREE - refuse single-arm reporting]"
elif _cfgs:
    arm += f"  [artifact {next(iter(_hashes))[:16]}]"
print(f"CONFIG {TAG} {arm}")

res = {}
for f0 in sorted(glob.glob(f"/root/out_{TAG}/*.0.jsonl")):
    dataset = f0.split("/")[-1].rsplit(".", 2)[0]
    if dataset not in D2M:
        continue
    rows = {}
    for f in glob.glob(f"/root/out_{TAG}/{dataset}.*.jsonl"):
        for line in open(f):
            try:
                o = json.loads(line)
                rows[o["idx"]] = o
            except Exception:
                pass
    idxs = sorted(rows)
    if not idxs:
        res[dataset] = {"n": 0, "score": None}
        continue
    preds = [rows[i]["pred"] for i in idxs]
    answers = [rows[i]["answers"] for i in idxs]
    res[dataset] = {
        "n": len(idxs),
        "score": scorer(dataset, preds, answers, rows[idxs[0]]["all_classes"]),
    }
print(f"RESULT {TAG} " + json.dumps(res))

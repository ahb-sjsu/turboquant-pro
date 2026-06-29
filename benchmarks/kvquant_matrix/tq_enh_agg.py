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

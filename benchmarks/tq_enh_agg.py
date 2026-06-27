#!/usr/bin/env python3
"""Aggregate a sharded enhanced-TurboQuant LongBench run into task scores."""

import glob
import json
import sys

sys.path.insert(0, "/root/LongBench/LongBench")
from metrics import classification_score, qa_f1_score  # noqa: E402

TAG = sys.argv[1]
D2M = {"qasper": qa_f1_score, "trec": classification_score, "triviaqa": qa_f1_score}


def scorer(dataset, preds, answers, allc):
    tot = 0.0
    for pred, gts in zip(preds, answers):
        s = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            pred = pred.lstrip("\n").split("\n")[0]
        for gt in gts:
            s = max(s, D2M[dataset](pred, gt, all_classes=allc))
        tot += s
    return round(100 * tot / len(preds), 2)


res = {}
for dataset in ["trec", "triviaqa", "qasper"]:
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

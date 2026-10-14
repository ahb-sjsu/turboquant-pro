#!/usr/bin/env python3
"""Stage AATC's rate-distortion calibration text (Part V baseline ``aatc``).

    python part5_aatc_text.py --out /data/kv5/aatc_text.jsonl [--expect SHA256]

The paper (arXiv 2608.14191, section 5) draws 256 sequences of 2048 tokens, half from
FineWeb sample-10BT (web text) and half from OpenR1-Math-220k (mathematical reasoning
traces). This takes the first FINEWEB_ROWS documents of FineWeb sample-10BT and the first
OPENR1_ROWS rows of OpenR1-Math-220k (default config) in the datasets' streaming order,
an OpenR1 row's text being its problem followed by its first generation (the reasoning
trace), and writes them as jsonl rows {"source", "id", "text"}. The harness packs each
source into 128 windows per model tokenizer (aatc.windows); these row counts give every
Tier A tokenizer more than enough tokens. The file's sha256 is printed; with --expect, a
different file fails the job (staging on NRP checks it against Atlas's).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

FINEWEB = ("HuggingFaceFW/fineweb", "sample-10BT")
OPENR1 = ("open-r1/OpenR1-Math-220k", "default")
FINEWEB_ROWS = 2000
OPENR1_ROWS = 600


def rows():
    from datasets import load_dataset

    fw = load_dataset(*FINEWEB, split="train", streaming=True)
    for i, r in enumerate(fw):
        if i >= FINEWEB_ROWS:
            break
        yield {"source": "fineweb", "id": r["id"], "text": r["text"]}
    r1 = load_dataset(*OPENR1, split="train", streaming=True)
    for i, r in enumerate(r1):
        if i >= OPENR1_ROWS:
            break
        yield {"source": "openr1-math", "id": r["uuid"],
               "text": r["problem"] + "\n\n" + r["generations"][0]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect", default="")
    a = ap.parse_args(argv)
    h = hashlib.sha256()
    n = 0
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows():
            line = json.dumps(r, ensure_ascii=False) + "\n"
            f.write(line)
            h.update(line.encode("utf-8"))
            n += 1
    sha = h.hexdigest()
    print(f"[aatc-text] {n} rows -> {a.out} sha256={sha}", flush=True)
    if a.expect and sha != a.expect:
        raise SystemExit(f"[aatc-text] sha256 {sha} != expected {a.expect}")
    return 0


if __name__ == "__main__":
    rc = main()
    # The datasets streaming threads abort the interpreter at shutdown (seen on Atlas,
    # datasets 4.3.0, after the file was complete), which would fail a staging Job: exit
    # at once, after every check above has passed and the output is flushed.
    sys.stdout.flush()
    os._exit(rc)

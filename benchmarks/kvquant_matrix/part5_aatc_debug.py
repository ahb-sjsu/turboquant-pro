#!/usr/bin/env python3
"""Diagnose AATC on one model on CPU (the 2026-10-13 smoke gave perplexity 3.9e5 at 2 bits).

    python part5_aatc_debug.py --model <path> --text <aatc_text.jsonl> --threads 16

Calibrates on a few short windows, then on one WikiText-2 test chunk reports perplexity
with no coding, with AATC at 16 bits (where the transform and quantizer must be nearly
lossless) and at 2 bits, keys only and keys and values, plus layer 0's key
reconstruction error. A diagnostic only: nothing here is a registered measurement.
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--center", type=int, default=1)
    ap.add_argument("--per-layer", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    model.requires_grad_(False)
    import aatc as AC

    os.environ.update({"AATC": "kv", "AATC_KEY_BITS": "2", "AATC_VAL_BITS": "2",
                       "AATC_CALIB": "/tmp/aatc_debug_cal.pt", "AATC_CENTER": str(a.center)})
    AC = importlib.reload(AC)
    wt = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
    tw = AC.windows(tok, [wt[:400000]], 4, a.seq)
    rows = AC.load_text(a.text)
    rd = AC.windows(tok, rows[:200], 4, a.seq) + AC.windows(tok, rows[-100:], 4, a.seq)
    cal = AC.calibrate(model, tw, rd, "")
    test = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(test[:20000], return_tensors="pt").input_ids[:, : a.seq]

    def ppl():
        with torch.no_grad():
            return math.exp(float(model(ids, labels=ids).loss))

    print(f"[debug] center={a.center} none", ppl(), flush=True)
    for kb, vb, mode in ((8, 0, "k"), (4, 0, "k"), (3, 0, "k"), (2, 0, "k"), (3, 3, "kv"),
                         (2, 2, "kv")):
        os.environ.update({"AATC": mode, "AATC_KEY_BITS": str(kb), "AATC_VAL_BITS": str(vb)})
        AC = importlib.reload(AC)
        AC.install(model, cal, 4, 128, lambda *x: None)
        print(f"[debug] center={a.center} {mode} k{kb} v{vb}", ppl(), flush=True)
        AC.remove()
    if not a.per_layer:
        return
    os.environ.update({"AATC": "k", "AATC_KEY_BITS": "3", "AATC_VAL_BITS": "0"})
    AC = importlib.reload(AC)
    for li in range(len(model.model.layers)):  # one layer at a time, keys at 3 bits
        AC.install(model, cal, 4, 128, lambda *x: None)
        for k, h in enumerate(list(AC._HOOKS)):
            if k != li:
                h.remove()
        print(f"[debug] center={a.center} only L{li:02d} keys k3", ppl(), flush=True)
        AC.remove()


if __name__ == "__main__":
    main()

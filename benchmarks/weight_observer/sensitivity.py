"""EXPLORATORY: measured single-matrix damage, the oracle for every additive predictor.

    python -m weight_observer.sensitivity --model-path M --text T --out O

For every decoder matrix ``m`` and bit width ``b`` in (3, 4, 5, 6), only ``m`` is quantized
(``quant.rtn``, the Part III codec) and the KL from the full-precision model is measured on the
registered evaluation sequences, exactly as ``run.py`` measures a variant. 8 bits is taken as
zero (uniform 8-bit costs 0.0005 nats per token for the whole model).

``sum_m KL_single(m, b_m)`` is the best any predictor that adds per-matrix costs can do. If it
ranks variants near perfectly, the diagonal Fisher's shortfall is per-matrix miscalibration
(the quadratic term misjudges some matrices); if it does no better than the Fisher, the
shortfall is interaction between matrices, which no additive predictor can capture. Resumable:
one line per (matrix, bits) in sensitivity.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from . import tables as T
from .measure import kl_per_sequence
from .quant import rtn

LEVELS = (3, 4, 5, 6)


def main(argv=None) -> int:
    from . import run as R

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    env = R.environment()
    ep = os.path.join(a.out, "env.json")
    if os.path.exists(ep) and json.load(open(ep)) != env:
        raise SystemExit(f"started on {json.load(open(ep))}, now {env}")
    json.dump(env, open(ep, "w"))
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model_path)
    ref = R.load(a.model_path, a.device)
    var = R.load(a.model_path, a.device)
    text = open(f"{a.text}/test.txt", encoding="utf-8").read()
    evalq = [c[None].to(a.device) for c in R.chunks(tok, text, R.N_EVAL)]
    rm, vm = T.linear_modules(ref), T.linear_modules(var)
    rp = os.path.join(a.out, "sensitivity.jsonl")
    done = set()
    if os.path.exists(rp):
        for line in open(rp, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["matrix"], r["bits"]))
            except (ValueError, KeyError):
                pass
    with open(rp, "a", encoding="utf-8") as fo, torch.no_grad():
        for name in rm:
            w = rm[name].weight
            for b in LEVELS:
                if (name, b) in done:
                    continue
                t0 = time.time()
                vm[name].weight.copy_(rtn(w, b).to(w.dtype))
                per = kl_per_sequence(ref, var, evalq)
                vm[name].weight.copy_(w)
                rec = {"matrix": name, "bits": b, "seqs": per}
                rec["seconds"] = round(time.time() - t0, 1)
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                kl = sum(s["kl_sum"] for s in per) / sum(s["tokens"] for s in per)
                print(
                    f"[sens] {name} b{b} kl/token {kl:.3e} {rec['seconds']}s",
                    flush=True,
                )
    print("SENS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Matched-bit ratios to three decimals, plus the top-1 flip range (read-only)."""

import glob
import json

for p in sorted(glob.glob("matched_bit_*.json")):
    d = json.load(open(p))
    by = {}
    for r in d["results"]:
        by.setdefault(r["bits"], {})[r["proj"]] = r
    out = []
    for b in sorted(by, reverse=True):
        x = by[b]
        if not all(k in x for k in "QKVO"):
            continue
        ok = (x["Q"]["out_kl"] + x["K"]["out_kl"]) / (
            x["V"]["out_kl"] + x["O"]["out_kl"]
        )
        wf = (x["Q"]["w_rel_fro"] + x["K"]["w_rel_fro"]) / (
            x["V"]["w_rel_fro"] + x["O"]["w_rel_fro"]
        )
        fl = [x[k]["out_top1_flip"] for k in "QKVO"]
        out.append(
            f"b{b} wf={wf:.3f} ok={ok:.3f} inv={1 / ok:.3f}"
            f" flips={min(fl):.3f}-{max(fl):.3f}"
        )
    print(d["model"], " | ".join(out))

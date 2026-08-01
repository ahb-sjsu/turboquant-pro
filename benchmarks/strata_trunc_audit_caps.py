#!/usr/bin/env python3
"""Arm D2 truncation at LaBSE's DEFAULT cap (256) vs its ARCHITECTURAL cap (512).

The tokenizer warns at 512, not 256 -- so sentence-transformers is capping
below what the position embeddings support. If raising max_seq_length to 512
makes truncation both rarer and more UNIFORM across languages, arm D2 may be
rescuable with no training at all.

CPU only. Same 30k seed-7 subsample as trunc_audit.py.
"""

import collections
import json

import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoTokenizer

SUB = 30000
CAPS = (256, 512)

cfg = AutoConfig.from_pretrained("sentence-transformers/LaBSE")
print(
    "LaBSE config max_position_embeddings =",
    getattr(cfg, "max_position_embeddings", "?"),
    flush=True,
)
st = SentenceTransformer("sentence-transformers/LaBSE", device="cpu")
print("sentence-transformers default max_seq_length =", st.max_seq_length, flush=True)
del st

ids = np.load("/archive/tqp_strata/sample_ids.npy")
labels = np.array(
    [ln.strip() for ln in open("/archive/tqp_strata/labels.txt", encoding="utf-8")]
)
rng = np.random.default_rng(7)
sel = np.sort(rng.choice(len(ids), size=min(SUB, len(ids)), replace=False))

conn = psycopg2.connect(dbname="atlas", user="claude")
cur = conn.cursor()
cur.execute(
    "SELECT id, content FROM ethics_chunks WHERE id = ANY(%s)",
    ([int(v) for v in ids[sel]],),
)
text = dict(cur.fetchall())
conn.close()

tok = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
per = collections.defaultdict(list)
for j, row in enumerate(sel):
    t = text.get(int(ids[row]))
    if t is not None:
        per[labels[row]].append(
            len(tok.encode(t, add_special_tokens=True, truncation=False))
        )
    if j % 10000 == 0:
        print(f"  tokenized {j}/{len(sel)}", flush=True)

out = {}
for lang, ns in sorted(per.items(), key=lambda kv: -len(kv[1])):
    a = np.array(ns)
    e = {
        "n_sampled": int(len(a)),
        "median_tokens": float(np.median(a)),
        "p90_tokens": float(np.percentile(a, 90)),
    }
    for cap in CAPS:
        e[f"frac_trunc_{cap}"] = float((a > cap).mean())
        e[f"kept_median_{cap}"] = float(np.median(np.minimum(a, cap) / a))
    out[lang] = e

hdr = f"{'language':<14}{'n':>7}{'med':>6}{'p90':>6}"
for cap in CAPS:
    hdr += f"{'%tr@' + str(cap):>9}{'kept':>7}"
print("\n" + hdr)
for lang, e in out.items():
    line = f"{lang:<14}{e['n_sampled']:>7}{e['median_tokens']:>6.0f}{e['p90_tokens']:>6.0f}"
    for cap in CAPS:
        line += f"{100 * e[f'frac_trunc_{cap}']:>8.1f}%{100 * e[f'kept_median_{cap}']:>6.0f}%"
    print(line)

big = [(k, v) for k, v in out.items() if v["n_sampled"] >= 150]
print(f"\nlanguages with >=150 sampled rows: n={len(big)}")
for cap in CAPS:
    f = np.array([v[f"frac_trunc_{cap}"] for _, v in big])
    k = np.array([v[f"kept_median_{cap}"] for _, v in big])
    print(
        f"  cap {cap}: trunc frac {f.min():.3f}-{f.max():.3f} "
        f"(spread {f.max() - f.min():.3f}); kept-median {k.min():.2f}-{k.max():.2f} "
        f"(spread {k.max() - k.min():.2f})"
    )

with open("/archive/tqp_strata/d2_truncation_audit_256_512.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("-> /archive/tqp_strata/d2_truncation_audit_256_512.json")

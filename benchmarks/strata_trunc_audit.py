#!/usr/bin/env python3
"""Per-language truncation audit for arm D2 (Ethics x LaBSE) BEFORE encoding.

LaBSE's max_seq_length is 256 tokens; Ethics content averages ~878 chars and
tokens-per-char varies by script by 3-4x. If truncation rates differ sharply
across languages, arm D2 cannot serve a per-language ROLE study -- the
truncation would manufacture the differentiation being measured. Measure first,
spend the GPU pass second.

CPU only. Uniform subsample of the committed seed-7 sample rows.
"""

import collections
import json

import numpy as np
import psycopg2
from transformers import AutoTokenizer

SUB = 30000
IDS = "/archive/tqp_strata/sample_ids.npy"
LABELS = "/archive/tqp_strata/labels.txt"
MAXLEN = 256

ids = np.load(IDS)
labels = np.array([ln.strip() for ln in open(LABELS, encoding="utf-8")])
rng = np.random.default_rng(7)
sel = np.sort(rng.choice(len(ids), size=min(SUB, len(ids)), replace=False))

conn = psycopg2.connect(dbname="atlas", user="claude")
cur = conn.cursor()
want = [int(v) for v in ids[sel]]
cur.execute("SELECT id, content FROM ethics_chunks WHERE id = ANY(%s)", (want,))
text = dict(cur.fetchall())
conn.close()
print(f"fetched {len(text)}/{len(want)} texts", flush=True)

tok = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
per = collections.defaultdict(list)
for row, i in zip(sel, range(len(sel))):
    t = text.get(int(ids[row]))
    if t is None:
        continue
    n = len(tok.encode(t, add_special_tokens=True, truncation=False))
    per[labels[row]].append(n)
    if i % 5000 == 0:
        print(f"  tokenized {i}/{len(sel)}", flush=True)

out = {}
for lang, ns in sorted(per.items(), key=lambda kv: -len(kv[1])):
    a = np.array(ns)
    out[lang] = {
        "n_sampled": int(len(a)),
        "median_tokens": float(np.median(a)),
        "p90_tokens": float(np.percentile(a, 90)),
        "frac_truncated": float((a > MAXLEN).mean()),
        "median_kept_frac": float(np.median(np.minimum(a, MAXLEN) / a)),
    }

print(f"\n{'language':<14}{'n':>7}{'med tok':>9}{'p90':>7}{'%trunc':>8}{'kept':>7}")
for lang, d in out.items():
    print(
        f"{lang:<14}{d['n_sampled']:>7}{d['median_tokens']:>9.0f}"
        f"{d['p90_tokens']:>7.0f}{100*d['frac_truncated']:>7.1f}%"
        f"{100*d['median_kept_frac']:>6.0f}%"
    )

fr = np.array([d["frac_truncated"] for d in out.values()])
big = [(k, v) for k, v in out.items() if v["n_sampled"] >= 150]
if big:
    fb = np.array([v["frac_truncated"] for _, v in big])
    print(
        f"\nover languages with >=150 sampled rows (n={len(big)}): "
        f"trunc frac min={fb.min():.3f} max={fb.max():.3f} spread={fb.max()-fb.min():.3f}"
    )
print(f"all languages: trunc frac min={fr.min():.3f} max={fr.max():.3f}")
with open("/archive/tqp_strata/d2_truncation_audit.json", "w") as f:
    json.dump(out, f, indent=2)
print("-> /archive/tqp_strata/d2_truncation_audit.json")

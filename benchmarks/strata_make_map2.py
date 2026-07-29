#!/usr/bin/env python3
"""Write the pre-declared SECONDARY area map for Ethics: tradition x century_bin
(PREREG Amendment 1 A1.3; measured 10 eligible strata at n_min=2000).

Row order is sample_ids.npy, identical to labels.txt and every Ethics arm.
"""
import numpy as np
import psycopg2

ids = np.load("/archive/tqp_strata/sample_ids.npy")
pos = {int(v): i for i, v in enumerate(ids)}
out = [None] * len(ids)

conn = psycopg2.connect(dbname="atlas", user="claude")
cur = conn.cursor()
CH = 20000
for s in range(0, len(ids), CH):
    cur.execute(
        "SELECT id, tradition, century FROM ethics_chunks WHERE id = ANY(%s)",
        ([int(v) for v in ids[s : s + CH]],),
    )
    for rid, trad, cent in cur.fetchall():
        bin_ = (cent // 5) if cent is not None else "na"
        out[pos[rid]] = f"{trad}|{bin_}"
conn.close()

missing = sum(1 for v in out if v is None)
if missing:
    raise SystemExit(f"FATAL: {missing} rows unmapped")
with open("/archive/tqp_strata/labels_tradcent.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")

import collections

c = collections.Counter(out)
elig = {k: v for k, v in c.items() if v >= 2000}
print(f"wrote labels_tradcent.txt: {len(out)} rows, {len(c)} areas, "
      f"{len(elig)} eligible at n>=2000")
for k, v in sorted(elig.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<28}{v:>8}")

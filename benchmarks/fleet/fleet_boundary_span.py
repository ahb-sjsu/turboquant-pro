# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""Which queries' tenth-to-eleventh gap the archived partials certify, with no new scan.

fleet_partials_audit.py takes the eleventh score of a query from the merge of the 500
per-server top-10 partials. The merge recovers the exact global top 10, but it holds
the exact global eleventh only when the top 10 spans at least two servers:

- If the top 10 spans two or more servers, each server holds at most nine of them, so
  on the server that holds the global eleventh at most nine rows outrank it. It is in
  that server's top 10, hence in its partial, and the merged eleventh is the global
  eleventh. The merged gap is the exact gap.
- If all ten lie on one server, that server's eleventh row is in no partial. The global
  eleventh is that row or the best row of another server, whichever scores higher, so
  the merged eleventh can be lower than the global one and the merged gap is only an
  upper bound on the exact gap.

For each query this reports the number of servers its reference top 10 spans, its
merged gap, and whether that gap is certified. It reads only the reference partials.

Usage: ``python fleet_boundary_span.py DIR TAG [--n 500]``
Writes ``span{TAG}.json`` in the current directory and prints RESULT_JSON.
"""

import argparse
import json
import os

import numpy as np

K = 10

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("tag")
ap.add_argument("--n", type=int, default=500)
a = ap.parse_args()

ids, scs = [], []
for s in range(a.n):
    with np.load(os.path.join(a.dir, f"ref{a.tag}_part_{s}.npz")) as z:
        ids.append(z["ids"])
        scs.append(z["scores"])
ids = np.stack(ids)  # (n, nq, K)
scs = np.stack(scs)
n, nq, k = scs.shape
assert k == K, scs.shape

server = np.repeat(np.arange(n), K)  # server of each candidate, in concatenation order
per_query = []
for q in range(nq):
    cand = scs[:, q, :].reshape(-1)  # (n*K,), server-major, as the audit concatenates
    order = np.argsort(-cand, kind="stable")
    top = order[:K]
    span = int(np.unique(server[top]).size)
    gap = float(cand[order[K - 1]] - cand[order[K]])
    per_query.append(
        {
            "query": q,
            "servers_in_top10": span,
            "merged_gap": gap,
            "certified": span >= 2,
            "top10_server": int(server[top[0]]) if span == 1 else None,
        }
    )

cert = [p for p in per_query if p["certified"]]
single = [p for p in per_query if not p["certified"]]
spans = np.array([p["servers_in_top10"] for p in per_query])
res = {
    "dir": os.path.abspath(a.dir),
    "tag": a.tag,
    "n_servers": n,
    "nq": nq,
    "queries_top10_on_one_server": len(single),
    "queries_certified": len(cert),
    "servers_in_top10_min": int(spans.min()),
    "servers_in_top10_median": float(np.median(spans)),
    "servers_in_top10_max": int(spans.max()),
    "certified_gap_min": min((p["merged_gap"] for p in cert), default=None),
    "merged_gap_min_all": min(p["merged_gap"] for p in per_query),
    "uncertified": [
        {
            "query": p["query"],
            "server": p["top10_server"],
            "merged_gap_upper_bound": p["merged_gap"],
        }
        for p in single
    ],
    "per_query": per_query,
}
tmp = f"span{a.tag}.json.tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
os.replace(tmp, f"span{a.tag}.json")
print(
    "RESULT_JSON "
    + json.dumps({k: v for k, v in res.items() if k not in ("per_query",)}),
    flush=True,
)

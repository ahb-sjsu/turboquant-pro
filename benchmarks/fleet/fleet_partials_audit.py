# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""Audit of the archived per-server partials of one run, with no new scan.

The score phase loads every partial it finds. This audit checks what it did not:

1. exactly one partial per server and phase, for servers 0..N-1, and no stray or
   temporary files beside them;
2. every partial has ``ids`` (int64) and ``scores`` (float32) of shape (nq, 10), with nq
   equal to the rows of the run's query file, so row q of every partial is query q;
3. every id lies in its server's row range, no id repeats within a row, and scores are
   finite and non-increasing along each row;
4. within a server, a row that appears in both the reference and a routed partial gets the
   same score up to floating-point reordering: the exact scan scores rows in fixed blocks and
   the routed pass gathers cell rows, so the sums differ in order. The audit reports how many
   such rows are bit-equal and the largest difference, in units in the last place and
   absolute, and checks that the routed i-th score never exceeds the reference i-th score by
   more than that difference (the routed set is a subset of the server's rows);
5. at the merge, the tenth and eleventh scores of the exact global top-10 differ, so no
   tie-break decides which row enters the reference, and their smallest gap exceeds twice
   the largest score difference of check 4, so that reordering cannot swap a row across
   the boundary;
6. optionally, every file matches the archive's ``SHA256SUMS``.

Usage: ``python fleet_partials_audit.py DIR TAG [--n 500] [--widths 32,128] [--rows 2e9]``
Writes ``audit{TAG}.json`` in the current directory and prints RESULT_JSON. Exit status 1
if any check fails.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

K = 10

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("tag")
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--widths", default="32,128")
ap.add_argument("--rows", type=float, default=2e9, help="rows per server")
ap.add_argument(
    "--queries", default=None, help="query .npy (default DIR/queries{TAG}.npy)"
)
ap.add_argument("--sha", action="store_true", help="verify SHA256SUMS in DIR")
a = ap.parse_args()

WIDTHS = [int(w) for w in a.widths.split(",")]
ROWS = int(a.rows)
fails = []


def fail(msg):
    fails.append(msg)


qpath = a.queries or os.path.join(a.dir, f"queries{a.tag}.npy")
nq = int(np.load(qpath, mmap_mode="r").shape[0])

phases = {"ref": f"ref{a.tag}"} | {f"p{w}": f"ivf{a.tag}_p{w}" for w in WIDTHS}
loaded = {}
for ph, stem in phases.items():
    pat = re.compile(rf"^{re.escape(stem)}_part_(\d+)\.npz$")
    files = sorted(os.listdir(a.dir))
    sids = [int(m.group(1)) for f in files if (m := pat.match(f))]
    stray = [f for f in files if f.startswith(stem + "_part_") and not pat.match(f)]
    if stray:
        fail(f"{ph}: stray files {stray[:5]}")
    if sorted(sids) != list(range(a.n)):
        missing = sorted(set(range(a.n)) - set(sids))
        dup = len(sids) - len(set(sids))
        fail(f"{ph}: {len(sids)} partials, missing {missing[:10]}, duplicates {dup}")
    ids = np.full((a.n, nq, K), -1, np.int64)
    scs = np.full((a.n, nq, K), -np.inf, np.float32)
    for s in sids:
        z = np.load(os.path.join(a.dir, f"{stem}_part_{s}.npz"))
        i, c = z["ids"], z["scores"]
        if i.dtype != np.int64 or c.dtype != np.float32:
            fail(f"{ph} {s}: dtypes {i.dtype} {c.dtype}")
        if i.shape != (nq, K) or c.shape != (nq, K):
            fail(f"{ph} {s}: shapes {i.shape} {c.shape}, expected {(nq, K)}")
            continue
        if ((i < s * ROWS) | (i >= (s + 1) * ROWS)).any():
            fail(f"{ph} {s}: ids outside the server's row range")
        if not np.isfinite(c).all():
            fail(f"{ph} {s}: non-finite scores")
        if (np.diff(c, axis=1) > 0).any():
            fail(f"{ph} {s}: scores not non-increasing along a row")
        if any(len(set(r)) < K for r in i.tolist()):
            fail(f"{ph} {s}: repeated id within a row")
        ids[s], scs[s] = i, c
    loaded[ph] = (ids, scs)

ref_ids, ref_scs = loaded["ref"]
shared_rows = bit_equal = 0
max_abs = 0.0
max_ulp = 0
for w in WIDTHS:
    r_ids, r_scs = loaded[f"p{w}"]
    for s in range(a.n):
        for q in range(nq):
            ref_map = dict(zip(ref_ids[s, q].tolist(), ref_scs[s, q]))
            for rid, rsc in zip(r_ids[s, q].tolist(), r_scs[s, q]):
                if rid in ref_map:
                    x, y = np.float32(ref_map[rid]), np.float32(rsc)
                    shared_rows += 1
                    bit_equal += bool(x == y)
                    max_abs = max(max_abs, abs(float(x) - float(y)))
                    ulp = abs(int(x.view(np.int32)) - int(y.view(np.int32)))
                    max_ulp = max(max_ulp, ulp)
for w in WIDTHS:
    over = loaded[f"p{w}"][1] - ref_scs
    if (over > max_abs).any():
        fail(
            f"p{w}: routed i-th score above reference i-th score by {float(over.max()):.3g}"
        )

# Ties at the reference merge boundary: the 10th and 11th of n x 10 candidates.
cand = np.concatenate([ref_scs[s] for s in range(a.n)], axis=1)  # (nq, n*10)
srt = -np.sort(-cand, axis=1)
boundary_ties = int((srt[:, K - 1] == srt[:, K]).sum())
margin_min = float((srt[:, K - 1] - srt[:, K]).min())
if boundary_ties:
    fail(f"{boundary_ties} queries tie at the reference top-10 boundary")
if not margin_min > 2 * max_abs:
    fail(
        f"boundary margin {margin_min:.3g} not above twice the score difference {max_abs:.3g}"
    )

sha_checked = 0
if a.sha:
    for line in open(os.path.join(a.dir, "SHA256SUMS"), encoding="utf-8"):
        h, name = line.split()
        p = os.path.join(a.dir, name)
        if not os.path.exists(p):
            fail(f"sha: {name} missing")
            continue
        if hashlib.sha256(open(p, "rb").read()).hexdigest() != h:
            fail(f"sha: {name} mismatch")
        sha_checked += 1

res = {
    "dir": os.path.abspath(a.dir),
    "tag": a.tag,
    "n_servers": a.n,
    "nq": nq,
    "phases": list(phases),
    "partials_checked": int(
        sum(
            len(glob.glob(os.path.join(a.dir, f"{st}_part_*.npz")))
            for st in phases.values()
        )
    ),
    "shared_rows": shared_rows,
    "shared_rows_bit_equal": bit_equal,
    "shared_rows_max_abs_diff": max_abs,
    "shared_rows_max_ulp": max_ulp,
    "reference_boundary_ties": boundary_ties,
    "reference_boundary_margin_min": margin_min,
    "sha256_files_checked": sha_checked,
    "failures": fails[:50],
    "n_failures": len(fails),
    "pass": not fails,
}
tmp = f"audit{a.tag}.json.tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
os.replace(tmp, f"audit{a.tag}.json")
print("RESULT_JSON " + json.dumps(res), flush=True)
sys.exit(0 if not fails else 1)

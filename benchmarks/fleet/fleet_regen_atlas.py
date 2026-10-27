# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""The Atlas half of the regenerated-index run: rebuild the shards the cluster pass deferred.

The cluster pass (``fleet_regen_score.py`` with ``TQP_DEFER_UNMATCHED=1``) leaves a shard that
no streamed variant reproduces and ends its log with ``DEFERRED <server> <j,j,...>``. Those are
the shards the first servers wrote as one block (see ``fleet_regen_build.py``), and the block
write peaks at 2.65 GiB, past the cluster's 2 GiB class. This script, run on Atlas:

  1. reads each server's DEFERRED line from its job log (``kubectl logs``), keeping a copy in
     ``deferred_<tag>.json`` so the list survives the job's deletion;
  2. runs ``fleet_regen_score.py`` with ``TQP_PART_KIND=atlas`` and ``TQP_SHARD_LIST`` for each
     server with deferred shards, block and streamed writes, both library versions, every
     kernel Atlas runs, at most ``TQP_ATLAS_WORKERS`` servers at once on cores 12 to 23;
  3. puts each ``regen<tag>_atlas_<sid>.npz`` in a config map ``tqp-fleet-atlas-<tag>-<sid>``
     and prints the ``TQP_ATLAS_MAPS`` value the merge job takes.

Resumable: a server whose Atlas partial exists and loads is skipped, and the score script
resumes inside a server from its own state file.

With ``--first <sid,...>`` it instead goes through those servers before the cluster does. The
first servers (0 to 7) wrote most of their early shards as one block, which the cluster cannot
build and Atlas builds in a minute or two, while each costs a cluster job ten minutes or more
of streamed builds before it is deferred (server 1, 2026-10-27). So every shard of each listed
server is built here with the block write, the other write is built as well to say whether the
record decides it (``TQP_CONFIRM_WRITE``), and a shard Atlas cannot reproduce is deferred
instead of searched. The shards Atlas reproduced go to ``atlas_first_<tag>.json``, which the
driver reads (``TQP_ATLAS_FIRST``) to have the cluster job skip them and take only the rest.

Usage: python fleet_regen_atlas.py <tag> [--fetch-only | --first <sid,...>]
"""

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

TAG = sys.argv[1]
NS = "ssu-atlas-ai"
W = os.environ.get("TQP_ATLAS_WORK", "/archive/experiments/tqp-fleet-1t/work")
CODE = os.environ.get("TQP_ATLAS_CODE", f"{W}/code_v")
OUTDIR = os.environ.get("TQP_ATLAS_RESULTS", f"{W}/rg_atlas_{TAG}")
HASHDIR = "/archive/experiments/tqp-fleet-1t/scale_transfer/1tnm-all"
PY = f"{W}/venv251/bin/python"
N_SRV = int(os.environ.get("TQP_N_SERVERS", "500"))
WORKERS = int(os.environ.get("TQP_ATLAS_WORKERS", "10"))  # thermal cap is 20 threads
LIST = f"{OUTDIR}/deferred_{TAG}.json"
# The released exact top-10s the codes are checked against (fleet_regen_check.py).
RELEASED = os.environ.get(
    "TQP_RELEASED",
    f"1t={W}/shared/results/queries1t.npy={W}/shared/results/ref1t_part_{{sid}}.npz,"
    f"1tnm={HASHDIR}/queries1tnm.npy={HASHDIR}/ref1tnm_part_{{sid}}.npz",
)


def fetch_deferred():
    have = json.load(open(LIST)) if os.path.exists(LIST) else {}
    for sid in range(N_SRV):
        if str(sid) in have:
            continue
        r = subprocess.run(
            ["kubectl", "-n", NS, "logs", f"job/aqx-regen{TAG}-{sid}"],
            capture_output=True,
            text=True,
        )
        m = re.search(rf"^DEFERRED {sid} ([0-9,]*)$", r.stdout, re.M)
        if m is None:
            print(f"server {sid}: no DEFERRED line (job unfinished or log gone)")
            continue
        have[str(sid)] = [int(x) for x in m.group(1).split(",") if x]
        with open(LIST + ".tmp", "w") as f:
            json.dump(have, f)
        os.replace(LIST + ".tmp", LIST)
    return have


def run(sid, js, first=False):
    out = f"{OUTDIR}/regen{TAG}_atlas_{sid}.npz"
    if os.path.exists(out):
        try:
            np.load(out)["shards"]
            return sid, "exists"
        except Exception:
            pass
    env = dict(
        os.environ,
        PYTHONPATH=f"{W}/repo_build:{CODE}",
        TQP_SERVER_ID=str(sid),
        TQP_RUN_TAG=TAG,
        TQP_PART_KIND="atlas",
        TQP_SHARD_LIST=",".join(map(str, js)),
        TQP_SHARDS_PER_SERVER="400",
        TQP_WRITES="block,stream",
        TQP_LIBS="pre,post",
        TQP_LIBDIR=f"/tmp/tqplib_{TAG}_{sid}",
        TQP_BOOT=f"{W}/shared/bootstrap",
        TQP_RESULTS=OUTDIR,
        TQP_HASH_FILE=f"{HASHDIR}/hash1tnm_part_{sid}.json",
        TQP_QUERIES_DIR=f"{W}/rg_results",
        TQP_QCACHE_NAME="queries1trg.npy",
        TQP_SCRATCH=f"/tmp/rga_{TAG}_{sid}",
        TQP_SPILL_DIR="/tmp",
        TQP_RELEASED=RELEASED,
        # One thread for the searches too, so ten servers hold ten cores (the builds set
        # their own count).
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    if first:
        env.update(TQP_WRITES="block", TQP_DEFER_UNMATCHED="1", TQP_CONFIRM_WRITE="1")
    with open(f"{OUTDIR}/atlas_{sid}.log", "a") as log:
        r = subprocess.run(
            ["taskset", "-c", "12-23", PY, f"{CODE}/fleet_regen_score.py"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return sid, f"rc={r.returncode}"


def publish(sid):
    name = f"tqp-fleet-atlas-{TAG}-{sid}"
    subprocess.run(
        ["kubectl", "-n", NS, "delete", "configmap", name, "--ignore-not-found"]
    )
    subprocess.run(
        [
            "kubectl", "-n", NS, "create", "configmap", name,
            f"--from-file={OUTDIR}/regen{TAG}_atlas_{sid}.npz",
        ],
        check=True,
    )  # fmt: skip
    return f"{sid}:{name}"


os.makedirs(OUTDIR, exist_ok=True)
if "--first" in sys.argv:
    sids = [int(x) for x in sys.argv[sys.argv.index("--first") + 1].split(",")]
    # TQP_FIRST_SHARDS limits it to a few shards, for a test.
    lim = os.environ.get("TQP_FIRST_SHARDS")
    all_js = [int(x) for x in lim.split(",")] if lim else list(range(400))
    with ThreadPoolExecutor(WORKERS) as ex:
        for sid, status in ex.map(lambda s: run(s, all_js, first=True), sids):
            print(f"server {sid}: {status}", flush=True)
    path = f"{OUTDIR}/atlas_first_{TAG}.json"
    done = json.load(open(path)) if os.path.exists(path) else {}
    for sid in sids:
        z = np.load(f"{OUTDIR}/regen{TAG}_atlas_{sid}.npz")
        ok = z["shards"][z["deferred"][z["shards"]] == 0]
        done[str(sid)] = [int(j) for j in ok]
        print(f"server {sid}: {len(ok)}/{len(z['shards'])} reproduced here", flush=True)
    with open(path + ".tmp", "w") as f:
        json.dump(done, f)
    os.replace(path + ".tmp", path)
    maps = [publish(sid) for sid in sids]
    print(f"TQP_ATLAS_FIRST={path}", flush=True)
    print("TQP_ATLAS_MAPS=" + ",".join(maps), flush=True)
    print("ATLAS_FIRST_DONE", flush=True)
    raise SystemExit(0)
deferred = fetch_deferred()
todo = {int(s): js for s, js in deferred.items() if js}
print(
    f"{len(deferred)}/{N_SRV} servers listed, {len(todo)} with deferred shards, "
    f"{sum(map(len, todo.values()))} shards",
    flush=True,
)
if "--fetch-only" in sys.argv or len(deferred) < N_SRV:
    raise SystemExit(
        0 if "--fetch-only" in sys.argv else "not every server is listed yet"
    )
with ThreadPoolExecutor(WORKERS) as ex:
    for sid, status in ex.map(lambda kv: run(*kv), sorted(todo.items())):
        print(f"server {sid}: {status}", flush=True)
bad = []
for sid, js in sorted(todo.items()):
    k = np.load(f"{OUTDIR}/regen{TAG}_atlas_{sid}.npz")["kernel"][js]
    bad += [(sid, j) for j, v in zip(js, k) if not v]
print(f"unreproduced after the Atlas pass: {len(bad)} {bad[:50]}", flush=True)
maps = [publish(sid) for sid in sorted(todo)]
print("TQP_ATLAS_MAPS=" + ",".join(maps), flush=True)
print("ATLAS_PASS_DONE", flush=True)

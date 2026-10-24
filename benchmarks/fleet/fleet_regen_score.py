# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""One server of a measurement over a regenerated index: reference and routed top-10 per shard.

The 500 index volumes of the trillion-row run were released after it was scored. The corpus is a
seed, so a new query set can still be measured against the same index: this job regenerates its
server's 400 shards one at a time into scratch space, exactly as ``fleet_build.py`` wrote them
(``write_shard_streaming`` over ``gen_block_bands``, the bootstrap's basis), assigns the shard's
rows to cells against the global coarse quantizer, and adopts the bootstrap's radius, so the shard
is the one the index held. On that one-shard index it runs the exact scan and the routed search
the first run used (``ShardedIndex.search`` with ``block`` and with ``nprobe``), keeps the best
ten per query across shards, and deletes the shard. Top-k merging across shards is exact, so the
result equals a search of the whole server.

Each regenerated shard file and its two cell sidecars are hashed and compared with the sha256 the
first run recorded for that server's index (``hash1tnm_part_{SID}.json``), so the output states,
shard by shard, whether the index measured is byte for byte the index that was released.

Resumable: after every shard the running state (top-10s, cell counts, hash matches, shards done)
is written atomically to ``regen{TAG}_state_{SID}.npz``; a restarted pod continues from the next
shard. The final partial is ``regen{TAG}_part_{SID}.npz``. Exempt class (1 CPU, 2 GiB).
"""

import hashlib
import json
import os
import shutil
import time
import uuid

import numpy as np
from fleet_common import (
    BOOT,
    RESULTS,
    SHARD_ROWS,
    SHARDS_PER_SERVER,
    drop_page_cache,
    gen_block_bands,
)

from turboquant_pro import ShardedIndex

K = 10
SID = int(os.environ["TQP_SERVER_ID"])
TAG = os.environ.get("TQP_RUN_TAG", "1trg")
NPROBES = [int(x) for x in os.environ.get("TQP_NPROBES", "32,128").split(",")]
BLOCK = int(os.environ.get("TQP_REF_BLOCK", "65536"))
SCRATCH = os.environ.get("TQP_SCRATCH", "/tmp/regen")
BOOT = os.environ.get("TQP_BOOT", BOOT)
RESULTS = os.environ.get("TQP_RESULTS", RESULTS)
SHARD_LIMIT = int(os.environ.get("TQP_SHARD_LIMIT", str(SHARDS_PER_SERVER)))
HASHES = os.environ.get("TQP_HASH_FILE", f"{RESULTS}/hash1tnm_part_{SID}.json")
BASIS = f"{BOOT}/shard_00000.tqe"
OUT = f"{RESULTS}/regen{TAG}_part_{SID}.npz"
STATE = f"{RESULTS}/regen{TAG}_state_{SID}.npz"
# The query set travels in the job's code map (/work), not on the shared volume.
QDIR = os.environ.get("TQP_QUERIES_DIR", RESULTS)
QUERIES = f"{QDIR}/{os.environ.get('TQP_QCACHE_NAME', f'queries{TAG}.npy')}"
ATTEMPT = f"{os.environ.get('HOSTNAME', 'local')}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
names = ["ref"] + [f"p{p}" for p in NPROBES]


def cgroup_peak_mb():
    """The control group's memory peak, page cache included, which the 2 GiB limit binds."""
    try:
        with open("/sys/fs/cgroup/memory.peak") as f:
            return int(f.read()) // 1048576
    except OSError:
        return -1


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def merge(a_ids, a_sc, b_ids, b_sc):
    """Best K of two (nq, K) top lists; -inf for empty slots, as the fleet merge treats NaN."""
    ids = np.concatenate([a_ids, b_ids], axis=1)
    sc = np.concatenate([a_sc, b_sc], axis=1)
    sc_f = np.where(np.isfinite(sc), sc, -np.inf)
    order = np.argsort(-sc_f, axis=1, kind="stable")[:, :K]
    return np.take_along_axis(ids, order, axis=1), np.take_along_axis(sc_f, order, 1)


def save_atomic(path, **arrays):
    """Write under a name unique to this attempt, then rename. Two live attempts for one
    server (a pod left running on a node that stopped responding, beside its replacement)
    never share a temporary file, and rename replaces the target whole."""
    tmp = f"{path}.{ATTEMPT}.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def intact(path):
    """An output counts as done only if it loads and carries every array the merge reads."""
    try:
        with np.load(path) as z:
            return all(f"{n}_ids" in z.files for n in names) and "hash_ok" in z.files
    except Exception:
        return False


if os.path.exists(OUT) and intact(OUT):
    print("partial exists and loads, skipping", flush=True)
    print("REGEN_PART_DONE", flush=True)
    raise SystemExit(0)

q = np.load(QUERIES).astype(np.float32)
nq = len(q)
cent = np.load(f"{BOOT}/coarse_centroids.npy")
with open(HASHES, encoding="utf-8") as f:
    recorded = json.load(f)["digest"]

if os.path.exists(STATE) and intact(STATE):
    st = np.load(STATE)
    done = int(st["shards_done"])
    top = {n: (st[f"{n}_ids"], st[f"{n}_sc"]) for n in names}
    ref_cells = st["ref_cells"]
    counts = st["counts"]
    hash_ok = st["hash_ok"]
    wall = float(st["wall_s"])
    print(f"resume at shard {done}", flush=True)
else:
    done = 0
    empty = (np.full((nq, K), -1, np.int64), np.full((nq, K), -np.inf))
    top = {n: (empty[0].copy(), empty[1].copy()) for n in names}
    ref_cells = np.full((nq, K), -1, np.int64)
    counts = np.zeros(len(cent), np.int64)
    hash_ok = np.zeros(
        (SHARDS_PER_SERVER, 3), np.int8
    )  # tqe, off, memb: 1 equal, -1 not
    wall = 0.0

print(
    f"server {SID}: {SHARD_LIMIT} shards, nq={nq}, widths {NPROBES}, rss={rss_mb():.0f}MB",
    flush=True,
)
for j in range(done, SHARD_LIMIT):
    t0 = time.time()
    g = SID * SHARDS_PER_SERVER + j
    d = os.path.join(SCRATCH, f"s{j:05d}")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    m = ShardedIndex.write_shard_streaming(
        d, gen_block_bands(g), j, ids_start=g * SHARD_ROWS, basis_from=BASIS
    )
    drop_page_cache(os.path.join(d, m["path"]))
    sh = ShardedIndex.finalize_manifest(d, [m])
    sh.build_ivf(centroids=cent)
    shutil.copy(f"{BOOT}/coarse_radius.npy", os.path.join(d, "coarse_radius.npy"))
    stem = os.path.splitext(m["path"])[0]
    for c, rel in enumerate((m["path"], stem + ".ivf.off.npy", stem + ".ivf.memb.npy")):
        hash_ok[j, c] = (
            1 if sha256(os.path.join(d, rel)) == recorded[rel]["sha256"] else -1
        )
    sh = ShardedIndex.open(
        os.path.join(d, "manifest.json"), mmap=True, max_open_shards=1
    )

    ids, sc = sh.search(q, k=K, block=BLOCK)
    # Cell of every row of this shard, for the reference neighbours it contributes.
    off = np.load(os.path.join(d, stem + ".ivf.off.npy"))
    memb = np.load(os.path.join(d, stem + ".ivf.memb.npy"))
    counts += np.diff(off)
    cell_of = np.empty(SHARD_ROWS, np.int16)
    cell_of[memb] = np.repeat(np.arange(len(off) - 1, dtype=np.int16), np.diff(off))
    shard_cells = np.where(
        ids >= 0, cell_of[np.clip(ids - g * SHARD_ROWS, 0, None)], -1
    )
    all_ids = np.concatenate([top["ref"][0], ids], axis=1)
    all_cells = np.concatenate([ref_cells, shard_cells], axis=1)
    all_sc = np.where(
        np.isfinite(np.concatenate([top["ref"][1], sc], axis=1)),
        np.concatenate([top["ref"][1], sc], axis=1),
        -np.inf,
    )
    order = np.argsort(-all_sc, axis=1, kind="stable")[:, :K]
    top["ref"] = (
        np.take_along_axis(all_ids, order, axis=1),
        np.take_along_axis(all_sc, order, axis=1),
    )
    ref_cells = np.take_along_axis(all_cells, order, axis=1)
    for p in NPROBES:
        ids_p, sc_p = sh.search(q, k=K, nprobe=p)
        top[f"p{p}"] = merge(*top[f"p{p}"], ids_p, sc_p)

    del sh
    shutil.rmtree(d, ignore_errors=True)
    wall += time.time() - t0
    arrays = {f"{n}_ids": top[n][0] for n in names}
    arrays.update({f"{n}_sc": top[n][1] for n in names})
    save_atomic(
        STATE,
        shards_done=np.int64(j + 1),
        ref_cells=ref_cells,
        counts=counts,
        hash_ok=hash_ok,
        wall_s=np.float64(wall),
        **arrays,
    )
    print(
        f"shard {j + 1}/{SHARD_LIMIT} (g={g}) {time.time() - t0:.1f}s "
        f"hash {hash_ok[j].tolist()} rss={rss_mb():.0f}MB cgroup_peak={cgroup_peak_mb()}MB",
        flush=True,
    )

st = dict(np.load(STATE))
st["nprobes"] = np.array(NPROBES)
save_atomic(OUT, **st)
n_ok = int((hash_ok[:SHARD_LIMIT] == 1).all(axis=1).sum())
print(
    f"hash: {n_ok}/{SHARD_LIMIT} shards byte-identical to the released index",
    flush=True,
)
print(f"wall_s={wall:.1f}", flush=True)
print("REGEN_PART_DONE", flush=True)

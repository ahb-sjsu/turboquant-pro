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

The shard a build writes depends on the OpenBLAS kernel the build process ran, on whether the
shard was written as one block or streamed, and for some shards on the build_ivf version (see
``fleet_regen_build.py``); the released shards were built on mixed hardware while the code
changed. So each shard is built in a subprocess under one variant (a forced kernel through
``OPENBLAS_CORETYPE``, a write, a library version) and its two cell files are compared with the
record; on a mismatch the next variant is tried, the one that matched the previous shard first.
The kernels tried are the distinct ones this node can run, found at start by forcing each name and
reading the core OpenBLAS reports. The variant that matched is recorded per shard (in ``kernel``);
a shard no variant reproduces is rebuilt under the node's own kernel and the staged library,
searched, and recorded as unmatched. The ``.tqe`` hash cannot match because the container stamps
its creation time.

Resumable: after every shard the running state (top-10s, cell counts, hash matches, shards done)
is written atomically to ``regen{TAG}_state_{SID}.npz``; a restarted pod continues from the next
shard. The final partial is ``regen{TAG}_part_{SID}.npz``. Exempt class (1 CPU, 2 GiB).
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np
from fleet_common import BOOT, RESULTS, SHARD_ROWS, SHARDS_PER_SERVER

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
HERE = os.path.dirname(os.path.abspath(__file__))
# OpenBLAS x86-64 kernel names; "" leaves the choice to OpenBLAS (the node's own kernel).
CANDIDATES = [
    "",
    "Haswell",
    "Zen",
    "SkylakeX",
    "Cooperlake",
    "SapphireRapids",
    "Sandybridge",
    "Nehalem",
    "Prescott",
]


# The thread count changes the build too. Under the Haswell kernel shard 2000 built six ways
# at 1 to 11 threads and one way at every count from 12 to 256 (Atlas, 2026-10-24). Every shard
# reproduced so far reproduced at one thread and none at 64, so one is the default.
THREADS = [int(x) for x in os.environ.get("TQP_BLAS_THREADS", "1").split(",")]
# How the shard was written and which build_ivf assigned it (see fleet_regen_build.py). The
# library versions are the sharded_index.py of 0d5178b ("pre", before the chunked decode) and of
# f99d0d5 ("post"), shipped beside this script and laid over the staged package.
WRITES = os.environ.get("TQP_WRITES", "stream,block").split(",")
LIBS = os.environ.get("TQP_LIBS", "post,pre").split(",")
LIBDIR = os.environ.get("TQP_LIBDIR", "/tmp/tqplib")


def stage_libs():
    """One copy of the staged package per library version, its sharded_index.py replaced."""
    import turboquant_pro

    src = os.path.dirname(turboquant_pro.__file__)
    print(
        f"staged sharded_index.py sha256 {sha256(f'{src}/sharded_index.py')}",
        flush=True,
    )
    for lib in LIBS:
        dst = os.path.join(LIBDIR, lib, "turboquant_pro")
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(
            os.path.join(HERE, f"sharded_index_{lib}.py"), f"{dst}/sharded_index.py"
        )
        print(
            f"library {lib}: sharded_index.py sha256 {sha256(f'{dst}/sharded_index.py')}"
        )


def kernel_env(name, threads=1, lib=None):
    env = dict(os.environ, OPENBLAS_VERBOSE="2", OPENBLAS_NUM_THREADS=str(threads))
    env.pop("OPENBLAS_CORETYPE", None)
    if name:
        env["OPENBLAS_CORETYPE"] = name
    if lib:
        path = [os.path.join(LIBDIR, lib), HERE, env.get("PYTHONPATH", "")]
        env["PYTHONPATH"] = os.pathsep.join(p for p in path if p)
    return env


def probe_kernels():
    """The distinct kernels this node runs: force each name, run a product, read the core."""
    seen = {}
    code = "import numpy as np; a=np.ones((64,64)); (a@a).sum(); print('PROBE_OK')"
    for name in CANDIDATES:
        r = subprocess.run(
            [sys.executable, "-c", code],
            env=kernel_env(name),
            capture_output=True,
            text=True,
            timeout=120,
        )
        core = re.search(r"Core: (\S+)", r.stdout + r.stderr)
        ok = r.returncode == 0 and "PROBE_OK" in r.stdout and core
        print(
            f"kernel {name or 'auto'}: {core.group(1) if core else '?'} rc={r.returncode}",
            flush=True,
        )
        if ok and core.group(1) not in seen:
            seen[core.group(1)] = name
    return seen  # reported core -> name that selects it


def build(g, j, d, name, threads=1, write="stream", lib=None):
    """Build the shard under one variant in a subprocess; its file name."""
    script = os.path.join(HERE, "fleet_regen_build.py")
    r = subprocess.run(
        [sys.executable, script, str(g), str(j), d, write],
        env=kernel_env(name, threads, lib),
        capture_output=True,
        text=True,
    )
    done = re.search(r"BUILD_DONE (\S+)", r.stdout)
    if r.returncode != 0 or not done:
        print(
            f"build g={g} {name or 'auto'}/{threads}/{write}/{lib} failed "
            f"rc={r.returncode}",
            flush=True,
        )
        print((r.stdout + r.stderr)[-2000:], flush=True)
        return None
    return done.group(1)


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
            need = ("hash_ok", "kernel", "trials")
            return all(f"{n}_ids" in z.files for n in names) and all(
                k in z.files for k in need
            )
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
    kernel = [str(x) for x in st["kernel"]]
    trials = st["trials"]
    wall = float(st["wall_s"])
    stage_s = st["stage_s"]
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
    kernel = [""] * SHARDS_PER_SERVER  # core that reproduced the shard, "" for none
    trials = np.zeros(SHARDS_PER_SERVER, np.int8)  # builds run for the shard
    wall = 0.0
    # build of the kept shard, builds discarded, hashing, exact scan, routed searches
    stage_s = np.zeros(5)

stage_libs()
cores = probe_kernels()  # reported core -> the OPENBLAS_CORETYPE name that selects it
# Every shard reproduced so far reproduced under Haswell or SkylakeX, so those go first.
first = [c for c in ("Haswell", "SkylakeX") if c in cores]
cores = {c: cores[c] for c in first + [c for c in cores if c not in first]}
# A variant is a kernel, a thread count, a write and a library, "Haswell/1/block/pre".
variants = {
    f"{c}/{t}/{w}/{lib}": (name, t, w, lib)
    for c, name in cores.items()
    for w in WRITES
    for lib in LIBS
    for t in THREADS
}
print(f"server {SID}: build variants on this node {list(variants)}", flush=True)
print(
    f"server {SID}: {SHARD_LIMIT} shards, nq={nq}, widths {NPROBES}, rss={rss_mb():.0f}MB",
    flush=True,
)
last = next((k for k in reversed(kernel[:done]) if k in variants), None)
for j in range(done, SHARD_LIMIT):
    t0 = time.time()
    g = SID * SHARDS_PER_SERVER + j
    d = os.path.join(SCRATCH, f"s{j:05d}")
    tries = ([last] if last else []) + [v for v in variants if v != last]
    kernel[j], trials[j], built_s, t_hash = "", 0, 0.0, 0.0
    for core in tries + [None]:  # None: nothing matched, rebuild under the node's own
        tb = time.time()
        path = build(g, j, d, *(variants[core] if core else ("", 1)))
        trials[j] += 1
        built_s = time.time() - tb
        stage_s[1] += built_s
        if path is None:
            if core is None:
                raise SystemExit(
                    f"shard {g} failed to build under the node's own kernel"
                )
            continue
        th = time.time()
        stem = os.path.splitext(path)[0]
        rels = (path, stem + ".ivf.off.npy", stem + ".ivf.memb.npy")
        for c, rel in enumerate(rels):
            hash_ok[j, c] = (
                1 if sha256(os.path.join(d, rel)) == recorded[rel]["sha256"] else -1
            )
        t_hash += time.time() - th
        if core is None or (hash_ok[j, 1:] == 1).all():
            kernel[j] = core or ""
            break
    stage_s[1] -= built_s  # the kept build is not discarded work
    stage_s[0] += built_s
    stage_s[2] += t_hash
    last = kernel[j] or last
    t_ivf = time.time()
    sh = ShardedIndex.open(
        os.path.join(d, "manifest.json"), mmap=True, max_open_shards=1
    )
    t_hash = time.time()

    ids, sc = sh.search(q, k=K, block=BLOCK)
    t_exact = time.time()
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
    t_routed = time.time()
    stage_s[3:] += np.diff([t_ivf, t_exact, t_routed])

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
        kernel=np.array(kernel),
        trials=trials,
        wall_s=np.float64(wall),
        stage_s=stage_s,
        **arrays,
    )
    print(
        f"shard {j + 1}/{SHARD_LIMIT} (g={g}) {time.time() - t0:.1f}s "
        f"variant {kernel[j] or 'NONE'} after {int(trials[j])} builds "
        f"(kept build {built_s:.1f}s, exact {t_exact - t_ivf:.1f}s, routed "
        f"{t_routed - t_exact:.1f}s) hash {hash_ok[j].tolist()} "
        f"rss={rss_mb():.0f}MB cgroup_peak={cgroup_peak_mb()}MB",
        flush=True,
    )

st = dict(np.load(STATE))
st["nprobes"] = np.array(NPROBES)
save_atomic(OUT, **st)
n_ok = int((hash_ok[:SHARD_LIMIT, 1:] == 1).all(axis=1).sum())
print(
    f"hash: {n_ok}/{SHARD_LIMIT} shards with both cell files equal to the released index's",
    flush=True,
)
ks, kn = np.unique(np.array(kernel[:SHARD_LIMIT]), return_counts=True)
print(f"variants: {dict(zip([str(k) or 'NONE' for k in ks], kn.tolist()))}", flush=True)
print(f"wall_s={wall:.1f}", flush=True)
print("REGEN_PART_DONE", flush=True)

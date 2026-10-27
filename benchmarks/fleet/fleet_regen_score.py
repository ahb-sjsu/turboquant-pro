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
first run recorded for that server's index (``hash1tnm_part_{SID}.json``). Equal cell files do
not fix the codes (shard 2 has the same cell files written as one block and streamed), and the
``.tqe`` hash cannot match because the container stamps its creation time, so with
``TQP_RELEASED`` the codes are checked too: every neighbour the released exact top-10s of the
two earlier query sets hold in the shard must come back with its released score bit for bit
(``fleet_regen_check.py``). The output states, shard by shard, both checks.

The shard a build writes depends on the OpenBLAS kernel the build process ran, on whether the
shard was written as one block or streamed, and for some shards on the build_ivf version (see
``fleet_regen_build.py``); the released shards were built on mixed hardware while the code
changed. So each shard is built in a subprocess under one variant (a forced kernel through
``OPENBLAS_CORETYPE``, a write, a library version) and its two cell files and released scores
are compared with the record; on a mismatch the next variant is tried, the one that matched the
previous shard first. A shard's position does not settle a write the record leaves open: on
server 0 streamed shards built under SkylakeX sit in runs inside the block-written range
(16-19, 28-33, 41-48, 53-62 of 0-99; Atlas and the cluster, 2026-10-25), so ``TQP_CONFIRM_WRITE``
reports such a shard as undecided instead.
The kernels tried are the distinct ones this node can run, found at start by forcing each name and
reading the core OpenBLAS reports. The variant that matched is recorded per shard (in ``kernel``);
a shard no variant reproduces is rebuilt under the node's own kernel and the staged library,
searched, and recorded as unmatched.

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
# The cluster pass ("part") defers a shard no variant reproduces instead of searching it, and
# the Atlas pass ("atlas") takes the deferred shards by index (TQP_SHARD_LIST) with the block
# write, which needs more memory than a cluster pod has. Each writes its own partial.
KIND = os.environ.get("TQP_PART_KIND", "part")
DEFER = os.environ.get("TQP_DEFER_UNMATCHED", "0") == "1"
SHARDS = (
    [int(x) for x in os.environ["TQP_SHARD_LIST"].split(",")]
    if os.environ.get("TQP_SHARD_LIST")
    else list(range(SHARD_LIMIT))
)
OUT = f"{RESULTS}/regen{TAG}_{KIND}_{SID}.npz"
STATE = (
    f"{RESULTS}/regen{TAG}_{'state' if KIND == 'part' else KIND + 'state'}_{SID}.npz"
)
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
# reproduced so far reproduced at one thread and none at 64, so one is the default here. The pool
# build pods set no count and ran what their node allowed: 15 of server 300's 17 cluster-deferred
# shards reproduce at 2 or 3 threads (Atlas, 2026-10-27), so the cluster pass runs 1,2,3.
THREADS = [int(x) for x in os.environ.get("TQP_BLAS_THREADS", "1").split(",")]
# How the shard was written and which build_ivf assigned it (see fleet_regen_build.py). The
# library versions are the sharded_index.py of 0d5178b ("pre", before the chunked decode) and of
# f99d0d5 ("post"), shipped beside this script and laid over the staged package.
WRITES = os.environ.get("TQP_WRITES", "stream,block").split(",")
LIBS = os.environ.get("TQP_LIBS", "post,pre").split(",")
LIBDIR = os.environ.get("TQP_LIBDIR", "/tmp/tqplib")
# The released exact top-10s the codes are checked against (see fleet_regen_check.py), as
# "name=queries=ref" items with {sid} in the ref path. Empty: the cell files alone decide.
RELEASED = [
    s.replace("{sid}", str(SID)).split("=")
    for s in os.environ.get("TQP_RELEASED", "").split(",")
    if s
]
# Search settings tried for them: kernel x threads x scan block. The reference pods set no
# thread count, so it was the pod's CPUs (1 after 8f25bc4, 6 before) or the node's; the block
# was 65536 after 8f25bc4 and the default before it.
SEARCH_THREADS = [
    int(x)
    for x in os.environ.get("TQP_SEARCH_THREADS", "1,6,2,3,4,8,12,16,64").split(",")
]
SEARCH_BLOCKS = os.environ.get("TQP_SEARCH_BLOCKS", "65536,none").split(",")
# A server's setting for a set is pinned once this many of its neighbours have matched under
# it; after that a mismatch condemns the codes instead of sending the search to other settings.
PIN_AFTER = int(os.environ.get("TQP_SEARCH_PIN_AFTER", "20"))
# The first servers wrote shards both ways. With TQP_CONFIRM_WRITE=1 an accepted shard is also
# built with the other write, under the same kernel and each library, and checked the same way,
# so the output says whether the record decides the write or leaves it undecided.
CONFIRM = os.environ.get("TQP_CONFIRM_WRITE", "0") == "1"
# Settings already known for this server ("1t=Haswell/6/none,..."), from another pass over
# it, start pinned: a few neighbours fit many settings, so a pass that starts without them on
# a shard holding few could accept one the server's reference never ran.
PRESET_PIN = dict(
    x.split("=") for x in os.environ.get("TQP_SEARCH_PIN", "").split(",") if x
)
# The shapes the reference jobs ran in, per set ("threads/block", joined by +): 6 CPUs at the
# default block before 8f25bc4 and 1 CPU at 65536 after it for 1t, and 1 CPU at 65536 for 1tnm,
# which ran after it. Under these kernels they are checked first, and the other settings only
# when none of them matches. On server 0 one 1tnm neighbour matched all 72 settings, so
# narrowing over every setting cost an hour a shard.
SEARCH_FIRST = dict(
    x.split("=")
    for x in os.environ.get("TQP_SEARCH_FIRST", "1t=6/none+1/65536,1tnm=1/65536").split(
        ","
    )
    if x
)
SEARCH_FIRST_KERNELS = os.environ.get(
    "TQP_SEARCH_FIRST_KERNELS", "Haswell,SkylakeX"
).split(",")


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


# OpenBLAS runs min(OPENBLAS_NUM_THREADS, CPUs in the affinity mask) threads, and a 1-CPU NRP
# pod is pinned to one core, so without this a 6-thread setting there runs at 1 thread and
# cannot reproduce the 6-CPU reference jobs' scores (server 12, 2026-10-26). fakecpus.so
# (fakecpus.c, shipped beside this script) reports the asked-for count instead.
FAKECPUS = os.path.join(HERE, "fakecpus.so")
ALLOWED_CPUS = len(os.sched_getaffinity(0))


def kernel_env(name, threads=1, lib=None):
    env = dict(os.environ, OPENBLAS_VERBOSE="2", OPENBLAS_NUM_THREADS=str(threads))
    env.pop("OPENBLAS_CORETYPE", None)
    env.pop("TQP_FAKE_CPUS", None)
    if name:
        env["OPENBLAS_CORETYPE"] = name
    if lib:
        path = [os.path.join(LIBDIR, lib), HERE, env.get("PYTHONPATH", "")]
        env["PYTHONPATH"] = os.pathsep.join(p for p in path if p)
    if threads > ALLOWED_CPUS and os.path.exists(FAKECPUS):
        env["LD_PRELOAD"] = FAKECPUS
        env["TQP_FAKE_CPUS"] = str(threads)
    return env


BLAS_THREADS_CODE = (
    "import ctypes, glob, os, numpy\n"
    "d = os.path.join(os.path.dirname(numpy.__file__), '..', 'numpy.libs')\n"
    "lib = ctypes.CDLL(glob.glob(os.path.join(d, '*openblas*'))[0])\n"
    "for n in ('scipy_openblas_get_num_threads64_', 'openblas_get_num_threads64_',\n"
    "          'openblas_get_num_threads'):\n"
    "    if hasattr(lib, n):\n"
    "        print('BLAS_THREADS', getattr(lib, n)())\n"
    "        break\n"
)


def check_blas_threads():
    """Stop unless OpenBLAS runs exactly the thread counts of the reference shapes and the
    builds here."""
    must = {int(x.split("/")[0]) for v in SEARCH_FIRST.values() for x in v.split("+")}
    must |= set(THREADS)  # the build thread counts too
    for t in sorted(set(SEARCH_THREADS) | must):
        r = subprocess.run(
            [sys.executable, "-c", BLAS_THREADS_CODE],
            env=kernel_env("", t),
            capture_output=True,
            text=True,
            timeout=120,
        )
        m = re.search(r"BLAS_THREADS (\d+)", r.stdout)
        got = int(m.group(1)) if m else None
        print(
            f"blas threads {t}: openblas runs {got} ({ALLOWED_CPUS} cpus allowed, "
            f"{'fakecpus' if kernel_env('', t).get('LD_PRELOAD') else 'no preload'})",
            flush=True,
        )
        if t in must and got != t:
            raise SystemExit(f"OpenBLAS runs {got} threads when asked for {t}")


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


def check(g, d, setting, sets):
    """Released neighbours of shard g matched by the build in d under one search setting:
    {set name: (equal, neighbours)}, or None if the check did not run."""
    name, threads, blk = setting
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "fleet_regen_check.py"), d, str(g), blk]
        + ["=".join(s) for s in sets],
        env=kernel_env(name, threads),
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or "CHECK_DONE" not in r.stdout:
        print(f"check g={g} {setting} failed rc={r.returncode}", flush=True)
        print((r.stdout + r.stderr)[-2000:], flush=True)
        return None
    return {
        m.group(1): (int(m.group(2)), int(m.group(3)))
        for m in re.finditer(r"^CHECK (\S+) (\d+) (\d+)$", r.stdout, re.M)
    }


def settings_matching(g, j, d, c, settings, stop_at_first):
    """Of these search settings, those under which the build in d gives every released
    neighbour of set c in shard g its released score, and the shard's neighbour count (0: none
    to match, None: no check ran). Keeps the best count per shard in score_eq / score_n.
    """
    s = RELEASED[c]
    hits, n = [], None
    for setting in settings:
        res = check(g, d, search_settings[setting], [s])
        if res is None:
            continue
        eq, n = res[s[0]]
        score_eq[j, c], score_n[j, c] = max(score_eq[j, c], eq), n
        if n == 0:
            return [], 0
        if eq == n:
            hits.append(setting)
            if stop_at_first:
                break
    return hits, n


def first_settings(key):
    """The search settings checked first for a set: its reference shapes under the first
    kernels, or every setting when the set names none or the node runs none of them."""
    shapes = SEARCH_FIRST.get(key, "").split("+")
    first = [
        x
        for x in search_settings
        if x.split("/")[0] in SEARCH_FIRST_KERNELS and x.split("/", 1)[1] in shapes
    ]
    return first or list(search_settings)


def codes_match(g, j, d):
    """Do the build's codes give every released neighbour of the shard its released score?
    A server's settings for a set narrow to those that matched every neighbour checked so far
    (a wrong setting matched 238 of 244 on server 0, so the first match is not the pin). The
    first candidates are the set's reference shapes (SEARCH_FIRST), and every other setting
    only if none of them matches. Until PIN_AFTER neighbours agree every candidate is checked,
    and a shard none of them matches is checked under every other setting; after that the
    first candidate to match decides. Records the counts and the setting per set; True if
    every set matched."""
    ok = True
    for c, s in enumerate(RELEASED):
        key = s[0]
        weak = pinned_n[c] < PIN_AFTER
        known = cands.get(key)
        score_eq[j, c], score_n[j, c], search_used[j][c] = 0, 0, ""
        first = known or first_settings(key)
        hits, n = settings_matching(g, j, d, c, first, stop_at_first=not weak)
        if n == 0:
            continue
        if not hits and not known:  # none of the reference shapes: every other setting
            rest = [x for x in search_settings if x not in first]
            hits, n = settings_matching(g, j, d, c, rest, stop_at_first=False)
            if hits:
                cands[key] = hits
        elif not hits and known and weak:  # the evidence for the candidates was thin
            rest = [x for x in search_settings if x not in known]
            hits, n = settings_matching(g, j, d, c, rest, stop_at_first=False)
            if hits:
                cands[key], pinned_n[c] = hits, 0
        elif hits and weak:
            cands[key] = hits
        if not hits:
            ok = False
            continue
        search_used[j][c] = hits[0]
        pinned_n[c] += n
        pin[key] = cands[key][0]
    return ok


def other_write_passes(g, j, d, core):
    """1 if the shard built with the other write, under the accepted kernel and either library,
    also has equal cell files and gives every released neighbour its score under the setting
    the accepted build matched it under (the record cannot tell the writes apart), else 0.
    """
    c, t, w, _ = core.split("/")
    other = "stream" if w == "block" else "block"
    alt = d + "_other"
    try:
        for lib in LIBS:
            v = f"{c}/{t}/{other}/{lib}"
            path = build(g, j, alt, *variants[v]) if v in variants else None
            if path is None:
                continue
            stem = os.path.splitext(path)[0]
            if any(
                sha256(os.path.join(alt, rel)) != recorded[rel]["sha256"]
                for rel in (stem + ".ivf.off.npy", stem + ".ivf.memb.npy")
            ):
                continue
            ok = True
            for k, s in enumerate(RELEASED):
                if score_n[j, k]:
                    res = check(g, alt, search_settings[search_used[j][k]], [s])
                    ok = ok and res is not None and res[s[0]][0] == res[s[0]][1]
            if ok:
                return 1
        return 0
    finally:
        shutil.rmtree(alt, ignore_errors=True)


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


def intact(path, state=False):
    """An output counts as done only if it loads and carries every array the merge reads. A
    running state also needs ``fake_cpus``: one written before fakecpus.so checked a 6-thread
    setting at 1 thread on a pinned pod and may have deferred shards that match."""
    try:
        with np.load(path) as z:
            need = (
                "hash_ok",
                "kernel",
                "trials",
                "deferred",
                "shards",
                "score_n",
                "search_cands",
            ) + (("fake_cpus",) if state else ())
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

if os.path.exists(STATE) and intact(STATE, state=True):
    st = np.load(STATE)
    done = int(st["shards_done"])
    top = {n: (st[f"{n}_ids"], st[f"{n}_sc"]) for n in names}
    ref_cells = st["ref_cells"]
    counts = st["counts"]
    hash_ok = st["hash_ok"]
    kernel = [str(x) for x in st["kernel"]]
    trials = st["trials"]
    deferred = st["deferred"]
    wall = float(st["wall_s"])
    stage_s = st["stage_s"]
    score_eq, score_n = st["score_eq"], st["score_n"]
    search_used = st["search_used"].tolist()
    pin = json.loads(str(st["search_pin"]))
    cands = json.loads(str(st["search_cands"]))
    pinned_n = st["pinned_n"]
    other_ok = st["other_ok"]
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
    deferred = np.zeros(SHARDS_PER_SERVER, np.int8)  # 1: left for the Atlas pass
    wall = 0.0
    # build of the kept shard, builds discarded, hashing and score checks, exact scan, routed
    stage_s = np.zeros(5)
    # Released neighbours per shard and set: how many, how many matched bit for bit, and the
    # search setting they matched under ("" if none or none to match).
    score_eq = np.zeros((SHARDS_PER_SERVER, len(RELEASED)), np.int16)
    score_n = np.zeros((SHARDS_PER_SERVER, len(RELEASED)), np.int16)
    search_used = [[""] * len(RELEASED) for _ in range(SHARDS_PER_SERVER)]
    pin = dict(PRESET_PIN)  # set name -> the setting pinned for this server
    cands = {
        k: [v] for k, v in PRESET_PIN.items()
    }  # set name -> settings still consistent
    pinned_n = np.array(  # neighbours matched under the pin; a preset pin is trusted
        [PIN_AFTER if s[0] in PRESET_PIN else 0 for s in RELEASED], np.int64
    )
    # 1: the other write passes too, 0: it does not, -1: not tried (TQP_CONFIRM_WRITE)
    other_ok = np.full(SHARDS_PER_SERVER, -1, np.int8)


def save_state(n_done):
    arrays = {f"{n}_ids": top[n][0] for n in names}
    arrays.update({f"{n}_sc": top[n][1] for n in names})
    save_atomic(
        STATE,
        shards_done=np.int64(n_done),
        shards=np.array(SHARDS, np.int64),
        ref_cells=ref_cells,
        counts=counts,
        hash_ok=hash_ok,
        kernel=np.array(kernel),
        trials=trials,
        deferred=deferred,
        wall_s=np.float64(wall),
        stage_s=stage_s,
        score_eq=score_eq,
        score_n=score_n,
        search_used=np.array(search_used, dtype="U32").reshape(score_n.shape),
        search_pin=np.array(json.dumps(pin)),
        search_cands=np.array(json.dumps(cands)),
        pinned_n=pinned_n,
        other_ok=other_ok,
        fake_cpus=np.int8(1),
        released_sets=np.array([s[0] for s in RELEASED], dtype="U16"),
        **arrays,
    )


# A missing released file would fail every check and defer every shard, so stop instead.
for s in RELEASED:
    for path in s[1:]:
        if not os.path.exists(path):
            raise SystemExit(f"released set {s[0]}: {path} not found")
        print(f"released {s[0]} {path} sha256 {sha256(path)}", flush=True)
check_blas_threads()
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
search_settings = {
    f"{c}/{t}/{b}": (name, t, b)
    for c, name in cores.items()
    for b in SEARCH_BLOCKS
    for t in SEARCH_THREADS
}
if RELEASED:
    print(
        f"server {SID}: released sets {[s[0] for s in RELEASED]}, "
        f"{len(search_settings)} search settings, pinned {pin}",
        flush=True,
    )
print(
    f"server {SID}: {len(SHARDS)} shards ({KIND}), nq={nq}, widths {NPROBES}, rss={rss_mb():.0f}MB",
    flush=True,
)
last = next((kernel[j] for j in reversed(SHARDS[:done]) if kernel[j] in variants), None)
for pos in range(done, len(SHARDS)):
    j = SHARDS[pos]
    t0 = time.time()
    g = SID * SHARDS_PER_SERVER + j
    d = os.path.join(SCRATCH, f"s{j:05d}")
    tries = ([last] if last else []) + [v for v in variants if v != last]
    kernel[j], trials[j], built_s, t_hash = "", 0, 0.0, 0.0
    # None: nothing matched, rebuild under the node's own kernel and the staged library
    # and search it, recorded unmatched; a deferring pass leaves the shard instead.
    for core in tries + ([] if DEFER else [None]):
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
        cells_ok = (hash_ok[j, 1:] == 1).all()
        # Equal cell files do not fix the codes; the released scores check those.
        codes_ok = (
            cells_ok and core is not None and (not RELEASED or codes_match(g, j, d))
        )
        t_hash += time.time() - th
        if core is None or codes_ok:
            kernel[j] = core or ""
            break
        if cells_ok:
            print(
                f"shard g={g} {core}: cell files equal, released scores "
                f"{score_eq[j].tolist()}/{score_n[j].tolist()}, next variant",
                flush=True,
            )
    stage_s[2] += t_hash
    if DEFER and not kernel[j]:
        deferred[j] = 1
        shutil.rmtree(d, ignore_errors=True)
        wall += time.time() - t0
        save_state(pos + 1)
        print(
            f"shard {pos + 1}/{len(SHARDS)} (g={g}) {time.time() - t0:.1f}s DEFERRED "
            f"after {int(trials[j])} builds, no variant matched",
            flush=True,
        )
        continue
    if CONFIRM and kernel[j]:
        tc = time.time()
        other_ok[j] = other_write_passes(g, j, d, kernel[j])
        stage_s[1] += time.time() - tc
    stage_s[1] -= built_s  # the kept build is not discarded work
    stage_s[0] += built_s
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
    save_state(pos + 1)
    print(
        f"shard {pos + 1}/{len(SHARDS)} (g={g}) {time.time() - t0:.1f}s "
        f"variant {kernel[j] or 'NONE'} after {int(trials[j])} builds "
        f"(kept build {built_s:.1f}s, exact {t_exact - t_ivf:.1f}s, routed "
        f"{t_routed - t_exact:.1f}s) hash {hash_ok[j].tolist()} "
        f"released {score_eq[j].tolist()}/{score_n[j].tolist()} {search_used[j]} "
        f"other write {'passes' if other_ok[j] == 1 else 'fails' if other_ok[j] == 0 else '-'} "
        f"rss={rss_mb():.0f}MB cgroup_peak={cgroup_peak_mb()}MB",
        flush=True,
    )

st = dict(np.load(STATE))
st["nprobes"] = np.array(NPROBES)
save_atomic(OUT, **st)
sel = np.array(SHARDS)
n_ok = int((hash_ok[sel, 1:] == 1).all(axis=1).sum())
print(
    f"hash: {n_ok}/{len(sel)} shards with both cell files equal to the released index's",
    flush=True,
)
if RELEASED:
    kept = deferred[sel] == 0  # a deferred shard was never checked here
    has = kept & (score_n[sel].sum(axis=1) > 0)
    good = (score_eq[sel] == score_n[sel]).all(axis=1) & (np.array(kernel)[sel] != "")
    print(
        f"codes: {int((has & good).sum())}/{int(kept.sum())} shards searched here with every "
        f"released neighbour bit-equal ({int(score_n[sel].sum())} neighbours), "
        f"{int((kept & ~has).sum())} hold none; pinned {pin}",
        flush=True,
    )
ks, kn = np.unique(np.array(kernel)[sel], return_counts=True)
print(f"variants: {dict(zip([str(k) or 'NONE' for k in ks], kn.tolist()))}", flush=True)
# The line the Atlas pass reads: this server's shards left for it, by index on the server.
print(f"DEFERRED {SID} {','.join(str(j) for j in sel[deferred[sel] == 1])}", flush=True)
print(f"wall_s={wall:.1f}", flush=True)
print("REGEN_PART_DONE", flush=True)

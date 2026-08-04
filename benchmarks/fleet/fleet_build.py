# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""One fleet build job: this server's 250M-row shard-range, built in place.

Regenerates its seeded row-range (zero corpus movement), writes 50 v3 shards
onto the server's own Linstor block PVC via ``write_shard(basis_from=…)``
(shared basis -> comparable scores), fills its slice of the CephFS cold store,
assigns cells against the **global** coarse quantizer, then adopts the
bootstrap's radius file so the entire fleet probes identical cells. Exports
the tiny coordinator artifacts (manifest + ``.ivf.off.npy`` sidecars) to the
shared volume.
"""

import glob
import json
import os
import shutil

import numpy as np
from fleet_common import (
    BOOT,
    SHARD_ROWS,
    SHARDS_PER_SERVER,
    SHARED,
    gen_block,
    write_original,
)

from turboquant_pro import ShardedIndex
from turboquant_pro.index_file import MAGIC

SID = int(os.environ["TQP_SERVER_ID"])
IDX = "/idx"
BASIS = f"{BOOT}/shard_00000.tqe"
# 10B runs skip the cold store (1.28 TB of fp32 originals is beyond quota;
# rerank/true-recall are already demonstrated at 1B) and export to their own
# shared dirs so the 1B artifacts stay intact.
WRITE_ORIGINALS = os.environ.get("TQP_WRITE_ORIGINALS", "1") == "1"
EXPORT_PREFIX = os.environ.get("TQP_EXPORT_PREFIX", "server_")


def _atomic_write_json(path, obj):
    """Write JSON via temp + rename so a kill mid-write can't leave a torn file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _shard_intact(path):
    """Cheap corruption gate for the resume scan: the shard file exists, is
    non-empty, and bears the TQIX magic. A done-sidecar can outlive a torn shard
    write (run-3's pod churn left JSON text where a TQIX container should be), and
    trusting the sidecar alone then crashes every restart at IVF-assign time with
    ``IndexCorruptionError``. Reading 4 bytes covers all three checks: a missing
    file raises ``OSError``, an empty/short one can't equal the magic."""
    try:
        with open(path, "rb") as f:
            return f.read(len(MAGIC)) == MAGIC
    except OSError:
        return False


# Shard-level resume. NRP preempts/sweeps long-running pods roughly every ~2 h,
# but a full 400-shard build is ~4-5 h, so a plain build never finishes: every
# restart begins again at shard 0. Instead, each finished shard drops a meta
# sidecar written LAST (after the .tqe is fully saved), so its presence
# guarantees a good shard; a restarted pod reloads those metas and rebuilds only
# the missing/interrupted shards. Progress now accumulates across pod lifetimes.
metas = []
for j in range(SHARDS_PER_SERVER):
    g = SID * SHARDS_PER_SERVER + j
    meta_path = os.path.join(IDX, f"shard_{j:05d}.meta.json")
    shard_path = os.path.join(IDX, f"shard_{j:05d}.tqe")
    if os.path.exists(meta_path):
        # The sidecar's presence is only trustworthy if the shard it vouches for
        # is actually a TQIX file — validate before skipping.
        if _shard_intact(shard_path):
            with open(meta_path, encoding="utf-8") as f:
                metas.append(json.load(f))
            print(f"shard {j + 1}/{SHARDS_PER_SERVER} (g={g}) resume-skip", flush=True)
            continue
        # Corrupt/torn shard behind a surviving done-marker: drop BOTH and fall
        # through to a full rebuild (gen_block is seeded, so the rows regenerate
        # identically and every other shard's IVF sidecars stay valid).
        print(
            f"CORRUPT shard {j + 1}/{SHARDS_PER_SERVER} (g={g}): {shard_path} "
            "failed the TQIX magic check but its done-sidecar survived; "
            "deleting shard + sidecar and rebuilding",
            flush=True,
        )
        for p in (shard_path, meta_path):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
    block = gen_block(g)
    m = ShardedIndex.write_shard(
        IDX,
        block,
        j,
        ids=np.arange(g * SHARD_ROWS, (g + 1) * SHARD_ROWS, dtype=np.int64),
        basis_from=BASIS,
        keep_originals=False,
    )
    if WRITE_ORIGINALS:
        write_original(g, block)
    _atomic_write_json(meta_path, m)  # completeness marker, written after the shard
    metas.append(m)
    print(f"shard {j + 1}/{SHARDS_PER_SERVER} (g={g}) written", flush=True)

# Finalize is re-runnable and gated by a marker so a resume past the shard phase
# doesn't redo the coarse assignment.
FINAL = os.path.join(IDX, ".finalized")
if os.path.exists(FINAL):
    print("finalize already done, resume-skip", flush=True)
else:
    sh = ShardedIndex.finalize_manifest(IDX, metas)
    print("assigning cells against the global coarse quantizer", flush=True)
    # resume=True: the ~84-min assignment of a server's 2B rows is the one remaining
    # non-resumable monolith. Per-shard sidecars are staged to /idx as they land, so a
    # preempted pod resumes the assignment instead of restarting it -- each shard is a
    # self-contained block, built and cell-assigned before the loop moves on.
    sh.build_ivf(centroids=np.load(f"{BOOT}/coarse_centroids.npy"), resume=True)
    # Fleet-wide consistent probing: every server (and the router) uses the
    # bootstrap's radius, not the locally-estimated one.
    shutil.copy(f"{BOOT}/coarse_radius.npy", f"{IDX}/coarse_radius.npy")

    out = f"{SHARED}/{EXPORT_PREFIX}{SID:03d}"
    os.makedirs(out, exist_ok=True)
    shutil.copy(f"{IDX}/manifest.json", out)
    for p in glob.glob(f"{IDX}/*.ivf.off.npy"):
        shutil.copy(p, out)
    open(FINAL, "w").close()

n_rows = SHARDS_PER_SERVER * SHARD_ROWS
nbytes = sum(os.path.getsize(p) for p in glob.glob(f"{IDX}/*"))
print(
    json.dumps(
        {
            "server": SID,
            "rows": n_rows,
            "index_bytes": nbytes,
            "bytes_per_row": round(nbytes / n_rows, 2),
        }
    ),
    flush=True,
)
print("BUILD_DONE", flush=True)

# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""Pack the shared volume's record into one archive, so the measurement is self-contained.

The bootstrap (basis, coarse quantizer, manifest), the query caches, every per-server partial
of every phase, the shortlists and regenerated rows, and every result JSON live on the shared
volume, not on the index volumes. This job writes ``{SHARED}/archive/<name>.tar.gz`` with a
sha256 manifest of every member, prints the total, and is the step before the archive is
copied off the cluster. Idempotent: skips if the archive exists.
"""

import hashlib
import json
import os
import tarfile
import time

from fleet_common import BOOT, RESULTS, SHARED

NAME = os.environ.get("TQP_ARCHIVE_NAME", "shared-fleet-" + time.strftime("%Y%m%d"))
OUT = f"{SHARED}/archive/{NAME}.tar.gz"
ROOTS = [BOOT, RESULTS]

if os.path.exists(OUT):
    print(f"archive exists: {OUT}", flush=True)
    print("ARCHIVE_DONE", flush=True)
    raise SystemExit(0)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
manifest = {}
t0 = time.time()
tmp = OUT + ".tmp"
with tarfile.open(tmp, "w:gz", compresslevel=6) as tar:
    for root in ROOTS:
        for dirpath, _dirs, files in os.walk(root):
            for fn in sorted(files):
                p = os.path.join(dirpath, fn)
                rel = os.path.relpath(p, SHARED)
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 22), b""):
                        h.update(chunk)
                manifest[rel] = {"sha256": h.hexdigest(), "bytes": os.path.getsize(p)}
                tar.add(p, arcname=rel)
    mpath = f"{SHARED}/archive/{NAME}.manifest.json"
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    tar.add(mpath, arcname=f"{NAME}.manifest.json")
os.replace(tmp, OUT)
h = hashlib.sha256()
with open(OUT, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 22), b""):
        h.update(chunk)
summary = {
    "archive": OUT,
    "files": len(manifest),
    "bytes_in": sum(v["bytes"] for v in manifest.values()),
    "bytes_out": os.path.getsize(OUT),
    "sha256": h.hexdigest(),
    "wall_s": round(time.time() - t0, 1),
}
print("ARCHIVE_JSON " + json.dumps(summary), flush=True)
print("ARCHIVE_DONE", flush=True)

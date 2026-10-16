#!/usr/bin/env python3
"""Keep a model's verified fp16 copy on its volume set, so later run pods skip building it.

    python part5_preload.py --src /tmp/model16/mistral-7b-instruct --vols /w0,/w1,/w2 \
        --key mistral-7b-instruct

A run pod's first minutes were building /tmp/model16 (part5_cache_fast.py: load, convert,
save, check) with its GPU idle, and NRP removes GPU pods that idle at start. The first pod
of each model runs this in the background once its copy is verified: each weight file goes
to the volume with the most free space, the small files to volume 0, every file is read
back and its sha256 compared with the local one, and only then is READY written, listing
each file, its volume and its sha256. Later pods link those files into /tmp/model16 and
start loading at once. A pod that dies mid-copy leaves no READY, and the next pod copies
again; a file is written under a temporary name and renamed when complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time

READY = "READY.json"
CHUNK = 64 << 20


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return h.hexdigest()


def copy(src: str, dest: str) -> None:
    """Plain writes, flushed and dropped from the page cache as they go (an exempt-size
    pod's memory counts its dirty pages)."""
    tmp = dest + ".part"
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        while True:
            b = fi.read(CHUNK)
            if not b:
                break
            fo.write(b)
            fo.flush()
            os.fsync(fo.fileno())
            os.posix_fadvise(fo.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    os.replace(tmp, dest)


def place(sizes: dict, free: dict) -> dict:
    """Weight file -> volume: largest file first, each onto the volume with most room."""
    out, room = {}, dict(free)
    for f in sorted(sizes, key=sizes.get, reverse=True):
        v = max(room, key=room.get)
        if room[v] < sizes[f]:
            raise SystemExit(f"[preload] no volume has room for {f} ({sizes[f]} bytes)")
        out[f] = v
        room[v] -= sizes[f]
    return out


def ready_files(vols: list, key: str):
    """{file: path} from a complete preload of ``key`` on ``vols``, else None."""
    p = os.path.join(vols[0], "model16", key, READY)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        rec = json.load(f)
    return {n: os.path.join(e["vol"], "model16", key, n) for n, e in rec["files"].items()}


def link(vols: list, key: str, dest: str) -> int:
    """Link a complete preload into ``dest``: 0, or 3 when there is none to link (the pod
    then builds its copy). Each file must be present at its recorded size."""
    files = ready_files(vols, key)
    if files is None:
        print(f"[preload] no fp16 copy of {key} on the set yet; building one", flush=True)
        return 3
    with open(os.path.join(vols[0], "model16", key, READY)) as f:
        rec = json.load(f)["files"]
    short = [n for n, p in files.items()
             if not os.path.exists(p) or os.path.getsize(p) != rec[n]["bytes"]]
    if short:
        print(f"[preload] {key}'s copy on the set is short ({short[:3]}); building one",
              flush=True)
        return 3
    os.makedirs(dest, exist_ok=True)
    for n, p in files.items():
        os.symlink(p, os.path.join(dest, n))
    print(f"[preload] linked {len(files)} files of {key} from the set", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="the verified local fp16 copy, to put on the set")
    ap.add_argument("--link", help="link the set's copy into this directory instead")
    ap.add_argument("--vols", required=True, help="the set's mount points, volume 0 first")
    ap.add_argument("--key", required=True)
    ap.add_argument("--reserve", type=int, default=1 << 30, help="bytes left free per volume")
    a = ap.parse_args(argv)
    vols = a.vols.split(",")
    if a.link:
        return link(vols, a.key, a.link)
    if not a.src:
        ap.error("--src or --link")
    if ready_files(vols, a.key) is not None:
        print(f"[preload] {a.key} already on the set", flush=True)
        return 0
    t0 = time.time()
    names = sorted(os.listdir(a.src))
    weights = {n: os.path.getsize(os.path.join(a.src, n)) for n in names
               if n.endswith(".safetensors")}
    free = {v: shutil.disk_usage(v).free - a.reserve for v in vols}
    where = {**place(weights, free), **{n: vols[0] for n in names if n not in weights}}
    rec = {}
    for n in names:
        d = os.path.join(where[n], "model16", a.key)
        os.makedirs(d, exist_ok=True)
        local = os.path.join(a.src, n)
        copy(local, os.path.join(d, n))
        want, got = sha256(local), sha256(os.path.join(d, n))
        if want != got:
            raise SystemExit(f"[preload] {n} on {where[n]} reads back {got[:12]}, "
                             f"not {want[:12]}")
        rec[n] = {"vol": where[n], "bytes": os.path.getsize(local), "sha256": want}
    out = os.path.join(vols[0], "model16", a.key, READY)
    with open(out + ".part", "w") as f:
        json.dump({"key": a.key, "files": rec, "written": time.strftime("%FT%TZ", time.gmtime())},
                  f, indent=1)
    os.replace(out + ".part", out)
    print(f"[preload] {a.key}: {len(rec)} files on {len(vols)} volumes, read back and "
          f"sha256-checked, in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

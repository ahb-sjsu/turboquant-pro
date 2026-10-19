#!/usr/bin/env python3
"""Keep a model's verified fp16 copy on its volume set, so later run pods skip building it.

    python part5_preload.py --src /tmp/model16/mistral-7b-instruct --vols /w0,/w1,/w2 \
        --key mistral-7b-instruct
    python part5_preload.py --restripe --vols /w0,/w1,/w2 --key mistral-7b-instruct

A run pod's first minutes were building /tmp/model16 (part5_cache_fast.py: load, convert,
save, check) with its GPU idle, and NRP removes GPU pods that idle at start. The first pod
of each model runs this in the background once its copy is verified, and later pods link
the set's copy into /tmp/model16 and start loading at once.

The copy is striped (2026-10-19). transformers 5.5 loads tensors in key order on four
threads, so with whole files on a volume each the four threads read one volume at a time
(~150 MB/s on linstor-unl) and Mistral-7B's 14.5 GB took ~100 s, at NRP's idle limit:
its resumed head pods were removed one after another. Here the tensors, in the loader's
order, are dealt onto one file per volume (each to the volume least full for its room,
``stripe``), so consecutive tensors are read from different volumes at once. Every tensor's bytes are copied unchanged into a safetensors
file with its own header, and a new index maps each tensor to its file; the files are
read back and their sha256 compared with the bytes they were built to hold, and only then
is READY written. A pod that dies mid-copy leaves no READY, and the next pod copies again;
a file is written under a temporary name and renamed when complete. ``--restripe`` builds
the striped copy from an earlier whole-file copy on the set (model16/), on a CPU pod.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import time

READY = "READY.json"
CHUNK = 64 << 20
OLD = "model16"  # whole-file copies (before 2026-10-19), still linkable
DIR = "model16s"  # striped copies
INDEX = "model.safetensors.index.json"
RESERVE = 1 << 30  # bytes left free on each volume


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


def natural_key(name: str):
    """transformers' load order (core_model_loading.dot_natural_key): dotted parts, numbers
    compared as numbers."""
    return [(0, int(p), "") if p.isdigit() else (1, 0, p) for p in re.split(r"\.", name)]


def read_header(path: str):
    """(header dict, data start) of a safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def tensors(src_files: list) -> list:
    """[(name, file, dtype, shape, begin, end)] of every tensor, in the loader's order."""
    out = []
    for path in src_files:
        head, start = read_header(path)
        for name, e in head.items():
            if name == "__metadata__":
                continue
            b, en = e["data_offsets"]
            out.append((name, path, e["dtype"], e["shape"], start + b, start + en))
    return sorted(out, key=lambda t: natural_key(t[0]))


def stripe(ts: list, room: list) -> list:
    """Tensor lists, one per volume. In load order each tensor goes to the volume that would
    be least full after it, relative to its room, so neighbouring tensors land on different
    volumes and each volume fills in proportion to its room."""
    parts, used = [[] for _ in room], [0] * len(room)
    for t in ts:
        size = t[5] - t[4]
        v = min(range(len(room)), key=lambda i: ((used[i] + size) / room[i], i))
        parts[v].append(t)
        used[v] += size
    return parts


def header_bytes(part: list) -> bytes:
    head, off = {"__metadata__": {"format": "pt"}}, 0
    for name, _, dtype, shape, b, e in part:
        head[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + e - b]}
        off += e - b
    raw = json.dumps(head, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)  # the data starts 8-byte aligned
    return struct.pack("<Q", len(raw)) + raw


def write_part(part: list, dest: str) -> str:
    """Write one striped file; returns the sha256 of the bytes it was built to hold. Writes
    are flushed and dropped from the page cache as they go (an exempt-size pod's memory
    counts its dirty pages)."""
    h = hashlib.sha256()
    tmp = dest + ".part"
    with open(tmp, "wb") as fo:
        hb = header_bytes(part)
        fo.write(hb)
        h.update(hb)
        for _, path, _, _, b, e in part:
            with open(path, "rb") as fi:
                fi.seek(b)
                left = e - b
                while left:
                    buf = fi.read(min(CHUNK, left))
                    if not buf:
                        raise SystemExit(f"[preload] {path} ends before byte {e}")
                    fo.write(buf)
                    h.update(buf)
                    left -= len(buf)
                os.posix_fadvise(fi.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            fo.flush()
            os.fsync(fo.fileno())
            os.posix_fadvise(fo.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    os.replace(tmp, dest)
    return h.hexdigest()


def copy_small(src: str, dest: str) -> str:
    tmp = dest + ".part"
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        fo.write(fi.read())
        fo.flush()
        os.fsync(fo.fileno())
    os.replace(tmp, dest)
    return sha256(src)


def build(src: dict, vols: list, key: str) -> int:
    """Stripe the copy whose files are ``src`` ({name: path}) onto ``vols``."""
    t0 = time.time()
    weights = sorted(p for n, p in src.items() if n.endswith(".safetensors"))
    ts = tensors(weights)
    room = []
    for v in vols:
        st = os.statvfs(v)
        room.append(max(1, st.f_bavail * st.f_frsize - RESERVE))
    parts = stripe(ts, room)
    need = [sum(e - b for *_, b, e in p) for p in parts]
    for v, n, r in zip(vols, need, room):
        if n > r:
            raise SystemExit(f"[preload] {v} has no room for its {n} bytes of {key}")
    rec, wmap = {}, {}
    for v, (vol, part) in enumerate(zip(vols, parts)):
        d = os.path.join(vol, DIR, key)
        os.makedirs(d, exist_ok=True)
        name = f"model-s{v + 1:02d}-of-{len(vols):02d}.safetensors"
        want = write_part(part, os.path.join(d, name))
        got = sha256(os.path.join(d, name))
        if want != got:
            raise SystemExit(f"[preload] {name} on {vol} reads back {got[:12]}, not {want[:12]}")
        rec[name] = {"vol": vol, "bytes": os.path.getsize(os.path.join(d, name)),
                     "sha256": want, "tensors": len(part)}
        wmap.update({t[0]: name for t in part})
    d0 = os.path.join(vols[0], DIR, key)
    for n, p in sorted(src.items()):
        if n.endswith(".safetensors") or n == INDEX or n == READY:
            continue
        rec[n] = {"vol": vols[0], "bytes": os.path.getsize(p),
                  "sha256": copy_small(p, os.path.join(d0, n))}
    index = {"metadata": {"total_size": sum(need)}, "weight_map": dict(sorted(wmap.items()))}
    with open(os.path.join(d0, INDEX + ".part"), "w") as f:
        json.dump(index, f, indent=1)
    os.replace(os.path.join(d0, INDEX + ".part"), os.path.join(d0, INDEX))
    rec[INDEX] = {"vol": vols[0], "bytes": os.path.getsize(os.path.join(d0, INDEX)),
                  "sha256": sha256(os.path.join(d0, INDEX))}
    out = os.path.join(d0, READY)
    with open(out + ".part", "w") as f:
        json.dump({"key": key, "layout": "striped", "files": rec,
                   "written": time.strftime("%FT%TZ", time.gmtime())}, f, indent=1)
    os.replace(out + ".part", out)
    print(f"[preload] {key}: {len(ts)} tensors striped over {len(vols)} volumes, read back "
          f"and sha256-checked, in {time.time() - t0:.0f}s", flush=True)
    return 0


def ready_files(vols: list, key: str, layout: str = DIR):
    """{file: path} from a complete copy of ``key`` on ``vols`` in ``layout``, else None."""
    p = os.path.join(vols[0], layout, key, READY)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        rec = json.load(f)
    return {n: os.path.join(e["vol"], layout, key, n) for n, e in rec["files"].items()}


def link(vols: list, key: str, dest: str) -> int:
    """Link a complete copy into ``dest``, striped if there is one: 0, or 3 when there is
    none to link (the pod then builds its copy). Each file must be at its recorded size."""
    for layout in (DIR, OLD):
        files = ready_files(vols, key, layout)
        if files is None:
            continue
        with open(os.path.join(vols[0], layout, key, READY)) as f:
            rec = json.load(f)["files"]
        short = [n for n, p in files.items()
                 if not os.path.exists(p) or os.path.getsize(p) != rec[n]["bytes"]]
        if short:
            print(f"[preload] {key}'s {layout} copy on the set is short ({short[:3]})",
                  flush=True)
            continue
        os.makedirs(dest, exist_ok=True)
        for n, p in files.items():
            os.symlink(p, os.path.join(dest, n))
        print(f"[preload] linked {len(files)} files of {key} ({layout}) from the set",
              flush=True)
        return 0
    print(f"[preload] no fp16 copy of {key} on the set yet; building one", flush=True)
    return 3


def bench(vols: list, key: str, workers: int = 4) -> int:
    """Read the set's copy as the loader does (every tensor, in its order, on ``workers``
    threads) twice: through mmap, as safe_open reads it, and with one pread per tensor, as
    the harness's FAST_LOAD reads it. Prints each pass's MB/s; the page cache is dropped
    before each pass and as it goes."""
    import mmap
    from concurrent.futures import ThreadPoolExecutor

    for layout in (DIR, OLD):
        files = ready_files(vols, key, layout)
        if files is not None:
            break
    else:
        raise SystemExit(f"[bench] no copy of {key} on the set")
    weights = sorted(p for n, p in files.items() if n.endswith(".safetensors"))
    ts = tensors(weights)
    fds = {p: os.open(p, os.O_RDONLY) for p in weights}
    maps = {p: mmap.mmap(fds[p], 0, prot=mmap.PROT_READ) for p in weights}

    def by_mmap(t):
        b, e = t[4], t[5]
        n = len(bytes(maps[t[1]][b:e]))  # a copy out of the mapping: page faults
        maps[t[1]].madvise(mmap.MADV_DONTNEED, b - b % mmap.PAGESIZE, e - b + b % mmap.PAGESIZE)
        os.posix_fadvise(fds[t[1]], b, e - b, os.POSIX_FADV_DONTNEED)
        return n

    def by_pread(t):
        b, e = t[4], t[5]
        buf, done = bytearray(e - b), 0
        view = memoryview(buf)
        while done < e - b:
            done += os.preadv(fds[t[1]], [view[done:]], b + done)
        os.posix_fadvise(fds[t[1]], b, e - b, os.POSIX_FADV_DONTNEED)
        return done

    total = sum(t[5] - t[4] for t in ts)
    for name, fn in (("mmap", by_mmap), ("pread", by_pread)):
        for p in weights:
            os.posix_fadvise(fds[p], 0, 0, os.POSIX_FADV_DONTNEED)
        t0 = time.time()
        with ThreadPoolExecutor(workers) as ex:
            got = sum(ex.map(fn, ts))
        dt = time.time() - t0
        assert got == total, (got, total)
        print(f"[bench] {key} ({layout}, {len(weights)} files, {workers} threads) {name}: "
              f"{total / 1e9:.2f} GB in {dt:.0f}s = {total / dt / 1e6:.0f} MB/s", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="the verified local fp16 copy, to put on the set")
    ap.add_argument("--restripe", action="store_true",
                    help="stripe the set's whole-file copy (model16/) instead")
    ap.add_argument("--link", help="link the set's copy into this directory instead")
    ap.add_argument("--bench", action="store_true",
                    help="time reading the set's copy through mmap and by pread instead")
    ap.add_argument("--vols", required=True, help="the set's mount points, volume 0 first")
    ap.add_argument("--key", required=True)
    a = ap.parse_args(argv)
    vols = a.vols.split(",")
    if a.link:
        return link(vols, a.key, a.link)
    if a.bench:
        return bench(vols, a.key)
    if ready_files(vols, a.key) is not None:
        print(f"[preload] {a.key} already striped on the set", flush=True)
        return 0
    if a.restripe:
        src = ready_files(vols, a.key, OLD)
        if src is None:
            raise SystemExit(f"[preload] no whole-file copy of {a.key} on the set to stripe")
    elif a.src:
        src = {n: os.path.join(a.src, n) for n in os.listdir(a.src)}
    else:
        ap.error("--src, --restripe or --link")
    return build(src, vols, a.key)


if __name__ == "__main__":
    raise SystemExit(main())

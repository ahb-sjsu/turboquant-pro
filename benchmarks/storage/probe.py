"""Exercise a storage path with known I/O, and record what each observer reports.

The storage counterpart of ``benchmarks/fabric``. There the traffic was counted at
the source and ``tqp fabric`` was checked against it. Here the pod counts every
byte and operation it issues, and three observers report on the same I/O:

- the application (this script): bytes and operations issued, wall time;
- the kernel's accounting for this process (``/proc/self/io``): ``read_bytes``
  and ``write_bytes`` are what reached the storage layer, so their ratio to the
  bytes issued shows page-cache hits, readahead and write amplification;
- the filesystem (``statvfs``): the used space it reports, sampled after each
  phase, which shows whether (and how late) written or deleted data is counted;

plus, from outside the pod, the cluster's CPU and memory metrics for it, which
``run_storage.sh`` samples on Atlas and compares with the CPU seconds this
process accounts for itself (the numbers NRP's utilisation enforcement uses).

Phases, each bounded by bytes AND by seconds, whichever comes first, and none of
them waiting on a timer (every wait is I/O):

1. ``seq_write``  write up to --write-mib in 4 MiB blocks, fsync every 64 MiB and
                  drop the written pages from the cache (so memory stays flat);
2. ``fsync``      N appends of 4 KiB, each followed by fdatasync (durability latency);
3. ``seq_read``   drop the file from the page cache, then read it back in 4 MiB blocks;
4. ``rand_read``  drop it again, then N reads of 4 KiB at random aligned offsets
                  (the shape of a rerank gather);
5. ``metadata``   create, stat and unlink N small files;
6. ``cleanup``    delete the file; used space is sampled once more.

Before anything is written the probe checks that --path is a mount of the
expected filesystem family, so a job whose volume did not attach fails instead of
measuring the container's own disk.

    python probe.py --path /mnt/probe --expect ceph --label rook-cephfs
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
import time

MIB = 2**20
BLOCK = 4 * MIB
SMALL = 4096


def _proc_io() -> dict:
    out = {}
    try:
        with open("/proc/self/io") as f:
            for line in f:
                k, v = line.split(":")
                out[k.strip()] = int(v)
    except OSError:
        pass
    return out


def _cpu_s() -> float | None:
    try:
        with open("/proc/self/stat") as f:
            s = f.read().rpartition(") ")[2].split()
        return (int(s[11]) + int(s[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def _rss_peak_mib() -> float | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def _statvfs(path) -> dict:
    s = os.statvfs(path)
    return {
        "t": time.time(),
        "size_bytes": s.f_blocks * s.f_frsize,
        "used_bytes": (s.f_blocks - s.f_bfree) * s.f_frsize,
        "free_bytes": s.f_bavail * s.f_frsize,
    }


def mount_of(path: str) -> dict | None:
    """The mount entry that holds ``path`` (longest matching mount point)."""
    real, best = os.path.realpath(path), None
    try:
        with open("/proc/self/mountinfo") as f:
            for line in f:
                left, _, right = line.partition(" - ")
                mp = left.split()[4]
                fstype, source = right.split()[:2]
                if real == mp or real.startswith(mp.rstrip("/") + "/"):
                    if best is None or len(mp) > len(best["mount_point"]):
                        best = {"mount_point": mp, "fstype": fstype, "source": source}
    except OSError:
        return None
    return best


def _pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, round(q / 100 * (len(s) - 1)))]


class Phase:
    """Wall time, CPU seconds and /proc/self/io deltas over a block of work."""

    def __init__(self, name, path):
        self.name, self.path = name, path

    def __enter__(self):
        self.t0, self.c0, self.io0 = time.perf_counter(), _cpu_s(), _proc_io()
        return self

    def __exit__(self, *exc):
        self.wall = time.perf_counter() - self.t0
        c1, io1 = _cpu_s(), _proc_io()
        self.cores = (
            (c1 - self.c0) / self.wall
            if c1 is not None and self.c0 is not None and self.wall > 0
            else None
        )
        self.io = {k: io1.get(k, 0) - self.io0.get(k, 0) for k in io1}
        self.fs_after = _statvfs(self.path)

    def record(self, **app) -> dict:
        return {
            "phase": self.name,
            "wall_s": self.wall,
            "cores": self.cores,
            "app": app,
            "kernel_io": self.io,
            "fs_after": self.fs_after,
        }


def _drop(fd, size):
    os.fsync(fd)
    os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)


def run(args) -> dict:
    path = args.path
    mnt = mount_of(path)
    ok = (
        mnt is not None
        and mnt["mount_point"] != "/"
        and any(e in mnt["fstype"] for e in args.expect.split(","))
    )
    doc = {
        "schema": "turboquant-pro/storage-probe",
        "schema_version": 1,
        "label": args.label,
        "host": socket.gethostname(),
        "node": os.environ.get("NODE_NAME"),
        "path": path,
        "mount": mnt,
        "expect": args.expect,
        "limits": {
            "write_mib": args.write_mib,
            "seconds": args.seconds,
            "fsync_n": args.fsync_n,
            "rand_n": args.rand_n,
            "meta_n": args.meta_n,
        },
        "t_start": time.time(),
        "fs_before": _statvfs(path),
        "phases": [],
    }
    if not ok:
        doc["error"] = (
            f"{path} is not a mount of the expected filesystem ({args.expect}); "
            f"found {mnt}. Refusing to measure the container's own disk."
        )
        return doc

    rnd = random.Random(args.seed)
    buf = bytearray(os.urandom(BLOCK))
    fn = os.path.join(path, f"probe-{os.getpid()}.bin")

    # 1. sequential write
    limit, deadline = args.write_mib * MIB, time.perf_counter() + args.seconds
    written = fsyncs = 0
    with Phase("seq_write", path) as ph:
        fd = os.open(fn, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            while written < limit and time.perf_counter() < deadline:
                buf[:8] = written.to_bytes(8, "little")  # every block distinct
                written += os.write(fd, buf)
                if written % (64 * MIB) == 0:
                    _drop(fd, written)
                    fsyncs += 1
            _drop(fd, written)
            fsyncs += 1
        finally:
            os.close(fd)
    doc["phases"].append(
        ph.record(bytes=written, fsyncs=fsyncs, mb_s=written / ph.wall / 1e6)
    )

    # 2. fsync latency
    lat = []
    with Phase("fsync", path) as ph:
        fd = os.open(fn, os.O_WRONLY | os.O_APPEND)
        try:
            small = os.urandom(SMALL)
            for _ in range(args.fsync_n):
                t = time.perf_counter()
                os.write(fd, small)
                os.fdatasync(fd)
                lat.append((time.perf_counter() - t) * 1e3)
                if time.perf_counter() > deadline + args.seconds:
                    break
        finally:
            os.close(fd)
    doc["phases"].append(
        ph.record(
            ops=len(lat),
            bytes=len(lat) * SMALL,
            ms_p50=_pct(lat, 50),
            ms_p99=_pct(lat, 99),
            ms_max=max(lat) if lat else None,
        )
    )
    size = written + len(lat) * SMALL

    # 3. sequential read, cold
    fd = os.open(fn, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        got, deadline = 0, time.perf_counter() + args.seconds
        with Phase("seq_read", path) as ph:
            while got < size and time.perf_counter() < deadline:
                chunk = os.pread(fd, BLOCK, got)
                if not chunk:
                    break
                got += len(chunk)
        doc["phases"].append(ph.record(bytes=got, mb_s=got / ph.wall / 1e6))

        # 4. random 4 KiB reads, cold
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        n_blocks = max(1, written // SMALL)
        lat, got, deadline = [], 0, time.perf_counter() + args.seconds
        with Phase("rand_read", path) as ph:
            for _ in range(args.rand_n):
                off = rnd.randrange(n_blocks) * SMALL
                t = time.perf_counter()
                got += len(os.pread(fd, SMALL, off))
                lat.append((time.perf_counter() - t) * 1e3)
                if time.perf_counter() > deadline:
                    break
        doc["phases"].append(
            ph.record(
                ops=len(lat),
                bytes=got,
                iops=len(lat) / ph.wall,
                ms_p50=_pct(lat, 50),
                ms_p99=_pct(lat, 99),
            )
        )
    finally:
        os.close(fd)

    # 5. metadata
    d = os.path.join(path, f"meta-{os.getpid()}")
    os.mkdir(d)
    names = [os.path.join(d, f"f{i:05d}") for i in range(args.meta_n)]
    small = os.urandom(1024)
    rates = {}
    with Phase("metadata", path) as ph:
        t = time.perf_counter()
        for n in names:
            with open(n, "wb") as f:
                f.write(small)
        rates["create_per_s"] = len(names) / (time.perf_counter() - t)
        t = time.perf_counter()
        for n in names:
            os.stat(n)
        rates["stat_per_s"] = len(names) / (time.perf_counter() - t)
        t = time.perf_counter()
        for n in names:
            os.unlink(n)
        rates["unlink_per_s"] = len(names) / (time.perf_counter() - t)
        os.rmdir(d)
    doc["phases"].append(ph.record(files=len(names), **rates))

    # 6. cleanup
    with Phase("cleanup", path) as ph:
        os.unlink(fn)
    doc["phases"].append(ph.record(bytes_deleted=size))

    doc["t_end"] = time.time()
    doc["usage"] = {"cpu_s": _cpu_s(), "peak_rss_mib": _rss_peak_mib()}
    return doc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--path", default=os.environ.get("PROBE_PATH", "/mnt/probe"))
    p.add_argument(
        "--expect",
        default=os.environ.get("PROBE_EXPECT", "ceph,ext4,xfs"),
        help="comma-separated fstype substrings the mount must match",
    )
    p.add_argument("--label", default=os.environ.get("PROBE_LABEL", "unlabelled"))
    p.add_argument(
        "--write-mib", type=int, default=int(os.environ.get("PROBE_MIB", 1024))
    )
    p.add_argument(
        "--seconds", type=float, default=float(os.environ.get("PROBE_S", 90))
    )
    p.add_argument("--fsync-n", type=int, default=200)
    p.add_argument("--rand-n", type=int, default=2000)
    p.add_argument("--meta-n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    doc = run(args)
    print(json.dumps(doc), flush=True)
    return 1 if "error" in doc else 0


if __name__ == "__main__":
    sys.exit(main())

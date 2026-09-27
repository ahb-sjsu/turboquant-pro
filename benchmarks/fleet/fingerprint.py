"""Where a fleet job ran, and what that place was like: one line per job.

The 1T measurement recorded 500 reference-scan wall times (1,062 to 16,308 s
for the same scan) but not which node ran each job, so that spread could not be
attributed after the fact. This module is the record every fleet job should
print. It is cheap (about 8 s), standard library only, and independent of the
scan:

- ``node``: the node name (from ``NODE_NAME`` when the Job sets it through the
  downward API, else unknown to the pod; the driver records it from the API),
  the kernel's hostname, and the CPU model and count the pod sees;
- ``cpu_speed``: a fixed single-core benchmark, SHA-256 over a 64 KiB buffer
  for ``bench_s`` seconds, as hashes per second. Measured across NRP nodes on
  2026-09-27 it ranged from 5,839 to 26,909, a 4.6x spread in the speed of
  the one core an exempt-class job gets;
- ``r_blk``: the median cold latency of one 4 KiB read at offset 0 of up to 50
  distinct files under the data path (page cache dropped first), the single
  coordinate that predicted random-read latency (Pearson 0.9995) and ordered
  sequential throughput (Spearman -1.0) for the 1T volumes on 2026-09-27;
- ``phases``: wall-clock marks a job adds as it goes (``mark("scan_start")``),
  plus the process CPU seconds and bytes read at each mark, so wall time can be
  split into fixed cost, CPU and I/O waiting.

Every value is what this pod measured on this node at this time. The benchmark
number is a relative speed, not a clock rate.

    from fingerprint import Fingerprint
    fp = Fingerprint(data_path="/idx")      # measures node, cpu_speed, r_blk
    fp.mark("scan_start"); ...; fp.mark("scan_end")
    fp.emit()                               # prints FINGERPRINT {...}
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import time

SMALL = 4096


def _cpu_s() -> float | None:
    try:
        with open("/proc/self/stat") as f:
            s = f.read().rpartition(") ")[2].split()
        return (int(s[11]) + int(s[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def _read_bytes() -> int | None:
    try:
        with open("/proc/self/io") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def cpu_model() -> dict:
    model, count = None, 0
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name") and model is None:
                    model = line.split(":", 1)[1].strip()
                if line.startswith("processor"):
                    count += 1
    except OSError:
        pass
    return {"model": model, "host_cpus": count or os.cpu_count()}


def cpu_speed(seconds: float = 5.0) -> dict:
    """Hashes per second on one core; relative, comparable across nodes."""
    data, n = os.urandom(1 << 16), 0
    t0 = time.perf_counter()
    c0 = _cpu_s()
    while time.perf_counter() - t0 < seconds:
        hashlib.sha256(data).digest()
        n += 1
    wall = time.perf_counter() - t0
    c1 = _cpu_s()
    return {
        "hashes_per_s": n / wall,
        "bench_s": wall,
        # below ~0.9 the core was shared or throttled while measuring
        "cores_during": (c1 - c0) / wall if c0 is not None and c1 is not None else None,
    }


def r_blk(data_path: str, n: int = 50, seed: int = 0) -> dict:
    """Median cold latency of one 4 KiB read at offset 0 of distinct files."""
    files = []
    for root, _, names in os.walk(data_path):
        for name in names:
            fp = os.path.join(root, name)
            try:
                if os.stat(fp).st_size >= SMALL:
                    files.append(fp)
            except OSError:
                continue
        if len(files) >= 20 * n:
            break
    if not files:
        return {"ms_p50": None, "n": 0, "reason": f"no files under {data_path}"}
    pick = random.Random(seed).sample(sorted(files), min(n, len(files)))
    lat = []
    for fp in pick:
        fd = os.open(fp, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            t = time.perf_counter()
            os.pread(fd, SMALL, 0)
            lat.append((time.perf_counter() - t) * 1e3)
        finally:
            os.close(fd)
    lat.sort()
    return {
        "ms_p50": lat[len(lat) // 2],
        "ms_p90": lat[min(len(lat) - 1, int(0.9 * len(lat)))],
        "n": len(lat),
    }


class Fingerprint:
    def __init__(self, data_path: str | None = None, bench_s: float = 5.0):
        self.t_created = time.time()
        self.phases: list = []
        self.mark("fingerprint_start")
        self.doc = {
            "schema": "turboquant-pro/fleet-fingerprint",
            "schema_version": 1,
            "node": os.environ.get("NODE_NAME"),
            "hostname": socket.gethostname(),
            "cpu": cpu_model(),
            "cpu_speed": cpu_speed(bench_s),
            "r_blk": r_blk(data_path) if data_path else None,
        }
        self.mark("fingerprint_end")

    def mark(self, name: str, **extra) -> None:
        self.phases.append(
            {
                "name": name,
                "t": time.time(),
                "cpu_s": _cpu_s(),
                "read_bytes": _read_bytes(),
                **extra,
            }
        )

    def emit(self, **result) -> dict:
        doc = {**self.doc, "phases": self.phases, **result}
        print("FINGERPRINT " + json.dumps(doc), flush=True)
        return doc

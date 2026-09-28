# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""The machine the console runs on, read from ``/proc`` and ``/sys``.

:class:`MachineMonitor` reads the kernel's own accounting and nothing else: CPU
time per logical CPU (``/proc/stat``), memory (``/proc/meminfo``,
``/proc/vmstat``), block devices (``/proc/diskstats``), network interfaces
(``/proc/net/dev``), temperature sensors (``/sys/class/hwmon``) and, when the
NVIDIA management library is installed, the GPUs (NVML, through ctypes). It needs
no package, starts no process and writes nothing.

Each poll is one ``turboquant-pro/machine-snapshot`` document. Kernel counters are
**measured**. Every per-second rate and every busy fraction is **derived** from
the change between two polls over the time between them, and is ``None`` on the
first poll and after a counter went backwards (a device or interface that
reappeared, a counter that wrapped). Temperatures and GPU readings are
**sampled**. A limit a sensor reports outside 0 to 150 degrees C (some NVMe
sensors report 65261.85) is treated as absent, not believed.

    mon = MachineMonitor()
    doc = mon.poll()           # counters and levels, no rates yet
    time.sleep(2)
    doc = mon.poll()           # rates over those 2 s

``root`` points the monitor at another tree laid out like ``/`` (the tests use
fixtures); ``gpu`` replaces the NVML reader.
"""

from __future__ import annotations

import ctypes
import os
import time
from collections import deque

SCHEMA = "turboquant-pro/machine-snapshot"
SCHEMA_VERSION = 1
SECTOR_BYTES = 512  # /proc/diskstats counts 512-byte sectors on every device
try:
    _PAGE = os.sysconf("SC_PAGE_SIZE")
except (AttributeError, ValueError, OSError):  # not a POSIX system
    _PAGE = 4096
PLAUSIBLE_C = (0.0, 150.0)  # a temperature limit outside this is not a limit
IFF_TUN = 0x0001  # linux/if_tun.h
ARPHRD_NONE = 65534  # linux/if_arp.h: no link-layer header

__all__ = [
    "MachineMonitor",
    "History",
    "SCHEMA",
    "parse_stat",
    "parse_meminfo",
    "parse_vmstat",
    "parse_diskstats",
    "parse_netdev",
]

# /proc/stat CPU fields, in order. guest and guest_nice are already inside user
# and nice, so they are not added again.
CPU_FIELDS = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")


# ------------------------------------------------------------------ parsers
def parse_stat(text: str) -> dict:
    """``/proc/stat``: {"cpu": {name: (user, ..., steal)}, "ctxt": n, ...}.
    ``cpu`` is the machine total; ``cpu0`` ... each logical CPU."""
    cpus, other = {}, {}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0].startswith("cpu"):
            vals = [int(v) for v in parts[1 : 1 + len(CPU_FIELDS)]]
            vals += [0] * (len(CPU_FIELDS) - len(vals))  # older kernels: fewer
            cpus[parts[0]] = tuple(vals)
        elif parts[0] in ("ctxt", "processes", "procs_running", "procs_blocked"):
            other[parts[0]] = int(parts[1])
    return {"cpu": cpus, **other}


def parse_meminfo(text: str) -> dict:
    """``/proc/meminfo`` in bytes (the file says kB and means KiB)."""
    out = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if not parts:
            continue
        scale = 1024 if len(parts) > 1 and parts[1] == "kB" else 1
        out[key.strip()] = int(parts[0]) * scale
    return out


def parse_vmstat(text: str) -> dict:
    """``/proc/vmstat``: name -> counter."""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[0]] = int(parts[1])
    return out


def parse_diskstats(text: str) -> dict:
    """``/proc/diskstats``: name -> the counters this monitor uses.

    Fields after major, minor and name (kernel Documentation/admin-guide/
    iostats.rst): reads, reads merged, sectors read, ms reading, writes, writes
    merged, sectors written, ms writing, I/Os in flight, ms doing I/O, weighted
    ms; newer kernels append discard and flush counters, not used here."""
    out = {}
    for line in text.splitlines():
        p = line.split()
        if len(p) < 14:
            continue
        v = [int(x) for x in p[3:14]]
        out[p[2]] = {
            "reads": v[0],
            "read_sectors": v[2],
            "read_ms": v[3],
            "writes": v[4],
            "write_sectors": v[6],
            "write_ms": v[7],
            "in_flight": v[8],
            "io_ms": v[9],
        }
    return out


DISK_COUNTERS = (
    "reads",
    "read_sectors",
    "read_ms",
    "writes",
    "write_sectors",
    "write_ms",
    "io_ms",
)
NET_FIELDS = ("rx_bytes", "rx_packets", "rx_errs", "rx_drop")
NET_TX = ("tx_bytes", "tx_packets", "tx_errs", "tx_drop")


def parse_netdev(text: str) -> dict:
    """``/proc/net/dev``: interface -> rx/tx bytes, packets, errors, drops."""
    out = {}
    for line in text.splitlines()[2:]:
        name, sep, rest = line.partition(":")
        if not sep:
            continue
        v = [int(x) for x in rest.split()]
        if len(v) < 16:
            continue
        out[name.strip()] = dict(zip(NET_FIELDS + NET_TX, v[0:4] + v[8:12]))
    return out


# ------------------------------------------------------------------ helpers
def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _int(path: str) -> int | None:
    t = _read(path)
    try:
        return int(t.strip()) if t is not None else None
    except ValueError:
        return None


def _limit(milli: int | None) -> float | None:
    """A sensor limit in degrees C, or None when absent or implausible."""
    if milli is None:
        return None
    c = milli / 1000.0
    return c if PLAUSIBLE_C[0] < c <= PLAUSIBLE_C[1] else None


def _rate(new, old, dt):
    """(new - old) / dt, or None: no earlier value, no time, or a counter that
    went backwards."""
    if old is None or new is None or dt is None or dt <= 0 or new < old:
        return None
    return (new - old) / dt


def _deltas(new: dict, old: dict | None, counters) -> dict | None:
    """The change in each of ``counters`` between two polls, or None when there
    is no earlier poll or any of them went backwards (a reset: nothing about the
    interval is known). Levels, which may fall, are not counters."""
    if not old or any(k not in old for k in counters):
        return None
    d = {k: new[k] - old[k] for k in counters}
    return d if all(v >= 0 for v in d.values()) else None


def _cpu_share(new: list, old: list) -> dict | None:
    """Busy, iowait and steal fractions of the CPU time several logical CPUs
    spent between two polls: their summed time, not a mean of their fractions."""
    if not new or len(new) != len(old) or any(o is None for o in old):
        return None
    d = [sum(a[i] - b[i] for a, b in zip(new, old)) for i in range(len(CPU_FIELDS))]
    total = sum(d)
    if total <= 0 or any(x < 0 for x in d):
        return None
    f = dict(zip(CPU_FIELDS, d))
    return {
        "busy": (total - f["idle"] - f["iowait"]) / total,
        "iowait": f["iowait"] / total,
        "steal": f["steal"] / total,
    }


def _is_disk(name: str) -> bool:
    """A block device worth a row: not a loop, RAM, optical or floppy device."""
    return not name.startswith(("loop", "ram", "sr", "fd", "zram"))


# ------------------------------------------------------------------ NVML
class _NvUtil(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _NvMem(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _NvMem2(ctypes.Structure):
    """nvmlMemory_v2_t: ``used`` excludes the memory the driver reserves, which
    the v1 call counts as used (seen on Atlas: v1 read about 280 MiB above what
    nvidia-smi shows, on each GPU). nvidia-smi reports the v2 figure."""

    _fields_ = [
        ("version", ctypes.c_uint),
        ("total", ctypes.c_ulonglong),
        ("reserved", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


# NVML_STRUCT_VERSION(Memory, 2): the struct's size with the version in the top byte
_NV_MEM2_VERSION = ctypes.sizeof(_NvMem2) | (2 << 24)


class NVMLReader:
    """The GPUs through NVML (``libnvidia-ml.so.1``), read-only. ``read()`` is a
    list of GPU dicts, or raises OSError with the reason it cannot read."""

    NVML_TEMPERATURE_GPU = 0
    NVML_TEMPERATURE_THRESHOLD_SLOWDOWN = 1

    def __init__(self, library: str = "libnvidia-ml.so.1"):
        self._lib = None
        self._handles: list = []
        self.reason: str | None = None
        try:
            lib = ctypes.CDLL(library)
        except OSError:
            self.reason = f"unavailable: no NVML ({library} not found)"
            return
        if lib.nvmlInit_v2() != 0:
            self.reason = "unavailable: NVML did not initialise (no driver?)"
            return
        n = ctypes.c_uint()
        if lib.nvmlDeviceGetCount_v2(ctypes.byref(n)) != 0:
            self.reason = "unavailable: NVML could not count devices"
            return
        for i in range(n.value):
            h = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) == 0:
                self._handles.append(h)
        self._lib = lib
        if not self._handles:
            self.reason = "no NVIDIA GPU"

    def read(self) -> list:
        if self._lib is None or not self._handles:
            raise OSError(self.reason or "unavailable")
        lib, out = self._lib, []
        for i, h in enumerate(self._handles):
            g: dict = {"index": i}
            name = ctypes.create_string_buffer(96)
            if lib.nvmlDeviceGetName(h, name, 96) == 0:
                g["name"] = name.value.decode(errors="replace")
            u = _NvUtil()
            if lib.nvmlDeviceGetUtilizationRates(h, ctypes.byref(u)) == 0:
                g["util"], g["mem_util"] = u.gpu / 100.0, u.memory / 100.0
            m2 = _NvMem2(version=_NV_MEM2_VERSION)
            get2 = getattr(lib, "nvmlDeviceGetMemoryInfo_v2", None)
            if get2 is not None and get2(h, ctypes.byref(m2)) == 0:
                g["mem_used"], g["mem_total"] = m2.used, m2.total
                g["mem_reserved"] = m2.reserved
            else:  # an older driver: v1, whose "used" includes the reserved part
                m = _NvMem()
                if lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(m)) == 0:
                    g["mem_used"], g["mem_total"] = m.used, m.total
            t = ctypes.c_uint()
            if (
                lib.nvmlDeviceGetTemperature(
                    h, self.NVML_TEMPERATURE_GPU, ctypes.byref(t)
                )
                == 0
            ):
                g["temp_c"] = float(t.value)
            if (
                lib.nvmlDeviceGetTemperatureThreshold(
                    h, self.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN, ctypes.byref(t)
                )
                == 0
            ):
                g["temp_limit_c"] = _limit(t.value * 1000)
            mw = ctypes.c_uint()
            if lib.nvmlDeviceGetPowerUsage(h, ctypes.byref(mw)) == 0:
                g["power_w"] = mw.value / 1000.0
            if lib.nvmlDeviceGetEnforcedPowerLimit(h, ctypes.byref(mw)) == 0:
                g["power_limit_w"] = mw.value / 1000.0
            out.append(g)
        return out


# ------------------------------------------------------------------ monitor
class MachineMonitor:
    """Polls the machine; each :meth:`poll` returns one snapshot document."""

    def __init__(self, root: str = "/", clock=time.monotonic, gpu=None):
        self.root = root
        self._clock = clock
        self._gpu = gpu if gpu is not None else NVMLReader()
        self._prev: dict | None = None
        self._prev_t: float | None = None
        self._packages = self._cpu_packages()

    def _p(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    def _cpu_packages(self) -> dict:
        """Logical CPU name -> physical package id (0 when the kernel does not say)."""
        out = {}
        base = self._p("sys", "devices", "system", "cpu")
        try:
            names = os.listdir(base)
        except OSError:
            return out
        for n in names:
            if n.startswith("cpu") and n[3:].isdigit():
                pkg = _int(os.path.join(base, n, "topology", "physical_package_id"))
                out[n] = pkg if pkg is not None else 0
        return out

    # ---- readers, each tolerant of a missing file (reported, not guessed)
    def _thermal(self) -> list:
        base = self._p("sys", "class", "hwmon")
        try:
            chips = sorted(os.listdir(base), key=lambda s: (len(s), s))
        except OSError:
            return []
        out = []
        for chip in chips:
            d = os.path.join(base, chip)
            name = (_read(os.path.join(d, "name")) or chip).strip()
            try:
                files = os.listdir(d)
            except OSError:
                continue
            sensors = []
            for f in files:
                if not (f.startswith("temp") and f.endswith("_input")):
                    continue
                stem = f[: -len("_input")]
                milli = _int(os.path.join(d, f))
                if milli is None:
                    continue
                c = milli / 1000.0
                if not -40.0 <= c <= PLAUSIBLE_C[1]:
                    continue
                label = (_read(os.path.join(d, stem + "_label")) or stem).strip()
                sensors.append(
                    {
                        "label": label,
                        "c": c,
                        "high_c": _limit(_int(os.path.join(d, stem + "_max"))),
                        "crit_c": _limit(_int(os.path.join(d, stem + "_crit"))),
                        "n": int(stem[4:]) if stem[4:].isdigit() else 0,
                    }
                )
            if sensors:
                sensors.sort(key=lambda s: s["n"])
                out.append({"chip": chip, "name": name, "sensors": sensors})
        return out

    def _nics(self, names) -> dict:
        """Interface -> {"class": physical | tunnel | virtual, "speed_mbps", "up"}."""
        out = {}
        for n in names:
            d = self._p("sys", "class", "net", n)
            # a tunnel carries packets with no link layer: a TUN device (IFF_TUN
            # in tun_flags; a TAP, IFF_TAP, is an Ethernet port such as a VM's)
            # or any interface of type ARPHRD_NONE (65534), as WireGuard is
            tun_flags = _read(os.path.join(d, "tun_flags"))
            try:
                tun = tun_flags is not None and int(tun_flags, 16) & IFF_TUN
            except ValueError:
                tun = False
            if os.path.exists(os.path.join(d, "device")):
                cls = "physical"
            elif tun or _int(os.path.join(d, "type")) == ARPHRD_NONE:
                cls = "tunnel"
            else:
                cls = "virtual"
            speed = _int(os.path.join(d, "speed"))
            state = (_read(os.path.join(d, "operstate")) or "").strip()
            out[n] = {
                "class": cls,
                "speed_mbps": speed if speed and speed > 0 else None,
                "up": state in ("up", "unknown"),
            }
        return out

    def poll(self) -> dict:
        now = self._clock()
        dt = None if self._prev_t is None else now - self._prev_t
        prev = self._prev or {}
        raw = {
            "stat": parse_stat(_read(self._p("proc", "stat")) or ""),
            "mem": parse_meminfo(_read(self._p("proc", "meminfo")) or ""),
            "vm": parse_vmstat(_read(self._p("proc", "vmstat")) or ""),
            "disk": parse_diskstats(_read(self._p("proc", "diskstats")) or ""),
            "net": parse_netdev(_read(self._p("proc", "net", "dev")) or ""),
        }
        doc = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "t": time.time(),
            "interval_s": dt,
            "cpu": self._cpu_doc(raw["stat"], prev.get("stat"), dt),
            "load": self._load(),
            "memory": self._mem_doc(raw["mem"], raw["vm"], prev.get("vm"), dt),
            "disks": self._disk_doc(raw["disk"], prev.get("disk"), dt),
            "net": self._net_doc(raw["net"], prev.get("net"), dt),
            "thermal": self._thermal(),
        }
        try:
            doc["gpus"], doc["gpu_reason"] = self._gpu.read(), None
        except OSError as e:
            doc["gpus"], doc["gpu_reason"] = [], str(e)
        self._prev, self._prev_t = raw, now
        return doc

    def _load(self):
        t = _read(self._p("proc", "loadavg"))
        try:
            return [float(x) for x in t.split()[:3]] if t else None
        except ValueError:
            return None

    def _cpu_doc(self, stat: dict, old: dict | None, dt) -> dict:
        cur, before = stat.get("cpu", {}), (old or {}).get("cpu", {})

        def share(names):
            return _cpu_share([cur[n] for n in names], [before.get(n) for n in names])

        logical = sorted((n for n in cur if n != "cpu"), key=lambda s: int(s[3:]))
        packages: dict = {}
        for n in logical:
            packages.setdefault(self._packages.get(n, 0), []).append(n)
        return {
            "total": share(["cpu"]) if "cpu" in cur else None,
            "logical": [{"cpu": n, "share": share([n])} for n in logical],
            "packages": [
                {"package": p, "cpus": names, "share": share(names)}
                for p, names in sorted(packages.items())
            ],
            "procs_running": stat.get("procs_running"),
            "procs_blocked": stat.get("procs_blocked"),
        }

    def _mem_doc(self, m: dict, vm: dict, old_vm: dict | None, dt) -> dict:
        old_vm = old_vm or {}
        total, avail = m.get("MemTotal"), m.get("MemAvailable")
        swap_t, swap_f = m.get("SwapTotal"), m.get("SwapFree")
        page = _PAGE  # pswpin / pswpout count pages
        return {
            "total": total,
            "available": avail,
            "used": None if total is None or avail is None else total - avail,
            "cached": (m.get("Cached") or 0) + (m.get("SReclaimable") or 0),
            "buffers": m.get("Buffers"),
            "dirty": m.get("Dirty"),
            "swap_total": swap_t,
            "swap_used": None if swap_t is None or swap_f is None else swap_t - swap_f,
            "major_faults_per_s": _rate(
                vm.get("pgmajfault"), old_vm.get("pgmajfault"), dt
            ),
            "swap_in_Bps": _scaled(
                _rate(vm.get("pswpin"), old_vm.get("pswpin"), dt), page
            ),
            "swap_out_Bps": _scaled(
                _rate(vm.get("pswpout"), old_vm.get("pswpout"), dt), page
            ),
        }

    def _disk_doc(self, cur: dict, old: dict | None, dt) -> list:
        old = old or {}
        out = []
        for name in sorted(cur):
            if not _is_disk(name) or not os.path.isdir(self._p("sys", "block", name)):
                continue  # partitions and pseudo devices are not rows
            c = cur[name]
            # a device with queue/iostats 0 keeps no statistics: its counters stay
            # 0 however busy it is, so every rate is unknown, not zero (seen on
            # Atlas: md127 keeps none, md1 and md5 do)
            keeps = _int(self._p("sys", "block", name, "queue", "iostats")) != 0
            d = None
            if keeps and dt and dt > 0:
                d = _deltas(c, old.get(name), DISK_COUNTERS)
            row = {"name": name, "in_flight": c["in_flight"]}  # measured, a level
            row["util_reason"] = (
                None if keeps else "keeps no I/O statistics (queue/iostats is 0)"
            )
            if d is None:
                row.update(read_Bps=None, write_Bps=None, read_iops=None)
                row.update(write_iops=None, await_ms=None, util=None)
            else:
                ios = d["reads"] + d["writes"]
                row.update(
                    read_Bps=d["read_sectors"] * SECTOR_BYTES / dt,
                    write_Bps=d["write_sectors"] * SECTOR_BYTES / dt,
                    read_iops=d["reads"] / dt,
                    write_iops=d["writes"] / dt,
                    # the mean time an I/O completed in this interval took
                    await_ms=(d["read_ms"] + d["write_ms"]) / ios if ios else None,
                    util=min(d["io_ms"] / (dt * 1000.0), 1.0),
                )
            out.append(row)
        return out

    def _net_doc(self, cur: dict, old: dict | None, dt) -> dict:
        old = old or {}
        names = sorted(n for n in cur if n != "lo")
        kind = self._nics(names)
        keys = NET_FIELDS + NET_TX
        rows, virtual = [], []
        for n in names:
            d = _deltas(cur[n], old.get(n), keys) if dt and dt > 0 else None
            rates = {k: None if d is None else d[k] / dt for k in keys}
            if kind[n]["class"] == "virtual":
                virtual.append(rates)
                continue
            link, speed = None, kind[n]["speed_mbps"]
            if speed and d is not None:
                link = max(rates["rx_bytes"], rates["tx_bytes"]) * 8 / (speed * 1e6)
            rows.append({"name": n, **kind[n], **rates, "link_util": link})
        # bridges, veth pairs and VM taps: one summed row, known only if all are
        summed = {
            k: (
                None
                if not virtual or any(r[k] is None for r in virtual)
                else sum(r[k] for r in virtual)
            )
            for k in keys
        }
        return {"interfaces": rows, "virtual": {"count": len(virtual), **summed}}


def _scaled(v, k):
    return None if v is None else v * k


# ------------------------------------------------------------------ history
class History:
    """The last ``n`` values of each series the machine panels draw."""

    def __init__(self, n: int = 256):
        self.n = n
        self.series: dict = {}

    def _add(self, key, v):
        self.series.setdefault(key, deque(maxlen=self.n)).append(v)

    def add(self, doc: dict) -> None:
        cpu = doc.get("cpu") or {}
        tot = cpu.get("total")
        self._add("cpu.busy", None if tot is None else tot["busy"])
        for p in cpu.get("packages") or []:
            s = p["share"]
            self._add(f"cpu.busy.{p['package']}", None if s is None else s["busy"])
        mem = doc.get("memory") or {}
        for k in ("used", "cached", "swap_used", "major_faults_per_s", "dirty"):
            self._add(f"mem.{k}", mem.get(k))
        for chip in doc.get("thermal") or []:
            for s in chip["sensors"]:
                self._add(f"temp.{chip['chip']}.{s['n']}", s["c"])
        for g in doc.get("gpus") or []:
            self._add(f"gpu.{g['index']}.util", g.get("util"))
            self._add(f"gpu.{g['index']}.temp", g.get("temp_c"))

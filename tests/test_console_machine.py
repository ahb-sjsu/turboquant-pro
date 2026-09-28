"""The machine source (console/machine.py) and its page (console/machine_view.py).

Every test runs against a fixture tree laid out like ``/`` (``/proc`` and
``/sys`` files written by hand, in the kernel's formats), with a fake clock, so
each derived number can be checked exactly: a rate is a counter change over the
time between two polls, and nothing else.
"""

from __future__ import annotations

import json
import os

import pytest

from turboquant_pro.cli import main
from turboquant_pro.console import machine_view as MV
from turboquant_pro.console import viewmodel as VM
from turboquant_pro.console.machine import (
    History,
    MachineMonitor,
    NVMLReader,
    parse_diskstats,
    parse_meminfo,
    parse_netdev,
    parse_stat,
)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class NoGPU:
    def read(self):
        raise OSError("unavailable: no NVML (test)")


class FakeGPU:
    def read(self):
        return [
            {"index": 0, "name": "Quadro GV100", "util": 0.5, "mem_util": 0.1,
             "mem_used": 2 * 2**30, "mem_total": 32 * 2**30, "temp_c": 86.0,
             "temp_limit_c": 88.0, "power_w": 40.7, "power_limit_w": 250.0},
        ]  # fmt: skip


def _w(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)


def _stat(cpus: dict) -> str:
    """cpus: name -> (user, nice, system, idle, iowait, irq, softirq, steal)."""
    total = [sum(v[i] for v in cpus.values()) for i in range(8)]
    lines = ["cpu  " + " ".join(map(str, total)) + " 0 0"]
    lines += [f"{n} " + " ".join(map(str, v)) + " 0 0" for n, v in cpus.items()]
    return "\n".join(lines + ["ctxt 1", "procs_running 3", "procs_blocked 1"]) + "\n"


NETDEV_HEAD = (
    "Inter-|   Receive                            |  Transmit\n"
    " face |bytes    packets errs drop fifo frame compressed multicast|bytes"
    "    packets errs drop fifo colls carrier compressed\n"
)


def _netdev(rows: dict) -> str:
    """rows: iface -> (rx_bytes, rx_packets, rx_errs, rx_drop, tx_bytes, tx_packets,
    tx_errs, tx_drop)."""
    out = NETDEV_HEAD
    for n, v in rows.items():
        rx, tx = v[:4], v[4:]
        out += (
            f"{n:>6}: " + " ".join(map(str, [*rx, 0, 0, 0, 0, *tx, 0, 0, 0, 0])) + "\n"
        )
    return out


def _disk(name, reads, rsect, rms, writes, wsect, wms, inflight, io_ms):
    return (
        f"   8       0 {name} {reads} 0 {rsect} {rms} {writes} 0 {wsect} {wms} "
        f"{inflight} {io_ms} {rms + wms} 0 0 0 0 0 0\n"
    )


@pytest.fixture
def tree(tmp_path):
    """A small machine: 2 packages x 2 CPUs, three disks and a loop device, a
    physical NIC, a tunnel, two virtual interfaces, two hwmon chips."""
    root = str(tmp_path)
    for i, pkg in enumerate([0, 0, 1, 1]):
        _w(
            root,
            f"sys/devices/system/cpu/cpu{i}/topology/physical_package_id",
            f"{pkg}\n",
        )
    for d in ("sda", "md1", "nvme0n1", "loop0"):
        os.makedirs(os.path.join(root, "sys/block", d), exist_ok=True)
    _w(root, "sys/block/md1/queue/iostats", "1\n")
    _w(root, "sys/block/nvme0n1/queue/iostats", "0\n")  # keeps no statistics
    for n in ("eno1", "tailscale0", "veth1", "docker0", "vnet7"):
        os.makedirs(os.path.join(root, "sys/class/net", n), exist_ok=True)
    os.makedirs(os.path.join(root, "sys/class/net/eno1/device"))
    _w(root, "sys/class/net/eno1/speed", "1000\n")
    _w(root, "sys/class/net/eno1/operstate", "up\n")
    # the values Atlas reports: a TUN device, and a VM's TAP (an Ethernet port)
    _w(root, "sys/class/net/tailscale0/tun_flags", "0x5001\n")
    _w(root, "sys/class/net/tailscale0/type", "65534\n")
    _w(root, "sys/class/net/vnet7/tun_flags", "0x5002\n")
    _w(root, "sys/class/net/vnet7/type", "1\n")
    _w(root, "sys/class/net/tailscale0/operstate", "unknown\n")
    # hwmon10 sorts after hwmon2 (numerically, not as text)
    _w(root, "sys/class/hwmon/hwmon2/name", "coretemp\n")
    for n, label, milli in (
        (1, "Package id 0", 71000),
        (2, "Core 0", 69000),
        (3, "Core 1", 74000),
    ):
        _w(root, f"sys/class/hwmon/hwmon2/temp{n}_input", f"{milli}\n")
        _w(root, f"sys/class/hwmon/hwmon2/temp{n}_label", label + "\n")
        _w(root, f"sys/class/hwmon/hwmon2/temp{n}_max", "82000\n")
        _w(root, f"sys/class/hwmon/hwmon2/temp{n}_crit", "100000\n")
    _w(root, "sys/class/hwmon/hwmon10/name", "nvme\n")
    _w(root, "sys/class/hwmon/hwmon10/temp1_input", "43850\n")
    _w(root, "sys/class/hwmon/hwmon10/temp1_label", "Composite\n")
    _w(root, "sys/class/hwmon/hwmon10/temp1_max", "84850\n")
    _w(root, "sys/class/hwmon/hwmon10/temp2_input", "42850\n")
    _w(root, "sys/class/hwmon/hwmon10/temp2_max", "65261850\n")  # as Atlas reports
    _w(root, "proc/loadavg", "1.50 1.25 1.00 2/900 12345\n")
    return root


def _poll_state(root, cpus, mem, vm, disks, net):
    _w(root, "proc/stat", _stat(cpus))
    _w(root, "proc/meminfo", mem)
    _w(root, "proc/vmstat", vm)
    _w(root, "proc/diskstats", disks)
    _w(root, "proc/net/dev", _netdev(net))


MEM = (
    "MemTotal:       16000000 kB\nMemFree:  1000000 kB\nMemAvailable:    6000000 kB\n"
    "Buffers:  100000 kB\nCached:  3000000 kB\nSReclaimable:  500000 kB\n"
    "Dirty:  2048 kB\nSwapTotal:  4000000 kB\nSwapFree:  3000000 kB\n"
)
CPUS0 = {f"cpu{i}": (0, 0, 0, 0, 0, 0, 0, 0) for i in range(4)}
NET0 = {n: (0,) * 8 for n in ("lo", "eno1", "tailscale0", "veth1", "docker0", "vnet7")}


def _first(root, clock, gpu=None):
    disks = (
        _disk("sda", 0, 0, 0, 0, 0, 0, 5, 0)
        + _disk("sda1", 0, 0, 0, 0, 0, 0, 0, 0)  # a partition: not in /sys/block
        + _disk("md1", 0, 0, 0, 0, 0, 0, 0, 0)
        + _disk("nvme0n1", 0, 0, 0, 0, 0, 0, 0, 0)
        + _disk("loop0", 0, 0, 0, 0, 0, 0, 0, 0)
    )
    _poll_state(root, CPUS0, MEM, "pgmajfault 100\npswpin 0\npswpout 0\n", disks, NET0)
    mon = MachineMonitor(root=root, clock=clock, gpu=gpu or NoGPU())
    return mon, mon.poll()


def _second(root, clock, mon, dt=2.0):
    clock.t += dt
    cpus = {
        # cpu0: 100 busy of 400; cpu1: 300 busy of 1000 (100 of it iowait-free);
        # package 1 idle; the package's share is its summed time
        "cpu0": (100, 0, 0, 300, 0, 0, 0, 0),
        "cpu1": (200, 0, 100, 600, 100, 0, 0, 0),
        "cpu2": (0, 0, 0, 1000, 0, 0, 0, 0),
        "cpu3": (0, 0, 0, 1000, 0, 0, 0, 0),
    }
    disks = (
        _disk("sda", 100, 2048, 300, 50, 4096, 200, 1, 1500)  # in flight fell 5 -> 1
        + _disk("sda1", 0, 0, 0, 0, 0, 0, 0, 0)
        + _disk("md1", 10, 80, 0, 0, 0, 0, 0, 0)
        + _disk("nvme0n1", 0, 0, 0, 0, 0, 0, 0, 0)
        + _disk("loop0", 9, 9, 9, 9, 9, 9, 0, 9)
    )
    net = dict(NET0)
    net["eno1"] = (2_000_000, 2000, 0, 2, 500_000, 1000, 0, 0)
    net["tailscale0"] = (10_000, 20, 0, 0, 20_000, 30, 0, 0)
    net["veth1"] = (100, 1, 0, 0, 300, 3, 0, 0)
    net["docker0"] = (200, 2, 0, 0, 100, 1, 0, 0)
    net["vnet7"] = (600, 6, 0, 0, 0, 0, 0, 0)
    _poll_state(root, cpus, MEM, "pgmajfault 160\npswpin 10\npswpout 20\n", disks, net)
    return mon.poll()


# ------------------------------------------------------------------ parsers
def test_parsers_read_the_kernels_formats():
    s = parse_stat(
        "cpu  10 1 2 30 4 0 5 6 7 0\ncpu0 1 0 0 3 0 0 0 0 0 0\nprocs_running 2\n"
    )
    assert s["cpu"]["cpu"] == (10, 1, 2, 30, 4, 0, 5, 6) and s["procs_running"] == 2
    assert parse_meminfo("MemTotal: 2 kB\nHugePages_Total: 0\n") == {
        "MemTotal": 2048,
        "HugePages_Total": 0,
    }
    d = parse_diskstats(_disk("sdb", 1, 2, 3, 4, 5, 6, 7, 8))
    assert d["sdb"]["read_sectors"] == 2 and d["sdb"]["io_ms"] == 8
    assert d["sdb"]["in_flight"] == 7 and d["sdb"]["write_ms"] == 6
    n = parse_netdev(_netdev({"eth0": (1, 2, 3, 4, 5, 6, 7, 8)}))
    assert n["eth0"] == dict(
        rx_bytes=1, rx_packets=2, rx_errs=3, rx_drop=4,
        tx_bytes=5, tx_packets=6, tx_errs=7, tx_drop=8,
    )  # fmt: skip


# ------------------------------------------------------------------ the monitor
def test_the_first_poll_has_levels_but_no_rates(tree):
    _, doc = _first(tree, Clock())
    assert doc["interval_s"] is None
    assert doc["cpu"]["total"] is None and doc["cpu"]["packages"][0]["share"] is None
    assert doc["memory"]["used"] == (16_000_000 - 6_000_000) * 1024  # a level: known
    assert doc["memory"]["major_faults_per_s"] is None
    sda = next(d for d in doc["disks"] if d["name"] == "sda")
    assert sda["in_flight"] == 5 and sda["read_Bps"] is None and sda["util"] is None
    assert all(r["rx_bytes"] is None for r in doc["net"]["interfaces"])
    assert doc["net"]["virtual"]["rx_bytes"] is None


def test_cpu_shares_are_summed_time_not_a_mean_of_fractions(tree):
    clock = Clock()
    mon, _ = _first(tree, clock)
    doc = _second(tree, clock, mon)
    c = doc["cpu"]
    by = {x["cpu"]: x["share"] for x in c["logical"]}
    assert by["cpu0"]["busy"] == pytest.approx(0.25)
    assert by["cpu1"]["busy"] == pytest.approx(0.3) and by["cpu1"]["iowait"] == 0.1
    pkg0 = c["packages"][0]
    assert pkg0["package"] == 0 and pkg0["cpus"] == ["cpu0", "cpu1"]
    # (100 + 300) busy of (400 + 1000): 0.2857, not the mean 0.275
    assert pkg0["share"]["busy"] == pytest.approx(400 / 1400)
    assert c["packages"][1]["share"]["busy"] == 0.0
    assert c["total"]["busy"] == pytest.approx(400 / 3400)
    assert doc["load"] == [1.5, 1.25, 1.0]


def test_disks_bytes_iops_wait_and_busy_time(tree):
    clock = Clock()
    mon, _ = _first(tree, clock)
    doc = _second(tree, clock, mon)
    names = [d["name"] for d in doc["disks"]]
    assert names == ["md1", "nvme0n1", "sda"]  # no loop device, no partition
    sda = next(d for d in doc["disks"] if d["name"] == "sda")
    assert sda["read_Bps"] == 2048 * 512 / 2 and sda["write_Bps"] == 4096 * 512 / 2
    assert sda["read_iops"] == 50 and sda["write_iops"] == 25
    assert sda["await_ms"] == pytest.approx((300 + 200) / 150)
    assert sda["util"] == 0.75  # 1500 ms busy of 2000
    assert sda["in_flight"] == 1  # a level that fell does not void the interval
    # md1 keeps statistics (as md1 and md5 do on Atlas): its busy share is real
    md = next(d for d in doc["disks"] if d["name"] == "md1")
    assert md["util_reason"] is None and md["read_iops"] == 5 and md["util"] == 0.0
    # nvme0n1 here keeps none (queue/iostats 0, as md127 on Atlas): unknown, not 0
    nv = next(d for d in doc["disks"] if d["name"] == "nvme0n1")
    assert nv["util_reason"] and nv["read_Bps"] is None and nv["util"] is None
    assert nv["await_ms"] is None


def test_a_counter_that_went_backwards_voids_the_interval(tree):
    clock = Clock()
    mon, _ = _first(tree, clock)
    _second(tree, clock, mon)
    clock.t += 2
    zero_disk = _disk("sda", 0, 0, 0, 0, 0, 0, 0, 0)
    _poll_state(tree, CPUS0, MEM, "pgmajfault 1\n", zero_disk, {"eno1": (0,) * 8})
    doc = mon.poll()
    sda = doc["disks"][0]
    assert sda["read_Bps"] is None and sda["util"] is None
    assert doc["cpu"]["total"] is None and doc["memory"]["major_faults_per_s"] is None
    assert doc["net"]["interfaces"][0]["rx_bytes"] is None


def test_network_classes_link_use_and_the_virtual_sum(tree):
    clock = Clock()
    mon, _ = _first(tree, clock)
    doc = _second(tree, clock, mon)
    rows = {r["name"]: r for r in doc["net"]["interfaces"]}
    assert set(rows) == {"eno1", "tailscale0"}  # lo excluded, virtual summed
    e = rows["eno1"]
    assert e["class"] == "physical" and e["up"] and e["speed_mbps"] == 1000
    assert e["rx_bytes"] == 1_000_000 and e["rx_drop"] == 1
    assert e["link_util"] == pytest.approx(1_000_000 * 8 / 1e9)
    assert rows["tailscale0"]["class"] == "tunnel" and rows["tailscale0"]["up"]
    assert rows["tailscale0"]["link_util"] is None  # no speed: no share of a link
    v = doc["net"]["virtual"]
    # the VM tap is summed with the bridges and veth pairs, not called a tunnel
    assert v["count"] == 3 and v["rx_bytes"] == (100 + 200 + 600) / 2
    assert v["tx_packets"] == (3 + 1) / 2


def test_memory_levels_and_rates(tree):
    clock = Clock()
    mon, _ = _first(tree, clock)
    m = _second(tree, clock, mon)["memory"]
    assert m["total"] == 16_000_000 * 1024 and m["available"] == 6_000_000 * 1024
    assert m["cached"] == (3_000_000 + 500_000) * 1024
    assert m["swap_used"] == 1_000_000 * 1024 and m["dirty"] == 2048 * 1024
    assert m["major_faults_per_s"] == 30.0
    assert m["swap_in_Bps"] == 5.0 * os.sysconf("SC_PAGE_SIZE")


def test_temperatures_in_degrees_with_only_plausible_limits(tree):
    _, doc = _first(tree, Clock())
    chips = doc["thermal"]
    assert [c["chip"] for c in chips] == ["hwmon2", "hwmon10"]  # numeric order
    core = chips[0]["sensors"]
    assert core[0] == {"label": "Package id 0", "c": 71.0, "high_c": 82.0,
                       "crit_c": 100.0, "n": 1}  # fmt: skip
    nv = {s["n"]: s for s in chips[1]["sensors"]}
    assert nv[1]["high_c"] == 84.85
    assert nv[2]["high_c"] is None and nv[2]["label"] == "temp2"  # 65261.85 C


def test_rates_scale_with_the_interval_and_shares_do_not(tree):
    """Metamorphic: the same counter changes over twice the time give half every
    rate (a device's busy share is one: busy ms per ms); CPU fractions and the
    mean wait per I/O are ratios of changes, so they do not move."""
    docs = []
    for dt in (2.0, 4.0):
        clock = Clock()
        mon, _ = _first(tree, clock)
        docs.append(_second(tree, clock, mon, dt=dt))
    a, b = docs
    da = {d["name"]: d for d in a["disks"]}
    db = {d["name"]: d for d in b["disks"]}
    assert db["sda"]["read_Bps"] == da["sda"]["read_Bps"] / 2
    assert db["sda"]["await_ms"] == da["sda"]["await_ms"]
    assert db["sda"]["util"] == da["sda"]["util"] / 2
    assert b["cpu"]["total"] == a["cpu"]["total"]
    assert b["memory"]["major_faults_per_s"] == a["memory"]["major_faults_per_s"] / 2
    ea = next(r for r in a["net"]["interfaces"] if r["name"] == "eno1")
    eb = next(r for r in b["net"]["interfaces"] if r["name"] == "eno1")
    assert eb["tx_bytes"] == ea["tx_bytes"] / 2


def test_a_reader_that_cannot_read_the_gpus_says_why(tree):
    _, doc = _first(tree, Clock())
    assert doc["gpus"] == [] and doc["gpu_reason"] == "unavailable: no NVML (test)"
    r = NVMLReader(library="libnvidia-ml-does-not-exist.so.1")
    assert "not found" in r.reason
    with pytest.raises(OSError, match="not found"):
        r.read()


# ------------------------------------------------------------------ the page
def _page(tree, gpu=None):
    clock = Clock()
    mon, first = _first(tree, clock, gpu=gpu)
    hist = History()
    hist.add(first)
    doc = _second(tree, clock, mon)
    hist.add(doc)
    return MV.panels({"machine": doc, "machine_hist": hist})


def test_every_machine_panel_is_the_one_shape_and_strict_json(tree):
    panels = _page(tree, gpu=FakeGPU())
    assert sorted(panels) == ["1", "2", "3", "4", "5", "6"]
    json.dumps(panels, allow_nan=False)
    for key, p in panels.items():
        assert p["state"] == "ok", key
        for r in p.get("rows") or []:
            assert len(r) == 6 and r[2] in ("meas", "samp", "deri"), (key, r)
        t = p.get("table")
        if t:
            assert all(len(row) == len(t["cols"]) for row in t["rows"]), key
            assert len(t["roles"]) == len(t["rows"])
    cpu = panels["1"]
    assert cpu["rows"][1][:2] == ["pkg 0", "28.6 %"]
    assert cpu["strips"][0] == ["pkg 0", [0.25, pytest.approx(0.3)]]
    assert "48" not in cpu["summary"][0] and "4 CPUs in 2 packages" in cpu["summary"][0]


def test_unknown_is_a_dash_never_a_zero(tree):
    clock = Clock()
    _, first = _first(tree, clock)
    panels = MV.panels({"machine": first, "machine_hist": None})
    assert panels["1"]["rows"][0][1] == "-"
    sda = next(r for r in panels["4"]["table"]["rows"] if r[0] == "sda")
    assert sda[1] == "-" and sda[7] == "5"  # rate unknown, queue (a level) known
    nv = next(r for r in panels["4"]["table"]["rows"] if r[0] == "nvme0n1")
    assert nv[6] == "n/a" and nv[1] == "-"
    assert panels["6"] == {"state": "none", "message": "unavailable: no NVML (test)"}


def test_colour_follows_each_sensors_own_limits(tree):
    panels = _page(tree, gpu=FakeGPU())
    rows = {r[0]: r for r in panels["2"]["rows"]}
    assert rows["pkg 0"][1] == "71 °C" and rows["pkg 0"][5] == ""  # 11 below high
    assert rows["pkg 0 core"][1] == "74 °C" and rows["pkg 0 core"][5] == ""
    assert rows["pkg 0 core"][4] == "Core 1, hottest of 2  high 82  crit 100"
    assert rows["gpu 0"][5] == "amber"  # 86, within 5 of its 88 slowdown limit
    assert rows["nvme temp2"][4] == "no limit reported"
    summary, role = panels["2"]["summary"]
    assert "gpu 0 at 86 °C, 2 °C below its slowdown mark" in summary
    assert role == "amber" and rows["gpu 0"][4] == "slowdown 88"
    assert MV._temp_role(82.0, 82.0, 100.0) == "red"
    assert MV._temp_role(77.0, 82.0, 100.0) == "amber"
    assert MV._temp_role(76.9, 82.0, 100.0) == ""


def test_without_a_machine_source_the_page_says_how_to_attach_one():
    p = MV.panels({})
    assert p["3"] == {"state": "none", "message": "not attached: start with --machine"}


# ------------------------------------------------------------------ pages, CLI
@pytest.mark.parametrize(
    "sources,names,index_panels",
    [
        ({"index": True}, ["index"], 8),
        ({"index": True, "nats": True}, ["index"], 9),
        ({"nats": True}, ["nats"], None),
        ({"machine": True}, ["machine"], None),
        ({"index": True, "machine": True, "nats": True}, ["index", "machine"], 9),
        ({"nats": True, "machine": True}, ["machine", "nats"], None),
    ],
)
def test_the_pages_follow_from_the_sources(sources, names, index_panels):
    pages = VM.pages(sources)
    assert [p["name"] for p in pages] == names
    if index_panels:
        assert len(pages[0]["panels"]) == index_panels
    h = VM.hello(sources)
    assert h["pages"] == pages and h["protocol"] == 2 and h["keys_machine"]


def test_the_machine_source_is_for_the_terminal_console(capsys):
    assert main(["console", "--machine", "--web"]) == 2
    assert "terminal console only" in capsys.readouterr().err
    assert main(["console", "--web"]) == 2
    assert "--machine" in capsys.readouterr().err

# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""The machine page: six panels from :mod:`console.machine` snapshots.

Every panel is one shape, which the terminal client draws with one renderer:

    {"state": "ok" | "none", "message": str,
     "summary": [text, role] | None, "title_extra": str,
     "rows": [[label, value, kind, series | None, note, role], ...],
     "strips": [[label, [fraction | None, ...]], ...],
     "table": {"cols": [[name, width, right], ...], "rows": [[cell, ...], ...],
               "roles": [role, ...]} | None,
     "note": str}

``kind`` is meas / samp / deri, as on every other panel; a value that is not
known is "-", never 0. Colour follows each sensor's own limits (amber within
5 degrees C of its high mark, red at or past it) and nothing else.
"""

from __future__ import annotations

from .tui import _si

TITLES = {
    "1": "1 CPU  busy per package and logical CPU",
    "2": "2 thermal  each sensor against its own limits",
    "3": "3 memory",
    "4": "4 disks",
    "5": "5 network",
    "6": "6 GPU",
}
PANELS = [1, 2, 3, 4, 5, 6]
KEYS = [
    ("q / Ctrl-C", "quit"),
    ("Ctrl-Z", "suspend to the shell (fg resumes)"),
    ("Tab / Shift-Tab", "next / previous panel"),
    ("digits", "focus a panel (this page's numbers)"),
    ("< / >", "previous / next page"),
    ("p", "pause / resume the display (polling goes on, so rates stay exact)"),
    ("P", "snapshot: write the screen as it is now to a .txt file"),
    ("?", "this help"),
    ("Esc", "close an overlay"),
]
WARN_C = 5.0  # amber this close to a sensor's own high mark


def _pct(f, d: int = 1) -> str:
    return "-" if f is None else f"{100.0 * f:.{d}f} %"


def _size(b) -> str:
    """Bytes as binary units (the kernel's own KiB, MiB, GiB)."""
    if b is None:
        return "-"
    for unit, f in (("TiB", 2**40), ("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if abs(b) >= f:
            return f"{b / f:.1f} {unit}"
    return f"{b:.0f} B"


def _temp(c) -> str:
    return "-" if c is None else f"{c:.0f} °C"


def _temp_role(c, high, crit) -> str:
    if c is None:
        return "dim"
    if (crit is not None and c >= crit) or (high is not None and c >= high):
        return "red"
    if high is not None and c >= high - WARN_C:
        return "amber"
    return ""


def _series(hist, key):
    if hist is None:
        return None
    return [None if v is None else float(v) for v in hist.series.get(key, [])][-256:]


def _empty(message: str) -> dict:
    return {"state": "none", "message": message}


def _poll_note(doc: dict) -> str:
    dt = doc.get("interval_s")
    return "" if not dt else f"rates over {dt:.1f} s polls"


def panels(st: dict) -> dict:
    """The machine page's panels, keyed "1" to "6"."""
    doc, hist = st.get("machine"), st.get("machine_hist")
    if doc is None:
        msg = "not attached: start with --machine"
        return {str(n): _empty(msg) for n in PANELS}
    return {
        "1": cpu(doc, hist),
        "2": thermal(doc, hist),
        "3": memory(doc, hist),
        "4": disks(doc),
        "5": network(doc),
        "6": gpu(doc),
    }


def cpu(doc: dict, hist=None) -> dict:
    c = doc.get("cpu") or {}
    pk = c.get("packages") or []
    load = doc.get("load")
    parts = [f"{len(c.get('logical') or [])} CPUs in {len(pk)} packages"]
    if load:
        parts.append("load " + " ".join(f"{x:.2f}" for x in load))
    if c.get("procs_running") is not None:
        parts.append(f"running {c['procs_running']}  blocked {c.get('procs_blocked')}")
    tot = c.get("total")
    rows = [
        [
            "all",
            _pct(tot and tot["busy"]),
            "deri",
            _series(hist, "cpu.busy"),
            "" if tot is None else f"iowait {_pct(tot['iowait'])}",
            "",
        ]
    ]
    for p in pk:
        s = p["share"]
        note = ""
        if s is not None:
            note = f"iowait {_pct(s['iowait'])}"
            if s["steal"] > 0:
                note += f"  steal {_pct(s['steal'])}"
        rows.append(
            [
                f"pkg {p['package']}",
                _pct(s and s["busy"]),
                "deri",
                _series(hist, f"cpu.busy.{p['package']}"),
                note,
                "",
            ]
        )
    by_name = {x["cpu"]: x["share"] for x in c.get("logical") or []}
    strips = [
        [
            f"pkg {p['package']}",
            [None if by_name.get(n) is None else by_name[n]["busy"] for n in p["cpus"]],
        ]
        for p in pk
    ]
    return {
        "state": "ok",
        "summary": ["  ".join(parts), "cyan"],
        "title_extra": _poll_note(doc),
        "rows": rows,
        "strips": strips,
        "note": "busy: time neither idle nor waiting on I/O; one cell per logical CPU",
    }


def thermal(doc: dict, hist=None) -> dict:
    rows, margins = [], []

    def add(label, c, high, crit, key, note_extra="", high_name="high"):
        lim = [
            f"{high_name} {high:.0f}" if high is not None else "",
            f"crit {crit:.0f}" if crit is not None else "",
        ]
        note = "  ".join(x for x in [note_extra, *lim] if x) or "no limit reported"
        rows.append(
            [
                label,
                _temp(c),
                "samp",
                _series(hist, key),
                note,
                _temp_role(c, high, crit),
            ]
        )
        ref = high if high is not None else crit
        if ref is not None and c is not None:
            which = high_name if high is not None else "crit"
            margins.append((ref - c, label, c, which))

    for chip in doc.get("thermal") or []:
        name, sensors = chip["name"], chip["sensors"]
        if name == "coretemp":
            pkg = [s for s in sensors if s["label"].startswith("Package")]
            cores = [s for s in sensors if s["label"].startswith("Core")]
            for s in pkg:
                p = s["label"].split()[-1]
                add(
                    f"pkg {p}",
                    s["c"],
                    s["high_c"],
                    s["crit_c"],
                    f"temp.{chip['chip']}.{s['n']}",
                )
            if cores:
                hot = max(cores, key=lambda s: s["c"])
                p = pkg[0]["label"].split()[-1] if pkg else chip["chip"]
                add(
                    f"pkg {p} core",
                    hot["c"],
                    hot["high_c"],
                    hot["crit_c"],
                    f"temp.{chip['chip']}.{hot['n']}",
                    f"{hot['label']}, hottest of {len(cores)}",
                )
        else:
            for s in sensors:  # an unlabelled sensor is named by its file: temp2
                add(
                    f"{name} {s['label']}"[:14],
                    s["c"],
                    s["high_c"],
                    s["crit_c"],
                    f"temp.{chip['chip']}.{s['n']}",
                )
    for g in doc.get("gpus") or []:
        add(
            f"gpu {g['index']}",
            g.get("temp_c"),
            g.get("temp_limit_c"),
            None,
            f"gpu.{g['index']}.temp",
            high_name="slowdown",
        )
    if not rows:
        return _empty("no temperature sensors under /sys/class/hwmon")
    summary = None
    if margins:
        m, label, c, which = min(margins)
        role = "red" if m <= 0 else ("amber" if m <= WARN_C else "green")
        where = "at or past" if m <= 0 else f"{m:.0f} °C below"
        summary = [
            f"closest to its limit: {label} at {c:.0f} °C, {where} its {which} mark",
            role,
        ]
    return {
        "state": "ok",
        "summary": summary,
        "title_extra": "",
        "rows": rows,
        "note": "limits are each sensor's own; implausible ones are not shown",
    }


def memory(doc: dict, hist=None) -> dict:
    m = doc.get("memory") or {}
    total, used = m.get("total"), m.get("used")
    frac = None if not total or used is None else used / total
    rows = [
        [
            "used",
            _size(used),
            "meas",
            _series(hist, "mem.used"),
            "total minus the kernel's MemAvailable",
            "",
        ],
        ["available", _size(m.get("available")), "meas", None, "", ""],
        [
            "cache",
            _size(m.get("cached")),
            "meas",
            _series(hist, "mem.cached"),
            "page cache + reclaimable slab",
            "",
        ],
        [
            "dirty",
            _size(m.get("dirty")),
            "meas",
            _series(hist, "mem.dirty"),
            "waiting to be written",
            "",
        ],
        [
            "swap used",
            _size(m.get("swap_used")),
            "meas",
            _series(hist, "mem.swap_used"),
            f"of {_size(m.get('swap_total'))}",
            "",
        ],
        [
            "major flt",
            _si(m.get("major_faults_per_s"), "/s"),
            "deri",
            _series(hist, "mem.major_faults_per_s"),
            "page faults that read a disk",
            "",
        ],
        ["swap in", _si(m.get("swap_in_Bps"), "B/s"), "deri", None, "", ""],
        ["swap out", _si(m.get("swap_out_Bps"), "B/s"), "deri", None, "", ""],
    ]
    return {
        "state": "ok",
        "summary": [f"{_size(used)} of {_size(total)} used ({_pct(frac, 0)})", "cyan"],
        "title_extra": _poll_note(doc),
        "rows": rows,
    }


DISK_COLS = [
    ["device", 9, False],
    ["read", 10, True],
    ["write", 10, True],
    ["r IOPS", 7, True],
    ["w IOPS", 7, True],
    ["wait", 8, True],
    ["util", 7, True],
    ["queue", 5, True],
]


def disks(doc: dict) -> dict:
    rows, roles = [], []
    for d in doc.get("disks") or []:
        util = d["util"]
        rows.append(
            [
                d["name"],
                _si(d["read_Bps"], "B/s"),
                _si(d["write_Bps"], "B/s"),
                "-" if d["read_iops"] is None else f"{d['read_iops']:.0f}",
                "-" if d["write_iops"] is None else f"{d['write_iops']:.0f}",
                "-" if d["await_ms"] is None else f"{d['await_ms']:.1f} ms",
                "n/a" if d["util_reason"] else _pct(util, 0),
                str(d["in_flight"]),
            ]
        )
        if d["util_reason"]:  # no statistics: unknown, shown dim, never as idle
            roles.append("dim")
        else:
            roles.append("amber" if util is not None and util >= 0.8 else "")
    if not rows:
        return _empty("no block devices in /proc/diskstats")
    return {
        "state": "ok",
        "title_extra": _poll_note(doc),
        "table": {"cols": DISK_COLS, "rows": rows, "roles": roles},
        "note": "wait: per I/O; util: time busy; n/a: device keeps no statistics",
    }


NET_COLS = [
    ["interface", 14, False],
    ["class", 8, False],
    ["rx", 10, True],
    ["tx", 10, True],
    ["rx pkt/s", 9, True],
    ["tx pkt/s", 9, True],
    ["err+drop", 9, True],
    ["link", 6, True],
]


def _errs(r):
    vals = [r.get(k) for k in ("rx_errs", "rx_drop", "tx_errs", "tx_drop")]
    return "-" if None in vals else f"{sum(vals):.1f}/s"


def network(doc: dict) -> dict:
    net = doc.get("net") or {}
    rows, roles = [], []
    for r in net.get("interfaces") or []:
        rows.append(
            [
                r["name"],
                r["class"] if r["up"] else f"{r['class']} (down)",
                _si(r["rx_bytes"], "B/s"),
                _si(r["tx_bytes"], "B/s"),
                _si(r["rx_packets"], ""),
                _si(r["tx_packets"], ""),
                _errs(r),
                "-" if r["link_util"] is None else _pct(r["link_util"], 0),
            ]
        )
        roles.append("" if r["up"] else "dim")
    v = net.get("virtual") or {}
    if v.get("count"):
        rows.append(
            [
                f"virtual ({v['count']})",
                "summed",
                _si(v.get("rx_bytes"), "B/s"),
                _si(v.get("tx_bytes"), "B/s"),
                _si(v.get("rx_packets"), ""),
                _si(v.get("tx_packets"), ""),
                _errs(v),
                "",
            ]
        )
        roles.append("dim")
    if not rows:
        return _empty("no network interfaces besides loopback")
    return {
        "state": "ok",
        "title_extra": _poll_note(doc),
        "table": {"cols": NET_COLS, "rows": rows, "roles": roles},
        "note": "physical NICs and tunnels one row each; bridges, veth and VM taps"
        " summed",
    }


GPU_COLS = [
    ["gpu", 4, False],
    ["name", 14, False],
    ["util", 6, True],
    ["memory", 20, True],
    ["temp", 11, True],
    ["power", 12, True],
]


def gpu(doc: dict) -> dict:
    gs = doc.get("gpus") or []
    if not gs:
        return _empty(doc.get("gpu_reason") or "no GPU")
    rows, roles = [], []
    for g in gs:
        mem = "-"
        if g.get("mem_total"):
            mem = f"{_size(g.get('mem_used'))} / {_size(g['mem_total'])}"
        lim = g.get("temp_limit_c")
        temp = _temp(g.get("temp_c")) + (f" / {lim:.0f}" if lim is not None else "")
        power = "-"
        if g.get("power_w") is not None:
            power = f"{g['power_w']:.0f} W"
            if g.get("power_limit_w"):
                power += f" / {g['power_limit_w']:.0f}"
        rows.append(
            [
                str(g["index"]),
                (g.get("name") or "-")[:14],
                _pct(g.get("util"), 0),
                mem,
                temp,
                power,
            ]
        )
        roles.append(_temp_role(g.get("temp_c"), lim, None))
    return {
        "state": "ok",
        "title_extra": "NVML, sampled",
        "table": {"cols": GPU_COLS, "rows": rows, "roles": roles},
        "note": "util: driver's last sample; temp / slowdown; power: now / limit",
    }

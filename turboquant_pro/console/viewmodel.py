# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""What the terminal client draws, as data.

The console engine (:mod:`.engine`) holds the session and its instruments; the
terminal client (``go/tqp-console``) owns the terminal and does all geometry and
drawing. Between them passes this view model: numbers already reduced to the
client's geometry (scope columns in divisions, spectrum columns in dB), and every
text that carries a unit, a scale or a meaning already formatted, so the
calibration is defined once, here, beside the instruments it describes.

A span is ``[text, role]``; a role is one of the colour roles of :mod:`.tui`
(``"cyan"``, ``"dim"``, ``"yellow_bold"``, ...) or ``null``. Every function is
pure: state in, JSON-ready dict out.
"""

from __future__ import annotations

import json
import time

import numpy as np

from . import tui
from .scope import COLORS as SCOPE_COLORS
from .scope import HDIV, SIGNALS, VDIV
from .scope_view import (
    COLOR_WORD,
    FFT_DB_SPAN,
    _fmt,
    _per_div,
)
from .scope_view import SOFTKEYS as SCOPE_SOFTKEYS
from .scope_view import _tick as scope_tick
from .spectrum_view import COLORS as SPEC_COLORS
from .spectrum_view import LABEL, MEANS, _db
from .spectrum_view import SOFTKEYS as SPEC_SOFTKEYS
from .spectrum_view import _tick as spec_tick

PROTOCOL = 1


def spans_of(cv: tui.Canvas, y0: int = 0, x0: int = 0, h=None, w=None) -> list:
    """Rows of spans for the rectangle of ``cv`` at (y0, x0), size h x w."""
    h = cv.h - y0 if h is None else h
    w = cv.w - x0 if w is None else w
    out = []
    for y in range(y0, y0 + h):
        row = cv.cells[y][x0 : x0 + w]
        spans, x = [], 0
        while x < len(row):
            role = row[x][1]
            j = x
            while j < len(row) and row[j][1] == role:
                j += 1
            spans.append(["".join(c for c, _ in row[x:j]), role])
            x = j
        out.append(spans)
    return out


# ----------------------------------------------------------------- header
def header(st: dict) -> dict:
    snap = st.get("snap") or {}
    wl = snap.get("workload", {})
    traces = st.get("traces", [])
    last = traces[-1] if traces else {}
    age = snap.get("last_trace_age_s")
    src = snap.get("sources") or {}
    if st.get("paused"):
        state = ["PAUSED", "amber"]
    elif src and not src.get("index"):
        state = ["no index", "dim"]
    elif age is not None and age < 3:
        state = ["live", "green"]
    else:
        state = ["waiting", "amber"] if age is None else [f"stale {age:.0f}s", "amber"]
    labels = [
        [f"[{state[0]}]", state[1]],
        [f"[mode: {wl.get('mode', '-')}]", "green" if wl.get("rerank") else "amber"],
        [f"[scan: {last.get('scan_path') or '-'}]", "cyan"],
    ]
    obs = ((st.get("readscope") or {}).get("observer") or {}).get("reference")
    if obs:
        labels.append(
            [f"[observer: {obs.get('observer')} {obs.get('sha256', '')[:8]}]", "purple"]
        )
    t = snap.get("t")
    stamp = time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(t)) if t else "--:--:--Z"
    return {"labels": labels, "stamp": stamp}


# ----------------------------------------------------------------- panels 1-6, 9
def system(st: dict) -> dict:
    """Panel 1: each reading with its unit and kind (meas / samp / deri)."""
    r = tui._readings(st.get("snap") or {})
    items = []
    for label, name, d in (
        ("QPS", "search.qps", 1),
        ("p50", "search.latency_ms.p50", 2),
        ("p95", "search.latency_ms.p95", 2),
        ("p99", "search.latency_ms.p99", 2),
        ("agree", "search.rerank_agreement", 2),
        ("compress", "index.compression_ratio", 1),
        ("rows", "index.rows", 0),
        ("CPU", "process.cpu_percent", 0),
        ("RSS", "process.rss_mb", 0),
    ):
        rd = r.get(name)
        if rd is None:
            continue
        unit = (
            "" if rd["unit"] == "fraction" else rd["unit"].replace("queries/s", "q/s")
        )
        items.append(
            [
                label,
                f"{tui.fmt(rd['value'], d)} {unit}".rstrip(),
                rd["kind"][:4],
                rd["value"] is None,
            ]
        )
    return {"items": items}


def throughput(st: dict) -> dict:
    """Panel 2: two series, one sample per second, newest last."""
    r = tui._readings(st.get("snap") or {})

    def ser(name, unit, d, key, hist, role):
        vals = [None if v is None else float(v) for v in st.get(hist, [])]
        now = r.get(key, {}).get("value")
        return {
            "name": name,
            "unit": unit,
            "now": tui.fmt(now, d),
            "digits": d,
            "role": role,
            "values": vals[-512:],
        }

    return {
        "series": [
            ser("QPS", "q/s", 1, "search.qps", "qps_hist", "cyan"),
            ser("p95 latency", "ms", 2, "search.latency_ms.p95", "p95_hist", "purple"),
        ],
        "note": "1 sample/s, bar height 0..max",
    }


def pipeline(st: dict) -> dict:
    snap = st.get("snap") or {}
    r = tui._readings(snap)
    wl = snap.get("workload", {})
    stages = []
    for s in ("encode", "scan", "rerank"):
        v = r.get(f"search.stage_ms.{s}", {}).get("value")
        stages.append([s, v, tui.fmt(v, 3)])
    k, rr = wl.get("k", "-"), wl.get("rerank", 0)
    note = (
        "no index attached"
        if not wl
        else f"top-{k} from {k * rr} reranked" if rr else f"top-{k}, approximate"
    )
    return {"stages": stages, "note": note}


def readscope(st: dict) -> dict:
    rs = st.get("readscope") or {}
    obs = (rs.get("observer") or {}).get("reference")
    rows = []
    if obs:
        cons = ", ".join(
            str(c.get("metric") or c.get("name") or c)
            for c in obs.get("consumers") or []
        )
        rows += [
            ["observer", f"{obs.get('observer')}  target {obs.get('target')}", None],
            ["sha256", obs.get("sha256", ""), None],
            ["consumers", cons or "-", None],
        ]
    else:
        rows.append(
            [
                "observer",
                "none loaded (--observer X.tqo): results are not tied to a declared "
                "reader",
                None,
            ]
        )
    cert = rs.get("certificate")
    if cert:
        cc = cert.get("certificate", {})
        rows.append(
            [
                "certificate",
                f"{'PASSED' if cert.get('passed') else 'NOT PASSED'}"
                f"  tau floor {cc.get('tau_floor')}",
                None,
            ]
        )
        for line in tui._validity_lines(rs.get("validity") or {}):
            k, v, *col = line
            rows.append([k, v, col[0] if col else None])
    for p in rs.get("provenance", []):
        val = p.get("sha256") or json.dumps({k: v for k, v in p.items() if k != "step"})
        rows.append([p["step"][:12], val, None])
    return {"rows": rows}


def index(st: dict) -> dict:
    snap = st.get("snap") or {}
    ie = snap.get("index", {})
    wl = snap.get("workload", {})
    if not ie:
        return {"rows": [], "empty": "no index attached (--index or --demo)"}
    rows = [
        ["kind", ie.get("kind")],
        ["rows", ie.get("rows")],
        ["dim", ie.get("dim")],
        ["metric", ie.get("metric")],
        ["bytes/row", ie.get("stored_bytes_per_row")],
        ["kernel", "AVX2" if ie.get("kernel") else "numpy"],
        ["workload", f"{wl.get('target_qps', '-')} qps, k={wl.get('k', '-')}"],
    ]
    return {"rows": [[k, str(v)] for k, v in rows]}


QUERY_COLS = [
    ["time", 12, False],
    ["trace", 16, False],
    ["row", 5, False],
    ["scan", 13, False],
    ["total", 8, True],
    ["encode", 8, True],
    ["scan ms", 8, True],
    ["rerank", 8, True],
    ["agree", 6, True],
]


def queries(st: dict, n: int = 64) -> dict:
    """Panel 6: newest first; ``ids`` lets the client ask to inspect a row."""
    rows, ids = [], []
    for t in list(reversed(st.get("traces", [])))[:n]:
        rows.append(
            [
                t["started_utc"][11:23],
                t["id"],
                str(t["params"].get("workload_row", "-")),
                t.get("scan_path") or "-",
                tui.fmt(t.get("total_ms"), 3),
                tui.fmt(tui._stage(t, "encode"), 3),
                tui.fmt(tui._stage(t, "scan"), 3),
                tui.fmt(tui._stage(t, "rerank"), 3),
                tui.fmt((t.get("results") or {}).get("rerank_agreement"), 2),
            ]
        )
        ids.append(t["id"])
    return {"cols": QUERY_COLS, "rows": rows, "ids": ids}


def nats(st: dict) -> dict:
    """Panel 9: every metric the fabric monitor has, one calibrated row each."""
    fab = st.get("fabric")
    if fab is None:
        return {"state": "none", "message": "not attached: start with --nats URL"}
    if not fab.get("reachable"):
        return {"state": "down", "message": f"{fab['source']['url']}: UNREACHABLE"}
    hist = st.get("fabric_hist")
    series = hist.series if hist is not None else {}
    leafs = fab.get("leafs") or []
    srv = fab.get("server") or {}
    js = fab.get("jetstream") or {}
    summary = (
        f"NATS {len(leafs)} leaf, {len(fab.get('connections') or [])} clients, "
        f"{srv.get('subscriptions', '-')} subs, {srv.get('slow_consumers', '-')} slow, "
        f"JS {js.get('streams', '-')} streams"
    )
    rows = []
    for label, key, unit, kind in tui.NATS_ROWS:
        vals = [None if v is None else float(v) for v in series.get(key, [])]
        cur = vals[-1] if vals else None
        real = [v for v in vals if v is not None]
        top = max(real) if len(real) >= 2 else None

        def show(v, unit=unit):
            return f"{tui.fmt(v, 1)} ms" if unit == "ms" else tui._si(v, unit)

        def scale(v, unit=unit):
            return tui.fmt(v, 1) if unit == "ms" else tui._si(v, "").strip()

        rows.append(
            [label, show(cur), kind, vals[-256:], None if top is None else scale(top)]
        )
    extra = (
        f"rates over {fab['interval_s']:.1f} s polls" if fab.get("interval_s") else ""
    )
    return {
        "state": "ok",
        "summary": [summary, "cyan" if leafs else "amber"],
        "title_extra": extra,
        "rows": rows,
    }


def strip(st: dict) -> str:
    """The instruments on a short terminal: one readout line each."""
    sc, an = st.get("scope"), st.get("analyzer")
    left = "7 scope: " + (
        f"{'RUN' if sc.running else 'STOP'} {sc.status}, {len(sc.segments)} segments"
        if sc is not None
        else "not running"
    )
    right = "8 spectrum: " + (
        f"{an.sweeps} sweeps" if an is not None and an.last is not None else "waiting"
    )
    return f"{left}   {right}   (focus 7 or 8, z to open)"


# ----------------------------------------------------------------- 7 scope
def scope(st: dict, now: float, gw: int, gh: int, zoom: bool = False) -> dict:
    """The scope reduced to a graticule ``gw`` x ``gh`` cells: per channel the
    min/max of each of ``2 gw`` sub-columns in divisions (0 = bottom edge,
    VDIV = top), the persistence cells, references, trigger marks, and the
    calibration (ticks, legend, notes)."""
    sc = st["scope"]
    sel = st.get("sel_ch", 0)
    tg = sc.trigger
    run = ("RUN", "green") if sc.running else ("STOP", "red")
    stat = {
        "trig'd": ("Trig'd", "green"),
        "auto": ("Auto", "amber"),
        "ready": ("Ready", "amber"),
        "armed": ("Armed", "cyan"),
        "stop": ("Stop", "red"),
    }[sc.status]
    slope = {"rising": "↑", "falling": "↓", "either": "↕"}[tg.slope]
    src_unit = SIGNALS[tg.source].unit
    status = [
        [f" {run[0]} ", run[1]],
        [f"{stat[0]} ", stat[1]],
        [f"H {_per_div(sc.s_per_div, 's')}  ", None],
        [f"T {tg.source} {slope} {_fmt(tg.level, src_unit)} ", "cyan"],
        [f"[{tg.kind} {tg.mode}] ", None],
        [f"[{sc.acquire}] ", None],
        [
            f"[persist {'inf' if sc.decay >= 1 else 'off' if sc.decay <= 0 else 'on'}]",
            "dim",
        ],
    ]
    if st.get("history") is not None:
        status.append([f"  HISTORY {st['history'] + 1}/{len(sc.segments)}", "purple"])
    out = {"status": status, "gw": gw, "gh": gh, "vdiv": VDIV, "hdiv": HDIV}
    if gw < 2 or gh < 2:
        return out
    if st.get("fft"):
        out["fft"] = _fft(st, now, gw)
        if zoom:
            out["softkeys"] = list(SCOPE_SOFTKEYS)
        return out

    channels = []
    for ci, ch in enumerate(sc.channels):
        role = SCOPE_COLORS[ci]
        entry = {"n": ci + 1, "signal": ch.signal, "role": role, "on": ch.on}
        if ch.on:
            cols = sc.columns(ch.signal, gw * 2, now)
            entry["cols"] = [
                (
                    None
                    if c is None
                    else [round(ch.to_div(c[0]), 4), round(ch.to_div(c[1]), 4)]
                )
                for c in cols
            ]
            entry["ground_div"] = round(ch.to_div(0.0), 4)
            persist = []
            if sc.decay > 0:
                grid = sc.persistence(ch, gw, gh, now)
                mx = grid.max() if grid.size else 0
                if mx > 0:
                    rr, cc = np.nonzero(grid / mx > 0.02)
                    for r_, c_ in zip(rr.tolist(), cc.tolist()):
                        # grid rows count up from the bottom; the client wants
                        # rows from the top of the graticule interior
                        persist.append(
                            [gh - 1 - r_, c_, 2 if grid[r_, c_] / mx >= 0.5 else 1]
                        )
            entry["persist"] = persist
            ref = sc.references.get(ch.signal)
            if ref is not None:
                entry["ref"] = [round(ch.to_div(ref[0]), 4), f" {ref[1]} "]
        channels.append(entry)
    out["channels"] = channels
    out["sel"] = sel + 1

    t0, t1 = sc.window(now)
    tsrc = next((c for c in sc.channels if c.signal == tg.source), None)
    if tg.kind != "logic" and tsrc is not None:
        out["trigger_level"] = {
            "div": round(tsrc.to_div(tg.level), 4),
            "role": SCOPE_COLORS[sc.channels.index(tsrc)],
        }
    rec = sc.record
    if rec is not None and rec.trigger_t is not None and (t0, t1) == (rec.t0, rec.t1):
        out["trigger_x"] = (rec.trigger_t - t0) / (t1 - t0)

    # calibration: the selected channel's scale on y, seconds on x, a legend
    chans = [(i, c) for i, c in enumerate(sc.channels) if c.on]
    if chans:
        ci, ch = next(((i, c) for i, c in chans if i == sel), chans[0])
        out["yunit"] = [SIGNALS[ch.signal].unit or "ratio", SCOPE_COLORS[ci]]
        out["yticks"] = [
            [k, scope_tick((k - VDIV / 2 - ch.position) * ch.scale)]
            for k in range(VDIV + 1)
        ]
        out["legend"] = [
            [
                f" {i + 1} {c.signal} {scope_tick(c.scale)} "
                f"{SIGNALS[c.signal].unit or 'ratio'}/div ",
                SCOPE_COLORS[i],
            ]
            for i, c in chans
        ]
    out["xticks"] = [
        [i / HDIV, "0 s" if i == HDIV else scope_tick(-(HDIV - i) * sc.s_per_div)]
        for i in range(HDIV + 1)
    ]
    out["notes"] = _scope_notes(sc)
    if zoom:
        out["side"] = _scope_side(sc, sel)
        out["meas"] = _scope_meas(sc, now)
        out["softkeys"] = list(SCOPE_SOFTKEYS)
    return out


def _scope_notes(sc) -> list:
    lines = []
    for ci, ch in enumerate(sc.channels):
        if not ch.on:
            continue
        spec = SIGNALS[ch.signal]
        unit = spec.unit or "ratio"
        lines.append(
            [
                f"{ci + 1} {COLOR_WORD.get(SCOPE_COLORS[ci], SCOPE_COLORS[ci])}: "
                f"{ch.signal}, {spec.description} ({unit}; "
                f"{_per_div(ch.scale, spec.unit)})",
                SCOPE_COLORS[ci],
            ]
        )
    lines.append(
        [
            f"x: time, newest at the right edge; {_per_div(sc.s_per_div, 's')}, "
            f"{_fmt(sc.s_per_div * HDIV, 's')} across. y: each channel on its own "
            "scale, its zero marked by its number on the left edge",
            "dim",
        ]
    )
    lines.append(
        [
            "▼ trigger point   ◀ trigger level   ░ where values fell recently   "
            "╌ reference line   i hides these notes",
            "dim",
        ]
    )
    return lines


def _scope_side(sc, sel: int) -> list:
    tg = sc.trigger
    src_unit = SIGNALS[tg.source].unit
    lines = []
    for ci, ch in enumerate(sc.channels):
        spec = SIGNALS[ch.signal]
        mark = ">" if ci == sel else " "
        lines.append(
            [f"{mark}CH{ci + 1} {ch.signal}", SCOPE_COLORS[ci] if ch.on else "dim"]
        )
        lines.append(
            [
                (
                    f"  {_per_div(ch.scale, spec.unit)}  pos {ch.position:+.1f}"
                    if ch.on
                    else "  off"
                ),
                None if ch.on else "dim",
            ]
        )
    lines.append(["", None])
    lines.append(["trigger", "bold"])
    rows = [
        f"{tg.kind}  {tg.mode}",
        f"src {tg.source}  {tg.slope}",
        f"level {_fmt(tg.level, src_unit)}",
        f"holdoff {tg.holdoff_s:g}s  pre {tg.position:.0%}",
    ]
    if tg.kind == "logic":
        rows[1:3] = [" AND ".join(f"{a}{o}{b}" for a, o, b in tg.conditions)]
    lines += [[f" {r}", None] for r in rows]
    lines.append(["", None])
    lines.append([f"segments {len(sc.segments)}", "purple"])
    return lines


def _scope_meas(sc, now: float) -> list:
    out = []
    for ch in [c for c in sc.channels if c.on][:2]:
        m = sc.measure(ch.signal, now)
        unit = SIGNALS[ch.signal].unit
        idx = sc.channels.index(ch)
        if m.get("n"):
            txt = (
                f"CH{idx + 1} {ch.signal}: mean {_fmt(m['mean'], unit)}  "
                f"min {_fmt(m['min'])}  max {_fmt(m['max'])}  "
                f"pk-pk {_fmt(m['pk-pk'])}  σ {_fmt(m['std'])}  "
                f"p99 {_fmt(m['p99'])}  n {m['n']}"
            )
            stt = sc.statistics(ch.signal, "max")
            if stt["count"]:
                txt += (
                    f"   | max over {stt['count']} acq: μ {_fmt(stt['mean'])} "
                    f"σ {_fmt(stt['std'])}"
                )
        else:
            txt = f"CH{idx + 1} {ch.signal}: no samples in the window"
        out.append([txt, SCOPE_COLORS[idx]])
    ms = sc.mask_summary()
    if ms:

        def lim(x):
            return "" if x is None else f"{x:g}"

        txt = "  ".join(
            f"mask {sig} {lim(d['limits'][0])}..{lim(d['limits'][1])}: "
            f"{d['violations']} fail / {d['of']}"
            for sig, d in ms.items()
        )
        out.append(
            [txt, "red" if any(d["violations"] for d in ms.values()) else "green"]
        )
    return out


def _fft(st: dict, now: float, gw: int) -> dict:
    sc = st["scope"]
    ci = st.get("sel_ch", 0)
    ch = sc.channels[ci]
    head = [
        f" FFT CH{ci + 1} {ch.signal}  Lomb-Scargle (irregular arrivals)  "
        f"window {_per_div(sc.s_per_div, 's')} x {HDIV}  {FFT_DB_SPAN:g} dB range",
        "cyan",
    ]
    res = sc.periodogram(ch.signal, now, nfreq=gw * 2)
    if res is None:
        return {
            "head": head,
            "message": ["fewer than 8 samples in the window", "amber"],
        }
    f, p = res
    pdb = 10 * np.log10(np.maximum(p, 1e-300))
    top = float(pdb.max())
    fracs = []
    for sx in range(gw * 2):
        i = min(len(f) - 1, int(sx / (gw * 2) * len(f)))
        fracs.append(
            round(
                float(
                    max(0.0, min(0.999, (pdb[i] - (top - FFT_DB_SPAN)) / FFT_DB_SPAN))
                ),
                4,
            )
        )
    k = int(np.argmax(p))
    pk = float(f[k])
    return {
        "head": head,
        "role": SCOPE_COLORS[ci],
        "fracs": fracs,  # 0 = bottom, 1 = top of the graticule
        "caption": [
            f"peak {pk:.3g} Hz (period {1 / pk:.3g} s), "
            f"{pdb[k] - np.median(pdb):+.1f} dB over the median   |   "
            f"0 .. {float(f[-1]):.3g} Hz",
            SCOPE_COLORS[ci],
        ],
    }


# ----------------------------------------------------------------- 8 spectrum
def spectrum(st: dict, gw: int, gh: int, zoom: bool = False, wf_rows: int = 0) -> dict:
    an = st["analyzer"]
    a, b = an.span()
    s = an.last
    run = ("RUN", "green") if an.running else ("STOP", "red")
    status = [
        [f" {run[0]} ", run[1]],
        [f"SPECTRUM sweeps {an.sweeps}  ", "purple"],
        [f"Ref {an.ref_db:g} dB  {an.db_div:g} dB/div  ", None],
        [f"Start {a}  Stop {b}  ", None],
        [
            (
                (
                    f"REF drift {100 * an.drift():.1f}%  "
                    if an.drift() is not None
                    else "REF  "
                )
                if an.reference is not None
                else ""
            ),
            "purple",
        ],
        [f"[{an.detector} detector] ", "dim"],
    ]
    if s is not None and b > a and gw * 2 < (b - a):
        status.append([f"[{(b - a) / (gw * 2):.1f} dirs/dot] ", "dim"])
    out = {"status": status, "gw": gw, "gh": gh, "vdiv": VDIV, "hdiv": HDIV}
    if s is None:
        out["reason"] = [
            st.get("spectrum_reason") or "waiting for the first sweep...",
            "amber",
        ]
        return out
    sel = st.get("sel_trace", 0)
    bottom = an.ref_db - an.db_div * VDIV
    out.update(
        {
            "ok": True,
            "bottom": bottom,
            "ref_db": an.ref_db,
            "db_div": an.db_div,
            "delta": any(t.mode == "delta" for t in an.traces),
            "limit_db": an.limit_db(),
            "sel": sel + 1,
        }
    )
    out["traces"] = [
        {
            "n": ti + 1,
            "role": SPEC_COLORS[ti],
            "mode": tr.mode,
            "cols": [
                None if v is None else round(v, 3) for v in an.columns(tr, gw * 2)
            ],
        }
        for ti, tr in enumerate(an.traces)
    ]
    marks = []
    for j, m in enumerate(an.markers[:2]):
        d = an.traces[sel].data
        if a <= m < b and d is not None:
            marks.append([(m - a) / max(b - a, 1), float(d[m]), "◆" if j == 0 else "◇"])
    out["markers"] = marks
    out["yticks"] = [
        [bottom + an.db_div * k, spec_tick(bottom + an.db_div * k)]
        for k in range(VDIV + 1)
    ]
    out["xticks"] = [
        [i / HDIV, f"{a + (b - a) * i / HDIV:.0f}"] for i in range(HDIV + 1)
    ]
    out["legend"] = [
        [
            f" T{i + 1} {LABEL.get(tr.source, tr.source)} "
            f"({'dB rel. ref' if tr.mode == 'delta' else 'dB'}) ",
            SPEC_COLORS[i],
        ]
        for i, tr in enumerate(an.traces)
        if tr.mode != "blank" and tr.data is not None
    ]
    out["notes"] = _spectrum_notes(an)
    lc = an.limit_check()
    real, pred = s.realised_total, s.predicted_total
    ro = [f"eff rank {s.effective_rank:.1f}/{s.sens.size}"]
    if pred is not None:
        ro.append(f"predicted D {pred:.3g}")
    ro.append(
        "realised D "
        + (f"{real:.3g}" if real is not None else "- (no codec to reconstruct with)")
    )
    if real and pred:
        ro.append(f"gap {10 * np.log10(real / pred):+.1f} dB")
    if lc["passed"] is not None:
        ro.append(
            f"limit {'PASS' if lc['passed'] else 'FAIL'} "
            f"{len(lc['fail'])}/{lc['checked']} dirs over"
        )
    for mk in an.marker_readout(sel):
        ro.append(
            f"M{mk['marker']} dir {mk['direction']} {_db(mk['db'])}"
            + (
                f" Δ {mk['delta_db']:+.1f} dB / {mk['delta_dirs']:+d} dirs"
                if "delta_db" in mk
                else ""
            )
        )
    out["readout"] = ["   ".join(ro), "red" if lc["passed"] is False else None]
    if zoom:
        out["names"] = "  ".join(
            f"T{i + 1} {tr.source}:{tr.mode}" for i, tr in enumerate(an.traces)
        )
        out["softkeys"] = list(SPEC_SOFTKEYS)
        if wf_rows > 0:
            out["waterfall"] = _waterfall(an, gw, wf_rows, bottom)
    return out


def _spectrum_notes(an) -> list:
    lines = []
    for i, tr in enumerate(an.traces):
        if tr.mode == "blank":
            continue
        lines.append(
            [
                f"T{i + 1} {SPEC_COLORS[i]}: {LABEL.get(tr.source, tr.source)}, "
                f"{MEANS.get(tr.source, '')} [{tr.mode}]",
                SPEC_COLORS[i],
            ]
        )
    lines.append(
        [
            "x: eigendirections of the observer's read operator, strongest first. "
            f"y: power in dB, {an.ref_db:g} dB at the top, "
            f"{an.db_div:g} dB per division",
            "dim",
        ]
    )
    if an.limit_db() is not None:
        lines.append(
            [
                "limit line: the water level; a direction above it is worth bits, "
                "one below it gets none.   i hides these notes",
                "dim",
            ]
        )
    else:
        lines.append(["i hides these notes", "dim"])
    return lines


def _waterfall(an, gw: int, rows: int, bottom: float) -> dict:
    a, b = an.span()
    wr = an.waterfall_range() or (bottom, an.ref_db)
    levels = []
    for spec in list(an.waterfall)[-rows:][::-1]:
        seg = spec[a:b]
        if not seg.size:
            levels.append([0] * gw)
            continue
        edges = np.linspace(0, seg.size, gw + 1)
        row = []
        for i in range(gw):
            lo, hi = int(edges[i]), max(int(edges[i + 1]), int(edges[i]) + 1)
            v = float(seg[lo : min(hi, seg.size)].max())
            f = (v - wr[0]) / (wr[1] - wr[0]) if wr[1] > wr[0] else 0.0
            row.append(max(0, min(4, int(f * 4 + 0.5))))
        levels.append(row)
    return {
        "label": f"waterfall: {LABEL.get(an.waterfall_source, '')}, newest at top, "
        f"{wr[0]:.0f}..{wr[1]:.0f} dB",
        "levels": levels,  # 0..4, the client's ramp " ░▒▓█"
    }


# ----------------------------------------------------------------- overlays
def fabric_screen(st: dict, w: int, h: int) -> dict:
    """The full NATS instrument (zoom on panel 9) as spans, h x w."""
    from . import fabric_view

    hist = st.get("fabric_hist") or fabric_view.History()
    cv = fabric_view.frame(st.get("fabric"), hist, w, h, tui.UNICODE)
    return {"spans": spans_of(cv)}


def inspect_sheet(st: dict, w: int, h: int) -> dict:
    """The inspect overlay for ``st["inspected"]`` (and ``st["replay"]``): the
    sheet's rectangle on a ``w`` x ``h`` screen and its spans."""
    if not st.get("inspected"):
        return {}
    cv = tui.Canvas(w, h)
    tui._overlay_inspect(cv, st, tui.UNICODE)
    # the rectangle tui._sheet placed it in (same arithmetic as _overlay_inspect)
    hh, ww = min(h - 2, cv.h - 2), min(min(w - 4, 110), cv.w - 4)
    y, x = max(1, (h - hh) // 2), max(2, (w - ww) // 2)
    return {"rect": [y, x, hh, ww], "spans": spans_of(cv, y, x, hh, ww)}


def hello() -> dict:
    from . import scope_view, spectrum_view

    return {
        "protocol": PROTOCOL,
        "keys": [list(k) for k in tui.KEYS],
        "keys_scope": [list(k) for k in scope_view.HELP],
        "keys_spectrum": [list(k) for k in spectrum_view.HELP],
        "zoomable": {str(k): v for k, v in tui.ZOOMABLE.items()},
        "titles": {
            "1": "1 system",
            "2": "2 throughput / latency",
            "3": "3 pipeline  ms/query",
            "4": "4 readscope  observer / certificate / provenance",
            "5": "5 index",
            "6": "6 query stream  Up/Down select, Enter inspect",
            "7": "7 scope  query signals in time  (z: full controls)",
            "8": "8 spectrum  what the observer reads, per direction"
            "  (z: full controls)",
            "9": "9 NATS fabric",
        },
        "help_note": "meas = timed directly, samp = from a sample, deri = computed; "
        "'-' = unavailable",
    }

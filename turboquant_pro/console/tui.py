"""Terminal console: btop-style panels over the same session the web view uses.

``tqp console`` runs this in the terminal (over SSH as well as locally): no browser, no
socket, no token. :func:`frame` turns the session state into a character grid with
semantic colours; it is pure, so the layout is tested at any terminal size without a
terminal. :func:`run` paints that grid with curses and handles the keys.
"""

from __future__ import annotations

import json
import locale
import time
from collections import deque

UNICODE = {
    "h": "─",
    "v": "│",
    "tl": "┌",
    "tr": "┐",
    "bl": "└",
    "br": "┘",
    "spark": " ▁▂▃▄▅▆▇█",
    "full": "█",
    "empty": "░",
    "up": "▲",
    "down": "▼",
    "dot": "·",
}
ASCII = {
    "h": "-",
    "v": "|",
    "tl": "+",
    "tr": "+",
    "bl": "+",
    "br": "+",
    "spark": " .:-=+*#%",
    "full": "#",
    "empty": ".",
    "up": "^",
    "down": "v",
    "dot": ".",
    "ascii": True,
}

MIN_W, MIN_H = 80, 24  # btop's own minimum; three panels abreast need it

KEYS = [
    ("q", "quit"),
    ("Tab / 1-9", "focus a panel (7 scope, 8 spectrum, 9 NATS)"),
    ("z", "zoom: the focused panel full screen with its own controls (Esc back)"),
    ("Up/Down j/k", "select a query"),
    ("Enter", "inspect the selected query"),
    ("r", "replay the query and compare"),
    ("e", "export the session as JSON to the current directory"),
    ("P", "snapshot: write the screen as it is now to a .txt file"),
    ("p", "pause / resume the display (the workload keeps running)"),
    ("i", "notes on the screen: what each graph shows"),
    ("?", "this help"),
    ("Esc", "close an overlay"),
]


def fmt(v, d: int = 2) -> str:
    if v is None:
        return "-"
    return f"{v:.0f}" if abs(v) >= 1000 else f"{v:.{d}f}"


def spark(values, width: int, g: dict) -> str:
    """The last ``width`` values as block characters scaled to their own maximum."""
    vals = [v for v in list(values)[-width:]]
    real = [v for v in vals if v is not None]
    if len(real) < 2:
        return "collecting".ljust(width)[:width]
    top = max(real) or 1.0
    ramp = g["spark"]
    out = "".join(
        " " if v is None else ramp[max(1, min(len(ramp) - 1, round(v / top * 8)))]
        for v in vals
    )
    return out.rjust(width)


def bar(frac: float, width: int, g: dict) -> str:
    n = max(0, min(width, round(frac * width)))
    return g["full"] * n + g["empty"] * (width - n)


class Canvas:
    """A W x H grid of (char, colour) cells; writes are clipped, never raise."""

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.cells = [[(" ", None) for _ in range(w)] for _ in range(h)]

    def put(self, y: int, x: int, text: str, color: str | None = None) -> None:
        if not 0 <= y < self.h:
            return
        for i, ch in enumerate(str(text)):
            if 0 <= x + i < self.w:
                self.cells[y][x + i] = (ch, color)

    def box(
        self,
        y: int,
        x: int,
        h: int,
        w: int,
        title: str,
        g: dict,
        color: str = "dim",
        focus: bool = False,
    ) -> None:
        if h < 2 or w < 4:
            return
        c = "cyan" if focus else color
        self.put(y, x, g["tl"] + g["h"] * (w - 2) + g["tr"], c)
        for r in range(1, h - 1):
            self.put(y + r, x, g["v"], c)
            self.put(y + r, x + w - 1, g["v"], c)
        self.put(y + h - 1, x, g["bl"] + g["h"] * (w - 2) + g["br"], c)
        if title:
            self.put(
                y, x + 2, f" {title} "[: max(0, w - 4)], "bold" if not focus else "cyan"
            )

    def text(self) -> list:
        return ["".join(ch for ch, _ in row) for row in self.cells]

    def blit(self, src: Canvas, y: int, x: int) -> None:
        """Copy ``src`` onto this canvas with its top-left at (y, x), clipped."""
        for r, row in enumerate(src.cells):
            yy = y + r
            if not 0 <= yy < self.h:
                continue
            for c, cell in enumerate(row):
                if 0 <= x + c < self.w:
                    self.cells[yy][x + c] = cell


def _brand(cv: Canvas) -> None:
    """The decorative title, only where the status line left blank space: status
    information always wins over decoration."""
    brand = " TurboQuant console  q quit  ? keys "
    x = cv.w - len(brand)
    if x > 0 and all(ch == " " for ch, _ in cv.cells[0][x - 1 :]):
        cv.put(0, x, brand, "dim")


def _readings(snap: dict) -> dict:
    return {r["name"]: r for r in snap.get("readings", [])}


def _stage(t: dict, name: str):
    s = next((s for s in t.get("stages", []) if s["name"] == name), None)
    return s["ms"] if s else None


_VALIDITY_COLOR = {"VALID": "green", "STALE": "red", "INCONCLUSIVE": "amber"}


def _validity_lines(v: dict) -> list:
    """The certificate's validity as panel lines. Four states, each a word, so none
    depends on colour: VALID, STALE (a check failed), INCONCLUSIVE (a check's own
    noise could reach its bar: no verdict) and UNCHECKED (nothing could be checked,
    which is never a pass)."""
    status = v.get("status", "UNCHECKED")
    out = [
        (
            "validity",
            f"{status}  {v.get('action') or ''}".rstrip(),
            _VALIDITY_COLOR.get(status, "dim"),
        )
    ]
    if v.get("reason"):
        out.append(("  why", v["reason"]))
    data = v.get("data") or {}
    if data.get("rows"):
        out.append(
            (
                "  checked on",
                f"{data['rows']} of {data['of']} rows of {data['source']} "
                f"({data['kind']})",
            )
        )
    elif data.get("reason"):
        out.append(("  checked on", data["reason"]))
    return out


def frame(st: dict, w: int, h: int, g: dict = UNICODE) -> Canvas:
    """The whole screen for state ``st``: one btop-style grid of numbered panels, or
    one panel maximised (``st["zoom"]``: "scope", "spectrum", "fabric", or None).

    ``st`` holds ``snap`` (a snapshot document), ``traces`` (newest last),
    ``readscope``, ``qps_hist`` / ``p95_hist``, ``scope`` / ``analyzer`` (the
    instruments), ``fabric`` / ``fabric_hist`` (the NATS source, if attached),
    ``sel``, ``focus`` (1-8), ``paused``, ``overlay`` (None | "inspect" | "help"),
    ``inspected``, ``replay``, ``message`` and ``annotate``."""
    cv = Canvas(w, h)
    if w < MIN_W or h < MIN_H:
        cv.put(
            0,
            0,
            f"terminal {w}x{h} is too small: {MIN_W}x{MIN_H} at least"[:w],
            "amber",
        )
        return cv
    zoom = zoom_of(st)
    if zoom:
        _zoomed(cv, st, g, zoom)
    else:
        _grid(cv, st, g)
    if st.get("message"):
        cv.put(h - 1, 2, f" {st['message']} "[: w - 4], "amber")
    if st.get("overlay") == "help":
        _overlay_help(cv, g, _help_for(zoom))
    elif st.get("overlay") == "inspect" and st.get("inspected"):
        _overlay_inspect(cv, st, g)
    return cv


# Panels, by number. 7 and 8 are the instruments; zooming (z) maximises the focused one.
PANELS = {
    1: "system",
    2: "throughput",
    3: "pipeline",
    4: "readscope",
    5: "index",
    6: "queries",
    7: "scope",
    8: "spectrum",
    9: "nats",
}
ZOOMABLE = {7: "scope", 8: "spectrum", 9: "fabric"}


def zoom_of(st: dict):
    """The maximised panel, or None for the grid. ``st["zoom"]`` when set; else
    the older ``view`` field (a setup file's): "scope" / "spectrum" / "overview"."""
    if "zoom" in st:
        return st["zoom"]
    v = st.get("view")
    return v if v in ("scope", "spectrum") else None


def _help_for(zoom):
    if zoom == "scope":
        from . import scope_view

        return scope_view.HELP
    if zoom == "spectrum":
        from . import spectrum_view

        return spectrum_view.HELP
    return KEYS


def _zoomed(cv: Canvas, st: dict, g: dict, zoom: str) -> None:
    """One panel on the whole screen, with its own controls."""
    if zoom == "scope":
        from . import scope_view

        scope_view.render(cv, st, g, st.get("now") or time.time())
    elif zoom == "spectrum":
        from . import spectrum_view

        spectrum_view.render(cv, st, g)
        spectrum_view.render_notes(cv, st, g)
    elif zoom == "fabric":
        from . import fabric_view

        hist = st.get("fabric_hist") or fabric_view.History()
        cv.blit(fabric_view.frame(st.get("fabric"), hist, cv.w, cv.h - 1, g), 0, 0)
        cv.put(cv.h - 1, 0, "[Esc back to the grid]  [i notes]  [q quit]", "dim")
    _brand(cv)


def _grid(cv: Canvas, st: dict, g: dict) -> None:
    """The btop grid. Rows: system / throughput / pipeline; the two instruments
    (scope, spectrum) when the terminal has room for them; readscope / index; the
    query stream. On a small terminal the instruments collapse to a one-line
    readout each (z still opens them full screen)."""
    w, h = cv.w, cv.h
    snap = st.get("snap") or {}
    _header(cv, st, snap)
    top_h = 9 if h >= 30 else 7
    if st.get("fabric") is not None and h >= 44:
        mid_h = 3 + len(NATS_ROWS)  # room for every NATS metric in panel 9
    else:
        mid_h = 10 if h >= 44 else 8 if h >= 30 else 6
    avail = h - 1 - top_h - mid_h
    inst_h = avail - max(6, avail // 3) if avail >= 22 else 0
    y = 1
    c = w // 3
    _p_system(cv, st, snap, y, 0, top_h, c, g)
    _p_throughput(cv, st, snap, y, c, top_h, c, g)
    _p_pipeline(cv, st, snap, y, 2 * c, top_h, w - 2 * c, g)
    y += top_h
    if inst_h:
        half = w // 2
        _p_instrument(cv, st, g, "scope", y, 0, inst_h, half)
        _p_instrument(cv, st, g, "spectrum", y, half, inst_h, w - half)
        y += inst_h
    else:
        _instrument_strip(cv, st, y, w)
        y += 1
    _p_readscope(cv, st, y, 0, mid_h, c, g)
    _p_index(cv, st, snap, y, c, mid_h, c, g)
    _p_nats(cv, st, y, 2 * c, mid_h, w - 2 * c, g)
    y += mid_h
    _p_queries(cv, st, y, 0, h - y, w, g)


def _header(cv: Canvas, st: dict, snap: dict) -> None:
    w = cv.w
    wl = snap.get("workload", {})
    traces = st.get("traces", [])
    last = traces[-1] if traces else {}
    x = 0
    cv.put(0, x, " TurboQuant ", "bold")
    x += 12
    cv.put(0, x, "console ", "cyan")
    x += 8
    age = snap.get("last_trace_age_s")
    src = snap.get("sources") or {}
    if st.get("paused"):
        state = ("PAUSED", "amber")
    elif src and not src.get("index"):
        state = ("no index", "dim")
    elif age is not None and age < 3:
        state = ("live", "green")
    else:
        state = ("waiting", "amber") if age is None else (f"stale {age:.0f}s", "amber")
    hint = "q quit  ? keys  z zoom  i notes  P snap"
    t = snap.get("t")
    stamp = time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(t)) if t else "--:--:--Z"
    hint = f"{stamp}  {hint}"
    if w - len(hint) < 24:  # narrow: keep the time and the essential keys
        hint = f"{stamp}  q quit  ? keys  z zoom"
    room = w - len(hint) - 3
    labels = [
        (f"[{state[0]}]", state[1]),
        (f"[mode: {wl.get('mode', '-')}]", "green" if wl.get("rerank") else "amber"),
        (f"[scan: {last.get('scan_path') or '-'}]", "cyan"),
    ]
    obs = ((st.get("readscope") or {}).get("observer") or {}).get("reference")
    if obs:
        labels.append(
            (f"[observer: {obs.get('observer')} {obs.get('sha256', '')[:8]}]", "purple")
        )
    for label, col in labels:
        if x + len(label) <= room:
            cv.put(0, x, label, col)
            x += len(label) + 1
    cv.put(0, w - len(hint) - 1, hint, "dim")


def _focus(st, n):
    return st.get("focus", 0) == n


def _p_system(cv, st, snap, y, x, hh, ww, g):
    cv.box(y, x, hh, ww, "1 system", g, focus=_focus(st, 1))
    r = _readings(snap)
    items = [
        ("QPS", "search.qps", 1),
        ("p50", "search.latency_ms.p50", 2),
        ("p95", "search.latency_ms.p95", 2),
        ("p99", "search.latency_ms.p99", 2),
        ("agree", "search.rerank_agreement", 2),
        ("compress", "index.compression_ratio", 1),
        ("rows", "index.rows", 0),
        ("CPU", "process.cpu_percent", 0),
        ("RSS", "process.rss_mb", 0),
    ]
    nats_rows = 0  # the fabric has its own panel (9)
    inner = ww - 2
    cols = 2 if inner >= 50 else 1
    rows_avail = hh - 2 - nats_rows
    cw = inner // cols
    for i, (label, name, d) in enumerate(items[: max(0, rows_avail) * cols]):
        rd = r.get(name)
        if rd is None:
            continue
        yy, xx = y + 1 + i % rows_avail, x + 1 + (i // rows_avail) * cw
        unit = (
            "" if rd["unit"] == "fraction" else rd["unit"].replace("queries/s", "q/s")
        )
        num = f"{fmt(rd['value'], d)} {unit}".rstrip()
        kind = rd["kind"][:4]
        right = xx + cw - 2
        cv.put(yy, xx + 1, label, "dim")
        cv.put(yy, right - len(kind), kind, "dim")
        cv.put(
            yy,
            right - len(kind) - 1 - len(num),
            num,
            "dim" if rd["value"] is None else None,
        )


def _si(v, unit: str) -> str:
    """A rate or size with a 1000-step prefix: 12.3 kB/s, 4.1 M/s."""
    if v is None:
        return "-"
    a = abs(v)
    for p, f in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if a >= f:
            return f"{v / f:.1f} {p}{unit}"
    return f"{v:.1f} {unit}" if a < 100 else f"{v:.0f} {unit}"


# Panel 9 rows: label, history key, unit, kind, and where the current value is.
NATS_ROWS = (
    ("msgs in", "in_msgs", "msg/s", "deri"),
    ("msgs out", "out_msgs", "msg/s", "deri"),
    ("bytes in", "in_bytes", "B/s", "deri"),
    ("bytes out", "out_bytes", "B/s", "deri"),
    ("leaf rtt", "leaf_rtt", "ms", "samp"),
    ("leaf msgs", "leaf_msgs", "msg/s", "deri"),
    ("leaf bytes", "leaf_bytes", "B/s", "deri"),
    ("connects", "connects", "/min", "deri"),
    ("pending", "pending", "B", "meas"),
    ("JS msgs", "js_msgs", "msg", "meas"),
)


def _p_nats(cv, st, y, x, hh, ww, g):
    """The NATS fabric (read-only, from the monitoring port): one calibrated row
    per metric (current value with unit and kind, a sparkline of the recent polls
    from 0 to the stated max) and a summary line. z opens the full instrument."""
    fab = st.get("fabric")
    title = "9 NATS fabric"
    if fab is not None and fab.get("interval_s"):
        title += f"  rates over {fab['interval_s']:.1f} s polls"
    cv.box(y, x, hh, ww, title, g, color="cyan", focus=_focus(st, 9))
    if fab is None:
        cv.put(y + 1, x + 2, "not attached: start with --nats URL"[: ww - 4], "dim")
        return
    if not fab.get("reachable"):
        cv.put(y + 1, x + 2, f"{fab['source']['url']}: UNREACHABLE"[: ww - 4], "red")
        return
    hist = (st.get("fabric_hist") or None) and st["fabric_hist"].series
    leafs = fab.get("leafs") or []
    srv = fab.get("server") or {}
    js = fab.get("jetstream") or {}
    summary = (
        f"NATS {len(leafs)} leaf, {len(fab.get('connections') or [])} clients, "
        f"{srv.get('subscriptions', '-')} subs, {srv.get('slow_consumers', '-')} slow"
    )
    more = f", JS {js.get('streams', '-')} streams"
    if len(summary) + len(more) <= ww - 4:
        summary += more
    cv.put(y + 1, x + 2, summary[: ww - 4], "cyan" if leafs else "amber")
    # columns: label (10), value with unit (11), kind (5), then the sparkline
    # and its scale ("max" in the row's own unit, bars from 0)
    val_w, max_w = 27, 11
    sw = max(0, ww - 4 - val_w - max_w)
    for i, (label, key, unit, kind) in enumerate(NATS_ROWS[: max(0, hh - 3)]):
        yy = y + 2 + i
        series = list(hist[key]) if hist and key in hist else []
        cur = series[-1] if series else None
        num = _si(cur, unit) if unit != "ms" else f"{fmt(cur, 1)} ms"
        cv.put(yy, x + 2, f"{label:<10}", "dim")
        cv.put(yy, x + 12, f"{num:>11}"[:11], None if cur is not None else "dim")
        cv.put(yy, x + 23, f" {kind}", "dim")
        if sw >= 4:
            cv.put(yy, x + 2 + val_w, spark(series, sw, g), "cyan")
            real = [v for v in series[-sw:] if v is not None]
            if len(real) >= 2:
                top = max(real)
                scale = _si(top, "").strip() if unit != "ms" else fmt(top, 1)
                cv.put(yy, x + 3 + val_w + sw, f"max {scale}"[: max_w - 1], "dim")


def _p_throughput(cv, st, snap, y, x, hh, ww, g):
    """Two sparklines, each calibrated: its name and unit, the current value, its
    scale (bars run from 0 at the baseline to the stated max), and the time span
    they actually cover (one sample per second, as many as fit)."""
    cv.box(y, x, hh, ww, "2 throughput / latency", g, focus=_focus(st, 2))
    r = _readings(snap)
    lab_w = 9  # scale labels right of the bars
    sw = max(4, ww - 4 - lab_w)
    rows = 2 if hh >= 9 else 1
    series = (
        ("QPS", "q/s", 1, "search.qps", "qps_hist", "cyan"),
        ("p95 latency", "ms", 2, "search.latency_ms.p95", "p95_hist", "purple"),
    )
    yy = y + 1
    for name, unit, d, key, hist_key, col in series:
        hist = list(st.get(hist_key, []))
        now_v = r.get(key, {}).get("value")
        cv.put(yy, x + 2, f"{name} {fmt(now_v, d)} {unit}", col)
        real = [v for v in hist[-sw:] if v is not None]
        top = max(real) if real else None
        line = spark(hist, sw, g)
        for k in range(rows):
            cv.put(yy + 1 + k, x + 2, line, col)
        if top is not None and len(real) >= 2:
            cv.put(yy + 1, x + 3 + sw, f"{fmt(top, d)}"[: lab_w - 1], "dim")
            cv.put(yy + rows, x + 3 + sw, f"0 {unit}"[: lab_w - 1], "dim")
        yy += rows + 1
    n = min(sw, len(list(st.get("qps_hist", []))))
    if yy < y + hh - 1:
        left = f"-{n} s" if n else "-"
        cv.put(yy, x + 2, left, "dim")
        cv.put(yy, x + 2 + sw - 3, "now", "dim")
        if st.get("annotate", True):
            mid = " 1 sample/s, bar height 0..max "
            cv.put(yy, x + 2 + max(len(left) + 1, (sw - len(mid)) // 2), mid, "dim")


def _p_pipeline(cv, st, snap, y, x, hh, ww, g):
    cv.box(y, x, hh, ww, "3 pipeline  ms/query", g, focus=_focus(st, 3))
    r = _readings(snap)
    wl = snap.get("workload", {})
    stages = [
        (s, r.get(f"search.stage_ms.{s}", {}).get("value"))
        for s in ("encode", "scan", "rerank")
    ]
    mx = max([v for _, v in stages if v is not None] or [1e-9])
    bw = max(4, ww - 20)
    for i, (s, v) in enumerate(stages):
        yy = y + 1 + i * (2 if hh >= 9 else 1)
        cv.put(yy, x + 2, f"{s:<7}", None)
        cv.put(yy, x + 9, bar(0 if v is None else v / mx, bw, g), "teal")
        cv.put(yy, x + 10 + bw, f"{fmt(v, 3):>7}", None)
    k, rr = wl.get("k", "-"), wl.get("rerank", 0)
    note = (
        "no index attached"
        if not wl
        else f"top-{k} from {k * rr} reranked" if rr else f"top-{k}, approximate"
    )
    cv.put(y + hh - 2, x + 2, note[: ww - 4], "dim")


def _p_instrument(cv, st, g, which, y, x, hh, ww):
    """An instrument drawn small in its panel. Its full controls are on the zoomed
    screen (focus the panel, press z)."""
    n = 7 if which == "scope" else 8
    title = (
        f"{n} scope  query signals in time"
        if which == "scope"
        else f"{n} spectrum  what the observer reads, per direction"
    )
    cv.box(y, x, hh, ww, f"{title}  (z: full controls)", g, focus=_focus(st, n))
    sub = Canvas(ww - 2, hh - 2)
    if which == "scope" and "scope" in st:
        from . import scope_view

        scope_view.render(sub, st, g, st.get("now") or time.time(), compact=True)
    elif which == "spectrum" and "analyzer" in st:
        from . import spectrum_view

        spectrum_view.render(sub, st, g, compact=True)
        spectrum_view.render_notes(sub, st, g)
    else:
        sub.put(1, 1, "instrument not running", "dim")
    cv.blit(sub, y + 1, x + 1)


def _instrument_strip(cv, st, y, w):
    """The instruments on a small terminal: one readout line each."""
    sc, an = st.get("scope"), st.get("analyzer")
    left = "7 scope: " + (
        f"{'RUN' if sc.running else 'STOP'} {sc.status}, {len(sc.segments)} segments"
        if sc is not None
        else "not running"
    )
    right = "8 spectrum: " + (
        f"{an.sweeps} sweeps" if an is not None and an.last is not None else "waiting"
    )
    cv.put(y, 1, f"{left}   {right}   (focus 7 or 8, z to open)"[: w - 2], "dim")


def _p_readscope(cv, st, y, x, hh, ww, g):
    cv.box(
        y,
        x,
        hh,
        ww,
        "4 readscope  observer / certificate / provenance",
        g,
        color="purple",
        focus=_focus(st, 4),
    )
    rs = st.get("readscope") or {}
    obs = (rs.get("observer") or {}).get("reference")
    lines = []
    if obs:
        cons = ", ".join(
            str(c.get("metric") or c.get("name") or c)
            for c in obs.get("consumers") or []
        )
        lines += [
            ("observer", f"{obs.get('observer')}  target {obs.get('target')}"),
            ("sha256", obs.get("sha256", "")),
            ("consumers", cons or "-"),
        ]
    else:
        lines.append(
            (
                "observer",
                "none loaded (--observer X.tqo): results are not tied to a declared "
                "reader",
            )
        )
    cert = rs.get("certificate")
    if cert:
        cc = cert.get("certificate", {})
        lines.append(
            (
                "certificate",
                f"{'PASSED' if cert.get('passed') else 'NOT PASSED'}"
                f"  tau floor {cc.get('tau_floor')}",
            )
        )
        lines.extend(_validity_lines(rs.get("validity") or {}))
    for p in rs.get("provenance", []):
        val = p.get("sha256") or json.dumps({k: v for k, v in p.items() if k != "step"})
        lines.append((p["step"][:12], val))
    for i, (kk, vv, *col) in enumerate(lines[: hh - 2]):
        cv.put(y + 1 + i, x + 2, f"{kk:<12}", "purple")
        cv.put(y + 1 + i, x + 15, str(vv)[: ww - 17], col[0] if col else None)


def _p_index(cv, st, snap, y, x, hh, ww, g):
    cv.box(y, x, hh, ww, "5 index", g, focus=_focus(st, 5))
    ie = snap.get("index", {})
    wl = snap.get("workload", {})
    if not ie:
        cv.put(y + 1, x + 2, "no index attached (--index or --demo)"[: ww - 4], "dim")
        return
    kvs = [
        ("kind", ie.get("kind")),
        ("rows", ie.get("rows")),
        ("dim", ie.get("dim")),
        ("metric", ie.get("metric")),
        ("bytes/row", ie.get("stored_bytes_per_row")),
        ("kernel", "AVX2" if ie.get("kernel") else "numpy"),
        ("workload", f"{wl.get('target_qps', '-')} qps, k={wl.get('k', '-')}"),
    ]
    for i, (kk, vv) in enumerate(kvs[: hh - 2]):
        cv.put(y + 1 + i, x + 2, f"{kk:<10}", "dim")
        cv.put(y + 1 + i, x + 13, str(vv)[: ww - 15], None)


def _p_queries(cv, st, y, x, hh, ww, g):
    traces = st.get("traces", [])
    cv.box(
        y,
        x,
        hh,
        ww,
        "6 query stream  Up/Down select, Enter inspect",
        g,
        focus=_focus(st, 6),
    )
    cols = [
        ("time", 12),
        ("trace", 16),
        ("row", 5),
        ("scan", 13),
        ("total", 8),
        ("encode", 8),
        ("scan ms", 8),
        ("rerank", 8),
        ("agree", 6),
    ]
    xs, xx = [], x + 2
    for name, cwid in cols:
        if xx + cwid > x + ww - 2:
            break
        xs.append((name, cwid, xx))
        xx += cwid + 1
    for name, cwid, xx in xs:
        cv.put(
            y + 1, xx, name.rjust(cwid) if cwid <= 8 and name != "row" else name, "dim"
        )
    rows = list(reversed(traces))[: max(0, hh - 3)]
    for i, t in enumerate(rows):
        vals = [
            t["started_utc"][11:23],
            t["id"],
            str(t["params"].get("workload_row", "-")),
            t.get("scan_path") or "-",
            fmt(t.get("total_ms"), 3),
            fmt(_stage(t, "encode"), 3),
            fmt(_stage(t, "scan"), 3),
            fmt(_stage(t, "rerank"), 3),
            fmt((t.get("results") or {}).get("rerank_agreement"), 2),
        ]
        sel = i == st.get("sel", 0)
        for (name, cwid, xx), v in zip(xs, vals):
            cv.put(
                y + 2 + i,
                xx,
                (v.rjust(cwid) if cwid <= 8 and name != "row" else v.ljust(cwid))[
                    :cwid
                ],
                "sel" if sel else None,
            )
        if sel:
            cv.put(y + 2 + i, x + 1, ">", "cyan")


def _sheet(cv: Canvas, title: str, g: dict, hh: int, ww: int):
    hh, ww = min(hh, cv.h - 2), min(ww, cv.w - 4)
    y, x = max(1, (cv.h - hh) // 2), max(2, (cv.w - ww) // 2)
    for r in range(hh):
        cv.put(y + r, x, " " * ww)
    cv.box(y, x, hh, ww, title, g, color="purple")
    return y, x, hh, ww


def _overlay_help(cv: Canvas, g: dict, keys: list = KEYS) -> None:
    y, x, hh, ww = _sheet(cv, "keys", g, len(keys) + 6, 76)
    for i, (k, what) in enumerate(keys):
        cv.put(y + 1 + i, x + 2, f"{k:<14}", "cyan")
        cv.put(y + 1 + i, x + 17, what[: ww - 19], None)
    note = (
        "meas = timed directly, samp = from a sample, deri = computed; "
        "'-' = unavailable"
    )
    cv.put(y + len(keys) + 2, x + 2, note[: ww - 4], "dim")


def _overlay_inspect(cv: Canvas, st: dict, g: dict) -> None:
    t = st["inspected"]
    y, x, hh, ww = _sheet(
        cv,
        f"query {t['id']}  (r replay, e export, Esc close)",
        g,
        cv.h - 2,
        min(cv.w - 4, 110),
    )
    row = y + 1
    info = [
        (
            "query",
            f"row {t['params'].get('workload_row', '-')}  sha256 "
            f"{t['input']['sha256'][:16]}  dim {t['input']['dim']}",
        ),
        ("params", json.dumps(t["params"])),
        ("scan path", f"{t.get('scan_path')}   total {fmt(t.get('total_ms'), 3)} ms"),
        ("observer", (t.get("observer") or {}).get("observer") or "none declared"),
    ]
    for k, v in info:
        cv.put(row, x + 2, f"{k:<10}", "purple")
        cv.put(row, x + 13, str(v)[: ww - 15], None)
        row += 1
    row += 1
    cv.put(row, x + 2, "stages", "bold")
    row += 1
    for s in t.get("stages", []):
        extra = f"  candidates {s['candidates']}" if "candidates" in s else ""
        cv.put(row, x + 4, f"{s['name']:<8}{fmt(s['ms'], 4):>9} ms{extra}", None)
        row += 1
    res = t.get("results") or {}
    row += 1
    if res.get("final"):
        cv.put(
            row,
            x + 2,
            f"results: approximate -> exact rerank  (agreement "
            f"{fmt(res.get('rerank_agreement'), 2)})",
            "bold",
        )
        row += 1
        cv.put(
            row,
            x + 4,
            f"{'rank':>4}  {'id':>7}  {'exact score':>12}  "
            f"{'approx rank':>11}  move",
            "dim",
        )
        row += 1
        for i, f in enumerate(res["final"]):
            if row >= y + hh - 4:
                break
            mv = f["rank_movement"]
            mtxt = (
                "new"
                if mv is None
                else (
                    f"{g['up']}{mv}"
                    if mv > 0
                    else f"{g['down']}{-mv}" if mv < 0 else "="
                )
            )
            ar = "-" if f["approx_rank"] is None else str(f["approx_rank"])
            cv.put(
                row,
                x + 4,
                f"{i:>4}  {f['id']:>7}  {fmt(f['exact_score'], 4):>12}  " f"{ar:>11}  ",
                None,
            )
            cv.put(
                row,
                x + 47,
                mtxt,
                "green" if mv and mv > 0 else "red" if mv and mv < 0 else None,
            )
            row += 1
    elif res.get("approximate"):
        cv.put(row, x + 2, "results: approximate scores (no rerank)", "bold")
        row += 1
        for i, a in enumerate(res["approximate"]):
            if row >= y + hh - 4:
                break
            cv.put(row, x + 4, f"{i:>4}  {a['id']:>7}  {fmt(a['score'], 4):>12}", None)
            row += 1
    rep = st.get("replay")
    if rep:
        ry = y + hh - 3
        if rep.get("error"):
            cv.put(ry, x + 2, ("replay: " + rep["error"])[: ww - 4], "red")
        else:
            d = rep["diff"]
            same = (
                "identical ids in identical order"
                if d["same_ids_in_order"]
                else f"overlap {d['overlap']}/{d['k']}, {len(d['moved'])} moved"
            )
            txt = (
                f"replay {rep['after']}: {same}; latency "
                f"{fmt(d['latency_ms']['before'], 3)} -> "
                f"{fmt(d['latency_ms']['after'], 3)} ms; nondeterminism: "
                f"{', '.join(rep['nondeterminism']) or 'none detected'}"
            )
            cv.put(
                ry, x + 2, txt[: ww - 4], "green" if d["same_ids_in_order"] else "amber"
            )


# --------------------------------------------------------------------- curses
def run(
    srv, export_dir: str = ".", setup: dict | None = None
) -> None:  # pragma: no cover - needs a terminal
    import curses

    locale.setlocale(locale.LC_ALL, "")
    g = UNICODE if "utf" in (locale.getpreferredencoding() or "").lower() else ASCII
    curses.wrapper(_loop, srv, g, export_dir, setup)


def _observer_sha(srv) -> str | None:
    return srv.observer.digest() if srv.observer is not None else None


def _feed_scope(st: dict, srv) -> None:
    """Hand the scope every trace finished since the last call (pulled on the UI
    thread, so the acquisition engine never sees two threads)."""
    docs = srv.tracer.traces()
    last = st.get("fed")
    start = 0
    if last is not None:
        for i in range(len(docs) - 1, -1, -1):
            if docs[i]["id"] == last:
                start = i + 1
                break
    for d in docs[start:]:
        st["scope"].feed(d)
    if docs:
        st["fed"] = docs[-1]["id"]


def color_pairs() -> dict:  # pragma: no cover - needs a terminal
    """The colour roles as curses attributes (call after curses.initscr)."""
    import curses

    pairs: dict = {}
    if curses.has_colors():
        curses.start_color()
        try:
            curses.use_default_colors()
            bg = -1
        except curses.error:
            bg = curses.COLOR_BLACK
        base = {
            "cyan": curses.COLOR_CYAN,
            "teal": curses.COLOR_CYAN,
            "purple": curses.COLOR_MAGENTA,
            "magenta": curses.COLOR_MAGENTA,
            "green": curses.COLOR_GREEN,
            "amber": curses.COLOR_YELLOW,
            "yellow": curses.COLOR_YELLOW,
            "red": curses.COLOR_RED,
            "dim": curses.COLOR_WHITE,
            "bold": curses.COLOR_WHITE,
            "grid": curses.COLOR_BLUE,
        }
        n = 1
        for name, col in base.items():
            curses.init_pair(n, col, bg)
            pairs[name] = curses.color_pair(n)
            pairs[f"{name}_dim"] = curses.color_pair(n) | curses.A_DIM
            pairs[f"{name}_bold"] = curses.color_pair(n) | curses.A_BOLD
            n += 1
        curses.init_pair(n, curses.COLOR_BLACK, curses.COLOR_CYAN)
        pairs["sel"] = curses.color_pair(n)
        pairs["dim"] |= curses.A_DIM
        pairs["grid"] |= curses.A_DIM
        pairs["bold"] |= curses.A_BOLD
    return pairs


def paint(scr, cv: Canvas, pairs: dict) -> None:  # pragma: no cover - terminal
    """Draw ``cv`` on the curses screen, one run of a colour at a time."""
    import curses

    h, w = scr.getmaxyx()
    scr.erase()
    for y, row in enumerate(cv.cells):
        x = 0
        while x < len(row):
            col = row[x][1]
            j = x
            while j < len(row) and row[j][1] == col:
                j += 1
            if y == h - 1 and j == w:
                j -= 1  # curses cannot write the bottom-right cell
            try:
                scr.addstr(y, x, "".join(c for c, _ in row[x:j]), pairs.get(col, 0))
            except curses.error:
                pass
            x = max(j, x + 1)
    scr.refresh()


_KEYNAMES = {259: "up", 258: "down", 260: "left", 261: "right", 32: "space"}


FABRIC_EVERY_S = 2.0  # NATS monitoring poll period in the console


def snapshot_txt(cv: Canvas, export_dir: str = ".") -> str:
    """Write the screen to ``tqp-console-<UTC stamp>.txt``; returns the message
    for the status line."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = f"{export_dir.rstrip('/')}/tqp-console-{stamp}.txt"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(line.rstrip() for line in cv.text()) + "\n")
    except OSError as e:
        return f"snapshot failed: {e}"
    return f"snapshot written: {path}"


def new_state(srv, setup: dict | None = None) -> dict:
    """The UI state for a session: the instruments, the panels' data, focus and
    overlays. Shared by the terminal UI and the vector (matplotlib) renderer."""
    from . import fabric_view
    from .scope import Scope
    from .spectrum import Analyzer

    st = {
        "zoom": None,
        "scope": Scope(),
        "analyzer": Analyzer(),
        "sel_trace": 0,
        "spectrum_reason": None,
        "sel_ch": 0,
        "history": None,
        "fed": None,
        "snap": None,
        "traces": [],
        "readscope": srv.readscope(),
        "sel": 0,
        "focus": 6,
        "fabric": None,
        "fabric_hist": fabric_view.History(),
        "paused": False,
        "overlay": None,
        "inspected": None,
        "replay": None,
        "message": "",
        "qps_hist": deque(maxlen=240),
        "p95_hist": deque(maxlen=240),
    }
    st["_clock"] = {
        "started": time.time(),
        "autoset_done": False,
        "tick": 0.0,
        "sweep": 0.0,
        "fabric": 0.0,
    }
    if srv.index is None:
        st["spectrum_reason"] = "no index attached (start with --index or --demo)"
    cert = srv.certificate or {}
    floor = (cert.get("certificate") or {}).get("tau_floor")
    if floor is not None:
        st["scope"].references["tau"] = (
            float(floor),
            "cert tau floor (anchor pairs: a reference, not a per-query bound)",
        )
    if setup is not None:
        from . import setup as SU

        warn = SU.apply(setup, st["scope"], st["analyzer"], _observer_sha(srv))
        st["zoom"] = zoom_of({"view": setup["view"]})
        # a recalled setup is not overridden by autoset
        st["_clock"]["autoset_done"] = True
        st["message"] = warn[0] if warn else "setup recalled"
    return st


def update(st: dict, srv, now: float) -> None:
    """Pull what is new from the session into ``st``: traces into the scope, a
    spectrum sweep every 2 s, a NATS poll every FABRIC_EVERY_S, a snapshot every
    second (held while paused)."""
    ck = st["_clock"]
    st["now"] = now
    _feed_scope(st, srv)
    st["scope"].tick(now)
    if not ck["autoset_done"] and now - ck["started"] > 3 and st["scope"].buf:
        st["scope"].autoset(now)
        ck["autoset_done"] = True
    if now - ck["sweep"] >= 2.0 and srv.index is not None:
        ck["sweep"] = now
        ref = st["analyzer"].reference
        sw, why = srv.spectrum_sweep(basis=None if ref is None else ref.basis)
        st["spectrum_reason"] = why
        if sw is not None:
            an = st["analyzer"]
            first = an.last is None
            an.feed(sw)
            if first:
                an.autoscale()
    if srv.fabric is not None and now - ck["fabric"] >= FABRIC_EVERY_S:
        ck["fabric"] = now
        doc = srv.fabric_poll()
        if doc is not None:
            st["fabric"] = doc
            st["fabric_hist"].add(doc)
    if now - ck["tick"] >= 1.0:
        ck["tick"] = now
        snap = srv.snapshot()
        rd = _readings(snap)
        st["qps_hist"].append(rd.get("search.qps", {}).get("value"))
        st["p95_hist"].append(rd.get("search.latency_ms.p95", {}).get("value"))
        if not st["paused"]:
            st["snap"], st["traces"] = snap, srv.tracer.traces(200)


def _loop(scr, srv, g, export_dir, setup=None):  # pragma: no cover - needs a terminal
    import curses

    from . import scope_view, spectrum_view

    curses.curs_set(0)
    scr.timeout(150)
    pairs = color_pairs()
    st = new_state(srv, setup)
    resumed = []  # set by SIGCONT: whatever the terminal showed meanwhile is stale

    import signal

    if hasattr(signal, "SIGCONT"):
        signal.signal(signal.SIGCONT, lambda *_: resumed.append(True))
    while True:
        now = time.time()
        update(st, srv, now)
        h, w = scr.getmaxyx()
        if resumed:  # stopped and continued (Ctrl+Z / fg, or a thermal pause)
            resumed.clear()
            curses.update_lines_cols()
            h, w = scr.getmaxyx()
            scr.clear()  # the next refresh repaints every cell, not just changes
        paint(scr, frame(st, w, h, g), pairs)
        ch = scr.getch()
        if ch == -1:
            continue
        st["message"] = ""
        name = _KEYNAMES.get(ch, chr(ch) if 32 <= ch < 127 else None)
        if ch in (ord("q"), ord("Q")):
            return
        if ch == 27:  # Esc: close an overlay, else leave the zoomed panel
            if st["overlay"] is None and st["zoom"] is not None:
                st["zoom"] = None
            st["overlay"], st["replay"] = None, None
            continue
        if ch == ord("?"):
            st["overlay"] = None if st["overlay"] == "help" else "help"
            continue
        if ch == ord("z") and st["overlay"] is None:
            if st["zoom"] is not None:
                st["zoom"] = None
            elif st["focus"] in ZOOMABLE:
                st["zoom"] = ZOOMABLE[st["focus"]]
            else:
                st["message"] = "z opens panels 7 (scope), 8 (spectrum), 9 (NATS)"
            continue
        if ch == ord("i"):
            st["annotate"] = not st.get("annotate", True)
            st["message"] = "notes " + ("on" if st["annotate"] else "off")
            continue
        if ch == ord("S"):
            from . import setup as SU

            stamp = time.strftime("%Y%m%dT%H%M%S")
            path = f"{export_dir.rstrip('/')}/tqp-console-{stamp}.tqs"
            try:
                SU.save(
                    path,
                    SU.to_dict(
                        st["scope"],
                        st["analyzer"],
                        (
                            st["zoom"]
                            if st["zoom"] in ("scope", "spectrum")
                            else "overview"
                        ),
                        _observer_sha(srv),
                    ),
                )
                st["message"] = f"setup saved: {path}"
            except (OSError, SU.SetupError) as e:
                st["message"] = f"setup not saved: {e}"
            continue
        if ch == ord("P"):
            h, w = scr.getmaxyx()
            st["message"] = snapshot_txt(
                frame(dict(st, message=""), w, h, g), export_dir
            )
            continue
        if ch == ord("e"):
            stamp = time.strftime("%Y%m%dT%H%M%S")
            path = f"{export_dir.rstrip('/')}/tqp-console-{stamp}.json"
            try:
                from .server import dumps

                with open(path, "w", encoding="utf-8") as f:
                    f.write(dumps(srv.export()))
                st["message"] = f"exported {path}"
            except OSError as e:
                st["message"] = f"export failed: {e}"
            continue
        if st["zoom"] == "fabric":
            continue
        if st["zoom"] == "spectrum":
            if name:
                st["message"] = spectrum_view.key(st, name)
            continue
        if st["zoom"] == "scope":
            sc = st["scope"]
            if ch in (10, 13, curses.KEY_ENTER):
                rec = sc.record
                t = srv.tracer.get(rec.trigger_id) if rec and rec.trigger_id else None
                if t:
                    st["inspected"], st["overlay"], st["replay"] = t, "inspect", None
                else:
                    st["message"] = "no trigger query to inspect (or it was evicted)"
            elif ch == ord("r") and st.get("inspected"):
                st["replay"] = srv.replay(st["inspected"]["id"])
            elif name:
                st["message"] = scope_view.key(st, name, now)
            continue
        visible = list(reversed(st["traces"]))
        if ch == ord("p"):
            st["paused"] = not st["paused"]
        elif ch in (curses.KEY_DOWN, ord("j")):
            st["sel"] = min(st["sel"] + 1, max(len(visible) - 1, 0))
        elif ch in (curses.KEY_UP, ord("k")):
            st["sel"] = max(st["sel"] - 1, 0)
        elif ch in (10, 13, curses.KEY_ENTER) and visible:
            st["inspected"], st["overlay"], st["replay"] = (
                visible[st["sel"]],
                "inspect",
                None,
            )
        elif ch == ord("r"):
            t = st["inspected"] or (visible[st["sel"]] if visible else None)
            if t:
                st["inspected"], st["overlay"] = t, "inspect"
                st["replay"] = srv.replay(t["id"])
        elif ch == 9:  # Tab
            st["focus"] = st["focus"] % len(PANELS) + 1
        elif ord("1") <= ch <= ord("9"):
            st["focus"] = ch - ord("0")

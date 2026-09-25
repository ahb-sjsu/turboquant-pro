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
    ("Tab / 1-6", "focus a panel"),
    ("Up/Down j/k", "select a query"),
    ("Enter", "inspect the selected query"),
    ("r", "replay the query and compare"),
    ("e", "export the session as JSON to the current directory"),
    ("p", "pause / resume the display (the workload keeps running)"),
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


def _readings(snap: dict) -> dict:
    return {r["name"]: r for r in snap.get("readings", [])}


def _stage(t: dict, name: str):
    s = next((s for s in t.get("stages", []) if s["name"] == name), None)
    return s["ms"] if s else None


def frame(st: dict, w: int, h: int, g: dict = UNICODE) -> Canvas:
    """The whole screen for state ``st``:
    ``snap`` (a snapshot document), ``traces`` (newest last), ``readscope``,
    ``qps_hist`` / ``p95_hist``, ``sel``, ``focus`` (1-6), ``paused``, ``overlay``
    (None | "inspect" | "help"), ``inspected``, ``replay``, ``message``."""
    cv = Canvas(w, h)
    if st.get("view") == "scope" and w >= MIN_W and h >= MIN_H:
        from . import scope_view

        scope_view.render(cv, st, g, st.get("now") or time.time())
        brand = " TurboQuant console  q quit  ? keys "
        if cv.w > 110:
            cv.put(0, cv.w - len(brand), brand, "dim")
        if st.get("message"):
            cv.put(cv.h - 2, 1, f" {st['message']} "[: cv.w - 2], "amber")
        if st.get("overlay") == "help":
            _overlay_help(cv, g, scope_view.HELP)
        elif st.get("overlay") == "inspect" and st.get("inspected"):
            _overlay_inspect(cv, st, g)
        return cv
    if w < MIN_W or h < MIN_H:
        cv.put(
            0,
            0,
            f"terminal {w}x{h} is too small: {MIN_W}x{MIN_H} at least"[:w],
            "amber",
        )
        return cv
    snap = st.get("snap") or {}
    r = _readings(snap)
    wl = snap.get("workload", {})
    traces = st.get("traces", [])
    last = traces[-1] if traces else {}

    # header -----------------------------------------------------------------
    x = 0
    cv.put(0, x, " TurboQuant ", "bold")
    x += 12
    cv.put(0, x, "console ", "cyan")
    x += 8
    age = snap.get("last_trace_age_s")
    state = (
        ("PAUSED", "amber")
        if st.get("paused")
        else (
            ("live", "green")
            if age is not None and age < 3
            else ("waiting", "amber") if age is None else (f"stale {age:.0f}s", "amber")
        )
    )
    room = w - len("q quit  ? keys") - 3
    for label, col in [
        (f"[{state[0]}]", state[1]),
        (f"[mode: {wl.get('mode', '-')}]", "green" if wl.get("rerank") else "amber"),
        (f"[scan: {last.get('scan_path') or '-'}]", "cyan"),
    ]:
        if x + len(label) <= room:
            cv.put(0, x, label, col)
            x += len(label) + 1
    obs = ((st.get("readscope") or {}).get("observer") or {}).get("reference")
    if obs:
        tag = f"[observer: {obs.get('observer')} {obs.get('sha256', '')[:8]}]"
        if x + len(tag) <= room:
            cv.put(0, x, tag, "purple")
            x += len(tag) + 1
    hint = "q quit  ? keys"
    cv.put(0, w - len(hint) - 1, hint, "dim")
    for xx in range(w - len(hint) - 2, w - len(hint) - 1):  # keep a gap before it
        cv.put(0, xx, " ")

    # geometry ---------------------------------------------------------------
    top_h = 9 if h >= 30 else 7
    mid_h = 8 if h >= 30 else 6
    y1, y2, y3 = 1, 1 + top_h, 1 + top_h + mid_h
    c = w // 3
    foc = st.get("focus", 0)

    # 1 system ---------------------------------------------------------------
    cv.box(y1, 0, top_h, c, "1 system", g, focus=foc == 1)
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
    inner = c - 2
    cols = 2 if inner >= 50 else 1
    rows_avail = top_h - 2
    cw = inner // cols
    for i, (label, name, d) in enumerate(items[: rows_avail * cols]):
        rd = r.get(name)
        if rd is None:
            continue
        yy, xx = y1 + 1 + i % rows_avail, 1 + (i // rows_avail) * cw
        unit = (
            "" if rd["unit"] == "fraction" else rd["unit"].replace("queries/s", "q/s")
        )
        val = fmt(rd["value"], d)
        cv.put(yy, xx + 1, label, "dim")
        num = f"{val} {unit}".rstrip()
        kind = rd["kind"][:4]
        right = xx + cw - 2
        cv.put(yy, right - len(kind), kind, "dim")
        cv.put(
            yy,
            right - len(kind) - 1 - len(num),
            num,
            "dim" if rd["value"] is None else None,
        )

    # 2 throughput / latency -------------------------------------------------
    cv.box(y1, c, top_h, c, "2 throughput / latency", g, focus=foc == 2)
    sw = c - 4
    q = r.get("search.qps", {}).get("value")
    p95 = r.get("search.latency_ms.p95", {}).get("value")
    cv.put(y1 + 1, c + 2, f"QPS {fmt(q, 1)}", "cyan")
    cv.put(y1 + 2, c + 2, spark(st.get("qps_hist", []), sw, g), "cyan")
    if top_h >= 9:
        cv.put(y1 + 3, c + 2, spark(st.get("qps_hist", []), sw, g), "cyan")
    ly = y1 + (4 if top_h >= 9 else 3)
    cv.put(ly, c + 2, f"p95 {fmt(p95, 2)} ms", "purple")
    cv.put(ly + 1, c + 2, spark(st.get("p95_hist", []), sw, g), "purple")
    if top_h >= 9:
        cv.put(ly + 2, c + 2, spark(st.get("p95_hist", []), sw, g), "purple")

    # 3 pipeline -------------------------------------------------------------
    pw = w - 2 * c
    cv.box(y1, 2 * c, top_h, pw, "3 pipeline  ms/query", g, focus=foc == 3)
    stages = [
        (s, r.get(f"search.stage_ms.{s}", {}).get("value"))
        for s in ("encode", "scan", "rerank")
    ]
    mx = max([v for _, v in stages if v is not None] or [1e-9])
    bw = max(4, pw - 20)
    for i, (s, v) in enumerate(stages):
        yy = y1 + 1 + i * (2 if top_h >= 9 else 1)
        cv.put(yy, 2 * c + 2, f"{s:<7}", None)
        cv.put(yy, 2 * c + 9, bar(0 if v is None else v / mx, bw, g), "teal")
        cv.put(yy, 2 * c + 10 + bw, f"{fmt(v, 3):>7}", None)
    k, rr = wl.get("k", "-"), wl.get("rerank", 0)
    note = f"top-{k} from {k * rr} reranked" if rr else f"top-{k}, approximate"
    cv.put(y1 + top_h - 2, 2 * c + 2, note[: pw - 4], "dim")

    # 4 readscope ------------------------------------------------------------
    rw = 2 * c
    cv.box(
        y2,
        0,
        mid_h,
        rw,
        "4 readscope  observer / certificate / provenance",
        g,
        color="purple",
        focus=foc == 4,
    )
    rs = st.get("readscope") or {}
    lines = []
    if obs:
        cons = ", ".join(
            str(x.get("metric") or x.get("name") or x)
            for x in obs.get("consumers") or []
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
                "none loaded (--observer X.tqo): results are not "
                "tied to a declared reader",
            )
        )
    cert = rs.get("certificate")
    if cert:
        cc = cert.get("certificate", {})
        lines.append(
            (
                "certificate",
                f"{'PASSED' if cert.get('passed') else 'NOT PASSED'}"
                f"  tau floor {cc.get('tau_floor')}  validity "
                f"{(rs.get('validity') or {}).get('status', 'UNCHECKED')}",
            )
        )
    for p in rs.get("provenance", []):
        val = p.get("sha256") or json.dumps({k: v for k, v in p.items() if k != "step"})
        lines.append((p["step"][:12], val))
    for i, (kk, vv) in enumerate(lines[: mid_h - 2]):
        cv.put(y2 + 1 + i, 2, f"{kk:<12}", "purple")
        cv.put(y2 + 1 + i, 15, str(vv)[: rw - 17], None)

    # 5 index ----------------------------------------------------------------
    iw = w - rw
    cv.box(y2, rw, mid_h, iw, "5 index", g, focus=foc == 5)
    ie = snap.get("index", {})
    kvs = [
        ("kind", ie.get("kind")),
        ("rows", ie.get("rows")),
        ("dim", ie.get("dim")),
        ("metric", ie.get("metric")),
        ("bytes/row", ie.get("stored_bytes_per_row")),
        ("kernel", "AVX2" if ie.get("kernel") else "numpy"),
        ("workload", f"{wl.get('target_qps', '-')} qps, k={wl.get('k', '-')}"),
    ]
    for i, (kk, vv) in enumerate(kvs[: mid_h - 2]):
        cv.put(y2 + 1 + i, rw + 2, f"{kk:<10}", "dim")
        cv.put(y2 + 1 + i, rw + 13, str(vv)[: iw - 15], None)

    # 6 query stream ---------------------------------------------------------
    sh = h - y3
    cv.box(
        y3, 0, sh, w, "6 query stream  Up/Down select, Enter inspect", g, focus=foc == 6
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
    xs, xx = [], 2
    for name, cwid in cols:
        if xx + cwid > w - 2:
            break
        xs.append((name, cwid, xx))
        xx += cwid + 1
    for name, cwid, xx in xs:
        cv.put(
            y3 + 1, xx, name.rjust(cwid) if cwid <= 8 and name != "row" else name, "dim"
        )
    rows = list(reversed(traces))[: max(0, sh - 3)]
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
                y3 + 2 + i,
                xx,
                (v.rjust(cwid) if cwid <= 8 and name != "row" else v.ljust(cwid))[
                    :cwid
                ],
                "sel" if sel else None,
            )
        if sel:
            cv.put(y3 + 2 + i, 1, ">", "cyan")
    if st.get("message"):
        cv.put(h - 1, 2, f" {st['message']} "[: w - 4], "amber")

    if st.get("overlay") == "help":
        _overlay_help(cv, g)
    elif st.get("overlay") == "inspect" and st.get("inspected"):
        _overlay_inspect(cv, st, g)
    return cv


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
def run(srv, export_dir: str = ".") -> None:  # pragma: no cover - needs a terminal
    import curses

    locale.setlocale(locale.LC_ALL, "")
    g = UNICODE if "utf" in (locale.getpreferredencoding() or "").lower() else ASCII
    curses.wrapper(_loop, srv, g, export_dir)


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


_KEYNAMES = {259: "up", 258: "down", 260: "left", 261: "right", 32: "space"}


def _loop(scr, srv, g, export_dir):  # pragma: no cover - needs a terminal
    import curses

    from . import scope_view
    from .scope import Scope

    curses.curs_set(0)
    scr.timeout(150)
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
    st = {
        "view": "scope",
        "scope": Scope(),
        "sel_ch": 0,
        "history": None,
        "fed": None,
        "snap": None,
        "traces": [],
        "readscope": srv.readscope(),
        "sel": 0,
        "focus": 6,
        "paused": False,
        "overlay": None,
        "inspected": None,
        "replay": None,
        "message": "",
        "qps_hist": deque(maxlen=240),
        "p95_hist": deque(maxlen=240),
    }
    started, autoset_done, last_tick = time.time(), False, 0.0
    while True:
        now = time.time()
        st["now"] = now
        _feed_scope(st, srv)
        st["scope"].tick(now)
        if not autoset_done and now - started > 3 and st["scope"].buf:
            st["scope"].autoset(now)
            autoset_done = True
        if now - last_tick >= 1.0:
            last_tick = now
            snap = srv.snapshot()
            rd = _readings(snap)
            st["qps_hist"].append(rd.get("search.qps", {}).get("value"))
            st["p95_hist"].append(rd.get("search.latency_ms.p95", {}).get("value"))
            if not st["paused"]:
                st["snap"], st["traces"] = snap, srv.tracer.traces(200)
        h, w = scr.getmaxyx()
        cv = frame(st, w, h, g)
        scr.erase()
        for y, row in enumerate(cv.cells):
            x = 0
            while x < len(row):  # paint runs of one colour at a time
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
        ch = scr.getch()
        if ch == -1:
            continue
        st["message"] = ""
        name = _KEYNAMES.get(ch, chr(ch) if 32 <= ch < 127 else None)
        if ch in (ord("q"), ord("Q")):
            return
        if ch == 27:  # Esc
            st["overlay"], st["replay"] = None, None
            continue
        if ch == ord("?"):
            st["overlay"] = None if st["overlay"] == "help" else "help"
            continue
        if ch == ord("v"):
            st["view"] = "overview" if st["view"] == "scope" else "scope"
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
        if st["view"] == "scope":
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
            st["focus"] = st["focus"] % 6 + 1
        elif ord("1") <= ch <= ord("6"):
            st["focus"] = ch - ord("0")

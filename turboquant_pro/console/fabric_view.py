# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""``tqp fabric``: the NATS fabric as a terminal instrument.

:func:`frame` is pure (a snapshot and its history in, a :class:`~.tui.Canvas`
out), so it is tested and screenshotted without a terminal; :func:`run` is the
curses loop around it. Four panels: the server, its leaf-node links, its client
connections, and the events between polls. Every rate is labelled with the
interval it was derived over; a value the server did not give is ``-``.
"""

from __future__ import annotations

import time
from collections import deque

from .tui import ASCII, MIN_H, MIN_W, UNICODE, Canvas, fmt, spark

HISTORY = 240  # polls kept for the sparklines

EVENT_COLOR = {
    "server_unreachable": "red",
    "server_restarted": "amber",
    "slow_consumers": "amber",
    "leaf_connected": "green",
    "leaf_disconnected": "red",
    "counter_reset": "amber",
}


def human_bytes(v) -> str:
    if v is None:
        return "-"
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(v) < 1000 or unit == "TB":
            return f"{v:.0f} {unit}" if unit == "B" else f"{v:.1f} {unit}"
        v /= 1000.0
    return "-"  # pragma: no cover


def human_s(v) -> str:
    if v is None:
        return "-"
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if v >= size:
            return f"{v / size:.1f}{unit}"
    return f"{v:.0f}s"


class History:
    """Per-poll series for the sparklines, plus the events seen so far."""

    def __init__(self, n: int = HISTORY):
        self.series = {
            k: deque(maxlen=n)
            for k in ("in_msgs", "out_msgs", "leaf_rtt", "leaf_msgs", "connects")
        }
        self.events: deque = deque(maxlen=200)

    def add(self, doc: dict) -> None:
        r = doc.get("rates") or {}
        leafs = doc.get("leafs") or []
        rtts = [lf["rtt_ms"] for lf in leafs if lf.get("rtt_ms") is not None]
        lm = [
            (lf["rates"].get("in_msgs_per_s"), lf["rates"].get("out_msgs_per_s"))
            for lf in leafs
        ]
        s = self.series
        s["in_msgs"].append(r.get("in_msgs_per_s"))
        s["out_msgs"].append(r.get("out_msgs_per_s"))
        s["connects"].append(r.get("connects_per_min"))
        s["leaf_rtt"].append(max(rtts) if rtts else None)
        s["leaf_msgs"].append(
            None if not lm or any(None in p for p in lm) else sum(a + b for a, b in lm)
        )
        for e in doc.get("events") or []:
            self.events.append((doc["t"], e))


def _rate(v, unit="/s") -> str:
    return "-" if v is None else f"{fmt(v, 1)}{unit}"


def frame(
    doc: dict | None,
    hist: History,
    w: int,
    h: int,
    g: dict = UNICODE,
    paused: bool = False,
) -> Canvas:
    cv = Canvas(w, h)
    if w < MIN_W or h < MIN_H:
        cv.put(0, 0, f"tqp fabric needs {MIN_W}x{MIN_H}; this terminal is {w}x{h}")
        return cv
    _status(cv, doc, paused)
    if doc is None:
        cv.put(2, 2, "first poll pending", "dim")
        return cv
    top = min(9, max(7, (h - 3) // 3))
    half = w // 2
    _server(cv, doc, hist, 1, 0, top, half, g)
    _leafs(cv, doc, hist, 1, half, top, w - half, g)
    ev_h = max(5, min(10, (h - 1 - top) // 3))
    conn_h = h - 1 - top - ev_h
    _clients(cv, doc, 1 + top, 0, conn_h, w, g)
    _events(cv, doc, hist, 1 + top + conn_h, 0, ev_h, w, g)
    return cv


def _status(cv: Canvas, doc, paused: bool) -> None:
    x = 0
    cv.put(0, x, "fabric ", "cyan")
    x += 7
    if doc is None:
        return
    src = doc.get("source") or {}
    if not doc.get("reachable"):
        state, col = "UNREACHABLE", "red"
    elif paused:
        state, col = "PAUSED", "amber"
    else:
        state, col = "live", "green"
    # Most important first: a narrow terminal clips the tail, never the state.
    parts = [
        (f"[{state}]", col),
        (
            (
                f" rates over {doc['interval_s']:.1f} s"
                if doc.get("interval_s")
                else " rates: need a second poll"
            ),
            None,
        ),
        (f"  as of {doc.get('as_of_utc', '-')[11:19]}Z", "dim"),
        (
            f"  nats {src.get('version') or '-'} up {human_s(src.get('uptime_s'))}",
            "dim",
        ),
    ]
    if src.get("redacted"):
        parts.append(("  IPs redacted", "purple"))
    parts.append((f"  {src.get('url')}", "dim"))
    for text, col in parts:
        cv.put(0, x, text, col)
        x += len(text)


def _server(cv, doc, hist, y, x, hh, ww, g):
    cv.box(y, x, hh, ww, "1 server", g)
    s, r = doc.get("server") or {}, doc.get("rates") or {}
    sw = max(8, ww - 34)
    rows = [
        (
            "connections",
            f"{fmt(s.get('connections'), 0)} open, "
            f"{_rate(r.get('connects_per_min'), '/min')} new",
            None,
        ),
        (
            "msgs in",
            _rate(r.get("in_msgs_per_s")),
            spark(hist.series["in_msgs"], sw, g),
        ),
        (
            "msgs out",
            _rate(r.get("out_msgs_per_s")),
            spark(hist.series["out_msgs"], sw, g),
        ),
        (
            "bytes in/out",
            f"{human_bytes(r.get('in_bytes_per_s'))}/s  "
            f"{human_bytes(r.get('out_bytes_per_s'))}/s",
            None,
        ),
        (
            "subscriptions",
            f"{fmt(s.get('subscriptions'), 0)}   slow consumers "
            f"{fmt(s.get('slow_consumers'), 0)} (since start)",
            None,
        ),
        (
            "process",
            f"mem {human_bytes(s.get('mem_bytes'))}  cpu "
            f"{fmt(s.get('cpu_percent'), 1)}%",
            None,
        ),
    ]
    js = doc.get("jetstream")
    if js:
        rows.append(
            (
                "jetstream",
                f"{fmt(js.get('streams'), 0)} streams, "
                f"{fmt(js.get('messages'), 0)} msgs, {human_bytes(js.get('bytes'))}",
                None,
            )
        )
    for i, (k, v, sp) in enumerate(rows[: hh - 2]):
        cv.put(y + 1 + i, x + 2, f"{k:<14}", "cyan")
        room = ww - 18 - (sw + 1 if sp else 0)
        cv.put(y + 1 + i, x + 16, str(v)[:room], None)
        if sp:
            cv.put(y + 1 + i, x + ww - 2 - sw, sp, "cyan")


def _leafs(cv, doc, hist, y, x, hh, ww, g):
    leafs = doc.get("leafs") or []
    cv.box(
        y, x, hh, ww, f"2 leaf links ({len(leafs)})", g, color="dim" if leafs else "red"
    )
    if "leafz" in (doc.get("errors") or {}):
        msg = f"cannot read /leafz: {doc['errors']['leafz']}"
        cv.put(y + 1, x + 2, msg[: ww - 4], "red")
        return
    if not leafs:
        cv.put(y + 1, x + 2, "no leaf node connected", "red")
        return
    sw = max(8, ww - 4)
    row = y + 1
    for lf in leafs:
        if row >= y + hh - 1:
            break
        rt = lf["rates"]
        cv.put(
            row,
            x + 2,
            f"rtt {fmt(lf.get('rtt_ms'), 1)} ms ({_rtt_note(lf)})  "
            f"{str(lf.get('name'))[:10]}  {lf.get('ip')}:{lf.get('port')}"[: ww - 4],
            None,
        )
        row += 1
        cv.put(
            row,
            x + 4,
            f"msgs {_rate(rt.get('in_msgs_per_s'))} in "
            f"{_rate(rt.get('out_msgs_per_s'))} out   "
            f"{human_bytes(rt.get('in_bytes_per_s'))}/s in "
            f"{human_bytes(rt.get('out_bytes_per_s'))}/s out"[: ww - 6],
            None,
        )
        row += 1
        comp = _compression(lf)
        if comp and row < y + hh - 2:
            cv.put(row, x + 4, comp[: ww - 6], "dim")
            row += 1
        cv.put(
            row,
            x + 4,
            f"totals {fmt(lf.get('in_msgs'), 0)} in "
            f"{fmt(lf.get('out_msgs'), 0)} out   subs "
            f"{', '.join(lf.get('subjects') or []) or '-'}"[: ww - 6],
            "dim",
        )
        row += 1
    if row < y + hh - 1:
        cv.put(row, x + 2, "rtt " + spark(hist.series["leaf_rtt"], sw - 4, g), "cyan")


def _compression(lf) -> str:
    """The link's compression, and what the byte figures mean under it: measured
    on the NRP leaf (benchmarks/RESULTS_fabric_leaf.md), the server's byte
    counters equal the payload bytes exactly on an s2-compressed link, so they
    are bytes before compression, not bytes on the wire."""
    c = lf.get("compression")
    if not c or c == "off":
        return c or ""
    return f"{c}: bytes are payload, before compression"


def _rtt_note(lf) -> str:
    """ "sampled", plus how long the value has not changed when that is long
    enough to suspect it is old (more than one server PING interval, 2 min)."""
    u = lf.get("rtt_unchanged_s")
    return "sampled" if not u or u < 120 else f"sampled, unchanged {human_s(u)}"


_COLS = (
    ("cid", 8),
    ("client", 22),
    ("rtt ms", 8),
    ("idle", 8),
    ("in/s", 9),
    ("out/s", 9),
    ("pending", 9),
)


def _clients(cv, doc, y, x, hh, ww, g):
    conns = doc.get("connections") or []
    cv.box(y, x, hh, ww, f"3 clients ({len(conns)})", g)
    col = x + 2
    for name, width in _COLS:
        cv.put(y + 1, col, f"{name:<{width}}", "bold")
        col += width
    cv.put(y + 1, col, "subjects", "bold")
    note = (doc.get("errors") or {}).get("connz")
    ordered = sorted(
        conns,
        key=lambda c: -(
            (c["rates"].get("in_msgs_per_s") or 0)
            + (c["rates"].get("out_msgs_per_s") or 0)
        ),
    )
    for i, c in enumerate(ordered[: max(0, hh - 3 - (1 if note else 0))]):
        rt = c["rates"]
        who = c.get("name") or c.get("lang") or "-"
        vals = (
            str(c.get("cid")),
            str(who),
            fmt(c.get("rtt_ms"), 2),
            human_s(c.get("idle_s")),
            fmt(rt.get("in_msgs_per_s"), 1),
            fmt(rt.get("out_msgs_per_s"), 1),
            human_bytes(c.get("pending_bytes")),
        )
        col = x + 2
        for (_, width), v in zip(_COLS, vals):
            cv.put(y + 2 + i, col, f"{v[: width - 1]:<{width}}", None)
            col += width
        cv.put(
            y + 2 + i, col, ", ".join(c.get("subjects") or [])[: ww - col - 2], "dim"
        )
    if note:
        cv.put(y + hh - 2, x + 2, note[: ww - 4], "amber")


def _events(cv, doc, hist, y, x, hh, ww, g):
    cv.box(y, x, hh, ww, "4 events since start", g)
    evs = list(hist.events)[-(hh - 2) :]
    if not evs:
        cv.put(y + 1, x + 2, "none: no link, restart or slow consumer seen", "dim")
    for i, (t, e) in enumerate(reversed(evs)):
        stamp = time.strftime("%H:%M:%S", time.gmtime(t))
        cv.put(y + 1 + i, x + 2, stamp, "dim")
        cv.put(y + 1 + i, x + 12, f"{e['kind']:<20}", EVENT_COLOR.get(e["kind"]))
        cv.put(y + 1 + i, x + 33, str(e.get("detail"))[: ww - 35], None)


def run(monitor, interval: float = 2.0) -> None:  # pragma: no cover - terminal
    import curses
    import locale

    locale.setlocale(locale.LC_ALL, "")
    g = UNICODE if "utf" in (locale.getpreferredencoding() or "").lower() else ASCII
    curses.wrapper(_loop, monitor, interval, g)


def _loop(scr, monitor, interval, g):  # pragma: no cover - needs a terminal
    import curses

    from .tui import color_pairs, paint

    curses.curs_set(0)
    scr.timeout(150)
    pairs = color_pairs()
    hist, doc, shown, paused, last = History(), None, None, False, 0.0
    while True:
        now = time.time()
        if now - last >= interval:
            last = now
            doc = monitor.poll()
            hist.add(doc)
            if not paused:
                shown = doc
        h, w = scr.getmaxyx()
        paint(scr, frame(shown, hist, w, h, g, paused), pairs)
        ch = scr.getch()
        if ch in (ord("q"), ord("Q")):
            return
        if ch in (ord("p"), ord("P")):
            paused = not paused
            shown = doc

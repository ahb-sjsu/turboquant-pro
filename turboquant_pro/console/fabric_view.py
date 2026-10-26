# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""The NATS fabric as a terminal instrument: panel 1 of ``tqp console --nats URL``
zoomed (z).

:func:`frame` is pure (a snapshot and its history in, a :class:`~.tui.Canvas`
out), so it is tested and screenshotted without a terminal. Four panels: the
server, its leaf-node links, its client connections, and the events between
polls. Every rate is labelled with the interval it was derived over; a value the
server did not give is ``-``.
"""

from __future__ import annotations

import time
from collections import deque

from .scope import Channel, ChannelSpec, Scope, Trigger
from .tui import MIN_W, UNICODE, Canvas, fmt, spark

HISTORY = 240  # polls kept for the sparklines
FIT_MIN_H = 12  # the least height the four panels fit in (each clips to its box)


def layout(h: int) -> tuple:
    """Rows for (server and leaf links, clients, events) in a frame ``h`` high,
    below its status line: the clients take what the other two leave."""
    top = max(4, min(9, (h - 1) * 3 // 10))
    ev = max(3, min(8, (h - 1 - top) // 4))
    return top, h - 1 - top - ev, ev


def need_h(doc: dict | None) -> int:
    """The least frame height that lists every client connection (and the
    error line, when the connections could not be read)."""
    if doc is None:
        return FIT_MIN_H
    rows = 3 + len(doc.get("connections") or [])  # box, header and one per client
    rows += 1 if (doc.get("errors") or {}).get("connz") else 0
    h = FIT_MIN_H
    while layout(h)[1] < rows and h < 400:
        h += 1
    return h


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


SERIES = (
    "in_msgs",
    "out_msgs",
    "in_bytes",
    "out_bytes",
    "leaf_rtt",
    "leaf_msgs",
    "leaf_bytes",
    "connects",
    "pending",
    "js_msgs",
)


def readings(doc: dict) -> dict:
    """The values one poll gives each series: the server's rates, the leaf links'
    summed rates and their slowest round trip, the largest pending bytes of any
    client, JetStream's message count. A value the poll does not give is None.
    The sparklines and the NATS page's scope both read these, so they cannot
    disagree."""
    r = doc.get("rates") or {}
    leafs = doc.get("leafs") or []
    rtts = [lf["rtt_ms"] for lf in leafs if lf.get("rtt_ms") is not None]
    lm = [
        (lf["rates"].get("in_msgs_per_s"), lf["rates"].get("out_msgs_per_s"))
        for lf in leafs
    ]
    lb = [
        (lf["rates"].get("in_bytes_per_s"), lf["rates"].get("out_bytes_per_s"))
        for lf in leafs
    ]
    pend = [
        c.get("pending_bytes")
        for c in doc.get("connections") or []
        if c.get("pending_bytes") is not None
    ]
    js = doc.get("jetstream") or {}

    def summed(pairs):
        return (
            None
            if not pairs or any(None in p for p in pairs)
            else sum(a + b for a, b in pairs)
        )

    return {
        "in_msgs": r.get("in_msgs_per_s"),
        "out_msgs": r.get("out_msgs_per_s"),
        "in_bytes": r.get("in_bytes_per_s"),
        "out_bytes": r.get("out_bytes_per_s"),
        "leaf_rtt": max(rtts) if rtts else None,
        "leaf_msgs": summed(lm),
        "leaf_bytes": summed(lb),
        "connects": r.get("connects_per_min"),
        "pending": max(pend) if pend else None,
        "js_msgs": js.get("messages"),
    }


class History:
    """Per-poll series for the sparklines, plus the events seen so far."""

    def __init__(self, n: int = HISTORY):
        self.series = {k: deque(maxlen=n) for k in SERIES}
        self.events: deque = deque(maxlen=200)

    def add(self, doc: dict) -> None:
        for k, v in readings(doc).items():
            self.series[k].append(v)
        for e in doc.get("events") or []:
            self.events.append((doc["t"], e))


# ---- the scope on the NATS page: one sample per poll -------------------------
def _reading(key):
    return lambda t: t["fabric"][key]


SCOPE_SIGNALS = {
    c.name: c
    for c in [
        ChannelSpec("in_msgs", "msg/s", _reading("in_msgs"), "messages in per second"),
        ChannelSpec(
            "out_msgs", "msg/s", _reading("out_msgs"), "messages out per second"
        ),
        ChannelSpec("in_bytes", "B/s", _reading("in_bytes"), "bytes in per second"),
        ChannelSpec("out_bytes", "B/s", _reading("out_bytes"), "bytes out per second"),
        ChannelSpec("leaf_rtt", "ms", _reading("leaf_rtt"), "slowest leaf round trip"),
        ChannelSpec(
            "leaf_msgs", "msg/s", _reading("leaf_msgs"), "leaf messages per second"
        ),
        ChannelSpec(
            "connects", "/min", _reading("connects"), "new connections per minute"
        ),
        ChannelSpec(
            "pending", "B", _reading("pending"), "largest pending bytes of a client"
        ),
    ]
}


def new_scope() -> Scope:
    """The NATS page's scope: messages in and out, bytes in and the leaf round
    trip on its four channels, triggering on messages in, 10 s/div to start
    (autoset rescales once it has samples)."""
    return Scope(
        signals=SCOPE_SIGNALS,
        channels=[
            Channel("in_msgs", 10.0),
            Channel("out_msgs", 10.0, on=False),
            Channel("in_bytes", 1000.0, on=False),
            Channel("leaf_rtt", 20.0, on=False),
        ],
        trigger=Trigger(source="in_msgs", level=10.0),
        s_per_div=10.0,
    )


def scope_sample(doc: dict, n: int) -> dict:
    """One poll as the scope's input: its time, an id, and its readings."""
    return {"started_unix": doc["t"], "id": f"poll-{n}", "fabric": readings(doc)}


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
    if w < MIN_W or h < FIT_MIN_H:
        cv.put(0, 0, f"tqp fabric needs {MIN_W}x{FIT_MIN_H}; this terminal is {w}x{h}")
        return cv
    _status(cv, doc, paused)
    if doc is None:
        cv.put(2, 2, "first poll pending", "dim")
        return cv
    top, conn_h, ev_h = layout(h)
    half = w // 2
    _server(cv, doc, hist, 1, 0, top, half, g)
    _leafs(cv, doc, hist, 1, half, top, w - half, g)
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

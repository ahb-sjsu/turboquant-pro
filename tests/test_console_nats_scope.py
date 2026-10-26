"""The NATS page's scope: the oscilloscope over the fabric's signals, one sample
per poll of the server's monitoring port.

It reads the same per-poll values as panel 9's sparklines (one function,
``fabric_view.readings``), so the two cannot disagree; it is a separate scope
from the index page's, so keys on one never move the other; and its FFT finds
the period of periodic traffic, as a scope's does.
"""

from __future__ import annotations

import math
import re

from turboquant_pro.console import engine as E
from turboquant_pro.console import fabric_view as FV
from turboquant_pro.console import tui
from turboquant_pro.console import viewmodel as VM
from turboquant_pro.console.fabric import FabricMonitor
from turboquant_pro.console.server import ConsoleServer

from .test_fabric import Clock, Fake


def _polls(n, fake, clock, per_poll=40):
    mon = FabricMonitor("http://x:8222", fetch=fake, clock=clock)
    docs = []
    for _ in range(n):
        docs.append(mon.poll())
        clock.t += 1.0
        fake.varz["in_msgs"] += per_poll
    return docs


def test_the_sparklines_and_the_scope_read_the_same_values():
    fake, clock = Fake(), Clock()
    docs = _polls(5, fake, clock)
    hist = FV.History()
    for d in docs:
        hist.add(d)
    for key in FV.SERIES:
        assert hist.series[key][-1] == FV.readings(docs[-1])[key], key
    s = FV.scope_sample(docs[-1], 5)
    assert s["started_unix"] == docs[-1]["t"] and s["id"] == "poll-5"
    assert FV.readings(docs[-1])["in_msgs"] == 40.0  # 40 more each 1 s poll


def test_the_nats_scope_has_its_own_signals_and_units():
    sc = FV.new_scope()
    assert set(ch.signal for ch in sc.channels) <= set(FV.SCOPE_SIGNALS)
    assert sc.signals["in_msgs"].unit == "msg/s" and sc.signals["leaf_rtt"].unit == "ms"
    assert sc.trigger.source == "in_msgs"
    fake, clock = Fake(), Clock()
    for n, d in enumerate(_polls(6, fake, clock), 1):
        sc.feed(FV.scope_sample(d, n))
    vals = [s.values["in_msgs"] for s in sc.buf]
    assert vals[0] is None and vals[1:] == [40.0] * 5  # no rate on the first poll


def test_the_fft_finds_the_period_of_periodic_traffic():
    """Messages in at 50 + 40 sin(2 pi t / 10) per second, one poll a second:
    the NATS scope's FFT puts its peak at 0.1 Hz, a 10 s period."""
    sc = FV.new_scope()
    sc.s_per_div = 20.0  # 200 s on screen
    t0 = 1_000_000.0
    for n in range(200):
        t = t0 + n
        rate = 50.0 + 40.0 * math.sin(2 * math.pi * n / 10.0)
        sc.feed({"started_unix": t, "id": f"poll-{n}", "fabric": {"in_msgs": rate}})
    now = t0 + 199.5
    st = {"scope": sc, "sel_ch": 0, "fft": True, "history": None}
    caption = VM.scope(st, now, 80, 16)["fft"]["caption"][0]
    peak = float(re.search(r"peak ([0-9.e+-]+) Hz", caption).group(1))
    # the frequency grid steps (0.5 - 1/T) / (2 gw - 1), about 0.003 Hz here
    assert abs(peak - 0.1) < 0.004, caption


def _engine_with_nats():
    fake, clock = Fake(), Clock()
    fab = FabricMonitor("http://x:8222", fetch=fake, clock=clock)
    srv = ConsoleServer(None, None, http=False, fabric=fab).start()
    e = E.Engine(srv)
    return e, srv, fake, clock


def test_the_nats_page_carries_its_scope_and_the_fabric():
    e, srv, fake, clock = _engine_with_nats()
    try:
        now = clock.t
        for _ in range(12):  # one poll per update at FABRIC_EVERY_S
            tui.update(e.st, srv, now)
            now += tui.FABRIC_EVERY_S
            clock.t += tui.FABRIC_EVERY_S
            fake.varz["in_msgs"] += 30
        hello = e.handle({"op": "hello"})
        assert [p["name"] for p in hello["pages"]] == ["nats"]
        assert hello["pages"][0]["panels"] == [7, 9]
        v = e.handle(
            {"op": "view", "page": "nats", "scope": [60, 10], "fabric": [100, 20]}
        )
        assert "p7" in v and "p9" in v and "fabric" in v and "p1" not in v
        assert v["p9"]["state"] == "ok" and v["p9"]["rows"]  # panel 9's rows
        legend = " ".join(t for t, _ in v["p7"]["legend"])
        assert "in_msgs" in legend and "latency" not in legend
        assert len(v["fabric"]["spans"]) == 20
        fi = e.st["fabric_inst"]
        assert fi["polls"] >= 10 and fi["autoset_done"]
    finally:
        srv.stop()


def test_keys_on_the_nats_scope_move_only_that_scope():
    e, srv, *_ = _engine_with_nats()
    try:
        index_scope, nats_scope = e.st["scope"], e.st["fabric_inst"]["scope"]
        before_i, before_n = index_scope.s_per_div, nats_scope.s_per_div
        r = e.handle({"op": "key", "zoom": "scope", "name": "right", "page": "nats"})
        assert nats_scope.s_per_div > before_n and index_scope.s_per_div == before_i
        assert "s/div" in r["message"]
        e.handle({"op": "key", "zoom": "scope", "name": "2", "page": "nats"})
        assert nats_scope.channels[1].on and not index_scope.channels[1].on
        assert (
            "poll"
            in e.handle(
                {"op": "key", "zoom": "scope", "name": "enter", "page": "nats"}
            )["message"]
        )
        assert (
            "index page"
            in e.handle({"op": "action", "name": "setup", "page": "nats"})["message"]
        )
    finally:
        srv.stop()


def test_the_fabric_fits_any_height_and_says_what_lists_every_client():
    """The fabric instrument lays out in any height from FIT_MIN_H (all four
    panels), and need_h is the least height at which every client is listed:
    one row fewer and the last one is cut."""
    fake, clock = Fake(), Clock()
    doc = _polls(2, fake, clock)[-1]
    one = doc["connections"][0]
    doc["connections"] = [dict(one, cid=100 + i, name=f"client-{i}") for i in range(10)]
    hist = FV.History()
    small = "\n".join(FV.frame(doc, hist, 120, FV.FIT_MIN_H).text())
    for title in ("1 server", "2 leaf links", "3 clients (10)", "4 events"):
        assert title in small, title
    need = FV.need_h(doc)
    full = "\n".join(FV.frame(doc, hist, 120, need).text())
    assert all(f"client-{i}" in full for i in range(10))
    short = "\n".join(FV.frame(doc, hist, 120, need - 1).text())
    assert sum(f"client-{i}" in short for i in range(10)) == 9
    for h in range(FV.FIT_MIN_H, 80):  # the three parts always fill the frame
        top, conn, ev = FV.layout(h)
        assert 1 + top + conn + ev == h and top >= 4 and ev >= 3 and conn >= 4

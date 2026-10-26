"""The console is one btop-style grid: numbered panels, the instruments among
them, z to maximise one, and sources (the index, a NATS server) each optional."""

from __future__ import annotations

import pytest

from turboquant_pro.cli import main
from turboquant_pro.console import tui
from turboquant_pro.console.fabric import FabricMonitor
from turboquant_pro.console.scope import Scope
from turboquant_pro.console.server import ConsoleServer
from turboquant_pro.console.spectrum import Analyzer

from .test_fabric import Clock, Fake


def _state(**kw):
    return {
        "scope": Scope(),
        "analyzer": Analyzer(),
        "now": 0.0,
        "snap": {},
        "traces": [],
        "readscope": {},
        "qps_hist": [],
        "p95_hist": [],
        "zoom": None,
        **kw,
    }


def _screen(st, w=160, h=48):
    lines = tui.frame(st, w, h).text()
    assert len(lines) == h and all(len(x) == w for x in lines)
    return "\n".join(lines)


def test_canvas_blit_copies_and_clips():
    a, b = tui.Canvas(6, 3), tui.Canvas(4, 2)
    b.put(0, 0, "abcd", "cyan")
    b.put(1, 0, "efgh")
    a.blit(b, 2, 4)  # only the top-left 2x1 of b lands
    assert a.text() == ["      ", "      ", "    ab"]
    assert a.cells[2][4] == ("a", "cyan")


@pytest.mark.parametrize("w,h", [(80, 24), (120, 40), (160, 48), (220, 60)])
def test_one_grid_with_the_panels_of_the_sources_attached(w, h):
    """Eight index panels; panel 9 only when a NATS source is attached."""
    screen = _screen(_state(), w, h)
    for title in ("1 system", "3 pipeline", "4 readscope", "5 index", "6 query"):
        assert title in screen
    assert ("7 scope" in screen) and ("8 spectrum" in screen)
    assert "z zoom" in screen.splitlines()[0]
    assert "9 NATS" not in screen
    with_nats = _screen(_state(snap={"sources": {"index": True, "nats": True}}), w, h)
    assert "9 NATS" in with_nats and "5 index" in with_nats


def test_the_instruments_in_the_grid_keep_their_notes():
    screen = _screen(_state())
    assert "x: time, newest at the right edge" in screen  # scope panel note
    assert "1 sample/s, bar height 0..max" in screen  # throughput note
    assert "x: time" not in _screen(_state(annotate=False))


def test_zoom_maximises_one_panel_and_the_view_field_still_maps():
    grid = _screen(_state())
    scope = _screen(_state(zoom="scope"))
    assert "1 system" in grid and "1 system" not in scope
    assert "Notes" in scope  # the scope's own softkeys, only when zoomed
    assert "Notes" not in grid
    legacy = dict(_state(), view="scope")
    del legacy["zoom"]
    assert tui.zoom_of(legacy) == "scope"
    assert tui.zoom_of({"view": "overview"}) is None
    assert tui.ZOOMABLE == {7: "scope", 8: "spectrum", 9: "fabric"}


def _fabric_doc():
    fake, clock = Fake(), Clock()
    m = FabricMonitor("http://x:8222", fetch=fake, clock=clock)
    m.poll()
    clock.t += 2.0
    fake.varz["in_msgs"] += 40
    return m.poll()


def test_panel_9_shows_every_nats_metric_calibrated():
    from turboquant_pro.console import fabric_view

    # no NATS source: the grid leaves panel 9 out; drawn on its own it says how
    assert "9 NATS" not in _screen(_state())
    alone = tui.Canvas(60, 6)
    tui.draw_panel(alone, _state(), 9)
    assert "not attached: start with --nats URL" in "\n".join(alone.text())
    fake, clock = Fake(), Clock()
    m = FabricMonitor("http://x:8222", fetch=fake, clock=clock)
    hist = fabric_view.History()
    for _ in range(4):
        doc = m.poll()
        hist.add(doc)
        clock.t += 2.0
        fake.varz["in_msgs"] += 40
        fake.varz["in_bytes"] += 4000
    screen = _screen(_state(fabric=doc, fabric_hist=hist))
    assert "9 NATS fabric  rates over 2.0 s polls" in screen
    assert "NATS 1 leaf, 1 clients, 102 subs, 0 slow" in screen
    for label in (
        "msgs in",
        "msgs out",
        "bytes in",
        "bytes out",
        "leaf rtt",
        "leaf msgs",
        "leaf bytes",
        "connects",
        "pending",
        "JS msgs",
    ):
        assert label in screen, label
    assert "20.0 msg/s" in screen and "2.0 kB/s" in screen  # value with its unit
    assert "78.4 ms" in screen and " samp" in screen and " deri" in screen
    assert "max 20.0" in screen  # the sparkline scale, in the row unit
    down = dict(doc, reachable=False)
    assert "UNREACHABLE" in _screen(_state(fabric=down))


def test_graphs_carry_scales_units_legends_and_the_time():
    import re

    st = _state()
    st["scope"].buf.clear()
    scope = _screen(dict(st, zoom="scope"), 160, 48)
    ch = st["scope"].channels[0]
    unit = tui_scope_unit(ch.signal)
    assert "0 s" in scope  # time axis ends at now
    assert f" 1 {ch.signal} " in scope and f"{unit}/div" in scope  # legend, scale
    assert unit in scope.splitlines()[1]  # the y axis names its unit
    stamp = r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\dZ"
    snap = {"t": 1_790_000_000.0}
    assert re.search(stamp, _screen(dict(st, snap=snap)).splitlines()[0])
    assert "--:--:--Z" in _screen(st).splitlines()[0]  # no data yet: no time


def tui_scope_unit(signal):
    from turboquant_pro.console.scope import SIGNALS

    return SIGNALS[signal].unit or "ratio"


def test_the_spectrum_axes_are_in_db_and_directions():
    from turboquant_pro.console import spectrum_view

    assert spectrum_view.YL == 7 and spectrum_view._tick(-57.5) == "-57.5"


def test_P_writes_the_screen_as_text(tmp_path):
    cv = tui.frame(_state(), 120, 40)
    msg = tui.snapshot_txt(cv, str(tmp_path))
    assert msg.startswith("snapshot written: ")
    path = msg.split(": ", 1)[1]
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert len(lines) == 40 and any("1 system" in line for line in lines)
    assert any("snapshot: write the screen" in k[1] for k in tui.KEYS)


def test_the_fabric_zooms_to_its_own_instrument():
    from turboquant_pro.console import fabric_view

    doc = _fabric_doc()
    hist = fabric_view.History()
    hist.add(doc)
    screen = _screen(_state(fabric=doc, fabric_hist=hist, zoom="fabric"), 120, 40)
    assert "2 leaf links (1)" in screen and "Esc back to the grid" in screen


def test_a_console_without_an_index_says_so_and_runs():
    fake, clock = Fake(), Clock()
    fab = FabricMonitor("http://x:8222", fetch=fake, clock=clock)
    s = ConsoleServer(None, None, http=False, fabric=fab).start()
    try:
        snap = s.snapshot()
        assert snap["sources"] == {
            "index": False,
            "nats": True,
            "machine": False,
            "dht": False,
        }
        assert snap["workload"] == {} and snap["paused"] is False
        sw, why = s.spectrum_sweep()
        assert sw is None and "no index attached" in why
        assert s.fabric_poll()["reachable"]
        screen = _screen(_state(snap=snap, fabric=s.fabric_poll()))
        assert "no index attached" in screen and "[no index]" in screen
    finally:
        s.stop()


def test_the_console_needs_a_source(capsys):
    assert main(["console", "--web"]) == 2
    assert "attach a source" in capsys.readouterr().err
    assert main(["console", "--web", "--nats", "http://x:8222"]) == 2
    assert "needs --index or --demo" in capsys.readouterr().err


def test_the_console_caps_its_own_blas_threads():
    """A monitor must not take the machine: numpy's OpenBLAS otherwise starts one
    thread per CPU for the console's own workload (seen: ~700% CPU at 20 q/s)."""
    import sys

    from turboquant_pro.cli import build_parser
    from turboquant_pro.console.threads import limit_blas_threads

    assert build_parser().parse_args(["console", "--demo"]).threads == 1
    done = limit_blas_threads(1)
    if sys.platform.startswith("linux"):
        assert not done.startswith("unchanged"), done
    assert main(["console", "--demo", "--web", "--threads", "0"]) == 2


def test_keys_drive_the_state_the_display_follows():
    """tui.handle_key is the reference key map (the Go client routes keys alike)."""
    st = dict(_state(), focus=6, overlay=None, replay=None, inspected=None, sel=0,
              paused=False)  # fmt: skip
    k = tui.handle_key
    assert k(st, None, "q", 0.0) == "quit"
    k(st, None, "7", 0.0)
    assert st["focus"] == 7
    k(st, None, "z", 0.0)
    assert st["zoom"] == "scope"
    k(st, None, "escape", 0.0)
    assert st["zoom"] is None
    k(st, None, "2", 0.0)  # on the scope, 2 is its channel 2, not panel 2
    assert st["focus"] == 7 and st["scope"].channels[1].on
    k(st, None, "tab", 0.0)  # Tab always moves on
    assert st["focus"] == 8
    k(st, None, "tab", 0.0)
    k(st, None, "5", 0.0)
    k(st, None, "z", 0.0)
    assert st["zoom"] is None and "z opens panels 7" in st["message"]
    k(st, None, "tab", 0.0)
    assert st["focus"] == 6
    k(st, None, "?", 0.0)
    assert st["overlay"] == "help"
    k(st, None, "7", 0.0)  # keys under an overlay do not move focus
    assert st["focus"] == 6
    k(st, None, "escape", 0.0)
    assert st["overlay"] is None
    k(st, None, "i", 0.0)
    assert st["annotate"] is False and st["message"] == "notes off"
    assert tui.in_foreground(-1) is False  # no terminal: never the foreground


def test_every_panel_draws_at_every_size():
    for n in range(1, 10):
        for w, h in ((20, 3), (60, 12), (120, 30)):
            cv = tui.Canvas(w, h)
            tui.draw_panel(cv, _state(), n)
            assert len(cv.text()) == h

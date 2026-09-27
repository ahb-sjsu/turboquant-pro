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
def test_one_grid_with_eight_numbered_panels(w, h):
    screen = _screen(_state(), w, h)
    for n in range(1, 9):
        assert f"{n} " in screen
    for title in ("1 system", "3 pipeline", "4 readscope", "5 index", "6 query"):
        assert title in screen
    assert ("7 scope" in screen) and ("8 spectrum" in screen)
    assert "z zoom" in screen.splitlines()[0]


def test_the_instruments_in_the_grid_keep_their_notes():
    screen = _screen(_state())
    assert "x: time, newest at the right edge" in screen  # scope panel note
    assert "last 4 min, newest right" in screen  # throughput note
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
    assert tui.ZOOMABLE == {1: "fabric", 7: "scope", 8: "spectrum"}


def _fabric_doc():
    fake, clock = Fake(), Clock()
    m = FabricMonitor("http://x:8222", fetch=fake, clock=clock)
    m.poll()
    clock.t += 2.0
    fake.varz["in_msgs"] += 40
    return m.poll()


def test_the_system_panel_says_whether_nats_is_attached():
    assert "NATS: not attached" in _screen(_state())
    doc = _fabric_doc()
    screen = _screen(_state(fabric=doc))
    assert "NATS 1 leaf, 1 clients" in screen and "leaf rtt 78.4 ms samp" in screen
    down = dict(doc, reachable=False)
    assert "UNREACHABLE" in _screen(_state(fabric=down))


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
        assert snap["sources"] == {"index": False, "nats": True}
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

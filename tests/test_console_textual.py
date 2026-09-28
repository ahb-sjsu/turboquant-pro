"""The console's terminal UI on Textual, driven headless by Textual's pilot."""

from __future__ import annotations

import asyncio
import time

import pytest

pytest.importorskip("textual")

from turboquant_pro.console import textual_app as TA  # noqa: E402
from turboquant_pro.console import tui  # noqa: E402
from turboquant_pro.console.server import ConsoleServer, demo_index  # noqa: E402


@pytest.fixture(scope="module")
def srv():
    index, Q, X, source, codec = demo_index(n=1500, dim=64, out_dim=32)
    s = ConsoleServer(
        index, Q, qps=50, k=5, rerank=3, originals=X, source=source, http=False,
        codec=codec,
    ).start()  # fmt: skip
    deadline = time.time() + 10
    while len(s.tracer.traces()) < 20 and time.time() < deadline:
        time.sleep(0.05)
    yield s
    s.stop()


def _run(coro):
    return asyncio.run(coro)


def test_rich_text_keeps_every_cell_and_colour():
    cv = tui.Canvas(6, 2)
    cv.put(0, 0, "ab", "cyan")
    cv.put(1, 2, "cd", "purple_dim")
    t = TA.to_text(cv)
    assert t.plain == "ab    \n  cd  "
    spans = {(s.start, s.end, str(s.style)) for s in t.spans}
    assert (0, 2, "cyan") in spans and (9, 11, "magenta dim") in spans
    assert TA.style_of("sel") == "black on cyan" and TA.style_of("nope") == ""


def test_key_names_are_the_ones_handle_key_takes():
    class E:
        def __init__(self, key, character):
            self.key, self.character = key, character

    assert TA.key_name(E("space", " ")) == "space"
    assert TA.key_name(E("enter", "\r")) == "enter"
    assert TA.key_name(E("P", "P")) == "P"
    assert TA.key_name(E("question_mark", "?")) == "?"
    assert TA.key_name(E("f1", None)) is None


def test_the_grid_zoom_overlay_and_quit_under_textual(srv, tmp_path):
    async def go():
        app = TA.ConsoleApp(srv, export_dir=str(tmp_path))
        async with app.run_test(size=(160, 48)) as pilot:
            await pilot.pause(0.5)
            panels = sorted(p.n for p in app.screen.query(TA.Panel))
            assert panels == list(range(1, 10))
            await pilot.press("7")
            assert app.st["focus"] == 7 and app.screen.focused.id == "p7"
            await pilot.press("z")
            assert isinstance(app.screen, TA.ZoomScreen)
            await pilot.press("escape")
            assert isinstance(app.screen, TA.GridScreen)
            await pilot.press("question_mark")
            assert isinstance(app.screen, TA.OverlayScreen)
            await pilot.press("escape")
            assert isinstance(app.screen, TA.GridScreen)
            await pilot.press("P")
            await pilot.pause(0.2)
            assert "snapshot written" in app.st["message"]
            await pilot.press("q")
        return app.return_code

    assert _run(go()) == 0
    names = {p.suffix for p in tmp_path.iterdir()}
    assert ".txt" in names and ".svg" in names  # P: text and Textual's own SVG


def test_ctrl_c_quits_with_status_130(srv):
    async def go():
        app = TA.ConsoleApp(srv)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            await pilot.press("ctrl+c")
        return app.return_code

    assert _run(go()) == 130


def test_a_short_terminal_folds_the_instruments_to_readout_lines(srv):
    async def go():
        app = TA.ConsoleApp(srv)
        async with app.run_test(size=(100, 26)) as pilot:
            await pilot.pause(0.3)
            return app.screen.query_one("#row2").size.height

    assert _run(go()) == 1


def test_a_headless_app_leaves_the_process_signal_handlers_alone(srv):
    """The background-exit rule is for a real terminal session only: a test (or
    any headless) app must not leave a SIGCONT handler that ends the process."""
    import signal

    before = signal.getsignal(signal.SIGCONT)

    async def go():
        app = TA.ConsoleApp(srv)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.2)
            during = signal.getsignal(signal.SIGCONT)
            await pilot.press("q")
        return during

    during = _run(go())
    assert during is before and signal.getsignal(signal.SIGCONT) is before

# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""The console's terminal UI on Textual.

Textual owns the terminal: raw mode and its restore, resize, the alternate
screen, focus, and (through its Linux driver) Ctrl-Z and ``fg``. The console
contributes three things:

* **Panels.** Each of the nine panels is a focusable widget that draws with the
  character panels in :mod:`.tui` (the same code the tests and the ``P``
  snapshot use) and hands Textual the result as Rich text, so axes, legends
  and units are one implementation.
* **State-driven screens.** A key changes the console state through
  :func:`.tui.handle_key`; the app then shows what the state asks for: the grid,
  a zoomed panel (``z``), or the help / inspect overlay.
* **One job-control rule Textual lacks.** Continued while in the background
  (after ``bg``, or an outside SIGSTOP then SIGCONT such as Atlas's thermal
  guardian), a full-screen program cannot draw: any terminal call draws SIGTTOU,
  and Textual's driver answers that by stopping again, which leaves a stopped
  job behind. Here, continued in the background, the console exits.
"""

from __future__ import annotations

import os
import signal
import time

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import ModalScreen, Screen
from textual.widget import Widget

from . import tui

# The character display's colour roles as Rich styles.
_BASE = {
    None: "",
    "cyan": "cyan",
    "teal": "dark_cyan",
    "purple": "magenta",
    "magenta": "magenta",
    "green": "green",
    "amber": "yellow",
    "yellow": "yellow",
    "red": "red",
    "dim": "grey50",
    "bold": "bold",
    "grid": "blue",
    "sel": "black on cyan",
}
_STYLE_CACHE: dict = {}


def style_of(role) -> str:
    """Rich style for a colour role, including the ``_dim`` / ``_bold`` variants."""
    if role in _STYLE_CACHE:
        return _STYLE_CACHE[role]
    if role in _BASE:
        out = _BASE[role]
    else:
        base, _, var = str(role).rpartition("_")
        out = f"{_BASE.get(base, '')} {var}".strip() if var in ("dim", "bold") else ""
    _STYLE_CACHE[role] = out
    return out


def to_text(cv: tui.Canvas) -> Text:
    """A Canvas as Rich text: one span per run of a colour, rows joined by newlines."""
    t = Text(no_wrap=True, overflow="crop", end="")
    for y, row in enumerate(cv.cells):
        x = 0
        while x < len(row):
            col = row[x][1]
            j = x
            while j < len(row) and row[j][1] == col:
                j += 1
            t.append("".join(c for c, _ in row[x:j]), style_of(col) or None)
            x = j
        if y < len(cv.cells) - 1:
            t.append("\n")
    return t


# ---------------------------------------------------------------- widgets
class Panel(Widget, can_focus=True):
    """Panel ``n`` of the grid."""

    DEFAULT_CSS = "Panel { width: 1fr; }"

    def __init__(self, n: int):
        super().__init__(id=f"p{n}")
        self.n = n

    def render(self) -> Text:
        cv = tui.Canvas(self.size.width, self.size.height)
        tui.draw_panel(cv, self.app.st, self.n, self.app.g)
        return to_text(cv)

    def on_focus(self) -> None:
        self.app.st["focus"] = self.n
        self.app.redraw()


class Line(Widget):
    """A one-row line: the header (status, UTC time, keys) or the message line."""

    DEFAULT_CSS = "Line { height: 1; width: 100%; }"

    def __init__(self, kind: str):
        super().__init__(id=kind)
        self.kind = kind

    def render(self) -> Text:
        st = self.app.st
        cv = tui.Canvas(self.size.width, 1)
        if self.kind == "header":
            tui._header(cv, st, st.get("snap") or {})
        elif st.get("message"):
            cv.put(0, 1, st["message"][: cv.w - 2], "amber")
        return to_text(cv)


class Full(Widget):
    """A whole-screen view drawn by :mod:`.tui`: a zoomed panel or an overlay."""

    DEFAULT_CSS = "Full { width: 100%; height: 1fr; }"

    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind

    def render(self) -> Text:
        st, g = self.app.st, self.app.g
        cv = tui.Canvas(self.size.width, self.size.height)
        if self.kind == "zoom":
            tui._zoomed(cv, st, g, st["zoom"])
        elif st.get("overlay") == "help":
            tui._overlay_help(cv, g, tui._help_for(st.get("zoom")))
        elif st.get("overlay") == "inspect" and st.get("inspected"):
            tui._overlay_inspect(cv, st, g)
        return to_text(cv)


# ---------------------------------------------------------------- screens
class KeysToState:
    """Every key goes to tui.handle_key; the app then follows the state."""

    def on_key(self, event) -> None:
        name = key_name(event)
        if name is None:
            return
        event.stop()
        event.prevent_default()
        self.app.key(name)


class GridScreen(KeysToState, Screen):
    def compose(self) -> ComposeResult:
        yield Line("header")
        with Horizontal(id="row1"):
            yield Panel(1)
            yield Panel(2)
            yield Panel(3)
        with Horizontal(id="row2"):
            yield Panel(7)
            yield Panel(8)
        with Horizontal(id="row3"):
            yield Panel(4)
            yield Panel(5)
            yield Panel(9)
        yield Panel(6)
        yield Line("status")

    def on_mount(self) -> None:
        self.query_one(f"#p{self.app.st['focus']}").focus()
        self.call_after_refresh(self.relayout)  # sizes are known after the first

    def on_resize(self, event) -> None:
        self.relayout()

    def relayout(self) -> None:
        """Row heights by terminal height, as the character grid sets them."""
        h = self.size.height
        rows = self.query("#row1")
        if h <= 0 or not rows:  # not laid out yet: on_resize will call again
            return
        top = 9 if h >= 30 else 7
        if self.app.st.get("fabric") is not None and h >= 44:
            mid = 3 + len(tui.NATS_ROWS)
        else:
            mid = 10 if h >= 44 else 8 if h >= 30 else 6
        room = h - 2 - top - mid - 5  # header, status, and the query stream's 5
        self.query_one("#row1").styles.height = top
        self.query_one("#row3").styles.height = mid
        # the instruments as panels when there is room for a graticule, else
        # one readout line each (z still opens them)
        self.query_one("#row2").styles.height = "2fr" if room >= 16 else 1


class ZoomScreen(KeysToState, Screen):
    def compose(self) -> ComposeResult:
        yield Full("zoom")
        yield Line("status")


class OverlayScreen(KeysToState, ModalScreen):
    DEFAULT_CSS = "OverlayScreen { background: $background 60%; }"

    def compose(self) -> ComposeResult:
        yield Full("overlay")


_NAMED = {"space", "up", "down", "left", "right", "enter", "escape", "tab"}


def key_name(event) -> str | None:
    """A Textual key event as the name :func:`.tui.handle_key` takes."""
    if event.key in _NAMED:
        return event.key
    ch = event.character
    if ch == " ":
        return "space"
    if ch and len(ch) == 1 and ch.isprintable():
        return ch
    return None


# ---------------------------------------------------------------- the app
class ConsoleApp(App):
    """The console session ``srv`` in the terminal."""

    TITLE = "TurboQuant console"
    CSS = """
    GridScreen, ZoomScreen { overflow: hidden hidden; layout: vertical; }
    #row1 { height: 9; }
    #row2 { height: 2fr; }
    #row3 { height: 13; }
    #row1 > Panel, #row2 > Panel, #row3 > Panel { height: 100%; }
    #p6 { width: 100%; height: 1fr; min-height: 4; }
    """
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("ctrl+c", "quit_console", "quit", priority=True, show=False),
        Binding("ctrl+z", "suspend_process", "suspend", priority=True, show=False),
    ]

    def __init__(self, srv, setup: dict | None = None, export_dir: str = "."):
        super().__init__()
        self.srv = srv
        self.export_dir = export_dir
        self.g = tui.UNICODE
        self.st = tui.new_state(srv, setup)

    def get_default_screen(self) -> Screen:
        return GridScreen()

    def on_mount(self) -> None:
        self._tick()
        self.set_interval(0.25, self._tick)
        self._follow()
        if not self.is_headless and _terminal_fd() is not None:
            self._restore_signals = install_job_control(self, _terminal_fd())

    def on_unmount(self) -> None:
        # the handlers belong to this app's terminal session, not the process
        restore = getattr(self, "_restore_signals", None)
        if restore is not None:
            restore()

    # ---- data
    def _tick(self) -> None:
        tui.update(self.st, self.srv, time.time())
        self.redraw()

    def redraw(self) -> None:
        for w in self.screen.query("Panel, Line, Full"):
            w.refresh()

    # ---- keys
    def key(self, name: str) -> None:
        size = (self.size.width, self.size.height)
        out = tui.handle_key(
            self.st, self.srv, name, time.time(), self.export_dir, size, self.g
        )
        if name == "P":  # also the screen exactly as Textual drew it
            try:
                svg = self.save_screenshot(path=self.export_dir)
                self.st["message"] += f"; {svg}"
            except OSError as e:
                self.st["message"] += f"; SVG not written: {e}"
        if out == "quit":
            self.exit(return_code=0)
            return
        self._follow()
        self.redraw()

    def action_quit_console(self) -> None:
        self.exit(return_code=130)  # as a shell reports Ctrl-C

    # ---- screens follow the state
    def _follow(self) -> None:
        want = [GridScreen]
        if self.st.get("zoom"):
            want.append(ZoomScreen)
        if self.st.get("overlay") and (
            self.st["overlay"] == "help" or self.st.get("inspected")
        ):
            want.append(OverlayScreen)
        have = [type(s) for s in self.screen_stack]
        while len(have) > 1 and have != want[: len(have)]:
            self.pop_screen()
            have.pop()
        for cls in want[len(have) :]:
            self.push_screen(cls())
        if isinstance(self.screen, GridScreen):
            self.screen.relayout()
            focus = self.screen.query(f"#p{self.st['focus']}")
            if focus and self.screen.focused is not focus.first():
                focus.first().focus()


def _terminal_fd() -> int | None:
    """The controlling terminal's descriptor (stdin), or None without one."""
    import sys

    try:
        fd = sys.__stdin__.fileno()
    except (AttributeError, ValueError, OSError):
        return None
    return fd if os.isatty(fd) else None


def install_job_control(app: ConsoleApp, fd: int):
    """Continued in the background: exit (a full-screen program cannot draw
    there). SIGTERM / SIGHUP: leave through Textual, which restores the terminal.
    Textual's own SIGCONT handling (resume after its Ctrl-Z) runs otherwise.

    Only for an app on a real terminal; returns a function that puts the
    previous handlers back (called when the app unmounts)."""
    if not hasattr(signal, "SIGCONT"):
        return None
    import asyncio

    loop = asyncio.get_running_loop()  # called from on_mount, inside the app loop
    names = ("SIGCONT", "SIGTERM", "SIGHUP")
    before = {n: signal.getsignal(getattr(signal, n)) for n in names}
    textual_cont = before["SIGCONT"]

    def on_cont(signum, frame):
        if not tui.in_foreground(fd):
            os._exit(128 + signal.SIGCONT)  # no terminal I/O: the shell owns it
        if callable(textual_cont):
            textual_cont(signum, frame)
        app.call_later(app.refresh, repaint=True, layout=True)

    def on_quit(signum, frame):
        code = 128 + signum
        loop.call_soon_threadsafe(lambda: app.exit(return_code=code))

    signal.signal(signal.SIGCONT, on_cont)
    for n in ("SIGTERM", "SIGHUP"):
        signal.signal(getattr(signal, n), on_quit)

    def restore():
        for n, h in before.items():
            signal.signal(getattr(signal, n), h if h is not None else signal.SIG_DFL)

    return restore


def run(srv, setup: dict | None = None, export_dir: str = ".") -> int:
    """Run the console in this terminal; returns the exit status."""
    app = ConsoleApp(srv, setup, export_dir)
    app.run()
    return app.return_code or 0

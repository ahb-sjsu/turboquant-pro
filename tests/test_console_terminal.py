"""The terminal UI as two processes: the engine (Python, this repository's
session and instruments) and the terminal client (go/tqp-console).

The engine's protocol is tested in process. The whole thing is tested end to
end under an interactive bash in a pseudo-terminal (tests/console_jobctl.py):
every job-control path must leave the terminal normal, the shell reading input,
and no client or engine process behind.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time

import pytest

from turboquant_pro.console import engine as E
from turboquant_pro.console import viewmodel as VM
from turboquant_pro.console.fabric import FabricMonitor
from turboquant_pro.console.server import ConsoleServer, demo_index

from .test_fabric import Clock, Fake


@pytest.fixture(scope="module")
def eng():
    index, Q, X, source, codec = demo_index(n=1500, dim=64, out_dim=32)
    fake, clock = Fake(), Clock()
    srv = ConsoleServer(
        index, Q, qps=50, k=5, rerank=3, originals=X, source=source, http=False,
        codec=codec, fabric=FabricMonitor("http://x:8222", fetch=fake, clock=clock),
    ).start()  # fmt: skip
    e = E.Engine(srv)
    threading.Thread(target=e.run, daemon=True).start()
    deadline = time.time() + 15
    while time.time() < deadline and (
        len(srv.tracer.traces()) < 30 or e.st["analyzer"].last is None
    ):
        time.sleep(0.1)
    time.sleep(1.0)
    yield e
    e.halt.set()
    srv.stop()


def test_hello_names_the_keys_titles_and_zoomable_panels(eng):
    h = eng.handle({"op": "hello"})
    assert h["protocol"] == VM.PROTOCOL and h["engine"]["pid"] == os.getpid()
    assert h["zoomable"] == {"7": "scope", "8": "spectrum", "9": "fabric"}
    assert set(h["titles"]) == {str(n) for n in range(1, 10)}
    assert any(k == "Ctrl-Z" for k, _ in h["keys"])


def test_a_view_is_the_data_for_the_geometry_asked(eng):
    v = eng.handle({"op": "view", "scope": [60, 10], "spectrum": [60, 9, 0]})
    assert {"header", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"} <= set(v)
    json.dumps(v, allow_nan=False)  # the wire format is strict JSON
    sc = v["p7"]
    assert sc["gw"] == 60 and sc["gh"] == 10
    on = [c for c in sc["channels"] if c["on"]]
    assert on and len(on[0]["cols"]) == 120  # two braille sub-columns per cell
    assert len(sc["yticks"]) == sc["vdiv"] + 1 and sc["xticks"][-1][1] == "0 s"
    assert any("/div" in text for text, _ in sc["legend"])
    sp = v["p8"]
    assert sp["ok"] and all(len(t["cols"]) == 120 for t in sp["traces"])
    assert sp["yticks"][0][0] == sp["bottom"] and "eff rank" in sp["readout"][0]
    rows = {r[0]: r for r in v["p9"]["rows"]}
    assert rows["leaf rtt"][1].endswith(" ms") and rows["leaf rtt"][2] == "samp"
    assert v["p6"]["ids"] and len(v["p6"]["rows"][0]) == len(VM.QUERY_COLS)


def test_zoomed_views_carry_their_extras(eng):
    v = eng.handle(
        {"op": "view", "grid": False, "scope": [80, 20], "scope_zoom": True,
         "spectrum": [80, 20, 5], "spectrum_zoom": True, "fabric": [120, 30]}
    )  # fmt: skip
    assert "p1" not in v
    assert v["p7"]["softkeys"] and v["p7"]["side"] and v["p7"]["meas"]
    assert len(v["p8"]["waterfall"]["levels"]) <= 5 and v["p8"]["softkeys"]
    assert len(v["fabric"]["spans"]) == 30


def test_keys_actions_and_inspect(eng):
    before = eng.st["scope"].s_per_div
    r = eng.handle({"op": "key", "zoom": "scope", "name": "right"})
    assert eng.st["scope"].s_per_div > before and "s/div" in r["message"]
    r = eng.handle({"op": "key", "zoom": "spectrum", "name": "l"})
    assert "limit line" in r["message"]
    assert eng.handle({"op": "action", "name": "pause"})["message"] == "display paused"
    assert eng.handle({"op": "action", "name": "pause"})["message"] == "live"
    tid = eng.srv.tracer.traces(1)[0]["id"]
    s = eng.handle({"op": "inspect", "id": tid, "replay": True, "w": 120, "h": 40})
    y, x, hh, ww = s["rect"]
    assert len(s["spans"]) == hh and hh <= 38 and ww <= 110
    text = "".join(t for row in s["spans"] for t, _ in row)
    assert f"query {tid}" in text and "identical" in text
    assert "no longer" in eng.handle({"op": "inspect", "id": "nope"})["message"]
    assert "unknown op" in eng.handle({"op": "bogus"})["error"]


def test_the_socket_is_private_and_answers_line_by_line(eng):
    d = tempfile.mkdtemp(prefix="tqp-console-")
    try:
        path = os.path.join(d, "engine.sock")
        srv = E.serve(eng, path)
        assert oct(os.stat(path).st_mode & 0o777) == "0o600"
        c = socket.socket(socket.AF_UNIX)
        c.connect(path)
        f = c.makefile("rwb")
        for req in ({"op": "hello"}, {"op": "view", "grid": False}):
            f.write(json.dumps(req).encode() + b"\n")
            f.flush()
            assert json.loads(f.readline())
        f.write(b"not json\n")
        f.flush()
        assert "error" in json.loads(f.readline())  # answers; never dies
        c.close()
        srv.shutdown()
        E._remove_own_files(path, None)
        assert not os.path.exists(d)  # an emptied private directory goes too
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_cleanup_never_removes_a_directory_that_is_not_empty(tmp_path):
    keep = tmp_path / "keep.txt"
    keep.write_text("x")
    sock = tmp_path / "engine.sock"
    sock.write_text("")
    E._remove_own_files(str(sock), None)
    assert keep.exists() and tmp_path.exists() and not sock.exists()


def test_without_a_built_client_the_console_says_how_to_build_it(monkeypatch, capsys):
    from turboquant_pro import cli

    monkeypatch.setenv("TQP_CONSOLE_CLIENT", "/nonexistent/tqp-console")
    assert cli.console_client_binary() is None
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)
    assert cli.main(["console", "--demo"]) == 2
    assert "go build" in capsys.readouterr().err


# ------------------------------------------------------------------ end to end
def _can_run_end_to_end() -> str | None:
    from turboquant_pro.cli import console_client_binary

    if not sys.platform.startswith("linux"):
        return "POSIX job control and /proc"
    if shutil.which("bash") is None:
        return "no bash"
    if console_client_binary() is None:
        return "the terminal client is not built (go/tqp-console)"
    return None


@pytest.mark.parametrize(
    "case",
    ["q", "Ctrl-C", "Ctrl-Z, fg, q", "outside SIGSTOP, SIGCONT, fg, q", "SIGHUP"],
)
def test_every_exit_leaves_the_terminal_normal_and_nothing_running(case):
    why = _can_run_end_to_end()
    if why:
        pytest.skip(why)
    sys.path.insert(0, os.path.dirname(__file__))
    import console_jobctl as J

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    action = dict(J.CASES)[case]
    cmd = f"{sys.executable} -m turboquant_pro.cli console --demo --qps 20\r".encode()
    r = J.run_case(case, action, cmd, repo, {"PYTHONPATH": repo})
    assert r["drew"], r
    assert r["terminal_normal"], r
    assert not r["forbidden_modes"], r
    assert r["shell_reads_input"], r
    assert not r["left_processes"], r

"""Console Phase 1: the local server's contract and security posture."""

from __future__ import annotations

import http.client
import json
import time

import pytest

from turboquant_pro import telemetry
from turboquant_pro.console.server import ConsoleServer, demo_index
from turboquant_pro.schemas import load_schema

jsonschema = pytest.importorskip("jsonschema")


@pytest.fixture(scope="module")
def srv():
    index, Q, X, source = demo_index(n=2000, dim=64, out_dim=32)
    s = ConsoleServer(index, Q, qps=100, k=5, rerank=3, originals=X, source=source)
    s.start()
    deadline = time.time() + 10
    while len(s.tracer.traces()) < 10 and time.time() < deadline:
        time.sleep(0.05)
    yield s
    s.stop()


def _get(srv, path, token=True, host=None, method="GET"):
    c = http.client.HTTPConnection(srv.host, srv.port, timeout=10)
    h = {"Host": host or f"localhost:{srv.port}"}
    if token:
        h["X-TQP-Token"] = srv.token if token is True else token
    c.request(method, path, headers=h)
    r = c.getresponse()
    body = r.read()
    c.close()
    return r.status, dict(r.getheaders()), body


def test_the_api_requires_the_session_token(srv):
    assert _get(srv, "/api/version", token=False)[0] == 401
    assert _get(srv, "/api/version", token="wrong")[0] == 401
    code, _, body = _get(srv, "/api/version")
    assert code == 200 and json.loads(body)["api_version"]


def test_a_foreign_host_header_is_refused(srv):
    """DNS rebinding: a page on evil.example resolving to 127.0.0.1 is refused even
    with a valid token."""
    assert _get(srv, "/api/version", host="evil.example")[0] == 403
    assert _get(srv, "/", host="evil.example")[0] == 403


def test_nothing_writes_and_static_files_stay_in_their_directory(srv):
    assert _get(srv, "/api/snapshot", method="POST")[0] == 405
    assert _get(srv, "/api/snapshot", method="DELETE")[0] == 405
    assert _get(srv, "/../server.py", token=False)[0] == 404
    assert _get(srv, "/%2e%2e/server.py", token=False)[0] == 404
    code, headers, body = _get(srv, "/", token=False)
    assert code == 200 and b"TurboQuant" in body
    assert "default-src 'self'" in headers["Content-Security-Policy"]


def test_snapshot_readings_and_traces_follow_the_contract(srv):
    snap = json.loads(_get(srv, "/api/snapshot")[2])
    rs = load_schema("metric_reading.schema.json")
    for r in snap["readings"]:
        jsonschema.validate(r, rs)
    names = {r["name"] for r in snap["readings"]}
    assert {"search.qps", "search.latency_ms.p95", "index.compression_ratio"} <= names
    assert snap["workload"]["mode"] == "exact rerank"
    traces = json.loads(_get(srv, "/api/traces?n=5")[2])
    ts = load_schema("query_trace.schema.json")
    for t in traces:
        jsonschema.validate(t, ts)
        assert "workload_row" in t["params"]
    one = json.loads(_get(srv, f"/api/trace/{traces[-1]['id']}")[2])
    assert one["id"] == traces[-1]["id"]
    assert _get(srv, "/api/trace/0000000000000000")[0] == 404


def test_replay_reproduces_a_deterministic_query_and_reports_what_was_pinned(srv):
    t = json.loads(_get(srv, "/api/traces?n=1")[2])[0]
    rep = json.loads(_get(srv, f"/api/replay/{t['id']}")[2])
    assert rep["same_input_sha256"] and rep["pinned"]["query"]
    assert rep["diff"]["same_ids_in_order"]
    assert rep["nondeterminism"] == []


def test_export_bundles_the_session(srv):
    doc = json.loads(_get(srv, "/api/export")[2])
    assert doc["schema"] == "turboquant-pro/console-export"
    assert doc["capabilities"]["features"]["operator_actions"] is False
    assert doc["traces"] and doc["readscope"]["observer"] is None


def test_the_stream_emits_snapshots(srv):
    c = http.client.HTTPConnection(srv.host, srv.port, timeout=10)
    c.request(
        "GET", "/api/stream", headers={"Host": "localhost", "X-TQP-Token": srv.token}
    )
    r = c.getresponse()
    assert r.status == 200 and r.getheader("Content-Type") == "text/event-stream"
    got = b""
    while b"event: snapshot" not in got:
        got += r.fp.readline()
    c.close()


def test_consoles_are_isolated_from_each_other_and_from_the_process_default():
    """Each console traces into its own tracer through a scope on its own thread: two
    consoles never see each other's searches, the process default sees none of them,
    and disabling or stopping anything leaves the others tracing."""
    telemetry.disable()
    default = telemetry.enable()
    a_idx, a_q, _, _ = demo_index(n=500, dim=32, out_dim=16, seed=1)
    b_idx, b_q, _, _ = demo_index(n=500, dim=32, out_dim=16, seed=2)
    a = ConsoleServer(a_idx, a_q, qps=100, http=False).start()
    b = ConsoleServer(b_idx, b_q, qps=100, http=False).start()
    try:
        deadline = time.time() + 10
        while min(len(a.tracer.traces()), len(b.tracer.traces())) < 5:
            assert time.time() < deadline
            time.sleep(0.05)
        a_hashes = {t["input"]["sha256"] for t in a.tracer.traces()}
        b_hashes = {t["input"]["sha256"] for t in b.tracer.traces()}
        assert a_hashes and b_hashes and not (a_hashes & b_hashes)
        assert default.traces() == []  # the consoles never touched the default
        telemetry.disable()  # a library user switching tracing off...
        n = len(a.tracer.traces())
        while len(a.tracer.traces()) <= n:  # ...does not blind a console
            assert time.time() < deadline + 10
            time.sleep(0.05)
        b.stop()
        n = len(a.tracer.traces())
        while len(a.tracer.traces()) <= n:  # nor does another console stopping
            assert time.time() < deadline + 20
            time.sleep(0.05)
    finally:
        a.stop()
        b.stop()  # stopping twice is harmless
    assert not a.workload.is_alive() and not b.workload.is_alive()
    assert telemetry.active() is None


def test_cli_parses_console():
    from turboquant_pro.cli import build_parser

    a = build_parser().parse_args(["console", "--demo"])
    assert a.demo and not a.web and a.host == "127.0.0.1"
    assert build_parser().parse_args(["console", "--demo", "--web"]).web
    with pytest.raises(SystemExit):
        build_parser().parse_args(["console"])


# --------------------------------------------------------------- terminal UI
from turboquant_pro.console import tui  # noqa: E402


@pytest.fixture(scope="module")
def tui_state():
    index, Q, X, source = demo_index(n=1500, dim=64, out_dim=32)
    s = ConsoleServer(
        index, Q, qps=50, k=5, rerank=3, originals=X, source=source, http=False
    ).start()
    deadline = time.time() + 10
    while len(s.tracer.traces()) < 20 and time.time() < deadline:
        time.sleep(0.05)
    snap = s.snapshot()
    st = {
        "snap": snap,
        "traces": s.tracer.traces(200),
        "readscope": s.readscope(),
        "qps_hist": [1.0, 2.0, 3.0, None, 2.5],
        "p95_hist": [1.0, 1.5, 1.2],
        "sel": 0,
        "focus": 6,
        "paused": False,
        "overlay": None,
        "inspected": None,
        "replay": None,
        "message": "",
    }
    yield s, st
    s.stop()


def test_a_terminal_session_opens_no_socket(tui_state):
    s, _ = tui_state
    assert s.httpd is None and s.port is None


@pytest.mark.parametrize("w,h", [(80, 24), (100, 30), (120, 40), (220, 60)])
def test_the_frame_fills_every_size_exactly_and_shows_every_panel(tui_state, w, h):
    _, st = tui_state
    lines = tui.frame(st, w, h).text()
    assert len(lines) == h and all(len(x) == w for x in lines)
    screen = "\n".join(lines)
    for title in (
        "1 system",
        "2 throughput",
        "3 pipeline",
        "4 readscope",
        "5 index",
        "6 query stream",
    ):
        assert title in screen
    assert "QPS" in screen and "encode" in screen and "ADCIndex" in screen
    assert "q quit  ? keys" in lines[0]  # header labels never overwrite the hint
    trace_id = st["traces"][-1]["id"]
    assert trace_id in screen  # the newest trace heads the stream


def test_too_small_says_so(tui_state):
    _, st = tui_state
    assert "too small" in tui.frame(st, 79, 30).text()[0]
    assert "too small" in tui.frame(st, 120, 23).text()[0]
    lines = tui.frame(st, 30, 5).text()
    assert len(lines) == 5 and all(len(x) == 30 for x in lines)


def test_overlays_inspect_with_replay_and_help(tui_state):
    s, st = tui_state
    t = s.tracer.traces(1)[0]  # fresh: the ring evicts old traces as the workload runs
    st2 = dict(st, overlay="inspect", inspected=t, replay=s.replay(t["id"]))
    screen = "\n".join(tui.frame(st2, 120, 40).text())
    assert f"query {t['id']}" in screen and "stages" in screen
    assert "exact rerank" in screen and "identical ids in identical order" in screen
    help_screen = "\n".join(tui.frame(dict(st, overlay="help"), 100, 30).text())
    assert "replay the query and compare" in help_screen


def test_ascii_fallback_and_primitives():
    assert tui.bar(0.5, 10, tui.ASCII) == "#####....."
    assert tui.spark([None], 8, tui.UNICODE).startswith("collect")
    sp = tui.spark([1, 2, 4, 8], 4, tui.UNICODE)
    assert len(sp) == 4 and sp[-1] == "█"
    assert tui.fmt(None) == "-" and tui.fmt(12345.6) == "12346"


def test_the_terminal_ui_starts_draws_and_quits_in_a_pty():
    """End to end on a pseudo-terminal: `tqp console --demo` draws, then `q` exits 0."""
    import os
    import subprocess
    import sys

    pty = pytest.importorskip("pty")
    if sys.platform.startswith("win"):
        pytest.skip("no pty on Windows")
    master, slave = pty.openpty()
    env = dict(
        os.environ,
        TERM="xterm-256color",
        LINES="40",
        COLUMNS="120",
        LANG="C.UTF-8",
        LC_ALL="C.UTF-8",
    )
    p = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from turboquant_pro.cli import main; "
            "raise SystemExit(main(['console', '--demo', '--qps', '50']))",
        ],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        close_fds=True,
    )
    os.close(slave)
    out = b""
    deadline = time.time() + 60
    import select

    while time.time() < deadline and b"query stream" not in out:
        r, _, _ = select.select([master], [], [], 0.5)
        if r:
            try:
                out += os.read(master, 65536)
            except OSError:
                break
    assert b"query stream" in out, out[-500:]
    os.write(master, b"q")
    assert p.wait(timeout=20) == 0
    os.close(master)

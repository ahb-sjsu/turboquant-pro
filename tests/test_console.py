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


def test_stop_releases_the_tracer():
    index, Q, X, _ = demo_index(n=500, dim=32, out_dim=16)
    s = ConsoleServer(index, Q, qps=10).start()
    assert telemetry.active() is s.tracer
    s.stop()
    assert telemetry.active() is None


def test_cli_parses_console():
    from turboquant_pro.cli import build_parser

    a = build_parser().parse_args(["console", "--demo", "--no-browser"])
    assert a.demo and a.host == "127.0.0.1"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["console"])

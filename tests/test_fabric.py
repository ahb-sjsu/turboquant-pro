"""The NATS fabric instrument (``tqp fabric``, console.fabric).

The monitor is driven by canned monitoring responses shaped like a real NATS
2.10 server's (a leaf link from an NRP namespace, local clients, JetStream), and
end to end through ``tqp fabric`` against a local HTTP server serving them.
"""

from __future__ import annotations

import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from turboquant_pro.cli import main
from turboquant_pro.console import fabric_view
from turboquant_pro.console.fabric import FabricMonitor, parse_duration
from turboquant_pro.schemas import load_schema, validate

jsonschema = pytest.importorskip("jsonschema")

VARZ = {
    "server_id": "NCDLUJ",
    "version": "2.10.24",
    "start": "2026-04-16T03:35:37Z",
    "uptime": "163d20h51m12s",
    "connections": 2,
    "total_connections": 1000,
    "leafnodes": 1,
    "subscriptions": 102,
    "slow_consumers": 0,
    "in_msgs": 1000,
    "out_msgs": 5000,
    "in_bytes": 10_000,
    "out_bytes": 90_000,
    "mem": 56_475_648,
    "cpu": 1.0,
}
LEAF = {
    "name": "NCHIOWFBU7ND",
    "is_spoke": False,
    "account": "$G",
    "ip": "152.55.153.65",
    "port": 7583,
    "rtt": "78.441859ms",
    "in_msgs": 10,
    "out_msgs": 20,
    "in_bytes": 1000,
    "out_bytes": 2000,
    "subscriptions": 2,
    "subscriptions_list": ["burst.submit", "$SYS.REQ.USER.INFO"],
    "compression": "s2_better",
}
CONN = {
    "cid": 7,
    "name": "nats-bursting",
    "lang": "go",
    "ip": "127.0.0.1",
    "rtt": "291µs",
    "uptime": "15d18h6m1s",
    "idle": "4m43s",
    "in_msgs": 100,
    "out_msgs": 100,
    "in_bytes": 5000,
    "out_bytes": 9000,
    "pending_bytes": 0,
    "subscriptions_list": ["burst.submit"],
}
JSZ = {
    "streams": 1,
    "consumers": 0,
    "messages": 80,
    "bytes": 4000,
    "account_details": [
        {"stream_detail": [{"name": "AGI_EVENTS", "state": {"messages": 80}}]}
    ],
}


class Fake:
    """Monitoring endpoints whose state a test moves between polls."""

    def __init__(self):
        self.varz, self.leafs, self.conns = dict(VARZ), [dict(LEAF)], [dict(CONN)]
        self.down = set()

    def __call__(self, path):
        ep = path.split("?")[0].strip("/")
        if ep in self.down:
            raise OSError(f"{ep} down")
        return copy.deepcopy(
            {
                "varz": self.varz,
                "leafz": {"leafs": self.leafs},
                "connz": {"connections": self.conns, "total": len(self.conns)},
                "jsz": JSZ,
            }[ep]
        )


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def mon():
    fake, clock = Fake(), Clock()
    return FabricMonitor("http://x:8222", fetch=fake, clock=clock), fake, clock


def _valid(doc):
    jsonschema.validate(doc, load_schema("fabric_snapshot.schema.json"))


@pytest.mark.parametrize(
    "text,seconds",
    [
        ("78.441859ms", 0.078441859),
        ("291µs", 291e-6),
        ("1m22s", 82.0),
        ("163d20h51m12s", 163 * 86400 + 20 * 3600 + 51 * 60 + 12),
        ("1.5s", 1.5),
        ("0s", 0.0),
    ],
)
def test_durations_parse(text, seconds):
    assert parse_duration(text) == pytest.approx(seconds)


@pytest.mark.parametrize("bad", ["", "abc", "12", "5x", "1m22", None, 3])
def test_non_durations_are_none(bad):
    assert parse_duration(bad) is None


def test_the_first_poll_has_counters_and_no_rates(mon):
    m, _, _ = mon
    doc = m.poll()
    _valid(doc)
    assert doc["reachable"] and doc["interval_s"] is None
    assert doc["server"]["in_msgs"] == 1000
    assert all(v is None for v in doc["rates"].values())
    lf = doc["leafs"][0]
    assert lf["rtt_ms"] == pytest.approx(78.441859)
    assert lf["subjects"] == sorted(LEAF["subscriptions_list"])
    assert doc["events"] == []
    why = {r["name"]: r.get("unavailable_reason") for r in m.readings(doc)}
    assert "a rate needs two" in why["nats.in_msgs_per_s"]


def test_rates_are_derived_over_the_poll_interval(mon):
    m, fake, clock = mon
    m.poll()
    clock.t += 2.0
    fake.varz.update(in_msgs=1100, out_msgs=5400, total_connections=1004)
    fake.leafs[0].update(in_msgs=30, out_bytes=2600)
    doc = m.poll()
    _valid(doc)
    assert doc["interval_s"] == pytest.approx(2.0)
    assert doc["rates"]["in_msgs_per_s"] == pytest.approx(50.0)
    assert doc["rates"]["out_msgs_per_s"] == pytest.approx(200.0)
    assert doc["rates"]["connects_per_min"] == pytest.approx(120.0)
    lr = doc["leafs"][0]["rates"]
    assert lr["in_msgs_per_s"] == pytest.approx(10.0)
    assert lr["out_bytes_per_s"] == pytest.approx(300.0)
    rd = {r["name"]: r for r in m.readings(doc)}
    assert rd["nats.leaf.in_msgs_per_s"]["value"] == pytest.approx(10.0)
    assert rd["nats.leaf.in_msgs_per_s"]["kind"] == "derived"
    assert rd["nats.leaf.rtt_ms"]["kind"] == "sampled"
    for r in rd.values():
        jsonschema.validate(r, load_schema("metric_reading.schema.json"))


def test_a_counter_going_backwards_gives_no_rate_and_an_event(mon):
    m, fake, clock = mon
    m.poll()
    clock.t += 2.0
    fake.leafs[0].update(in_msgs=1)  # the same link, its counter reset
    doc = m.poll()
    assert doc["leafs"][0]["rates"]["in_msgs_per_s"] is None
    assert [e["kind"] for e in doc["events"]] == ["counter_reset"]


def test_a_leaf_reconnect_is_a_disconnect_and_a_connect(mon):
    m, fake, clock = mon
    m.poll()
    clock.t += 2.0
    fake.leafs[0].update(port=9999, in_msgs=0, out_msgs=0)
    doc = m.poll()
    kinds = sorted(e["kind"] for e in doc["events"])
    assert kinds == ["leaf_connected", "leaf_disconnected"]
    assert doc["leafs"][0]["rates"]["in_msgs_per_s"] is None  # a new link
    clock.t += 2.0
    fake.leafs = []
    doc = m.poll()
    assert [e["kind"] for e in doc["events"]] == ["leaf_disconnected"]
    why = {r["name"]: r.get("unavailable_reason") for r in m.readings(doc)}
    assert why["nats.leaf.rtt_ms"] == "no leaf node connected"


def test_a_restart_resets_the_baseline(mon):
    m, fake, clock = mon
    m.poll()
    clock.t += 2.0
    fake.varz.update(start="2026-09-27T00:00:00Z", in_msgs=5)
    doc = m.poll()
    assert [e["kind"] for e in doc["events"]] == ["server_restarted"]
    assert doc["interval_s"] is None and doc["rates"]["in_msgs_per_s"] is None


def test_slow_consumers_are_an_event(mon):
    m, fake, clock = mon
    m.poll()
    clock.t += 2.0
    fake.varz["slow_consumers"] = 3
    assert m.poll()["events"] == [{"kind": "slow_consumers", "detail": "3 new"}]


def test_unreachable_is_a_state_and_every_reading_says_why(mon):
    m, fake, clock = mon
    m.poll()
    fake.down.add("varz")
    clock.t += 2.0
    doc = m.poll()
    _valid(doc)
    assert not doc["reachable"] and "varz down" in doc["errors"]["varz"]
    assert [e["kind"] for e in doc["events"]] == ["server_unreachable"]
    rd = m.readings(doc)
    assert all(
        r["value"] is None and "unreachable" in r["unavailable_reason"] for r in rd
    )


def test_a_section_that_cannot_be_read_says_so(mon):
    m, fake, _ = mon
    fake.down.update({"leafz", "jsz"})
    doc = m.poll()
    _valid(doc)
    assert doc["leafs"] == [] and doc["jetstream"] is None
    assert set(doc["errors"]) == {"leafz", "jsz"}


def test_redaction_hashes_addresses():
    m = FabricMonitor("http://x", fetch=Fake(), redact=True)
    doc = m.poll()
    assert doc["source"]["redacted"]
    assert doc["leafs"][0]["ip"].startswith("ip:")
    assert "152.55" not in json.dumps(doc)


@pytest.mark.parametrize("w,h", [(80, 24), (120, 40), (200, 60)])
def test_the_frame_fills_the_terminal_and_names_what_it_shows(mon, w, h):
    m, fake, clock = mon
    hist = fabric_view.History()
    for _ in range(3):
        hist.add(m.poll())
        clock.t += 2.0
        fake.varz["in_msgs"] += 40
    doc = m.poll()
    hist.add(doc)
    lines = fabric_view.frame(doc, hist, w, h).text()
    assert len(lines) == h and all(len(x) == w for x in lines)
    screen = "\n".join(lines)
    for label in ("1 server", "2 leaf links (1)", "3 clients (1)", "4 events"):
        assert label in screen
    assert "rtt 78.4 ms (sampled)" in screen
    assert "rates over 2.0 s" in screen


def test_no_leaf_is_said_not_hidden(mon):
    m, fake, _ = mon
    fake.leafs = []
    screen = "\n".join(
        fabric_view.frame(m.poll(), fabric_view.History(), 100, 30).text()
    )
    assert "no leaf node connected" in screen


# ---- end to end: tqp fabric against a local monitoring port -------------------


@pytest.fixture
def server():
    fake = Fake()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            try:
                body = json.dumps(fake(self.path)).encode()
                self.send_response(200)
            except OSError:
                body = b"{}"
                self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", fake
    httpd.shutdown()
    httpd.server_close()


def test_tqp_fabric_writes_a_valid_snapshot_with_its_invocation(
    server, tmp_path, capsys
):
    url, _ = server
    out = str(tmp_path / "fabric.json")
    argv = ["fabric", "--url", url, "--interval", "0.2", "--out", out]
    assert main(argv) == 0
    assert "1 leaf link(s)" in capsys.readouterr().out
    doc = json.loads(open(out, encoding="utf-8").read())
    assert validate(doc)["status"] == "valid"
    assert doc["interval_s"] == pytest.approx(0.2, abs=0.15)
    assert doc["invocation"]["argv"] == ["tqp", *argv]


def test_tqp_fabric_exits_1_when_the_port_is_unreachable(capsys):
    assert main(["fabric", "--url", "http://127.0.0.1:9", "--interval", "0.05",
                 "--timeout", "0.5", "--once"]) == 1  # fmt: skip
    assert "UNREACHABLE" in capsys.readouterr().out


def test_an_unchanging_rtt_is_reported_as_possibly_old(mon):
    m, fake, clock = mon
    assert m.poll()["leafs"][0]["rtt_unchanged_s"] is None  # first sight
    clock.t += 300.0
    doc = m.poll()
    assert doc["leafs"][0]["rtt_unchanged_s"] == pytest.approx(300.0)
    screen = "\n".join(fabric_view.frame(doc, fabric_view.History(), 160, 40).text())
    assert "sampled, unchanged 5.0m" in screen
    clock.t += 2.0
    fake.leafs[0]["rtt"] = "80ms"  # the server measured it again
    assert m.poll()["leafs"][0]["rtt_unchanged_s"] == 0.0

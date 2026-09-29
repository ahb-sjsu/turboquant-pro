"""The DHT page: the console's reader (console/dht.py) and panels
(console/dht_view.py) against the tqp-dht daemon's own snapshot and server.

The contract test builds a snapshot with the plugin's snapshot(), serves it with
the plugin's SnapshotServer and reads it with the console's DhtMonitor: what the
daemon writes and what the console reads cannot drift apart. No libtorrent: the
daemon imports it only to run a session.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from turboquant_pro.cli import main
from turboquant_pro.console import dht_view as DV
from turboquant_pro.console import viewmodel as VM
from turboquant_pro.console.dht import DhtMonitor, History

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "plugins", "tqp-dht")
)
from tqp_dht import daemon as D  # noqa: E402
from tqp_dht.krpc import LookupTracer, bencode  # noqa: E402

OURS = bytes(20)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _metrics(msgs_in, nodes):
    return {
        "dht.dht_nodes": {"value": nodes, "type": "gauge"},
        "dht.dht_node_cache": {"value": 40, "type": "gauge"},
        "dht.dht_messages_in": {"value": msgs_in, "type": "counter"},
        "dht.dht_messages_out": {"value": msgs_in // 2, "type": "counter"},
        "dht.dht_bytes_in": {"value": msgs_in * 100, "type": "counter"},
        "dht.dht_bytes_out": {"value": msgs_in * 50, "type": "counter"},
        "dht.dht_get_peers_in": {"value": msgs_in // 4, "type": "counter"},
        "dht.dht_invalid_get_peers": {"value": 2, "type": "counter"},
    }


def _tracer():
    """Two lookups, one finished: bencoded packets as the daemon would see them."""
    tr = LookupTracer({OURS}, idle_s=10.0)
    target = bytes([0xF0]) * 20

    def near(k):  # an id sharing exactly k leading bits with target
        t = int.from_bytes(target, "big")
        return (t ^ (1 << (159 - k))).to_bytes(20, "big")

    def q(tid, tgt):
        a = {b"id": OURS, b"target": tgt}
        return bencode({b"t": tid, b"y": b"q", b"q": b"find_node", b"a": a})

    def r(tid, rid):
        return bencode({b"t": tid, b"y": b"r", b"r": {b"id": rid}})

    for i, k in enumerate((4, 11, 19)):
        tr.observe(q(bytes([i]), target), i)
        tr.observe(r(bytes([i]), near(k)), i + 0.5)
    other = bytes([0x0F]) * 20
    tr.observe(q(b"z", other), 30.0)  # the first lookup goes idle: finished
    tr.observe(q(b"y", other), 30.1)  # a second walk, live
    tr.observe(q(b"x", other), 30.2)
    return tr


def _swarms():
    row = {"name": "debian-13.7.0-amd64-netinst.iso", "state": "seeding",
           "progress": 1.0, "peers": 4, "seeds": 1, "swarm_seeds": 300,
           "swarm_peers": 12, "upload_Bps": 150000, "download_Bps": 0,
           "uploaded": 2_000_000, "downloaded": 700_000_000, "ratio": 0.003,
           "sha256": {"status": "verified"}}  # fmt: skip
    bad = dict(row, name="archlinux.iso", sha256={"status": "mismatch"})
    return [row, bad]


@pytest.fixture
def served():
    srv = D.SnapshotServer(port=0).start()
    node = {"ids": [OURS.hex()], "listen_port": 6881, "dht_running": True}
    routing = [
        {"bucket": i, "nodes": n, "replacements": 1} for i, n in enumerate([8, 8, 5, 2])
    ]

    def publish(msgs_in, nodes):
        m = _metrics(msgs_in, nodes)
        active = [{"type": "get_peers"}]
        tables = [routing, routing[:1]]  # two DHT nodes: the fuller one is drawn
        labels = {(bytes([0xF0]) * 20).hex(): "debian-13.7.0-amd64-netinst.iso"}
        doc = D.snapshot(
            node, m, tables, active, _tracer(), _swarms(), 1.0, None, labels
        )
        srv.publish(doc)

    yield srv, publish
    srv.stop()


def test_the_reader_turns_counters_into_rates_and_keeps_gauges(served):
    srv, publish = served
    clock = Clock()
    mon = DhtMonitor(f"http://127.0.0.1:{srv.port}", clock=clock)
    publish(1000, 150)
    first = mon.poll()
    assert first["reachable"] and first["interval_s"] is None
    assert first["rates"]["dht.dht_messages_in"] is None  # no rate on the first poll
    assert "dht.dht_nodes" not in first["rates"]  # a gauge is a level, not a rate
    clock.t += 2.0
    publish(1400, 160)
    second = mon.poll()
    assert second["interval_s"] == 2.0
    assert second["rates"]["dht.dht_messages_in"] == 200.0
    assert second["rates"]["dht.dht_bytes_in"] == 20000.0
    clock.t += 2.0
    publish(10, 160)  # the daemon restarted: its counters went backwards
    assert mon.poll()["rates"]["dht.dht_messages_in"] is None


def test_what_the_reader_cannot_use_is_said():
    mon = DhtMonitor(
        "http://127.0.0.1:9", fetch=lambda: (_ for _ in ()).throw(OSError("refused"))
    )
    doc = mon.poll()
    assert not doc["reachable"] and "refused" in doc["error"]
    assert (
        "not a tqp-dht" in DhtMonitor(fetch=lambda: b'{"schema": "x"}').poll()["error"]
    )
    assert "not a tqp-dht" in DhtMonitor(fetch=lambda: b"[1, 2]").poll()["error"]
    panels = DV.panels({"dht": doc})
    assert all(
        p["state"] == "none" and "UNREACHABLE" in p["message"] for p in panels.values()
    )
    assert DV.panels({})["5"]["message"] == "not attached: start with --dht URL"


def test_the_five_panels_from_the_daemons_own_snapshot(served):
    srv, publish = served
    clock = Clock()
    mon, hist = DhtMonitor(f"http://127.0.0.1:{srv.port}", clock=clock), History()
    for n in (1000, 1400):
        publish(n, 150)
        doc = mon.poll()
        hist.add(doc)
        clock.t += 2.0
    p = DV.panels({"dht": doc, "dht_hist": hist})
    json.dumps(p, allow_nan=False)
    assert sorted(p) == ["1", "2", "3", "4", "5"] and all(
        x["state"] == "ok" for x in p.values()
    )
    rows = {r[0]: r for r in p["1"]["rows"]}
    assert rows["nodes"][1:3] == ["150", "meas"] and rows["msgs in"][2] == "deri"
    assert rows["msgs in"][1] == "200 /s" and rows["invalid"][1] == "0.0 /s"
    assert "no port mapping, outbound only" in p["1"]["summary"][0]
    assert (
        p["2"]["summary"][0] == "23 nodes in 4 buckets  (fullest of 2 DHT nodes: 23, 8)"
    )
    assert p["2"]["strips"][0] == ["fill", [1.0, 1.0, 0.625, 0.25]]
    looks = p["3"]["table"]["rows"]
    assert (
        looks[0][-1] == "live" and looks[1][-1] == "done" and looks[1][5] == "19 bits"
    )
    assert looks[1][1] == "debian-13.7" and looks[0][1] == "-"  # an announce, named
    assert "0 refresh probes" in p["3"]["summary"][0]
    assert "paired 3/3 (100 %)" in p["3"]["summary"][0]
    strips = p["4"]["strips"]
    assert strips[0][1] == [4 / 32, 11 / 32, 19 / 32]  # best prefix after each response
    assert "median best prefix 19 bits over 1 finished" in p["4"]["summary"][0]
    sw = p["5"]["table"]
    assert sw["roles"] == ["green", "red"] and sw["rows"][0][5] == "300/12"


@pytest.mark.parametrize(
    "sources,names",
    [({"dht": True}, ["dht"]), ({"index": True, "machine": True, "dht": True},
                                ["index", "machine", "dht"])],
)  # fmt: skip
def test_the_dht_page_follows_the_source(sources, names):
    pages = VM.pages(sources)
    assert [p["name"] for p in pages] == names
    assert pages[-1]["panels"] == [1, 2, 3, 4, 5] and pages[-1]["titles"][
        "4"
    ].startswith("4 convergence")


def test_the_dht_source_is_for_the_terminal_console(capsys):
    assert main(["console", "--dht", "http://127.0.0.1:8290", "--web"]) == 2
    assert "terminal console only" in capsys.readouterr().err

"""The daemon's pure parts: the pinned list, node ids, swarm rows, the snapshot
and its server. libtorrent is not needed (it is imported only by run())."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from tqp_dht import daemon as D
from tqp_dht.krpc import LookupTracer, bencode


def test_the_pinned_torrents_are_allowed_and_well_formed():
    assert D.check_torrents(D.TORRENTS) == []
    assert all(t["sha256"] and t["torrent"].startswith("https://") for t in D.TORRENTS)


@pytest.mark.parametrize(
    "change,why",
    [
        ({"torrent": "http://cdimage.debian.org/x.torrent"}, "https"),
        ({"torrent": "https://evil.example/x.torrent"}, "https"),
        ({"sha256": "ABC"}, "sha256"),
        ({"name": "../etc/passwd"}, "plain name"),
    ],
)
def test_anything_else_is_refused(change, why):
    t = dict(D.TORRENTS[0], **change)
    assert any(why in b for b in D.check_torrents([t]))


def test_the_same_name_twice_is_refused():
    assert any("twice" in b for b in D.check_torrents([D.TORRENTS[0]] * 2))


def test_the_session_opens_nothing_on_the_router_and_is_capped():
    s = D.SESSION
    assert s["enable_upnp"] is False and s["enable_natpmp"] is False
    assert s["enable_lsd"] is False and s["enable_dht"] is True
    assert s["upload_rate_limit"] == 2_000_000 and s["connections_limit"] == 50


def test_the_snapshot_reports_the_settings_read_back():
    held = D.applied({**D.SESSION, "enable_upnp": True})
    assert held["upnp"] is True and held["natpmp"] is False and held["dht"] is True
    assert held["upload_rate_limit"] == 2_000_000
    doc = D.snapshot({}, {}, [], [], LookupTracer(), [], t=1.0, limits=held)
    assert doc["limits"] is held
    assert (
        D.snapshot({}, {}, [], [], LookupTracer(), [], t=1.0)["limits"]["upnp"] is False
    )


def test_our_node_ids_from_either_state_format():
    a, b = bytes([1]) * 20, bytes([2]) * 20
    assert D.our_node_ids(bencode({b"node-id": a})) == {a}
    nids = [[b"1.2.3.4", a], [b"::1", b]]
    assert D.our_node_ids(bencode({b"nids": nids})) == {a, b}
    assert D.our_node_ids(b"garbage") == set()
    # the form libtorrent 2.1.1 returns once the DHT runs: a dictionary
    assert D.our_node_ids({"node-id": a.decode("latin-1")}) == {a}
    assert D.our_node_ids({b"nids": [(b"1.2.3.4", a)]}) == {a}
    # 2.1.1 with one DHT node per listen socket: a list of ids
    assert D.our_node_ids({b"node-id": [a, b, b"short"]}) == {a, b}
    # as 2.1.1 writes them: each the 20-byte id, then the node's address
    entries = [a + bytes([192, 168, 0, 7]), b + bytes(16)]
    assert D.our_node_ids({b"node-id": entries}) == {a, b}
    assert D.our_node_ids(None) == set() and D.our_node_ids({"node-id": 5}) == set()


def test_a_swarm_row_says_what_is_unknown():
    st = SimpleNamespace(
        name="x.iso", state="torrent_status.states.seeding", progress=1.0,
        num_peers=3, num_seeds=1, num_complete=-1, num_incomplete=12,
        upload_rate=1000, download_rate=0, all_time_upload=500,
        all_time_download=0,
    )  # fmt: skip
    row = D.swarm_row(st, {"status": "verified"})
    assert row["state"] == "seeding" and row["swarm_seeds"] is None
    assert row["swarm_peers"] == 12 and row["ratio"] is None  # nothing downloaded


def test_the_snapshot_server_serves_one_path_read_only_on_the_loopback():
    srv = D.SnapshotServer(port=0).start()
    try:
        assert srv.httpd.server_address[0] == "127.0.0.1"
        doc = D.snapshot({"ids": []}, {}, [], [], LookupTracer(), [], t=1.0)
        json.dumps(doc, allow_nan=False)
        srv.publish(doc)
        base = f"http://127.0.0.1:{srv.port}"
        got = json.loads(urllib.request.urlopen(base + "/snapshot", timeout=5).read())
        assert got["schema"] == D.SCHEMA and got["limits"]["upnp"] is False
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/other", timeout=5)
        assert e.value.code == 404
        req = urllib.request.Request(base + "/snapshot", data=b"x", method="POST")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=5)
        assert e.value.code == 405
    finally:
        srv.stop()

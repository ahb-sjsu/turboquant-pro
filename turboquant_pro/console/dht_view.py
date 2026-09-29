# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""The DHT page: five panels from :mod:`console.dht` snapshots, in the one
PanelView shape the machine page uses (see :mod:`console.machine_view`).

A Kademlia lookup is a routed nearest-neighbour search: each response brings
nodes whose ids share more leading bits with the target. Panels 3 and 4 show
that walk, rebuilt by the daemon from its own packets (derived); the rest is
libtorrent's own accounting (gauges measured, counters as derived rates).
"""

from __future__ import annotations

import statistics

from .machine_view import _empty, _size
from .tui import _si

TITLES = {
    "1": "1 DHT node",
    "2": "2 routing table",
    "3": "3 lookups  rebuilt from our node's packets",
    "4": "4 convergence  best shared prefix after each response",
    "5": "5 swarms",
}
PANELS = [1, 2, 3, 4, 5]
PREFIX_FULL = 32  # a convergence cell is full at this many shared bits


def _series(hist, key):
    if hist is None:
        return None
    return [None if v is None else float(v) for v in hist.series.get(key, [])][-256:]


def _rate(doc, name):
    return (doc.get("rates") or {}).get(name)


def panels(st: dict) -> dict:
    doc, hist = st.get("dht"), st.get("dht_hist")
    if doc is None:
        return {str(n): _empty("not attached: start with --dht URL") for n in PANELS}
    if not doc.get("reachable"):
        why = f"{doc['source']['url']}: UNREACHABLE ({doc.get('error') or 'no answer'})"
        return {str(n): _empty(why) for n in PANELS}
    s = doc["state"]
    return {
        "1": node(doc, s, hist),
        "2": routing(s),
        "3": lookups(s, doc["t"]),
        "4": convergence(s),
        "5": swarms(s),
    }


def node(doc: dict, s: dict, hist=None) -> dict:
    m = s.get("metrics") or {}
    n = s.get("node") or {}
    ids = n.get("ids") or []
    lim = s.get("limits") or {}
    head = f"id {ids[0][:12]}…" if ids else "id not known yet"
    summary = (
        f"{head} ({len(ids)} id{'s' if len(ids) != 1 else ''})  port "
        f"{n.get('listen_port', '-')}  DHT {'on' if n.get('dht_running') else 'OFF'}"
        f"  no port mapping, outbound only"
        if not (lim.get("upnp") or lim.get("natpmp"))
        else f"{head}  port mapping ON"
    )

    def gauge(name):
        return (m.get(name) or {}).get("value")

    def per_s(label, name, note="", key=None):
        v = _rate(doc, name)
        return [
            label,
            _si(v, "/s"),
            "deri",
            _series(hist, key) if key else None,
            note,
            "",
        ]

    invalid = [_rate(doc, k) for k in m if k.startswith("dht.dht_invalid_")]
    inv = None if not invalid or None in invalid else sum(invalid)
    nodes = gauge("dht.dht_nodes")
    held = (
        f"cache {gauge('dht.dht_node_cache') or 0:.0f}"
        f"  torrents {gauge('dht.dht_torrents') or 0:.0f}"
    )
    rows = [
        ["nodes", "-" if nodes is None else f"{nodes:.0f}", "meas",
         _series(hist, "nodes"), held, ""],
        per_s("msgs in", "dht.dht_messages_in", key="msgs_in"),
        per_s("msgs out", "dht.dht_messages_out", key="msgs_out"),
        ["bytes in", _si(_rate(doc, "dht.dht_bytes_in"), "B/s"), "deri",
         _series(hist, "bytes_in"), "", ""],
        ["bytes out", _si(_rate(doc, "dht.dht_bytes_out"), "B/s"), "deri",
         _series(hist, "bytes_out"), "", ""],
        per_s("get_peers", "dht.dht_get_peers_in", "asked of us"),
        per_s("find_node", "dht.dht_find_node_in", "asked of us"),
        per_s("announce", "dht.dht_announce_peer_in", "asked of us"),
        per_s("our queries", "dht.dht_get_peers_out", "get_peers we sent"),
        ["invalid", _si(inv, "/s"), "deri", None, "malformed queries from others", ""],
    ]  # fmt: skip
    return {"state": "ok", "summary": [summary, "cyan"], "title_extra": _poll(doc),
            "rows": rows}  # fmt: skip


def _poll(doc):
    dt = doc.get("interval_s")
    return f"rates over {dt:.1f} s polls" if dt else "rates: need a second poll"


def routing(s: dict) -> dict:
    rt = s.get("routing_table") or []
    if not rt:
        return _empty("routing table empty: the node is still bootstrapping")
    most = max(b["nodes"] for b in rt) or 1
    total = sum(b["nodes"] for b in rt)
    sizes = [sum(b["nodes"] for b in t) for t in s.get("routing_tables") or [rt]]
    nodes = (
        ""
        if len(sizes) < 2
        else (f"  (fullest of {len(sizes)} DHT nodes: {', '.join(map(str, sizes))})")
    )
    rows = [[str(b["bucket"]), str(b["nodes"]), str(b["replacements"])] for b in rt]
    return {
        "state": "ok",
        "summary": [f"{total} nodes in {len(rt)} buckets{nodes}", "cyan"],
        "strips": [["fill", [b["nodes"] / most for b in rt]]],
        "table": {
            "cols": [
                ["bucket", 7, True],
                ["nodes", 6, True],
                ["replacements", 13, True],
            ],
            "rows": rows,
            "roles": [""] * len(rows),
        },
        "note": "bucket i holds nodes sharing i leading bits with ours; cell: nodes /"
        " the fullest bucket",
    }


def lookups(s: dict, now: float) -> dict:
    looks = s.get("lookups") or []
    tr = s.get("tracer") or {}
    paired, unpaired = tr.get("responses_paired", 0), tr.get("responses_unpaired", 0)
    share = f"{100 * paired / (paired + unpaired):.0f} %" if paired + unpaired else "-"
    summary = (
        f"{len(s.get('active_lookups') or [])} running;"
        f" paired {paired}/{paired + unpaired} ({share});"
        f" {tr.get('probes', 0)} refresh probes"
    )
    if not looks:
        return {"state": "ok", "summary": [summary, "cyan"],
                "note": "no lookup seen yet: they start as the node bootstraps and"
                " announces"}  # fmt: skip
    rows, roles = [], []
    for lk in looks:
        rows.append([
            lk["method"], (lk.get("for") or "-")[:11], lk["target"][:10],
            str(lk["queries"]), str(lk["responses"]),
            "-" if lk["best_prefix"] is None else f"{lk['best_prefix']} bits",
            str(lk.get("near_target", 0)),
            f"{max(0.0, now - lk['last']):.0f} s", "live" if lk["live"] else "done",
        ])  # fmt: skip
        roles.append("" if lk["live"] else "dim")
    return {
        "state": "ok",
        "summary": [summary, "cyan"],
        "table": {
            "cols": [["method", 10, False], ["for", 12, False], ["target", 11, False],
                     ["queries", 8, True], ["responses", 10, True], ["best", 9, True],
                     ["near", 5, True], ["idle", 6, True], ["", 5, False]],
            "rows": rows,
            "roles": roles,
        },  # fmt: skip
        "note": "derived: queries paired with responses by transaction id; a lookup"
        " is three or more queries toward one target",
    }


def convergence(s: dict) -> dict:
    looks = [lk for lk in (s.get("lookups") or []) if lk["curve"]]
    if not looks:
        return _empty("no lookup has had a response yet")
    strips = [
        [lk["target"][:6], [min(1.0, b / PREFIX_FULL) for b in lk["curve"]]]
        for lk in looks
    ]
    finals = [lk["best_prefix"] for lk in looks if not lk["live"]]
    if finals:
        med = statistics.median(finals)
        summary = (
            f"median best prefix {med:.0f} bits over {len(finals)} finished lookups:"
            f" about 2^{med:.0f} = {_si(2.0 ** med, '')} nodes (rough estimate)"
        )
    else:
        summary = "no lookup has finished yet"
    return {
        "state": "ok",
        "summary": [summary, "cyan"],
        "strips": strips,
        "note": f"one cell per response, its height the best prefix so far (full at"
        f" {PREFIX_FULL} bits); ids chosen near the target (over 40 shared bits,"
        " the near column) are left out",
    }


def swarms(s: dict) -> dict:
    sw = s.get("swarms") or []
    if not sw:
        return _empty("no torrents")
    rows, roles = [], []
    for t in sw:
        v = t.get("sha256") or {}
        status = v.get("status", "-")
        swarm = (
            f"{t['swarm_seeds'] if t['swarm_seeds'] is not None else '?'}/"
            f"{t['swarm_peers'] if t['swarm_peers'] is not None else '?'}"
        )
        rows.append([
            t["name"][:30], t["state"], f"{100 * t['progress']:.1f} %",
            str(t["peers"]), str(t["seeds"]), swarm,
            _si(t["upload_Bps"], "B/s"), _si(t["download_Bps"], "B/s"),
            "-" if t["ratio"] is None else f"{t['ratio']:.2f}", status,
        ])  # fmt: skip
        roles.append(
            {"mismatch": "red", "error": "red", "verified": "green"}.get(status, "")
        )
    up = sum(t["uploaded"] for t in sw)
    return {
        "state": "ok",
        "summary": [f"{len(sw)} images, {_size(up)} uploaded this session", "cyan"],
        "table": {
            "cols": [["image", 31, False], ["state", 12, False], ["done", 8, True],
                     ["peers", 6, True], ["seeds", 6, True], ["swarm s/p", 10, True],
                     ["up", 10, True], ["down", 10, True], ["ratio", 6, True],
                     ["sha256", 9, False]],
            "rows": rows,
            "roles": roles,
        },  # fmt: skip
        "note": "sha256: the finished image against the hash its project published",
    }

# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""The NATS fabric, read through the server's monitoring port.

A NATS server with ``http_port`` set serves its own state as JSON: ``/varz``
(the server), ``/leafz`` (leaf-node links, such as the one an NRP namespace
dials into), ``/connz`` (client connections) and ``/jsz`` (JetStream).
:class:`FabricMonitor` polls those endpoints and nothing else. It opens no NATS
connection, subscribes to nothing and publishes nothing, so watching the fabric
cannot change what flows over it, and it reads sizes and counts, never message
content.

Each poll is one ``turboquant-pro/fabric-snapshot`` document. Counters are
**measured** (the server's own totals); rates are **derived** from the change
between two polls over the time between them, and are ``null`` on the first
poll, after a counter went backwards (a reconnect or restart) and for a link that
was not there last time; round-trip times are **sampled** (the server's last
PING to that peer). Changes between polls (a leaf link appearing or going, a
server restart, new slow consumers) are reported as events.

    mon = FabricMonitor("http://127.0.0.1:8222")
    doc = mon.poll()           # counters, no rates yet
    time.sleep(2)
    doc = mon.poll()           # rates over those 2 s, and any events
    mon.readings(doc)          # metric readings (telemetry.metrics)
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request

SCHEMA = "turboquant-pro/fabric-snapshot"
SCHEMA_VERSION = 1

__all__ = ["FabricMonitor", "SCHEMA", "parse_duration"]

_DURATION = re.compile(r"(\d+(?:\.\d+)?)(d|h|ms|m|µs|μs|us|ns|s)")
_UNIT_S = {
    "d": 86400.0,
    "h": 3600.0,
    "m": 60.0,
    "s": 1.0,
    "ms": 1e-3,
    "µs": 1e-6,
    "μs": 1e-6,
    "us": 1e-6,
    "ns": 1e-9,
}


def parse_duration(text) -> float | None:
    """Seconds in a NATS duration string (``"78.44ms"``, ``"487µs"``,
    ``"1m22s"``, ``"163d20h51m12s"``); None when it is not one."""
    if not isinstance(text, str) or not text:
        return None
    pos, total = 0, 0.0
    for m in _DURATION.finditer(text):
        if m.start() != pos:
            return None
        total += float(m.group(1)) * _UNIT_S[m.group(2)]
        pos = m.end()
    return total if pos == len(text) else None


def _http_fetch(base: str, timeout: float):
    def fetch(path: str) -> dict:
        with urllib.request.urlopen(base.rstrip("/") + path, timeout=timeout) as r:
            return json.load(r)

    return fetch


def _ms(text) -> float | None:
    s = parse_duration(text)
    return None if s is None else s * 1e3


_COUNTERS = ("in_msgs", "out_msgs", "in_bytes", "out_bytes")


def _rates(cur: dict, prev: dict | None, dt: float | None) -> dict:
    """Per-second change of each counter; None where it cannot be derived."""
    out = {}
    for c in _COUNTERS:
        a, b = cur.get(c), (prev or {}).get(c)
        ok = prev is not None and dt and dt > 0 and a is not None and b is not None
        out[f"{c}_per_s"] = (a - b) / dt if ok and a >= b else None
    return out


def _reset(cur: dict, prev: dict | None) -> bool:
    return prev is not None and any(
        cur.get(c) is not None and prev.get(c) is not None and cur[c] < prev[c]
        for c in _COUNTERS
    )


class FabricMonitor:
    """Polls one NATS server's monitoring port (read-only)."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:8222",
        *,
        timeout: float = 3.0,
        redact: bool = False,
        fetch=None,
        clock=time.time,
    ):
        self.url = url
        self.redact = redact
        self._fetch = fetch or _http_fetch(url, timeout)
        self._clock = clock
        self._prev: dict | None = None  # last reachable poll's raw state
        self._rtt_seen: dict = {}  # leaf key -> (rtt text, first poll it had it)

    # -------------------------------------------------------------- polling
    def _ip(self, ip):
        if not self.redact or not ip:
            return ip
        return "ip:" + hashlib.sha256(str(ip).encode()).hexdigest()[:8]

    def poll(self) -> dict:
        """One snapshot document. The server section is required; a leaf,
        connection or JetStream section that cannot be read carries its error
        instead of being left out."""
        t = self._clock()
        doc = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "source": {"url": self.url, "redacted": self.redact},
            "t": t,
            "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)),
            "reachable": False,
            "interval_s": None,
            "server": None,
            "rates": None,
            "leafs": [],
            "connections": [],
            "jetstream": None,
            "errors": {},
            "events": [],
        }
        try:
            varz = self._fetch("/varz")
        except Exception as e:  # noqa: BLE001 - unreachable is a state, not a crash
            doc["errors"]["varz"] = f"{type(e).__name__}: {e}"
            if self._prev is not None:
                doc["events"].append({"kind": "server_unreachable", "detail": str(e)})
            return doc
        doc["reachable"] = True
        prev = self._prev
        restarted = prev is not None and prev["varz"].get("start") != varz.get("start")
        if restarted:
            doc["events"].append(
                {"kind": "server_restarted", "detail": f"start {varz.get('start')}"}
            )
            prev = None  # nothing before a restart is comparable
        dt = None if prev is None else t - prev["t"]
        doc["interval_s"] = dt

        doc["source"].update(
            server_id=varz.get("server_id"),
            server_name=varz.get("server_name"),
            version=varz.get("version"),
            start=varz.get("start"),
            uptime_s=parse_duration(varz.get("uptime")),
        )
        server = {
            k: varz.get(k)
            for k in (
                "connections",
                "total_connections",
                "leafnodes",
                "subscriptions",
                "slow_consumers",
                *_COUNTERS,
            )
        }
        server["mem_bytes"] = varz.get("mem")
        server["cpu_percent"] = varz.get("cpu")
        doc["server"] = server
        rates = _rates(server, prev and prev["server"], dt)
        tc, ptc = server.get("total_connections"), (prev or {}).get("server", {})
        ptc = ptc.get("total_connections") if ptc else None
        rates["connects_per_min"] = (
            60.0 * (tc - ptc) / dt
            if dt and tc is not None and ptc is not None and tc >= ptc
            else None
        )
        doc["rates"] = rates
        if prev is not None:
            new_slow = (server.get("slow_consumers") or 0) - (
                prev["server"].get("slow_consumers") or 0
            )
            if new_slow > 0:
                doc["events"].append(
                    {"kind": "slow_consumers", "detail": f"{new_slow} new"}
                )

        doc["leafs"], leaf_raw = self._leafs(doc, prev, dt)
        doc["connections"], conn_raw = self._connections(doc, prev, dt)
        doc["jetstream"] = self._jetstream(doc)
        self._prev = {
            "t": t,
            "varz": varz,
            "server": server,
            "leafs": leaf_raw,
            "conns": conn_raw,
        }
        return doc

    def _leafs(self, doc, prev, dt):
        try:
            leafz = self._fetch("/leafz?subs=1")
        except Exception as e:  # noqa: BLE001
            doc["errors"]["leafz"] = f"{type(e).__name__}: {e}"
            return [], (prev or {}).get("leafs", {})
        before = (prev or {}).get("leafs", {})
        raw, out = {}, []
        for lf in leafz.get("leafs") or []:
            key = f"{lf.get('name')}@{lf.get('ip')}:{lf.get('port')}"
            counters = {c: lf.get(c) for c in _COUNTERS}
            raw[key] = counters
            old = before.get(key)
            if prev is not None and old is None:
                doc["events"].append(
                    {"kind": "leaf_connected", "detail": self._leaf_label(lf)}
                )
            if _reset(counters, old):
                doc["events"].append(
                    {"kind": "counter_reset", "detail": self._leaf_label(lf)}
                )
            # The server refreshes a link's RTT when it PINGs; a value that has
            # not changed across polls may be old, so say for how long it has not.
            rtt, seen = lf.get("rtt"), self._rtt_seen.get(key)
            if seen is None or seen[0] != rtt:
                self._rtt_seen[key] = (rtt, doc["t"])
                unchanged = None if seen is None else 0.0
            else:
                unchanged = doc["t"] - seen[1]
            out.append(
                {
                    "name": lf.get("name"),
                    "account": lf.get("account"),
                    "ip": self._ip(lf.get("ip")),
                    "port": lf.get("port"),
                    "is_spoke": lf.get("is_spoke"),
                    "rtt_ms": _ms(rtt),
                    "rtt_unchanged_s": unchanged,
                    **counters,
                    "subscriptions": lf.get("subscriptions"),
                    "subjects": sorted(lf.get("subscriptions_list") or []),
                    "compression": lf.get("compression"),
                    "rates": _rates(counters, old, dt),
                }
            )
        if prev is not None:
            for key in set(before) - set(raw):
                doc["events"].append(
                    {"kind": "leaf_disconnected", "detail": key.split("@")[0][:12]}
                )
        return out, raw

    def _leaf_label(self, lf) -> str:
        return f"{str(lf.get('name'))[:12]} from {self._ip(lf.get('ip'))}"

    def _connections(self, doc, prev, dt):
        try:
            connz = self._fetch("/connz?subs=1&limit=1024")
        except Exception as e:  # noqa: BLE001
            doc["errors"]["connz"] = f"{type(e).__name__}: {e}"
            return [], (prev or {}).get("conns", {})
        before = (prev or {}).get("conns", {})
        raw, out = {}, []
        for c in connz.get("connections") or []:
            cid = c.get("cid")
            counters = {k: c.get(k) for k in _COUNTERS}
            raw[cid] = counters
            out.append(
                {
                    "cid": cid,
                    "name": c.get("name"),
                    "lang": c.get("lang"),
                    "ip": self._ip(c.get("ip")),
                    "rtt_ms": _ms(c.get("rtt")),
                    "uptime_s": parse_duration(c.get("uptime")),
                    "idle_s": parse_duration(c.get("idle")),
                    **counters,
                    "pending_bytes": c.get("pending_bytes"),
                    "subjects": sorted(c.get("subscriptions_list") or []),
                    "rates": _rates(counters, before.get(cid), dt),
                }
            )
        if connz.get("total") is not None and connz["total"] > len(out):
            doc["errors"]["connz"] = (
                f"{connz['total'] - len(out)} connections beyond the listing limit "
                "are not shown"
            )
        return out, raw

    def _jetstream(self, doc):
        try:
            js = self._fetch("/jsz?streams=1")
        except Exception as e:  # noqa: BLE001 - JetStream is optional
            doc["errors"]["jsz"] = f"{type(e).__name__}: {e}"
            return None
        streams = []
        for acc in js.get("account_details") or []:
            for s in acc.get("stream_detail") or []:
                st = s.get("state") or {}
                streams.append(
                    {
                        "name": s.get("name"),
                        "messages": st.get("messages"),
                        "bytes": st.get("bytes"),
                        "consumers": st.get("consumer_count"),
                    }
                )
        return {
            "streams": js.get("streams"),
            "consumers": js.get("consumers"),
            "messages": js.get("messages"),
            "bytes": js.get("bytes"),
            "stream_detail": streams,
        }

    # ------------------------------------------------------------- readings
    def readings(self, doc: dict) -> list:
        """The snapshot as metric readings (each with unit, window and kind)."""
        from ..telemetry.metrics import reading

        t = doc["t"]
        if not doc["reachable"]:
            why = f"NATS monitoring unreachable: {doc['errors'].get('varz')}"
            return [reading(n, None, reason=why, as_of=t) for n in FABRIC_METRICS]
        s, r, leafs = doc["server"], doc["rates"], doc["leafs"]
        first = "first poll: a rate needs two"

        def rate(v, why=first):
            return v, (None if v is not None else why)

        def leaf_sum(key):
            if not leafs:
                return None, "no leaf node connected"
            vals = [lf["rates"][key] for lf in leafs]
            if any(v is None for v in vals):
                return None, "a leaf link is new or reset since the last poll"
            return sum(vals), None

        rtts = [lf["rtt_ms"] for lf in leafs if lf["rtt_ms"] is not None]
        values = {
            "nats.leafnodes": (s.get("leafnodes"), None),
            "nats.leaf.rtt_ms": (
                max(rtts) if rtts else None,
                None if rtts else "no leaf node connected",
            ),
            "nats.leaf.in_msgs_per_s": leaf_sum("in_msgs_per_s"),
            "nats.leaf.out_msgs_per_s": leaf_sum("out_msgs_per_s"),
            "nats.leaf.in_bytes_per_s": leaf_sum("in_bytes_per_s"),
            "nats.leaf.out_bytes_per_s": leaf_sum("out_bytes_per_s"),
            "nats.in_msgs_per_s": rate(r.get("in_msgs_per_s")),
            "nats.out_msgs_per_s": rate(r.get("out_msgs_per_s")),
            "nats.connections": (s.get("connections"), None),
            "nats.connects_per_min": rate(r.get("connects_per_min")),
            "nats.slow_consumers": (s.get("slow_consumers"), None),
        }
        out = []
        for name, (v, why) in values.items():
            if v is None and why is None:
                why = "the server did not report it"
            out.append(reading(name, v, reason=why, as_of=t))
        return out


FABRIC_METRICS = (
    "nats.leafnodes",
    "nats.leaf.rtt_ms",
    "nats.leaf.in_msgs_per_s",
    "nats.leaf.out_msgs_per_s",
    "nats.leaf.in_bytes_per_s",
    "nats.leaf.out_bytes_per_s",
    "nats.in_msgs_per_s",
    "nats.out_msgs_per_s",
    "nats.connections",
    "nats.connects_per_min",
    "nats.slow_consumers",
)

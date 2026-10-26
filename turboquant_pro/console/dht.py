# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""The BitTorrent DHT, read from the tqp-dht daemon's snapshot.

The daemon (``plugins/tqp-dht``) runs the torrent session and serves its state
on the loopback. :class:`DhtMonitor` polls ``GET /snapshot`` and nothing else:
it opens no torrent session and sends nothing to the DHT, so watching cannot
change what it watches (the same stance as :mod:`console.fabric`).

Each poll is one ``turboquant-pro/dht-snapshot`` document. The daemon's gauges
(nodes known) are **measured**; its counters (messages, bytes, queries) are
turned into per-second rates, **derived** from the change between two polls,
``None`` on the first poll and after a counter went backwards (a restart).
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections import deque

SCHEMA = "turboquant-pro/dht-snapshot"
SCHEMA_VERSION = 1
DAEMON_KIND = "tqp-dht/state"  # what the daemon serves; read here, not written

__all__ = ["DhtMonitor", "History", "SCHEMA"]


def _http_fetch(url: str, timeout: float):
    def fetch() -> bytes:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310
            return r.read(8 << 20)

    return fetch


class DhtMonitor:
    """Polls a tqp-dht daemon; each :meth:`poll` returns one snapshot document."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:8290",
        *,
        timeout: float = 3.0,
        fetch=None,
        clock=time.time,
    ):
        self.url = url.rstrip("/")
        if not self.url.endswith("/snapshot"):
            self.url += "/snapshot"
        self._fetch = fetch or _http_fetch(self.url, timeout)
        self._clock = clock
        self._prev: dict | None = None  # the last reachable poll's metrics
        self._prev_t: float | None = None

    def poll(self) -> dict:
        t = self._clock()
        doc = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "source": {"url": self.url},
            "t": t,
            "reachable": False,
            "interval_s": None,
            "state": None,
            "rates": {},
            "error": None,
        }
        try:
            raw = json.loads(self._fetch())
        except (OSError, ValueError) as e:
            doc["error"] = str(e) or type(e).__name__
            self._prev = self._prev_t = None
            return doc
        kind = raw.get("schema") if isinstance(raw, dict) else type(raw).__name__
        if kind != DAEMON_KIND:
            doc["error"] = f"not a tqp-dht snapshot ({kind!r})"
            return doc
        metrics = raw.get("metrics") or {}
        dt = None if self._prev_t is None else t - self._prev_t
        prev = self._prev or {}
        rates = {}
        for name, m in metrics.items():
            if m.get("type") != "counter":
                continue
            old = (prev.get(name) or {}).get("value")
            v = m.get("value")
            ok = old is not None and v is not None and dt and dt > 0 and v >= old
            rates[name] = (v - old) / dt if ok else None
        doc.update(reachable=True, interval_s=dt, state=raw, rates=rates)
        self._prev, self._prev_t = metrics, t
        return doc


SERIES = ("nodes", "msgs_in", "msgs_out", "bytes_in", "bytes_out")


class History:
    """The last ``n`` values of each series the DHT panels draw."""

    def __init__(self, n: int = 256):
        self.series = {k: deque(maxlen=n) for k in SERIES}

    def add(self, doc: dict) -> None:
        m = ((doc.get("state") or {}).get("metrics")) or {}
        r = doc.get("rates") or {}
        vals = {
            "nodes": (m.get("dht.dht_nodes") or {}).get("value"),
            "msgs_in": r.get("dht.dht_messages_in"),
            "msgs_out": r.get("dht.dht_messages_out"),
            "bytes_in": r.get("dht.dht_bytes_in"),
            "bytes_out": r.get("dht.dht_bytes_out"),
        }
        for k, v in vals.items():
            self.series[k].append(v)

# tqp-dht: a BitTorrent DHT source for the TurboQuant Pro console
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""The DHT's own messages, read: bencode, KRPC (BEP 5), and lookups rebuilt from
the packets our node sends and receives.

A Kademlia lookup is a routed nearest-neighbour search over 160-bit node ids,
where the distance between two ids is their XOR. Each response to a
``find_node`` or ``get_peers`` query returns nodes closer to the target; the
**shared prefix** of two ids (the number of leading bits they agree on,
160 minus the bit length of their XOR) is how close, 0 far away and 160 equal.

libtorrent reports each lookup's progress (outstanding, responses, timeouts)
but not the distance at each step. The packets carry it, so this module pairs
our outgoing queries with their responses by transaction id and records, per
lookup, the best shared prefix reached after each response. It is pure: bytes
in, numbers out, no libtorrent, so it is tested on packets built by hand.
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque

ID_BITS = 160
COMPACT_NODE = 26  # BEP 5: 20-byte id, 4-byte IPv4 address, 2-byte port
LOOKUP_METHODS = (b"find_node", b"get_peers")
# A lookup opens with several queries at once (libtorrent's branching factor); one
# or two queries to a target are a bucket-refresh probe or its retry, not a walk
MIN_QUERIES = 3
# An id sharing more than NEAR_BITS leading bits with the target was chosen near it
# (as index crawlers and Sybil nodes do): with N nodes the chance that a random id
# shares that many is about N / 2**NEAR_BITS, under 1e-3 for the ~2**30 nodes of the
# public DHT. Such ids are counted apart and never enter the convergence.
NEAR_BITS = 40


class BencodeError(ValueError):
    pass


def bdecode(data: bytes):
    """Decode one bencoded value that is the whole of ``data`` (strict: trailing
    bytes, leading zeros and non-bytes dictionary keys are errors)."""
    value, end = _decode(data, 0)
    if end != len(data):
        raise BencodeError(f"{len(data) - end} trailing bytes")
    return value


def _decode(d: bytes, i: int):
    if i >= len(d):
        raise BencodeError("truncated")
    c = d[i : i + 1]
    if c == b"i":
        end = d.index(b"e", i)
        s = d[i + 1 : end]
        if not s or (s != b"0" and s.lstrip(b"-").startswith(b"0")) or s == b"-0":
            raise BencodeError(f"bad integer {s!r}")
        return int(s), end + 1
    if c == b"l":
        out, i = [], i + 1
        while d[i : i + 1] != b"e":
            v, i = _decode(d, i)
            out.append(v)
        return out, i + 1
    if c == b"d":
        out, i = {}, i + 1
        while d[i : i + 1] != b"e":
            k, i = _decode(d, i)
            if not isinstance(k, bytes):
                raise BencodeError("dictionary key is not a string")
            out[k], i = _decode(d, i)
        return out, i + 1
    if c.isdigit():
        colon = d.index(b":", i)
        n = int(d[i:colon])
        start = colon + 1
        if start + n > len(d):
            raise BencodeError("string runs past the end")
        return d[start : start + n], start + n
    raise BencodeError(f"unexpected byte {c!r} at {i}")


def bencode(v) -> bytes:
    """Encode (for tests and for building packets): bytes, int, list, dict."""
    if isinstance(v, bytes):
        return str(len(v)).encode() + b":" + v
    if isinstance(v, int):
        return b"i" + str(v).encode() + b"e"
    if isinstance(v, list):
        return b"l" + b"".join(bencode(x) for x in v) + b"e"
    if isinstance(v, dict):
        return b"d" + b"".join(bencode(k) + bencode(v[k]) for k in sorted(v)) + b"e"
    raise TypeError(type(v).__name__)


def shared_prefix(a: bytes, b: bytes) -> int:
    """Leading bits two ids agree on: 160 - bitlen(a XOR b)."""
    return ID_BITS - (int.from_bytes(a, "big") ^ int.from_bytes(b, "big")).bit_length()


def compact_nodes(blob: bytes) -> list:
    """BEP 5 compact node info: (id, "a.b.c.d", port) per 26 bytes."""
    if len(blob) % COMPACT_NODE:
        raise BencodeError(f"compact nodes length {len(blob)} is not a multiple of 26")
    out = []
    for k in range(0, len(blob), COMPACT_NODE):
        e = blob[k : k + COMPACT_NODE]
        out.append(
            (e[:20], ".".join(str(x) for x in e[20:24]), int.from_bytes(e[24:], "big"))
        )
    return out


class Lookup:
    """One lookup: its target, and the best shared prefix after each response."""

    __slots__ = ("method", "target", "started", "last", "curve", "queries", "near")

    def __init__(self, method: str, target: bytes, t: float):
        self.method, self.target = method, target
        self.started = self.last = t
        self.curve: list = []  # best shared prefix so far, one entry per response
        self.queries = 0
        self.near = 0  # ids returned that share more than NEAR_BITS with the target

    def summary(self) -> dict:
        return {
            "method": self.method,
            "target": self.target.hex(),
            "started": self.started,
            "last": self.last,
            "queries": self.queries,
            "responses": len(self.curve),
            "best_prefix": self.curve[-1] if self.curve else None,
            "curve": list(self.curve),
            "near_target": self.near,
        }


class LookupTracer:
    """Rebuilds our node's lookups from its DHT packets.

    ``our_ids`` are this node's ids (one per address family); a query whose
    ``a.id`` is one of them is ours. A response is paired with the query of the
    same transaction id sent less than ``timeout_s`` before it; the lookup is
    identified by its target, and a target idle for ``idle_s`` starts a new one.
    Unpaired responses are counted, not guessed at.
    """

    def __init__(self, our_ids=(), timeout_s: float = 15.0, idle_s: float = 30.0,
                 keep: int = 64, clock=time.time):  # fmt: skip
        self.our_ids = set(our_ids)
        self.timeout_s, self.idle_s = timeout_s, idle_s
        self._clock = clock
        self._pending: dict = {}  # tid -> (target, t, method)
        self._live: OrderedDict = OrderedDict()  # (method, target) -> Lookup
        self.done: deque = deque(maxlen=keep)
        self.counts = {"packets": 0, "undecodable": 0, "queries_out": 0,
                       "queries_in": 0, "responses_paired": 0,
                       "responses_unpaired": 0, "errors": 0,
                       "probes": 0}  # fmt: skip

    def observe(self, pkt: bytes, t: float | None = None) -> None:
        t = self._clock() if t is None else t
        self.counts["packets"] += 1
        try:
            m = bdecode(pkt)
        except (BencodeError, ValueError, IndexError):
            self.counts["undecodable"] += 1
            return
        if not isinstance(m, dict):
            self.counts["undecodable"] += 1
            return
        kind, tid = m.get(b"y"), m.get(b"t")
        if kind == b"q":
            a = m.get(b"a") or {}
            if a.get(b"id") in self.our_ids:
                self._query(m.get(b"q"), a, tid, t)
            else:
                self.counts["queries_in"] += 1
        elif kind == b"r":
            self._response(m.get(b"r") or {}, tid, t)
        elif kind == b"e":
            self.counts["errors"] += 1
        self._expire(t)

    def _query(self, method, a, tid, t):
        self.counts["queries_out"] += 1
        if method not in LOOKUP_METHODS or not isinstance(tid, bytes):
            return
        target = a.get(b"target") if method == b"find_node" else a.get(b"info_hash")
        if not isinstance(target, bytes) or len(target) != 20:
            return
        self._pending[tid] = (target, t, method.decode())
        key = (method.decode(), target)
        lk = self._live.get(key)
        if lk is None:
            lk = self._live[key] = Lookup(method.decode(), target, t)
        lk.queries += 1
        lk.last = t

    def _response(self, r, tid, t):
        sent = self._pending.pop(tid, None) if isinstance(tid, bytes) else None
        if sent is None or t - sent[1] > self.timeout_s or r.get(b"id") in self.our_ids:
            self.counts["responses_unpaired"] += 1
            return
        target, _, method = sent
        lk = self._live.get((method, target))
        if lk is None:
            self.counts["responses_unpaired"] += 1
            return
        self.counts["responses_paired"] += 1
        ids = []
        rid = r.get(b"id")
        if isinstance(rid, bytes) and len(rid) == 20:
            ids.append(rid)
        blob = r.get(b"nodes")
        if isinstance(blob, bytes):
            try:
                ids += [nid for nid, _, _ in compact_nodes(blob)]
            except BencodeError:
                pass
        best = lk.curve[-1] if lk.curve else 0
        for i in ids:
            p = shared_prefix(i, target)
            if p > NEAR_BITS:
                lk.near += 1
            else:
                best = max(best, p)
        lk.curve.append(best)
        lk.last = t

    def _expire(self, t):
        for tid in [k for k, v in self._pending.items() if t - v[1] > self.timeout_s]:
            del self._pending[tid]
        for key in [k for k, lk in self._live.items() if t - lk.last > self.idle_s]:
            lk = self._live.pop(key)
            if lk.queries >= MIN_QUERIES:
                self.done.append(lk)
            else:  # a single query to a target: a bucket-refresh probe, counted
                self.counts["probes"] += 1

    def lookups(self, n: int = 16) -> list:
        """The ``n`` most recent lookups, newest first: live ones, then done. A
        lookup is a walk: at least MIN_QUERIES queries toward one target; the
        one-query probes that refresh the routing table are counted, not listed."""
        live = [lk for lk in self._live.values() if lk.queries >= MIN_QUERIES]
        live.sort(key=lambda lk: -lk.last)
        done = sorted(self.done, key=lambda lk: -lk.last)
        return [
            lk.summary() | {"live": i < len(live)} for i, lk in enumerate(live + done)
        ][:n]

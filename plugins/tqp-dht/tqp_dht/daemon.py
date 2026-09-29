# tqp-dht: a BitTorrent DHT source for the TurboQuant Pro console
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""The DHT daemon: one libtorrent session that seeds a fixed list of official
open-source images, and serves its own state, read-only, on the loopback.

    tqp-dht serve --data /archive/tqp-dht        # http://127.0.0.1:8290/snapshot

It is the only process that talks to peers; ``tqp console --dht URL`` reads the
snapshot and sends nothing, so watching cannot change what it watches.

Operation, by default and by design:

- **Seeds only what the projects publish**, each ``.torrent`` fetched over HTTPS
  from an allowed host and each finished image checked against the SHA-256 the
  project published (pinned in :data:`TORRENTS`); a mismatch stops that torrent.
- **Caps**: upload 2 MB/s, download 20 MB/s, 50 connections.
- **Nothing opened on the router**: UPnP, NAT-PMP and local peer discovery off
  (libtorrent turns all three on by default), no port forward assumed. The DHT
  and seeding then work outbound only.
- **The snapshot server** binds 127.0.0.1, answers ``GET /snapshot`` and nothing
  else, and carries no secrets.

libtorrent is imported only by :func:`run`, so everything else here (the
configuration checks, the snapshot, the server) is tested without it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import traceback
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .krpc import LookupTracer, bdecode

SCHEMA = "tqp-dht/state"
SCHEMA_VERSION = 1
PORT = 8290

# The images seeded, pinned: the torrent each project publishes and the SHA-256
# each project lists for the image (fetched 2026-09-29 from the URLs in "sums").
TORRENTS = [
    {
        "name": "debian-13.7.0-amd64-netinst.iso",
        "torrent": "https://cdimage.debian.org/debian-cd/13.7.0/amd64/bt-cd/"
        "debian-13.7.0-amd64-netinst.iso.torrent",
        "sha256": "a7ef94ac2fb9a7fec454552abd629b7cc9d5155c886165a45649f5ce6167e355",
        "sums": "https://cdimage.debian.org/debian-cd/13.7.0/amd64/iso-cd/SHA256SUMS",
    },
    {
        "name": "archlinux-2026.09.01-x86_64.iso",
        "torrent": "https://archlinux.org/releng/releases/2026.09.01/torrent/",
        "sha256": "be8458032f8105e60ee2a3067f950b6e3c007ee51b38dac50e8b48e765561c91",
        "sums": "https://archlinux.org/releng/releases/json/",
    },
]
ALLOWED_HOSTS = {"cdimage.debian.org", "archlinux.org"}

LIMITS = {
    "upload_rate_limit": 2_000_000,  # bytes/s
    "download_rate_limit": 20_000_000,
    "connections_limit": 50,
}
SESSION = {
    **LIMITS,
    "enable_upnp": False,
    "enable_natpmp": False,
    "enable_lsd": False,
    "enable_dht": True,
    "listen_interfaces": "0.0.0.0:6881,[::]:6881",
    "user_agent": "tqp-dht",
}


def check_torrents(torrents: list) -> list:
    """Every reason the list may not be used (empty: it may)."""
    bad, names = [], set()
    for i, t in enumerate(torrents):
        where = f"torrent {i} ({t.get('name', '?')})"
        url = urllib.parse.urlparse(t.get("torrent", ""))
        if url.scheme != "https" or url.hostname not in ALLOWED_HOSTS:
            bad.append(f"{where}: not an https URL on {sorted(ALLOWED_HOSTS)}")
        if not re.fullmatch(r"[0-9a-f]{64}", t.get("sha256", "")):
            bad.append(f"{where}: sha256 is not 64 lowercase hex digits")
        name = t.get("name", "")
        if not name or "/" in name or name.startswith("."):
            bad.append(f"{where}: the file name must be a plain name")
        if name in names:
            bad.append(f"{where}: the same name twice")
        names.add(name)
    return bad


def _atomic(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def dump_inputs(directory: str, tracer: LookupTracer, t: float) -> dict:
    """What the Observation Theory measurement reads, written atomically: the
    node ids seen (20 bytes each, oldest first), our lookup targets (as JSON
    lines), and a meta record naming both by SHA-256, so a campaign can freeze
    and cite exactly the node set it measured."""
    os.makedirs(directory, exist_ok=True)
    nodes = b"".join(tracer.seen)
    _atomic(os.path.join(directory, "nodes.bin"), nodes)
    targets = "".join(
        json.dumps({"t": ts, "target": tg.hex(), "method": m}) + "\n"
        for ts, tg, m in tracer.targets
    ).encode()
    _atomic(os.path.join(directory, "targets.jsonl"), targets)
    meta = {
        "t": t,
        "n_nodes": len(tracer.seen),
        "n_targets": len(tracer.targets),
        "our_ids": sorted(i.hex() for i in tracer.our_ids),
        "nodes_sha256": hashlib.sha256(nodes).hexdigest(),
        "targets_sha256": hashlib.sha256(targets).hexdigest(),
    }
    _atomic(os.path.join(directory, "meta.json"), json.dumps(meta, indent=1).encode())
    return meta


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _key(d: dict, name: str):
    """A dictionary value under ``name`` whether its keys are bytes or str."""
    v = d.get(name.encode())
    return d.get(name) if v is None else v


def our_node_ids(dht_state) -> set:
    """This node's DHT ids (one per address family), from libtorrent's DHT state:
    bencoded bytes, or the dictionary libtorrent returns once the DHT runs (seen
    on 2.1.1: 'node-id' a list, one entry per DHT node, one node per listen
    socket, each entry the node's 20-byte id followed by its address, 4 or 16
    bytes: on Atlas the first ended in c0a80007, 192.168.0.7),
    with 'node-id' / 'node-id6' or 'nids' as [address, id] pairs.
    Anything else reads as no ids, never as an error."""
    st = dht_state
    if isinstance(st, (bytes, bytearray)):
        try:
            st = bdecode(bytes(st))
        except ValueError:
            return set()
    if not isinstance(st, dict):
        return set()

    def as_id(v):
        if isinstance(v, str):
            v = v.encode("latin-1")
        if not isinstance(v, bytes):
            return None
        if len(v) in (24, 36):  # 2.1.1: the 20-byte id, then the node's address
            v = v[:20]
        return v if len(v) == 20 else None

    ids = set()
    for k in ("node-id", "node-id6"):  # one id, or (2.1.1) one per DHT node
        v = _key(st, k)
        for x in v if isinstance(v, list) else [v]:
            ids.add(as_id(x))
    for pair in _key(st, "nids") or []:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            ids.add(as_id(pair[1]))
    ids.discard(None)
    return ids


def swarm_row(st, expected: dict) -> dict:
    """One torrent's line in the snapshot, from a libtorrent torrent_status-like
    object and the verification state kept for it."""
    up, down = st.all_time_upload, st.all_time_download
    return {
        "name": st.name,
        "state": str(st.state).rsplit(".", 1)[-1],
        "progress": st.progress,
        "peers": st.num_peers,
        "seeds": st.num_seeds,
        "swarm_seeds": st.num_complete if st.num_complete >= 0 else None,
        "swarm_peers": st.num_incomplete if st.num_incomplete >= 0 else None,
        "upload_Bps": st.upload_rate,
        "download_Bps": st.download_rate,
        "uploaded": up,
        "downloaded": down,
        "ratio": (up / down) if down else None,
        "sha256": expected,
    }


def applied(settings: dict) -> dict:
    """The limits and switches as the session holds them (read back, not the
    values asked for): upnp, natpmp and lsd False mean nothing was opened."""
    out = {k: settings.get(k) for k in LIMITS}
    for k in ("upnp", "natpmp", "lsd", "dht"):
        out[k] = settings.get(f"enable_{k}")
    return out


def snapshot(
    node: dict,
    metrics: dict,
    routing: list,
    active: list,
    tracer: LookupTracer,
    swarms: list,
    t: float | None = None,
    limits: dict | None = None,
    labels: dict | None = None,
) -> dict:
    """The daemon's state as one document (what ``GET /snapshot`` returns).
    ``limits`` is what the session read back (see :func:`applied`); ``labels``
    maps a target (hex) to what it is, our torrents' info hashes to their names,
    so a lookup for one reads as that torrent's announce."""
    looks = [lk | {"for": (labels or {}).get(lk["target"])} for lk in tracer.lookups()]
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "t": time.time() if t is None else t,
        "node": node,
        "limits": limits if limits is not None else applied(SESSION),
        "metrics": metrics,
        # one DHT node per listen socket, each with its own table; the fullest
        # is the one the console draws, all are here
        "routing_tables": routing,
        "routing_table": max(
            routing, key=lambda t: sum(b["nodes"] for b in t), default=[]
        ),
        "active_lookups": active,
        "lookups": looks,
        "tracer": dict(tracer.counts),
        "swarms": swarms,
    }


class SnapshotServer:
    """``GET /snapshot`` on 127.0.0.1: the latest snapshot as JSON; any other
    path is 404 and any other method 405. Nothing it serves is secret, and it
    changes nothing."""

    def __init__(self, port: int = PORT):
        self._doc = b"{}"
        self._lock = threading.Lock()
        owner = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path.split("?", 1)[0] != "/snapshot":
                    self.send_error(404)
                    return
                with owner._lock:
                    body = owner._doc
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _refuse(self):
                self.send_error(405)

            do_POST = do_PUT = do_DELETE = do_PATCH = _refuse  # noqa: N815

            def log_message(self, *a):  # quiet: the daemon has its own log
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.port = self.httpd.server_address[1]

    def publish(self, doc: dict) -> None:
        body = json.dumps(doc, allow_nan=False).encode()
        with self._lock:
            self._doc = body

    def start(self) -> SnapshotServer:
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


# ------------------------------------------------------------------ run
def _fetch(url: str, limit: int = 4 << 20) -> bytes:
    host = urllib.parse.urlparse(url).hostname
    if host not in ALLOWED_HOSTS:
        raise ValueError(f"{host} is not an allowed host")
    with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 (https, allowlist)
        body = r.read(limit + 1)
    if len(body) > limit:
        raise ValueError(f"{url}: larger than {limit} bytes")
    return body


def _log(msg: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + msg, flush=True)


DUMP_EVERY_S = 600  # how often the measurement's inputs are written


def run(
    data_dir: str,
    port: int = PORT,
    torrents: list = TORRENTS,
    log=_log,
    observe_dir: str | None = None,
) -> None:
    """Run the session until SIGTERM or SIGINT. One loop, once a second: ask for
    DHT and session statistics, read every alert, rebuild the snapshot."""
    import signal

    import libtorrent as lt  # the only libtorrent import

    bad = check_torrents(torrents)
    if bad:
        raise SystemExit("tqp-dht: " + "; ".join(bad))
    os.makedirs(data_dir, exist_ok=True)
    cat = lt.alert_category
    mask = cat.dht | cat.dht_log | cat.stats | cat.status | cat.error | cat.storage
    ses = lt.session({**SESSION, "alert_mask": mask, "alert_queue_size": 20000})
    held = applied(ses.get_settings())
    wrong = sorted(k for k in ("upnp", "natpmp", "lsd") if held[k])
    if wrong:  # never run with the router opened, whatever the library did
        raise SystemExit(f"tqp-dht: the session did not take {wrong} = off")
    log(f"session settings read back: {held}")
    metric_type = {
        m.name: str(m.type).rsplit(".", 1)[-1] for m in lt.session_stats_metrics()
    }
    handles, verify, labels = {}, {}, {}
    for t in torrents:
        ti = lt.torrent_info(_fetch(t["torrent"]))
        h = ses.add_torrent({"ti": ti, "save_path": data_dir})
        handles[t["name"]] = h
        labels[str(ti.info_hash())] = t["name"]
        verify[t["name"]] = {"expected": t["sha256"], "status": "pending"}
        log(f"added {t['name']} ({ti.total_size() / 1e9:.2f} GB)")
    server = SnapshotServer(port).start()
    log(f"snapshot on http://127.0.0.1:{server.port}/snapshot")
    tracer = LookupTracer()
    stop = threading.Event()
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, lambda *_: stop.set())

    def check(name, h):
        path = os.path.join(data_dir, name)
        try:
            got = sha256_of(path)
        except OSError as e:
            verify[name].update(status="error", error=str(e))
            return
        ok = got == verify[name]["expected"]
        verify[name].update(status="verified" if ok else "mismatch", got=got)
        if not ok:
            h.pause()  # a file that is not what the project published is not seeded
        log(f"{name}: sha256 {'verified' if ok else 'MISMATCH, paused'}")

    live = {"metrics": {}, "routing": [], "active": [], "ids_at": 0.0, "errors": 0}
    live["dumped"] = time.time()

    def tick(now: float) -> None:
        """One second of the session: node ids (each minute), every alert, the
        swarms, and the snapshot rebuilt from them."""
        if now - live["ids_at"] > 60:
            state = ses.dht_state()
            if not live["ids_at"]:  # once: what this libtorrent's DHT state holds
                keys = sorted(map(str, state)) if isinstance(state, dict) else []
                log(f"dht state: {type(state).__name__}, keys {keys}")
                if isinstance(state, dict):
                    v = _key(state, "node-id")
                    shape = (
                        [(type(x).__name__, len(x)) for x in v][:4]
                        if isinstance(v, list)
                        else ""
                    )
                    log(f"node-id: {type(v).__name__} len {len(v or '')} {shape}")
            tracer.our_ids = our_node_ids(state) or tracer.our_ids
            live["ids_at"] = now
        ses.post_dht_stats()
        ses.post_session_stats()
        alerts = ses.pop_alerts()
        tables, active = [], []
        if not live.get("logged_stats"):  # once: how many DHT nodes report
            sizes = [
                len(a.routing_table)
                for a in alerts
                if type(a).__name__ == "dht_stats_alert"
            ]
            if sizes:
                log(f"dht_stats alerts in one tick: {len(sizes)}, buckets {sizes}")
                live["logged_stats"] = True
        for a in alerts:
            kind = type(a).__name__
            if kind == "dht_pkt_alert":
                tracer.observe(bytes(a.pkt_buf), now)
            elif kind == "dht_stats_alert":  # one per DHT node, every tick
                tables.append(
                    [
                        {
                            "bucket": i,
                            "nodes": b["num_nodes"],
                            "replacements": b["num_replacements"],
                        }
                        for i, b in enumerate(a.routing_table)
                    ]
                )
                active.extend(dict(r) for r in a.active_requests)
            elif kind == "session_stats_alert":
                live["metrics"] = {
                    k: {"value": v, "type": metric_type.get(k, "counter")}
                    for k, v in a.values.items()
                    if k.startswith("dht.")
                }
        if tables:
            live["routing"], live["active"] = tables, active
        swarms = []
        for name, h in handles.items():
            st = h.status()
            if st.is_seeding and verify[name]["status"] == "pending":
                verify[name]["status"] = "checking"
                threading.Thread(target=check, args=(name, h), daemon=True).start()
            swarms.append(swarm_row(st, dict(verify[name])))
        node = {
            "ids": sorted(i.hex() for i in tracer.our_ids),
            "listen_port": ses.listen_port() if hasattr(ses, "listen_port") else None,
            "dht_running": ses.is_dht_running(),
            "loop_errors": live["errors"],
        }
        if observe_dir and now - live["dumped"] >= DUMP_EVERY_S:
            live["dumped"] = now
            meta = dump_inputs(observe_dir, tracer, now)
            log(f"inputs written: {meta['n_nodes']} nodes, {meta['n_targets']} targets")
        m, rt, act = live["metrics"], live["routing"], live["active"]
        server.publish(snapshot(node, m, rt, act, tracer, swarms, now, held, labels))

    while not stop.is_set():
        try:
            tick(time.time())
        except Exception:  # one bad reading must not stop the seeding: log, count
            live["errors"] += 1
            log("loop error (the session goes on):\n" + traceback.format_exc())
        stop.wait(1.0)
    log("stopping")
    ses.pause()
    server.stop()


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="tqp-dht", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the session and serve its snapshot")
    s.add_argument("--data", required=True, help="where the images are kept")
    s.add_argument("--port", type=int, default=PORT, help="snapshot port on 127.0.0.1")
    s.add_argument(
        "--observe",
        metavar="DIR",
        help="write the Observation Theory measurement's inputs here every 10 min",
    )
    sub.add_parser("check", help="check the pinned torrent list and exit")
    args = p.parse_args(argv)
    if args.cmd == "check":
        bad = check_torrents(TORRENTS)
        print("\n".join(bad) or f"{len(TORRENTS)} torrents, all pinned and allowed")
        return 1 if bad else 0
    run(args.data, args.port, observe_dir=args.observe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

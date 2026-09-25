"""Local console server: standard library only, read-only, token-gated.

Security posture (docs/DESIGN_console.md section 2.7):

- binds to 127.0.0.1 unless told otherwise; remote use goes through an SSH tunnel;
- every ``/api`` request needs the per-session token (header ``X-TQP-Token``), compared
  in constant time; the page receives it in the URL fragment, which browsers never send
  to a server, so it does not land in access logs;
- the ``Host`` header must name the bound address or ``localhost`` (DNS rebinding);
- no endpoint changes state: nothing from the browser is executed, and replay re-runs a
  query the workload already ran;
- responses carry a Content-Security-Policy restricting the page to its own origin.

The console hosts the workload it observes: :class:`Workload` replays a query file
against the index at a target rate, so traces and metrics are live without any other
process.
"""

from __future__ import annotations

import hmac
import json
import mimetypes
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .. import telemetry
from ..telemetry.metrics import reading

STATIC = Path(__file__).with_name("static")


def _clean(o):
    """JSON-safe copy: NaN and infinities become null (they are not JSON)."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return _clean(o.item())
    return o


def dumps(o) -> str:
    return json.dumps(_clean(o), allow_nan=False)


MAX_TRACES_PER_S = 20  # stream throttle; the ring keeps everything up to its capacity


class Workload(threading.Thread):
    """Replays ``queries`` against ``index.search`` at ``qps`` (one query per call),
    cycling through the rows. The row each call used is recorded on its trace
    (``params.workload_row``) so the console can replay it."""

    def __init__(
        self,
        index,
        queries: np.ndarray,
        qps: float = 20.0,
        k: int = 10,
        rerank: int = 0,
        originals: np.ndarray | None = None,
    ):
        super().__init__(daemon=True, name="tqp-console-workload")
        self.index = index
        self.queries = np.ascontiguousarray(queries, dtype=np.float32)
        self.qps = max(float(qps), 0.1)
        self.k, self.rerank, self.originals = k, rerank, originals
        self.row = 0
        self.errors = 0
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._search = threading.Lock()  # replay and workload must not interleave

    def run_one(self, row: int):
        q = self.queries[row : row + 1]
        with self._search:
            tr = telemetry.active()
            before = len(tr.traces()) if tr else 0
            if self.rerank and self.originals is not None:
                self.index.search(
                    q, k=self.k, rerank=self.rerank, originals=self.originals
                )
            else:
                self.index.search(q, k=self.k)
            if tr:
                docs = tr.traces()
                if len(docs) > before:
                    docs[-1]["params"]["workload_row"] = row
                    return docs[-1]
        return None

    def run(self):
        period = 1.0 / self.qps
        nxt = time.perf_counter()
        while not self._stop.is_set():
            if not self._paused.is_set():
                try:
                    self.run_one(self.row)
                except Exception as e:  # keep serving; the error is shown, not hidden
                    self.errors += 1
                    self.last_error = f"{type(e).__name__}: {e}"
                self.row = (self.row + 1) % len(self.queries)
            nxt += period
            delay = nxt - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
            else:
                nxt = time.perf_counter()

    def stop(self):
        self._stop.set()

    def state(self) -> dict:
        return {
            "target_qps": self.qps,
            "rows": len(self.queries),
            "row": self.row,
            "k": self.k,
            "rerank": self.rerank,
            "mode": (
                "exact rerank"
                if self.rerank and self.originals is not None
                else "approximate"
            ),
            "errors": self.errors,
            "last_error": self.last_error,
        }


_PROC: list = []  # one psutil.Process: cpu_percent measures between successive calls


def _process_readings() -> list:
    try:
        import psutil
    except ImportError:
        why = "psutil not installed (pip install psutil)"
        return [
            reading("process.cpu_percent", None, reason=why),
            reading("process.rss_mb", None, reason=why),
        ]
    if not _PROC:
        _PROC.append(psutil.Process())
        _PROC[0].cpu_percent(interval=None)  # primes; its own return value is 0.0
        cpu = reading("process.cpu_percent", None, reason="priming the first interval")
    else:
        cpu = reading("process.cpu_percent", _PROC[0].cpu_percent(interval=None))
    return [cpu, reading("process.rss_mb", _PROC[0].memory_info().rss / 2**20)]


def _index_entity(index) -> dict:
    ent = dict(index._trace_identity()) if hasattr(index, "_trace_identity") else {}
    ent["kind"] = type(index).__name__
    if hasattr(index, "stats"):
        try:
            ent["stats"] = index.stats()
        except Exception as e:
            ent["stats_error"] = str(e)
    return ent


def _diff(a: dict | None, b: dict | None) -> dict:
    """Result comparison of two traces of the same query (requirements QI-003)."""

    def ids(d):
        r = (d or {}).get("results") or {}
        rows = r.get("final") or r.get("approximate") or []
        return [x["id"] for x in rows]

    x, y = ids(a), ids(b)
    moved = {
        i: (x.index(i), y.index(i)) for i in set(x) & set(y) if x.index(i) != y.index(i)
    }
    return {
        "same_ids_in_order": x == y,
        "overlap": len(set(x) & set(y)),
        "k": max(len(x), len(y)),
        "only_before": [i for i in x if i not in y],
        "only_after": [i for i in y if i not in x],
        "moved": {str(i): {"before": p, "after": q} for i, (p, q) in moved.items()},
        "latency_ms": {
            "before": (a or {}).get("total_ms"),
            "after": (b or {}).get("total_ms"),
        },
        "scan_path": {
            "before": (a or {}).get("scan_path"),
            "after": (b or {}).get("scan_path"),
        },
    }


class ConsoleServer:
    def __init__(
        self,
        index,
        queries: np.ndarray,
        *,
        qps: float = 20.0,
        k: int = 10,
        rerank: int = 0,
        originals: np.ndarray | None = None,
        observer=None,
        certificate: dict | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        token: str | None = None,
        sample_rate: float = 1.0,
        source: dict | None = None,
    ):
        self.index = index
        self.token = token or secrets.token_urlsafe(24)
        self.observer = observer  # an ObserverContract or None
        self.certificate = certificate
        self.source = source or {}
        ref = observer.reference() if observer is not None else None
        self.tracer = telemetry.enable(rate=sample_rate, observer=ref)
        self.workload = Workload(index, queries, qps, k, rerank, originals)
        self.started = time.time()
        self.replays: dict[str, dict] = {}
        self.httpd = ThreadingHTTPServer((host, port), self._handler())
        self.httpd.daemon_threads = True
        self.host, self.port = self.httpd.server_address[:2]

    # ------------------------------------------------------------- documents
    def snapshot(self) -> dict:
        tr = self.tracer
        readings = tr.snapshot() + _process_readings()
        ent = _index_entity(self.index)
        stats = ent.get("stats") or {}
        cr = stats.get("compression_ratio")
        if cr is None and ent.get("stored_bytes_per_row") and ent.get("dim"):
            src_dim = self.source.get("dim") or ent["dim"]
            cr = 4.0 * src_dim / ent["stored_bytes_per_row"]
        readings.append(
            reading(
                "index.compression_ratio",
                cr,
                reason=None if cr else "index reports no stored bytes per row",
            )
        )
        readings.append(
            reading(
                "index.rows",
                ent.get("rows") or stats.get("n_rows"),
                reason=(
                    None
                    if (ent.get("rows") or stats.get("n_rows"))
                    else "index reports no row count"
                ),
            )
        )
        last = tr.traces(1)
        return {
            "t": time.time(),
            "uptime_s": time.time() - self.started,
            "readings": readings,
            "index": ent,
            "workload": self.workload.state(),
            "paused": self.workload._paused.is_set(),
            "last_trace_age_s": (
                (time.time() - last[0]["started_unix"]) if last else None
            ),
            "tracer": {
                "rate": tr.rate,
                "capacity": tr._ring.maxlen,
                "captured": len(tr.traces()),
            },
        }

    def readscope(self) -> dict:
        out = {
            "observer": None,
            "certificate": None,
            "validity": None,
            "provenance": [],
        }
        if self.observer is not None:
            out["observer"] = {
                "contract": self.observer.as_dict(),
                "reference": self.observer.reference(),
            }
            out["provenance"].append(
                {"step": "observer contract", "sha256": self.observer.digest()}
            )
        if self.certificate is not None:
            c = self.certificate
            out["certificate"] = c
            out["validity"] = c.get("validity")
            for side in ("original", "reconstructed"):
                sha = ((c.get("inputs") or {}).get(side) or {}).get("sha256")
                if sha:
                    out["provenance"].append(
                        {"step": f"certificate input: {side}", "sha256": sha}
                    )
        if self.source:
            out["provenance"].insert(0, {"step": "source", **self.source})
        return out

    def replay(self, trace_id: str) -> dict:
        """Re-run a traced workload query under the current configuration; compare."""
        before = self.tracer.get(trace_id)
        if before is None:
            return {"error": "no such trace in the ring (it may have been evicted)"}
        row = before["params"].get("workload_row")
        if row is None:
            return {
                "error": "this trace did not come from the console workload, so its "
                "query is known only by sha256 and cannot be replayed"
            }
        after = self.workload.run_one(int(row))
        if after is None:
            return {"error": "the replayed call was not traced"}
        after["params"]["replay_of"] = trace_id
        same_input = after["input"]["sha256"] == before["input"]["sha256"]
        res = {
            "before": trace_id,
            "after": after["id"],
            "same_input_sha256": same_input,
            "pinned": {
                "query": same_input,
                "index": after["index"] == before["index"],
                "params": {
                    k: v for k, v in before["params"].items() if k != "workload_row"
                },
            },
            "nondeterminism": (
                []
                if after["index"] == before["index"]
                else ["the index changed between the two runs"]
            ),
            "diff": _diff(before, after),
        }
        self.replays[after["id"]] = res
        return res

    def export(self) -> dict:
        return {
            "schema": "turboquant-pro/console-export",
            "schema_version": 1,
            "capabilities": telemetry.capabilities(),
            "snapshot": self.snapshot(),
            "readscope": self.readscope(),
            "traces": self.tracer.traces(200),
            "replays": list(self.replays.values()),
        }

    # ------------------------------------------------------------------ http
    def _handler(self):
        srv = self

        class H(BaseHTTPRequestHandler):
            server_version = "tqp-console"

            def log_message(self, *a):  # quiet; the console is the log
                pass

            def _headers(self, code: int, ctype: str, extra: dict | None = None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; img-src 'self' data:; style-src 'self'; "
                    "script-src 'self'; connect-src 'self'; frame-ancestors 'none'",
                )
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()

            def _json(self, obj, code: int = 200):
                body = dumps(obj).encode()
                self._headers(code, "application/json")
                self.wfile.write(body)

            def _host_ok(self) -> bool:
                host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
                return host in {"localhost", "127.0.0.1", "::1", srv.host}

            def _authed(self) -> bool:
                got = self.headers.get("X-TQP-Token") or ""
                return hmac.compare_digest(got.encode(), srv.token.encode())

            def do_GET(self):  # noqa: N802
                if not self._host_ok():
                    return self._json({"error": "bad Host header"}, 403)
                u = urlparse(self.path)
                p = u.path
                if not p.startswith("/api/"):
                    return self._static(p)
                if not self._authed():
                    return self._json({"error": "missing or wrong session token"}, 401)
                qs = parse_qs(u.query)
                try:
                    if p == "/api/version":
                        return self._json(telemetry.capabilities())
                    if p == "/api/snapshot":
                        return self._json(srv.snapshot())
                    if p == "/api/traces":
                        n = min(int(qs.get("n", ["100"])[0]), 2048)
                        return self._json(srv.tracer.traces(n))
                    if p.startswith("/api/trace/"):
                        d = srv.tracer.get(p.rsplit("/", 1)[1])
                        return self._json(
                            d or {"error": "not found"}, 200 if d else 404
                        )
                    if p == "/api/readscope":
                        return self._json(srv.readscope())
                    if p.startswith("/api/replay/"):
                        return self._json(srv.replay(p.rsplit("/", 1)[1]))
                    if p == "/api/export":
                        return self._json(srv.export())
                    if p == "/api/stream":
                        return self._stream()
                except (ValueError, KeyError) as e:
                    return self._json({"error": str(e)}, 400)
                return self._json({"error": "unknown endpoint"}, 404)

            def do_POST(self):  # noqa: N802
                self._json({"error": "the console is read-only"}, 405)

            do_PUT = do_DELETE = do_PATCH = do_POST

            def _static(self, p: str):
                name = "index.html" if p in ("/", "/index.html") else p.lstrip("/")
                f = (STATIC / name).resolve()
                if STATIC.resolve() not in f.parents or not f.is_file():
                    return self._json({"error": "not found"}, 404)
                ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                self._headers(200, ctype)
                self.wfile.write(f.read_bytes())

            def _stream(self):
                """Server-sent events: a snapshot every second and traces as they finish
                (at most MAX_TRACES_PER_S per second; the rest stay in the ring)."""
                self._headers(200, "text/event-stream", {"Connection": "keep-alive"})
                pending: list = []
                lock = threading.Lock()

                def on_trace(doc):
                    with lock:
                        if len(pending) < MAX_TRACES_PER_S:
                            pending.append(doc)

                srv.tracer.subscribe(on_trace)
                try:
                    while True:
                        with lock:
                            batch, pending[:] = list(pending), []
                        for d in batch:
                            self._event("trace", d)
                        self._event("snapshot", srv.snapshot())
                        time.sleep(1.0)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    srv.tracer.unsubscribe(on_trace)

            def _event(self, name: str, obj):
                data = dumps(obj)
                self.wfile.write(f"event: {name}\ndata: {data}\n\n".encode())
                self.wfile.flush()

        return H

    # ------------------------------------------------------------- lifecycle
    @property
    def url(self) -> str:
        host = "localhost" if self.host in ("127.0.0.1", "::1") else self.host
        return f"http://{host}:{self.port}/#token={self.token}"

    def start(self) -> ConsoleServer:
        self.workload.start()
        threading.Thread(
            target=self.httpd.serve_forever, daemon=True, name="tqp-console-http"
        ).start()
        return self

    def stop(self) -> None:
        self.workload.stop()
        self.httpd.shutdown()
        self.httpd.server_close()
        if telemetry.active() is self.tracer:
            telemetry.disable()


def demo_index(
    n: int = 20000, dim: int = 128, out_dim: int = 64, bits: int = 3, seed: int = 0
):
    """A synthetic index and query set for ``tqp console --demo``: clustered unit
    vectors, so rerank moves results and the panels have something to show."""
    from ..adc_index import ADCIndex
    from ..pca import PCAMatryoshka

    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((32, dim)).astype(np.float32)
    X = centers[rng.integers(0, 32, n)] + 0.6 * rng.standard_normal((n, dim)).astype(
        np.float32
    )
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    nq = min(512, n)
    Q = X[rng.choice(n, nq, replace=False)] + 0.05 * rng.standard_normal(
        (nq, dim)
    ).astype(np.float32)
    pca = PCAMatryoshka(input_dim=dim, output_dim=out_dim)
    pca.fit(X[: min(n, 5000)])
    idx = ADCIndex(pca.with_quantizer(bits=bits)).add(X)
    return (
        idx,
        Q,
        X,
        {
            "name": "demo (synthetic clustered unit vectors)",
            "rows": n,
            "dim": dim,
            "seed": seed,
        },
    )


__all__ = ["ConsoleServer", "Workload", "demo_index", "dumps"]

"""Query traces: stage timings, the scan path taken, approximate vs exact results.

Tracing is off by default. An instrumented search calls :func:`begin`, which
returns ``None`` unless a tracer is enabled and samples the call; every later
hook is guarded by ``if tr:``, so a disabled tracer costs one global read per
search and one truth test per stage.

A trace records the query batch by sha256 and shape. The vectors themselves are
kept only when the tracer was enabled with ``capture_vectors=True``
(requirements NFR-006).

Usage::

    from turboquant_pro import telemetry
    tracer = telemetry.enable(rate=1.0)
    index.search(queries, k=10, rerank=5, originals=corpus)
    tracer.traces()[-1]          # a turboquant-pro/query-trace document
    tracer.snapshot()            # metric readings, each with unit, window and kind
"""

from __future__ import annotations

import hashlib
import random
import threading
import time
import uuid
from collections import deque

import numpy as np

from .metrics import Window, reading

SCHEMA = "turboquant-pro/query-trace"
SCHEMA_VERSION = 1
SCAN_PATHS = ("kernel", "kernel_pruned", "numpy", "exact")
_ACTIVE: Tracer | None = None
_LOCK = threading.Lock()


def _now_iso(t: float) -> str:
    return (
        time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int(t % 1 * 1e3):03d}Z"
    )


class QueryTrace:
    """One search call. Stages are recorded with :meth:`lap`, which times the interval
    since the previous lap (or since :func:`begin`)."""

    __slots__ = ("_tracer", "doc", "_t0", "_last")

    def __init__(
        self, tracer: Tracer, component: str, queries, params: dict, index: dict | None
    ):
        q = np.ascontiguousarray(np.asarray(queries, dtype=np.float32))
        if q.ndim == 1:
            q = q[None, :]
        now = time.time()
        self._tracer = tracer
        self.doc = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "id": uuid.uuid4().hex[:16],
            "component": component,
            "started_utc": _now_iso(now),
            "started_unix": now,
            "input": {
                "n_queries": int(q.shape[0]),
                "dim": int(q.shape[1]),
                "sha256": hashlib.sha256(q.tobytes()).hexdigest(),
            },
            "params": {k: v for k, v in params.items() if v is not None},
            "index": index or {},
            "observer": tracer.observer,
            "scan_path": None,
            "stages": [],
            "results": None,
            "sampled": tracer.rate < 1.0,
        }
        if tracer.capture_vectors:
            self.doc["input"]["vectors"] = q.tolist()
        self._t0 = self._last = time.perf_counter()

    def lap(self, name: str, **attrs) -> None:
        t = time.perf_counter()
        self.doc["stages"].append({"name": name, "ms": (t - self._last) * 1e3, **attrs})
        self._last = t

    def set(self, **kv) -> None:
        self.doc.update(kv)

    def results(
        self,
        approx_ids,
        approx_scores,
        final_ids=None,
        exact_scores=None,
        k: int | None = None,
    ) -> None:
        """Top-k of the FIRST query of the batch (bounded size): the approximate list,
        and when reranked the final list with exact scores and each result's rank
        movement (approximate position minus final position; positive moved up)."""
        a_ids = [int(x) for x in np.asarray(approx_ids)[0] if x >= 0]
        a_sc = [float(x) for x in np.asarray(approx_scores)[0][: len(a_ids)]]
        k = k or len(a_ids)
        out = {
            "k": k,
            "approximate": [{"id": i, "score": s} for i, s in zip(a_ids, a_sc)],
        }
        if final_ids is not None:
            pos = {i: r for r, i in enumerate(a_ids)}
            f_ids = [int(x) for x in np.asarray(final_ids)[0] if x >= 0]
            ex = [] if exact_scores is None else [float(x) for x in exact_scores]
            out["final"] = [
                {
                    "id": i,
                    "exact_score": ex[r] if r < len(ex) else None,
                    "approx_rank": pos.get(i),
                    "rank_movement": None if i not in pos else pos[i] - r,
                }
                for r, i in enumerate(f_ids)
            ]
            top = set(a_ids[:k])
            out["rerank_agreement"] = (
                len(top & set(f_ids[:k])) / max(len(f_ids[:k]), 1) if f_ids else None
            )
        self.doc["results"] = out

    def finish(self) -> dict:
        self.doc["total_ms"] = (time.perf_counter() - self._t0) * 1e3
        self._tracer._record(self.doc)
        return self.doc


class Tracer:
    """Samples search calls into a bounded ring of traces and rolling metric windows."""

    def __init__(
        self,
        rate: float = 1.0,
        capacity: int = 2048,
        capture_vectors: bool = False,
        window_s: float = 60.0,
        seed: int | None = None,
        observer: dict | None = None,
    ):
        if not 0.0 < rate <= 1.0:
            raise ValueError("rate must be in (0, 1]")
        self.rate = rate
        self.capture_vectors = capture_vectors
        self.observer = observer  # an ObserverContract.reference() block, if any
        self._rng = random.Random(seed)
        self._ring: deque = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._listeners: list = []
        self.windows = {
            "latency": Window(window_s),
            "queries": Window(window_s),
            "agreement": Window(window_s),
        }
        self.stage_windows: dict[str, Window] = {}
        self.window_s = window_s
        self.started = time.time()

    def sampled(self) -> bool:
        return self.rate >= 1.0 or self._rng.random() < self.rate

    def subscribe(self, fn) -> None:
        """``fn(trace_doc)`` after every finished trace (the console's live stream)."""
        with self._lock:
            self._listeners.append(fn)

    def unsubscribe(self, fn) -> None:
        with self._lock:
            if fn in self._listeners:
                self._listeners.remove(fn)

    def _record(self, doc: dict) -> None:
        nq = max(doc["input"]["n_queries"], 1)
        t = doc["started_unix"]
        with self._lock:
            self._ring.append(doc)
            self.windows["latency"].add(doc["total_ms"] / nq, nq, t)
            self.windows["queries"].add(nq, nq, t)
            for st in doc["stages"]:
                w = self.stage_windows.setdefault(st["name"], Window(self.window_s))
                w.add(st["ms"] / nq, nq, t)
            r = doc.get("results") or {}
            if r.get("rerank_agreement") is not None:
                self.windows["agreement"].add(r["rerank_agreement"], 1, t)
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(doc)
            except Exception:  # a broken listener must never break a search
                pass

    def traces(self, n: int | None = None) -> list:
        with self._lock:
            ring = list(self._ring)
        return ring if n is None else ring[-n:]

    def get(self, trace_id: str) -> dict | None:
        with self._lock:
            return next((d for d in self._ring if d["id"] == trace_id), None)

    def snapshot(self) -> list:
        """Metric readings for everything the traces measure, each with its spec."""
        now = time.time()
        kind = "sampled" if self.rate < 1.0 else None
        with self._lock:
            lat = self.windows["latency"]
            qn = self.windows["queries"]
            span = min(self.window_s, max(now - self.started, 1e-9))
            nq = qn.weight(now)
            out = [
                reading(
                    "search.qps",
                    nq / span / self.rate,
                    n=nq,
                    kind=kind or "measured",
                    as_of=now,
                ),
            ]
            for p, name in ((50, "p50"), (95, "p95"), (99, "p99")):
                v = lat.percentile(p, now)
                out.append(
                    reading(
                        f"search.latency_ms.{name}",
                        v,
                        n=nq,
                        kind=kind,
                        as_of=now,
                        reason=None if v is not None else "no queries in window",
                    )
                )
            for stage in ("encode", "scan", "rerank"):
                w = self.stage_windows.get(stage)
                v = w.mean(now) if w else None
                out.append(
                    reading(
                        f"search.stage_ms.{stage}",
                        v,
                        kind=kind,
                        as_of=now,
                        reason=(
                            None
                            if v is not None
                            else f"no '{stage}' stage " "in window"
                        ),
                    )
                )
            ag = self.windows["agreement"].mean(now)
            out.append(
                reading(
                    "search.rerank_agreement",
                    ag,
                    as_of=now,
                    reason=None if ag is not None else "no reranked queries in window",
                )
            )
        return out


def enable(
    rate: float = 1.0,
    capacity: int = 2048,
    capture_vectors: bool = False,
    window_s: float = 60.0,
    seed: int | None = None,
    observer: dict | None = None,
) -> Tracer:
    """Install a process-wide tracer (replacing any previous one) and return it."""
    global _ACTIVE
    t = Tracer(rate, capacity, capture_vectors, window_s, seed, observer)
    with _LOCK:
        _ACTIVE = t
    return t


def install(tracer: Tracer) -> Tracer:
    """Make an existing tracer the process-wide one (a console re-asserts its own)."""
    global _ACTIVE
    with _LOCK:
        _ACTIVE = tracer
    return tracer


def disable() -> None:
    global _ACTIVE
    with _LOCK:
        _ACTIVE = None


def active() -> Tracer | None:
    return _ACTIVE


def begin(component: str, queries, index: dict | None = None, **params):
    """A :class:`QueryTrace` if tracing is on and this call is sampled, else None."""
    t = _ACTIVE
    if t is None or not t.sampled():
        return None
    return QueryTrace(t, component, queries, params, index)

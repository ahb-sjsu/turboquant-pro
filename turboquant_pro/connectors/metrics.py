# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Connector observability (2.0 roadmap P1-M3).

A platform connector an operator cannot see is a platform connector an
operator cannot safely run. :class:`ConnectorMetrics` is the single counter
surface for the KV tier — every store and connector event lands here, and
both export formats (dict for logs/JSON, Prometheus text for scrapes) are
first-class. Thread-safe; latencies keep a bounded reservoir for p50/p95/p99.

The counter vocabulary is the roadmap's list, implemented: hits and misses
*by cause* (a miss is never anonymous — empty, corrupt, incompatible,
timeout, or admission-declined), logical vs physical bytes, save/load
latencies, integrity failures, evictions, backpressure events.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager

import numpy as np

__all__ = ["ConnectorMetrics"]

_MISS_CAUSES = ("empty", "corrupt", "incompatible", "timeout", "declined")
_RESERVOIR = 1024


class ConnectorMetrics:
    """Thread-safe counters + latency reservoirs for the KV tier."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._c: dict[str, float] = {
            "saves": 0,
            "loads": 0,
            "hits": 0,
            "evictions": 0,
            "integrity_failures": 0,
            "backpressure_blocked": 0,
            "backpressure_dropped": 0,
            "bytes_logical": 0,
            "bytes_physical": 0,
            "records_persisted": 0,
            "records_restored": 0,
        }
        for cause in _MISS_CAUSES:
            self._c[f"misses_{cause}"] = 0
        self._lat: dict[str, deque] = {
            "save": deque(maxlen=_RESERVOIR),
            "load": deque(maxlen=_RESERVOIR),
        }

    # ------------------------------------------------------------- record
    def inc(self, name: str, by: float = 1) -> None:
        with self._lock:
            if name not in self._c:
                raise KeyError(f"unknown counter {name!r}")
            self._c[name] += by

    def miss(self, cause: str) -> None:
        if cause not in _MISS_CAUSES:
            raise KeyError(f"unknown miss cause {cause!r}; one of {_MISS_CAUSES}")
        self.inc(f"misses_{cause}")

    def bytes_saved(self, logical: int, physical: int) -> None:
        with self._lock:
            self._c["bytes_logical"] += logical
            self._c["bytes_physical"] += physical

    @contextmanager
    def timed(self, op: str):
        """``with metrics.timed("save"): ...`` — records seconds on exit."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            with self._lock:
                self._lat[op].append(dt)

    # ------------------------------------------------------------- export
    def _quantiles(self, op: str) -> dict[str, float]:
        with self._lock:
            xs = np.asarray(self._lat[op], dtype=np.float64)
        if len(xs) == 0:
            return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan")}
        return {
            "p50": float(np.percentile(xs, 50)),
            "p95": float(np.percentile(xs, 95)),
            "p99": float(np.percentile(xs, 99)),
        }

    def to_dict(self) -> dict:
        with self._lock:
            out = dict(self._c)
        total_miss = sum(out[f"misses_{c}"] for c in _MISS_CAUSES)
        looks = out["hits"] + total_miss
        out["misses_total"] = total_miss
        out["hit_rate"] = out["hits"] / looks if looks else float("nan")
        out["effective_expansion"] = (
            out["bytes_logical"] / out["bytes_physical"]
            if out["bytes_physical"]
            else float("nan")
        )
        for op in ("save", "load"):
            for q, v in self._quantiles(op).items():
                out[f"{op}_latency_{q}_s"] = v
        return out

    def to_prometheus(self, prefix: str = "tqp_kv") -> str:
        """Prometheus text exposition (counters + summary quantiles)."""
        d = self.to_dict()
        lines: list[str] = []
        for name, val in sorted(d.items()):
            if name.startswith(("save_latency", "load_latency")):
                continue
            metric = f"{prefix}_{name}"
            kind = "gauge" if name in ("hit_rate", "effective_expansion") else "counter"
            lines.append(f"# TYPE {metric} {kind}")
            v = 0.0 if isinstance(val, float) and np.isnan(val) else val
            lines.append(f"{metric} {v}")
        for op in ("save", "load"):
            metric = f"{prefix}_{op}_latency_seconds"
            lines.append(f"# TYPE {metric} summary")
            for q, key in (("0.5", "p50"), ("0.95", "p95"), ("0.99", "p99")):
                v = d[f"{op}_latency_{key}_s"]
                v = 0.0 if isinstance(v, float) and np.isnan(v) else v
                lines.append(f'{metric}{{quantile="{q}"}} {v}')
        return "\n".join(lines) + "\n"

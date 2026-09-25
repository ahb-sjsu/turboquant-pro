"""Metric specifications and rolling windows (requirements section 10).

Every metric the console shows is a :class:`MetricSpec` in :data:`REGISTRY`:
name, unit, aggregation, window, source, update interval, kind, and for quality
metrics the reference it is judged against. :func:`reading` joins a value to its
spec, so a number never travels without its unit, window, freshness and kind. A
value the runtime cannot measure is reported as unavailable (``value: None``
with a reason), never as zero.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

KINDS = ("measured", "estimated", "sampled", "derived")


@dataclass(frozen=True)
class MetricSpec:
    name: str
    unit: str
    aggregation: str  # "p50" | "p95" | "p99" | "mean" | "rate" | "last" | "ratio"
    window_s: float | None  # None: a point value, not a window
    source: str
    interval_s: float
    kind: str
    description: str
    reference: str | None = None  # quality metrics: what they are judged against

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: kind must be one of {KINDS}")


_W = 60.0
_TRACE = "telemetry.trace (finished query traces)"

REGISTRY: dict[str, MetricSpec] = {
    s.name: s
    for s in [
        MetricSpec(
            "search.qps",
            "queries/s",
            "rate",
            _W,
            _TRACE,
            1.0,
            "measured",
            "queries completed per second over the window",
        ),
        MetricSpec(
            "search.latency_ms.p50",
            "ms",
            "p50",
            _W,
            _TRACE,
            1.0,
            "measured",
            "median wall time of a search call, per query",
        ),
        MetricSpec(
            "search.latency_ms.p95",
            "ms",
            "p95",
            _W,
            _TRACE,
            1.0,
            "measured",
            "95th percentile wall time of a search call, per query",
        ),
        MetricSpec(
            "search.latency_ms.p99",
            "ms",
            "p99",
            _W,
            _TRACE,
            1.0,
            "measured",
            "99th percentile wall time of a search call, per query",
        ),
        MetricSpec(
            "search.stage_ms.encode",
            "ms",
            "mean",
            _W,
            _TRACE,
            1.0,
            "measured",
            "query rotation and projection, per query",
        ),
        MetricSpec(
            "search.stage_ms.scan",
            "ms",
            "mean",
            _W,
            _TRACE,
            1.0,
            "measured",
            "compressed-domain scan and candidate cut, per query",
        ),
        MetricSpec(
            "search.stage_ms.rerank",
            "ms",
            "mean",
            _W,
            _TRACE,
            1.0,
            "measured",
            "exact rescoring of candidates, per query",
        ),
        MetricSpec(
            "search.rerank_agreement",
            "fraction",
            "mean",
            _W,
            _TRACE,
            1.0,
            "derived",
            "share of the approximate top-k kept in the reranked top-k",
            reference="exact rescoring of the index's own candidates, "
            "not ground-truth neighbours",
        ),
        MetricSpec(
            "index.compression_ratio",
            "x",
            "last",
            None,
            "index.stats()",
            5.0,
            "derived",
            "original fp32 bytes over stored code bytes",
        ),
        MetricSpec(
            "index.rows",
            "rows",
            "last",
            None,
            "index.stats()",
            5.0,
            "measured",
            "rows in the open index",
        ),
        MetricSpec(
            "process.cpu_percent",
            "%",
            "last",
            1.0,
            "psutil (optional)",
            1.0,
            "sampled",
            "this process's CPU use, 100 = one core",
        ),
        MetricSpec(
            "process.rss_mb",
            "MiB",
            "last",
            None,
            "psutil (optional)",
            1.0,
            "measured",
            "this process's resident memory",
        ),
    ]
}


def reading(
    name: str,
    value,
    *,
    n: int | None = None,
    kind: str | None = None,
    reason: str | None = None,
    as_of: float | None = None,
) -> dict:
    """A value with its spec. ``kind`` overrides the spec's (a sampled trace
    stream turns 'measured' into 'sampled'); ``value=None`` needs a ``reason``."""
    spec = REGISTRY[name]
    if value is None and not reason:
        raise ValueError(f"{name}: an unavailable reading must say why")
    out = asdict(spec)
    out.update(
        value=None if value is None else float(value),
        n=n,
        kind=kind or spec.kind,
        as_of=time.time() if as_of is None else as_of,
    )
    if reason:
        out["unavailable_reason"] = reason
    return out


class Window:
    """Timestamped samples kept for ``window_s`` seconds (bounded by ``cap``)."""

    def __init__(self, window_s: float = _W, cap: int = 100_000):
        self.window_s = window_s
        self._d: deque = deque(maxlen=cap)

    def add(self, value: float, weight: int = 1, t: float | None = None) -> None:
        self._d.append((time.time() if t is None else t, float(value), int(weight)))

    def _live(self, now: float) -> list:
        lo = now - self.window_s
        while self._d and self._d[0][0] < lo:
            self._d.popleft()
        return list(self._d)

    def values(self, now: float | None = None) -> np.ndarray:
        return np.array(
            [v for _, v, _ in self._live(time.time() if now is None else now)]
        )

    def weight(self, now: float | None = None) -> int:
        return sum(w for _, _, w in self._live(time.time() if now is None else now))

    def percentile(self, q: float, now: float | None = None):
        v = self.values(now)
        return float(np.percentile(v, q)) if len(v) else None

    def mean(self, now: float | None = None):
        v = self.values(now)
        return float(v.mean()) if len(v) else None

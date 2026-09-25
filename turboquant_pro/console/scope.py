"""The oscilloscope: acquisition engine over the query-trace stream.

Design: docs/DESIGN_console.md section 5.1.

Operators already know this instrument, so the vocabulary is the scope's own:

- **channels** are per-query signals extracted from traces (latency, stage times,
  candidates, rerank agreement, rank movement, score error), each with a vertical scale
  in units/div and a position, over 8 vertical divisions;
- the **time base** is seconds/div over 10 horizontal divisions of query arrival time;
- the **trigger** fires on an edge (a level crossed with a slope), a pulse (a level held
  for at least N consecutive queries) or a logic condition (fields ANDed, including
  non-numeric ones such as the scan path), with a holdoff and a trigger position
  (the pre-trigger share of the record). Modes: **auto** (sweep without a trigger),
  **normal** (update only on a trigger), **single** (arm, capture one record around
  the trigger, stop): the way a one-shot is caught;
- **acquisition** is sample, **peak detect** (each screen column keeps its min and max,
  so a one-query spike survives any time base) or average;
- **persistence** is a decaying time x value hit histogram (the digital phosphor): how
  often a value occurs shows as intensity, so rare events are dim, not lost;
- **measurements** per channel carry statistics across acquisitions;
- **segments** keep every triggered record (segmented memory), and **masks** count
  limit violations (pass/fail testing), optionally stopping on the first.

Pure and time-injected: every method takes ``now``, so it is tested with synthetic
signals the way a scope is checked against a function generator.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

HDIV, VDIV = 10, 8


# ------------------------------------------------------------------ channels
def _per_query(t: dict, v):
    return None if v is None else v / max(t["input"]["n_queries"], 1)


def _stage(name):
    def f(t):
        s = next((s for s in t.get("stages", []) if s["name"] == name), None)
        return None if s is None else _per_query(t, s["ms"])

    return f


def _candidates(t):
    s = next((s for s in t.get("stages", []) if s["name"] == "scan"), None)
    return None if s is None or "candidates" not in s else float(s["candidates"])


def _agree(t):
    return (t.get("results") or {}).get("rerank_agreement")


def _move(t):
    fin = (t.get("results") or {}).get("final") or []
    mv = [abs(r["rank_movement"]) for r in fin if r.get("rank_movement") is not None]
    return float(max(mv)) if mv else None


def _err(t):
    """Approximate minus exact score of the final top result, when both are known."""
    r = t.get("results") or {}
    fin, app = r.get("final") or [], r.get("approximate") or []
    if not fin or fin[0].get("exact_score") is None:
        return None
    a = next((x["score"] for x in app if x["id"] == fin[0]["id"]), None)
    return None if a is None else a - fin[0]["exact_score"]


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    unit: str
    extract: object
    description: str


SIGNALS: dict[str, ChannelSpec] = {
    c.name: c
    for c in [
        ChannelSpec(
            "latency",
            "ms",
            lambda t: _per_query(t, t.get("total_ms")),
            "wall time of the search, per query",
        ),
        ChannelSpec("encode", "ms", _stage("encode"), "query projection, per query"),
        ChannelSpec("scan", "ms", _stage("scan"), "compressed scan, per query"),
        ChannelSpec("rerank", "ms", _stage("rerank"), "exact rescoring, per query"),
        ChannelSpec("candidates", "", _candidates, "candidates cut from the scan"),
        ChannelSpec("agree", "", _agree, "share of approximate top-k kept by rerank"),
        ChannelSpec("move", "ranks", _move, "largest rank movement in the rerank"),
        ChannelSpec("err", "", _err, "approximate minus exact score, top result"),
    ]
}
COLORS = ("yellow", "cyan", "magenta", "green")  # Tek CH1..CH4


@dataclass
class Channel:
    signal: str
    scale: float = 1.0  # units per division
    position: float = 0.0  # divisions from centre
    on: bool = True

    def to_div(self, v: float) -> float:
        """Vertical position in divisions, 0 = bottom edge, VDIV = top edge."""
        return v / self.scale + VDIV / 2 + self.position


def step_125(x: float) -> float:
    """The smallest 1-2-5 value >= x (the scale steps every scope uses)."""
    if x <= 0:
        return 1.0
    e = math.floor(math.log10(x))
    for m in (1, 2, 5, 10):
        if m * 10**e >= x * (1 - 1e-12):
            return m * 10**e
    return 10 ** (e + 1)


def step(scale: float, up: bool) -> float:
    """The next 1-2-5 step above (``up``) or below ``scale``."""
    e = math.floor(math.log10(scale))
    m = round(scale / 10**e)
    seq = [1, 2, 5]
    i = seq.index(m) if m in seq else 0
    if up:
        return seq[i + 1] * 10**e if i < 2 else 10 ** (e + 1)
    return seq[i - 1] * 10**e if i > 0 else 5 * 10 ** (e - 1)


def lomb_scargle(t: np.ndarray, y: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """The Lomb-Scargle periodogram (Scargle 1982), normalised by the variance.

    Query arrivals are not evenly spaced, so a plain FFT (which assumes they are) is
    the wrong transform, and resampling onto a grid would bias it. Lomb-Scargle is the
    least-squares fit of a sinusoid at each frequency to the samples where they fall;
    on evenly spaced samples at the Fourier frequencies it equals the classical
    periodogram |FFT(y - mean)|^2 / (N var), which the tests check.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    y = y - y.mean()
    var = y.var()
    if var <= 0 or len(y) < 2:
        return np.zeros(len(freqs))
    w = 2 * np.pi * np.asarray(freqs, dtype=np.float64)[:, None]
    tau = np.arctan2(np.sin(2 * w * t).sum(1), np.cos(2 * w * t).sum(1))[:, None] / (
        2 * w
    )
    c, s = np.cos(w * (t - tau)), np.sin(w * (t - tau))
    num_c, num_s = (c * y).sum(1) ** 2, (s * y).sum(1) ** 2
    den_c, den_s = (c * c).sum(1), (s * s).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(den_c > 0, num_c / den_c, 0) + np.where(
            den_s > 0, num_s / den_s, 0
        )
    return p / (2 * var)


# ------------------------------------------------------------------- trigger
_OPS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    ">": lambda a, b: a is not None and a > b,
    ">=": lambda a, b: a is not None and a >= b,
    "<": lambda a, b: a is not None and a < b,
    "<=": lambda a, b: a is not None and a <= b,
}


@dataclass
class Trigger:
    kind: str = "edge"  # "edge" | "pulse" | "logic"
    source: str = "latency"
    level: float = 10.0
    slope: str = "rising"  # "rising" | "falling" | "either"
    width: int = 3  # pulse: consecutive queries beyond the level
    conditions: list = field(default_factory=list)  # logic: [(field, op, value)]
    mode: str = "auto"  # "auto" | "normal" | "single"
    holdoff_s: float = 0.0
    position: float = 0.2  # pre-trigger share of the record


@dataclass
class Sample:
    t: float
    trace_id: str
    values: dict
    fields: dict


@dataclass
class Record:
    """One acquisition: the samples in [t0, t1], and the trigger that framed it."""

    t0: float
    t1: float
    samples: list
    trigger_t: float | None = None
    trigger_id: str | None = None


class Scope:
    def __init__(self, buffer: int = 200_000):
        self.channels = [
            Channel("latency", 2.0),
            Channel("scan", 2.0, on=False),
            Channel("agree", 0.2, on=False),
            Channel("move", 2.0, on=False),
        ]
        self.s_per_div = 1.0
        self.h_position = 0.0  # seconds; the record's trigger point moves by this
        self.trigger = Trigger()
        self.acquire = "peak"  # "sample" | "peak" | "average"
        self.running = True
        self.status = "auto"  # "auto" | "ready" | "trig'd" | "armed" | "stop"
        self.buf: deque = deque(maxlen=buffer)
        self.segments: deque = deque(maxlen=500)
        self.record: Record | None = None  # what the screen shows when triggered
        self._pending: Record | None = None  # a trigger waiting for its post-trigger
        self._last_trig = -math.inf
        self._run_len = 0  # pulse trigger: consecutive samples beyond the level
        self.masks: dict = {}  # signal -> (low, high)
        self.stop_on_fail = False
        self.violations: dict = {}
        self.persist: dict = {}  # signal -> hit grid
        self.decay = 0.85  # per update; 1.0 = infinite persistence
        self.stats: dict = {}

    # --------------------------------------------------------------- input
    @property
    def span(self) -> float:
        return self.s_per_div * HDIV

    def feed(self, trace: dict, now: float | None = None) -> None:
        """One finished trace into acquisition memory; evaluates trigger and masks."""
        vals = {}
        for name, spec in SIGNALS.items():
            try:
                v = spec.extract(trace)
            except (KeyError, TypeError, IndexError):
                v = None
            vals[name] = None if v is None or not math.isfinite(v) else float(v)
        s = Sample(
            trace["started_unix"],
            trace["id"],
            vals,
            {"scan_path": trace.get("scan_path"), **(trace.get("params") or {})},
        )
        prev = self.buf[-1] if self.buf else None
        # time advances to this sample first: a record whose post-trigger ended before
        # it must complete before this sample's own trigger is judged
        self.tick(s.t)
        self.buf.append(s)
        self._mask(s)
        if (
            self.running
            and self._fires(prev, s)
            and s.t - self._last_trig >= (self.trigger.holdoff_s)
        ):
            self._last_trig = s.t
            if self._pending is None and self.status != "stop":
                pre = self.trigger.position * self.span
                self._pending = Record(
                    s.t - pre - self.h_position,
                    s.t - pre - self.h_position + self.span,
                    [],
                    s.t,
                    s.trace_id,
                )
        self.tick(now if now is not None else s.t)

    def _fires(self, prev: Sample | None, s: Sample) -> bool:
        tg = self.trigger
        if tg.kind == "logic":
            return bool(tg.conditions) and all(
                _OPS[op]({**s.fields, **s.values}.get(f), v)
                for f, op, v in tg.conditions
            )
        v = s.values.get(tg.source)
        if tg.kind == "pulse":
            beyond = v is not None and (
                v > tg.level if tg.slope != "falling" else v < tg.level
            )
            self._run_len = self._run_len + 1 if beyond else 0
            return self._run_len == tg.width
        p = None if prev is None else prev.values.get(tg.source)
        if v is None or p is None:
            return False
        rising = p < tg.level <= v
        falling = p > tg.level >= v
        return {"rising": rising, "falling": falling, "either": rising or falling}[
            tg.slope
        ]

    def tick(self, now: float) -> None:
        """Advance time: complete a pending record once its post-trigger has passed."""
        if not self.running:
            self.status = "stop"
            return
        pend = self._pending
        if pend is not None and now >= pend.t1:
            pend.samples = [s for s in self.buf if pend.t0 <= s.t <= pend.t1]
            self.record = pend
            self.segments.append(pend)
            self._pending = None
            self._measure(pend.samples)
            if self.trigger.mode == "single":
                self.running = False
                self.status = "stop"
                return
            self.status = "trig'd"
            return
        if pend is not None:
            self.status = "trig'd"
        elif self.trigger.mode == "auto":
            self.status = "auto"
        elif self.trigger.mode == "single":
            self.status = "armed"
        else:
            self.status = "ready"

    # ------------------------------------------------------------- control
    def run_stop(self) -> None:
        self.running = not self.running
        self.status = "stop" if not self.running else "ready"

    def single(self) -> None:
        """Arm for one acquisition: capture the next trigger, then stop."""
        self.trigger.mode = "single"
        self.running = True
        self._pending = None
        self.status = "armed"

    def force(self, now: float) -> None:
        """Force a trigger now (the front-panel Force key)."""
        pre = self.trigger.position * self.span
        self._pending = Record(now - pre, now - pre + self.span, [], now, None)
        self.running = True

    def autoset(self, now: float) -> None:
        """Scales from the data: ~200 queries across the screen, each channel's
        peak-to-peak over about 6 divisions, trigger level at the source's median."""
        recent = [s for s in self.buf if s.t >= now - 60]
        if len(recent) >= 2:
            rate = len(recent) / max(recent[-1].t - recent[0].t, 1e-6)
            self.s_per_div = step_125(200 / rate / HDIV)
        for ch in self.channels:
            v = [
                s.values[ch.signal]
                for s in recent
                if s.values.get(ch.signal) is not None
            ]
            if not v:
                continue
            lo, hi = min(v), max(v)
            ch.scale = step_125(max(hi - lo, abs(hi) * 0.1, 1e-9) / 6)
            ch.position = -((lo + hi) / 2) / ch.scale
        src = [
            s.values[self.trigger.source]
            for s in recent
            if s.values.get(self.trigger.source) is not None
        ]
        if src:
            self.trigger.level = float(np.median(src))

    # ------------------------------------------------------------- display
    def window(self, now: float) -> tuple:
        """The time span on screen: the triggered record when there is one and the
        mode is not auto; otherwise the last span, rolling."""
        if self.record is not None and (
            self.trigger.mode != "auto" or not self.running
        ):
            return self.record.t0, self.record.t1
        t1 = now - self.h_position
        return t1 - self.span, t1

    def columns(self, signal: str, width: int, now: float) -> list:
        """Per screen column: (min, max, mean, n), or None where the column is empty.
        Peak detect keeps min and max; sample keeps the last value; average the mean."""
        t0, t1 = self.window(now)
        src = (
            self.record.samples
            if (
                self.record is not None and (t0, t1) == (self.record.t0, self.record.t1)
            )
            else [s for s in self.buf if t0 <= s.t <= t1]
        )
        cols: list = [None] * width
        acc: list = [[] for _ in range(width)]
        for s in src:
            v = s.values.get(signal)
            if v is None:
                continue
            i = min(width - 1, max(0, int((s.t - t0) / (t1 - t0) * width)))
            acc[i].append(v)
        for i, vs in enumerate(acc):
            if not vs:
                continue
            if self.acquire == "peak":
                cols[i] = (min(vs), max(vs), sum(vs) / len(vs), len(vs))
            elif self.acquire == "average":
                m = sum(vs) / len(vs)
                cols[i] = (m, m, m, len(vs))
            else:
                cols[i] = (vs[-1], vs[-1], vs[-1], len(vs))
        return cols

    def persistence(self, ch: Channel, width: int, rows: int, now: float) -> np.ndarray:
        """The phosphor for one channel: hits per (row, column), decayed each update."""
        grid = self.persist.get(ch.signal)
        if grid is None or grid.shape != (rows, width):
            grid = np.zeros((rows, width))
        grid *= self.decay
        for i, c in enumerate(self.columns(ch.signal, width, now)):
            if c is None:
                continue
            lo = int(ch.to_div(c[0]) / VDIV * rows)
            hi = int(ch.to_div(c[1]) / VDIV * rows)
            for r in range(max(lo, 0), min(hi, rows - 1) + 1):
                grid[r, i] += 1.0
        self.persist[ch.signal] = grid
        return grid

    # ------------------------------------------------------------------ FFT
    def periodogram(self, signal: str, now: float, nfreq: int = 256):
        """The spectrum of a channel over the screen's time window: Lomb-Scargle,
        from 1/T to the mean Nyquist rate n/(2T). None with fewer than 8 samples."""
        t0, t1 = self.window(now)
        pts = [
            (s.t, s.values[signal])
            for s in self.buf
            if t0 <= s.t <= t1 and s.values.get(signal) is not None
        ]
        if len(pts) < 8:
            return None
        t = np.array([p[0] for p in pts]) - pts[0][0]
        y = np.array([p[1] for p in pts])
        T = max(t[-1] - t[0], 1e-9)
        f = np.linspace(1.0 / T, len(t) / (2.0 * T), nfreq)
        return f, lomb_scargle(t, y, f)

    # -------------------------------------------------------- measurements
    MEASURES = ("mean", "min", "max", "pk-pk", "std", "p50", "p95", "p99")

    def measure(self, signal: str, now: float) -> dict:
        t0, t1 = self.window(now)
        v = np.array(
            [
                s.values[signal]
                for s in self.buf
                if t0 <= s.t <= t1 and s.values.get(signal) is not None
            ]
        )
        if not len(v):
            return {"n": 0}
        return {
            "n": int(len(v)),
            "mean": float(v.mean()),
            "min": float(v.min()),
            "max": float(v.max()),
            "pk-pk": float(v.max() - v.min()),
            "std": float(v.std()),
            "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)),
        }

    def _measure(self, samples: list) -> None:
        """Statistics across acquisitions: each completed record adds one value of each
        measurement (count, mean, min, max, sigma of that measurement)."""
        for ch in self.channels:
            v = [
                s.values[ch.signal]
                for s in samples
                if s.values.get(ch.signal) is not None
            ]
            if not v:
                continue
            m = {
                "mean": float(np.mean(v)),
                "max": float(np.max(v)),
                "pk-pk": float(np.ptp(v)),
            }
            for k, x in m.items():
                st = self.stats.setdefault((ch.signal, k), [])
                st.append(x)

    def statistics(self, signal: str, what: str) -> dict:
        xs = self.stats.get((signal, what), [])
        if not xs:
            return {"count": 0}
        a = np.array(xs)
        return {
            "count": len(a),
            "mean": float(a.mean()),
            "min": float(a.min()),
            "max": float(a.max()),
            "std": float(a.std()),
        }

    # ------------------------------------------------------------------ mask
    def _mask(self, s: Sample) -> None:
        for sig, (lo, hi) in self.masks.items():
            v = s.values.get(sig)
            if v is None:
                continue
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                self.violations.setdefault(sig, []).append(s.trace_id)
                if self.stop_on_fail and self.running:
                    self.force(s.t)  # capture around the failure, then stop
                    self.trigger.mode = "single"

    def mask_summary(self) -> dict:
        n = len(self.buf)
        return {
            sig: {
                "limits": self.masks[sig],
                "violations": len(ids),
                "of": n,
                "last": ids[-1] if ids else None,
            }
            for sig, ids in ((s, self.violations.get(s, [])) for s in self.masks)
        }

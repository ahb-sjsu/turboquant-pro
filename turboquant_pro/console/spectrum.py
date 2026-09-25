"""The spectrum analyzer: what the observer reads, direction by direction.

Design: docs/DESIGN_console.md section 5.2. The x axis is the eigendirection index of
the observer's read operator ``P_C`` (the "frequency"), in descending sensitivity; the y
axis is power in dB. Every trace is a quantity :mod:`turboquant_pro.read_allocation`
already defines, computed by the same functions, so the analyzer shows the water-filling
problem rather than a lookalike of it:

- ``sens``      lambda_i, the eigenvalues of ``P_C`` (``read_allocation._spectrum``);
- ``var``       sigma_i^2, source variance along each direction (``_variance_along``);
- ``weighted``  w_i = lambda_i sigma_i^2, what the reader sees of the source; the water
  level theta of :func:`~turboquant_pro.read_allocation.allocate_bits` is its reference
  line, and directions below it are starved by an optimal allocation;
- ``predicted`` w_i 2^(-2 b_i), the distortion the allocation's model predicts;
- ``noise``     lambda_i u_i' Sigma_delta u_i, the distortion the consumer actually
  feels along each direction from a codec's real error
  (``read_operators.error_covariance``).
  It sums exactly to :func:`~turboquant_pro.read_allocation.realised_distortion`,
  because tr(P Sigma) = sum_i lambda_i u_i' Sigma u_i in P's eigenbasis.

The instrument is a spectrum analyzer's: four traces, each clear/write, max hold, min
hold, average or blank; reference level and dB/div; start and stop over the direction
index; a peak or average detector when there are more directions than columns; markers
with peak search, next peak and a delta marker; a limit line with pass/fail; and a
waterfall of past sweeps, which is drift made visible. Pure: sweeps are computed by
:func:`sweep`, and the :class:`Analyzer` only accumulates and reduces them.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..read_allocation import _spectrum, _variance_along, allocate_bits
from ..read_operators import error_covariance

SOURCES = ("sens", "var", "weighted", "predicted", "noise")
FLOOR = 1e-30  # below any meaningful power; dB of zero is reported at this floor


def db(x) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), FLOOR))


@dataclass
class Sweep:
    """One measurement across every direction of the read operator."""

    t: float
    sens: np.ndarray
    var: np.ndarray
    weighted: np.ndarray
    basis: np.ndarray
    bits: np.ndarray | None = None
    predicted: np.ndarray | None = None
    noise: np.ndarray | None = None
    water_level: float | None = None
    effective_rank: float = 0.0
    budget_bits: float | None = None
    notes: dict = field(default_factory=dict)

    def source(self, name: str) -> np.ndarray | None:
        return getattr(self, name)

    @property
    def realised_total(self) -> float | None:
        return None if self.noise is None else float(self.noise.sum())

    @property
    def predicted_total(self) -> float | None:
        return None if self.predicted is None else float(self.predicted.sum())


def read_operator_from_queries(queries: np.ndarray) -> np.ndarray:
    """``E[q q']``: the read operator of an inner-product retrieval consumer, whose
    scores are ``q . x`` and so respond to a perturbation of ``x`` along ``u`` in
    proportion to ``E[(q . u)^2] = u' E[q q'] u``."""
    q = np.asarray(queries, dtype=np.float64).reshape(-1, np.shape(queries)[-1])
    return q.T @ q / max(len(q), 1)


def sweep(
    read_operator: np.ndarray,
    sample: np.ndarray,
    reconstructed: np.ndarray | None = None,
    budget_bits: float | None = None,
    t: float | None = None,
) -> Sweep:
    """Measure every trace once. ``sample`` is the source (rows of vectors);
    ``reconstructed`` its codec output, for the ``noise`` trace; ``budget_bits`` the
    total bits over all directions, for the water level and ``predicted``."""
    lam, basis = _spectrum(read_operator)
    var = _variance_along(basis, sample, lam.size)
    s = Sweep(
        t=time.time() if t is None else t,
        sens=lam,
        var=var,
        weighted=lam * var,
        basis=basis,
    )
    s1, s2 = float(lam.sum()), float((lam**2).sum())
    s.effective_rank = (s1 * s1 / s2) if s2 > 0 else 0.0
    if budget_bits is not None:
        alloc = allocate_bits(
            read_operator, budget_bits=budget_bits, activations=sample
        )
        s.bits, s.water_level, s.budget_bits = (
            alloc.bits,
            alloc.water_level,
            budget_bits,
        )
        s.predicted = (
            alloc.sensitivity * alloc.variance * np.power(2.0, -2.0 * alloc.bits)
        )
    if reconstructed is not None:
        sig = error_covariance(sample, reconstructed)
        s.noise = lam * np.einsum("ji,jk,ki->i", basis, sig, basis)
    return s


@dataclass
class Trace:
    source: str
    mode: str = "write"  # "write" | "maxhold" | "minhold" | "average" | "blank"
    avg_n: int = 16
    data: np.ndarray | None = None  # dB, per direction
    count: int = 0

    def update(self, values_db: np.ndarray | None) -> None:
        if values_db is None or self.mode == "blank":
            return
        if (
            self.data is None
            or self.data.shape != values_db.shape
            or self.mode == "write"
        ):
            self.data, self.count = values_db.copy(), 1
            return
        self.count += 1
        if self.mode == "maxhold":
            np.maximum(self.data, values_db, out=self.data)
        elif self.mode == "minhold":
            np.minimum(self.data, values_db, out=self.data)
        elif self.mode == "average":
            # exponential average in power, as analyzers do, not in dB
            k = min(self.count, self.avg_n)
            p = (1 - 1 / k) * 10 ** (self.data / 10) + (1 / k) * 10 ** (values_db / 10)
            self.data = db(p)

    def clear(self) -> None:
        self.data, self.count = None, 0


class Analyzer:
    def __init__(self, waterfall: int = 64):
        self.traces = [
            Trace("weighted"),
            Trace("noise"),
            Trace("predicted"),
            Trace("noise", mode="blank"),
        ]
        self.ref_db = 0.0  # top of the screen
        self.db_div = 10.0
        self.vdiv = 8
        self.start, self.stop = 0, None  # direction index span; stop None = all
        self.detector = "peak"  # "peak" | "average" (when directions > columns)
        self.limit: float | None = None  # dB; None = the water level when known
        self.limit_on = True
        self.limit_trace = 1  # trace checked against the limit line
        # direction indices: markers[0] is the reference once a delta marker exists,
        # markers[1] the active (delta) marker; the readout is active minus reference
        self.markers: list = []
        self.waterfall: deque = deque(maxlen=waterfall)
        self.waterfall_source = "sens"
        self.last: Sweep | None = None
        self.sweeps = 0
        self.running = True

    def feed(self, s: Sweep) -> None:
        if not self.running:
            return
        self.last = s
        self.sweeps += 1
        for tr in self.traces:
            v = s.source(tr.source)
            tr.update(None if v is None else db(v))
        wf = s.source(self.waterfall_source)
        if wf is not None:
            self.waterfall.append(db(wf))

    # ------------------------------------------------------------- display
    def span(self) -> tuple:
        n = 0 if self.last is None else self.last.sens.size
        stop = n if self.stop is None else min(self.stop, n)
        return max(0, min(self.start, stop - 1)), stop

    def columns(self, trace: Trace, width: int) -> list:
        """Per screen column the dB value, reduced over the directions the column
        covers by the detector (peak keeps the largest, average the power mean)."""
        if trace.data is None or trace.mode == "blank":
            return [None] * width
        a, b = self.span()
        seg = trace.data[a:b]
        if not seg.size:
            return [None] * width
        edges = np.linspace(0, seg.size, width + 1)
        out = []
        for i in range(width):
            lo, hi = int(edges[i]), max(int(edges[i + 1]), int(edges[i]) + 1)
            chunk = seg[lo : min(hi, seg.size)]
            if not chunk.size:
                out.append(None)
            elif self.detector == "peak":
                out.append(float(chunk.max()))
            else:
                out.append(float(db(np.mean(10 ** (chunk / 10)))))
        return out

    def autoscale(self) -> None:
        """Reference level at the highest visible point, rounded up to 10 dB; the
        dB/div so the lowest meaningful point is on screen."""
        vals = [
            t.data[self.span()[0] : self.span()[1]]
            for t in self.traces
            if t.data is not None and t.mode != "blank"
        ]
        if not vals:
            return
        v = np.concatenate(vals)
        v = v[v > db(FLOOR) + 1]
        if not v.size:
            return
        self.ref_db = float(np.ceil(v.max() / 10) * 10)
        rng = self.ref_db - float(np.percentile(v, 2))
        self.db_div = float(
            next((d for d in (1, 2, 5, 10, 20) if d * self.vdiv >= rng), 20)
        )

    def waterfall_range(self) -> tuple | None:
        """The waterfall's own colour scale (min, max dB over its history, floor
        excluded): it is not the trace graticule's, since its source need not be on
        screen."""
        if not self.waterfall:
            return None
        v = np.concatenate([w for w in self.waterfall])
        v = v[v > float(db(FLOOR)) + 1]
        if not v.size:
            return None
        lo, hi = float(np.percentile(v, 1)), float(v.max())
        return (lo, hi if hi > lo else lo + 1.0)

    def limit_db(self) -> float | None:
        if not self.limit_on:
            return None
        if self.limit is not None:
            return self.limit
        s = self.last
        return None if s is None or s.water_level is None else float(db(s.water_level))

    def limit_check(self) -> dict:
        """Pass/fail of the limit trace against the limit line, per direction."""
        lim = self.limit_db()
        tr = self.traces[self.limit_trace]
        if lim is None or tr.data is None:
            return {"limit_db": lim, "checked": 0, "fail": [], "passed": None}
        a, b = self.span()
        fail = [int(i) for i in range(a, b) if tr.data[i] > lim]
        return {"limit_db": lim, "checked": b - a, "fail": fail, "passed": not fail}

    # ------------------------------------------------------------- markers
    def peak_search(self, trace: int = 0, after: int | None = None) -> int | None:
        """The highest point of a trace in the span (after ``after`` for 'next peak':
        the highest local maximum strictly below the current marker's level)."""
        d = self.traces[trace].data
        if d is None:
            return None
        a, b = self.span()
        idx = np.arange(a, b)
        if after is not None:
            ref = d[after]
            peaks = [
                i
                for i in idx
                if (i == a or d[i] >= d[i - 1])
                and (i == b - 1 or d[i] >= d[i + 1])
                and d[i] < ref
            ]
            return int(max(peaks, key=lambda i: d[i])) if peaks else None
        return int(idx[np.argmax(d[a:b])])

    def marker_readout(self, trace: int = 0) -> list:
        d = self.traces[trace].data
        out = []
        for j, m in enumerate(self.markers):
            if d is None or not 0 <= m < d.size:
                continue
            row = {"marker": j + 1, "direction": m, "db": float(d[m])}
            if j == 1:
                row["delta_db"] = float(d[m] - d[self.markers[0]])
                row["delta_dirs"] = m - self.markers[0]
            out.append(row)
        return out

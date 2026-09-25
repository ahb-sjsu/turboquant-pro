"""The spectrum analyzer screen of the terminal console (DESIGN_console 5.2).

Laid out like a swept analyzer: a status line (RUN/STOP, sweep count, reference level,
dB/div, start and stop, detector), a 10 x 8 graticule with up to four traces in braille,
the limit line (by default the allocation's water level), markers with a delta readout,
a waterfall of past sweeps, the readouts that matter here (effective rank of the read
operator, predicted vs realised consumer distortion, limit pass/fail) and softkeys.
:func:`render` and :func:`key` are pure.
"""

from __future__ import annotations

import numpy as np

from .spectrum import SOURCES, Analyzer

HDIV, VDIV = 10, 8
BRAILLE = 0x2800
_DOT = {
    (0, 0): 0x01,
    (0, 1): 0x02,
    (0, 2): 0x04,
    (1, 0): 0x08,
    (1, 1): 0x10,
    (1, 2): 0x20,
    (0, 3): 0x40,
    (1, 3): 0x80,
}
COLORS = ("yellow", "cyan", "magenta", "green")
LABEL = {
    "sens": "sensitivity lambda",
    "var": "variance sigma^2",
    "weighted": "weighted lambda*sigma^2",
    "predicted": "predicted distortion",
    "noise": "realised noise",
}
MODES = ("write", "maxhold", "minhold", "average", "delta", "blank")
SOFTKEYS = (
    "Spc Run/Stop",
    "a Autoscale",
    "1-4 Trace",
    "m Mode",
    "c Source",
    "k Peak",
    "n Next",
    "d Delta",
    "l Limit",
    "[ ] Span",
    "v View",
    "? Keys",
)
HELP = [
    ("Space", "run / stop sweeping"),
    ("a", "autoscale reference level and dB/div"),
    ("1-4", "select a trace"),
    ("m", "trace mode: write, max hold, min hold, average, blank"),
    ("c", "trace source: sensitivity, variance, weighted, predicted, noise"),
    ("x", "clear the traces (restart holds and averages)"),
    ("Up / Down", "reference level by one division"),
    ("+ / -", "dB per division (1-2-5 steps)"),
    ("[ / ]", "narrow / widen the span of directions shown"),
    (", / .", "move the span"),
    ("k", "marker to the peak of the selected trace"),
    ("n", "marker to the next lower peak"),
    ("d", "delta: freeze M1 as the reference; k / n then move M2 (reads M2 - M1)"),
    (
        "l",
        "limit line: water level theta / off (optimal per-direction D = min(w, theta))",
    ),
    ("w", "waterfall source"),
    (
        "R",
        "store the last sweep as the reference (again: clear). Later sweeps are "
        "measured in its directions; delta traces show now minus reference",
    ),
    ("v", "view: scope / spectrum / overview"),
    ("q", "quit"),
]


def _db(v):
    return "-" if v is None else f"{v:.1f} dB"


def render(cv, st: dict, g: dict) -> None:
    an: Analyzer = st["analyzer"]
    w, h = cv.w, cv.h
    wf_h = max(3, (h - 6) // 4) if h >= 30 else 0
    gw, gh = w - 2, h - 6 - (wf_h + 1 if wf_h else 0)
    y0 = 1

    # status line -------------------------------------------------------------
    a, b = an.span()
    s = an.last
    run = ("RUN", "green") if an.running else ("STOP", "red")
    parts = [
        (f" {run[0]} ", run[1]),
        (f"SPECTRUM sweeps {an.sweeps}  ", "purple"),
        (f"Ref {an.ref_db:g} dB  {an.db_div:g} dB/div  ", None),
        (f"Start {a}  Stop {b}  ", None),
        (
            (
                (
                    f"REF drift {100 * an.drift():.1f}%  "
                    if an.drift() is not None
                    else "REF  "
                )
                if an.reference is not None
                else ""
            ),
            "purple",
        ),
        (f"[{an.detector} detector] ", "dim"),
    ]
    if s is not None and b > a and gw * 2 < (b - a):
        parts.append((f"[{(b - a) / (gw * 2):.1f} dirs/dot] ", "dim"))
    x = 0
    for text, col in parts:
        cv.put(0, x, text, col)
        x += len(text)
    if s is None:
        cv.put(
            2, 2, st.get("spectrum_reason") or "waiting for the first sweep...", "amber"
        )
        return

    # graticule ---------------------------------------------------------------
    cv.box(y0, 0, gh + 2, gw + 2, "", g, color="dim")
    for j in range(gh):
        for i in range(gw):
            on_v = (i * HDIV) % gw < HDIV
            on_h = (j * VDIV) % gh < VDIV
            if on_v and on_h:
                cv.put(y0 + 1 + j, 1 + i, "+" if g.get("ascii") else "┼", "grid")
            elif on_v and j % 2 == 0 or on_h and i % 2 == 0:
                cv.put(y0 + 1 + j, 1 + i, "." if g.get("ascii") else "·", "grid")
    bottom = an.ref_db - an.db_div * VDIV

    def row_of(dbv: float, sub: int, delta: bool = False) -> int:
        """dB to a row index counted from the bottom, in ``sub`` units per cell. A
        delta trace is a difference, not a level: it gets its own axis, 0 dB at the
        centre and the same dB/div."""
        if delta:
            return int((dbv / (an.db_div * VDIV) + 0.5) * gh * sub)
        return int((dbv - bottom) / (an.ref_db - bottom) * gh * sub)

    if any(t.mode == "delta" for t in an.traces):
        cy = gh - 1 - row_of(0.0, 1, delta=True)
        cv.put(y0 + 1 + cy, 2, " Δ 0 dB ", "purple")

    # limit line ----------------------------------------------------------------
    lim = an.limit_db()
    if lim is not None:
        r = gh - 1 - row_of(lim, 1)
        if 0 <= r < gh:
            for i in range(0, gw, 2):
                cv.put(y0 + 1 + r, 1 + i, "-" if g.get("ascii") else "─", "red")
            cv.put(y0 + 1 + r, gw - 11, " water lvl ", "red")

    # traces ----------------------------------------------------------------------
    sel = st.get("sel_trace", 0)
    for ti, tr in enumerate(an.traces):
        cols = an.columns(tr, gw * 2)
        col = COLORS[ti]
        dl = tr.mode == "delta"
        if g.get("ascii"):
            for i, v in enumerate(cols[::2]):
                if v is not None:
                    r = gh - 1 - row_of(v, 1, dl)
                    if 0 <= r < gh:
                        cv.put(y0 + 1 + r, 1 + i, "*", col)
            continue
        cells: dict = {}
        prev = None
        for sx, v in enumerate(cols):
            if v is None:
                prev = None
                continue
            ry = row_of(v, 4, dl)
            lo, hi = (ry, ry) if prev is None else (min(ry, prev), max(ry, prev))
            prev = ry
            for sy in range(max(lo, 0), min(hi, gh * 4 - 1) + 1):  # join to the last
                cy, cx = gh - 1 - sy // 4, sx // 2
                cells[(cy, cx)] = cells.get((cy, cx), 0) | _DOT[(sx % 2, 3 - sy % 4)]
        for (cy, cx), bits in cells.items():
            cv.put(
                y0 + 1 + cy,
                1 + cx,
                chr(BRAILLE + bits),
                f"{col}_bold" if ti == sel else col,
            )

    # markers ---------------------------------------------------------------------
    for j, m in enumerate(an.markers[:2]):
        if not a <= m < b:
            continue
        cx = int((m - a) / max(b - a, 1) * gw)
        d = an.traces[sel].data
        if d is None:
            continue
        r = gh - 1 - row_of(float(d[m]), 1)
        glyph = ("M" if j == 0 else "D") if g.get("ascii") else ("◆" if j == 0 else "◇")
        if 0 <= r < gh:
            cv.put(y0 + 1 + r, 1 + cx, glyph, "bold")

    # waterfall -------------------------------------------------------------------
    y = y0 + gh + 2
    if wf_h:
        ramp = " .:-=+*#%@" if g.get("ascii") else " ░▒▓█"
        wr = an.waterfall_range() or (bottom, an.ref_db)  # its own colour scale
        cv.put(
            y,
            1,
            f"waterfall: {LABEL.get(an.waterfall_source, '')}, newest at top, "
            f"{wr[0]:.0f}..{wr[1]:.0f} dB",
            "dim",
        )
        rows = list(an.waterfall)[-wf_h:][::-1]
        for k, spec in enumerate(rows):
            seg = spec[a:b]
            if not seg.size:
                continue
            edges = np.linspace(0, seg.size, gw + 1)
            for i in range(gw):
                lo, hi = int(edges[i]), max(int(edges[i + 1]), int(edges[i]) + 1)
                v = float(seg[lo : min(hi, seg.size)].max())
                f = (v - wr[0]) / (wr[1] - wr[0])
                ch = ramp[max(0, min(len(ramp) - 1, int(f * (len(ramp) - 1) + 0.5)))]
                cv.put(y + 1 + k, 1 + i, ch, "cyan")
        y += wf_h + 1

    # readouts --------------------------------------------------------------------
    names = "  ".join(
        f"T{i + 1} {tr.source}:{tr.mode}" for i, tr in enumerate(an.traces)
    )
    cv.put(y, 1, names[: w - 2], None)
    lc = an.limit_check()
    real, pred = s.realised_total, s.predicted_total
    ro = [f"eff rank {s.effective_rank:.1f}/{s.sens.size}"]
    if pred is not None:
        ro.append(f"predicted D {pred:.3g}")
    ro.append(
        "realised D "
        + (f"{real:.3g}" if real is not None else "- (no codec to reconstruct with)")
    )
    if real and pred:
        ro.append(f"gap {10 * np.log10(real / pred):+.1f} dB")
    if lc["passed"] is not None:
        ro.append(
            f"limit {'PASS' if lc['passed'] else 'FAIL'} "
            f"{len(lc['fail'])}/{lc['checked']} dirs over"
        )
    for mk in an.marker_readout(sel):
        ro.append(
            f"M{mk['marker']} dir {mk['direction']} {_db(mk['db'])}"
            + (
                f" Δ {mk['delta_db']:+.1f} dB / {mk['delta_dirs']:+d} dirs"
                if "delta_db" in mk
                else ""
            )
        )
    cv.put(y + 1, 1, "   ".join(ro)[: w - 2], "red" if lc["passed"] is False else None)

    x = 0
    for k in SOFTKEYS:
        if x + len(k) + 3 > w:
            break
        cv.put(h - 1, x, f"[{k}]", "dim")
        x += len(k) + 3


def _step125(v: float, up: bool) -> float:
    seq = [1, 2, 5, 10, 20]
    i = min(range(len(seq)), key=lambda j: abs(seq[j] - v))
    return float(seq[min(i + 1, len(seq) - 1)] if up else seq[max(i - 1, 0)])


def key(st: dict, k: str) -> str:
    an: Analyzer = st["analyzer"]
    sel = st.setdefault("sel_trace", 0)
    tr = an.traces[sel]
    if k == "space":
        an.running = not an.running
        return "RUN" if an.running else "STOP"
    if k == "a":
        an.autoscale()
        return f"ref {an.ref_db:g} dB, {an.db_div:g} dB/div"
    if k in "1234" and len(k) == 1:
        st["sel_trace"] = int(k) - 1
        t = an.traces[st["sel_trace"]]
        return f"T{k} {t.source} ({t.mode})"
    if k == "m":
        tr.mode = MODES[(MODES.index(tr.mode) + 1) % len(MODES)]
        tr.clear()
        return f"T{sel + 1} {tr.mode}"
    if k == "c":
        tr.source = SOURCES[(SOURCES.index(tr.source) + 1) % len(SOURCES)]
        tr.clear()
        return f"T{sel + 1} = {LABEL[tr.source]}"
    if k == "x":
        for t in an.traces:
            t.clear()
        return "traces cleared"
    if k in ("up", "down"):
        an.ref_db += an.db_div if k == "up" else -an.db_div
        return f"ref {an.ref_db:g} dB"
    if k in ("+", "-"):
        an.db_div = _step125(an.db_div, up=(k == "+"))
        return f"{an.db_div:g} dB/div"
    if k in ("[", "]", ",", "."):
        n = 0 if an.last is None else an.last.sens.size
        a, b = an.span()
        width = b - a
        if k == "[":
            width = max(4, width // 2)
        elif k == "]":
            width = min(n, width * 2)
        shift = {",": -max(1, width // 4), ".": max(1, width // 4)}.get(k, 0)
        a = max(0, min(n - width, a + shift))
        an.start, an.stop = a, a + width
        return f"span {an.start}..{an.stop}"
    if k in ("k", "n"):
        active = len(an.markers) - 1 if an.markers else 0  # the delta marker if any
        after = an.markers[active] if (k == "n" and an.markers) else None
        if k == "n" and after is None:
            return "place a marker first (k)"
        m = an.peak_search(sel, after=after)
        if m is None:
            return "no data on this trace" if k == "k" else "no lower peak"
        if an.markers:
            an.markers[active] = m
        else:
            an.markers = [m]
        return f"M{active + 1} dir {m}"
    if k == "d":
        if not an.markers:
            return "place a marker first (k)"
        if len(an.markers) == 1:
            an.markers = [an.markers[0], an.markers[0]]  # reference, then active
            return "delta marker on: k / n now move M2; the readout is M2 - M1"
        an.markers = an.markers[:1]
        return "delta marker off"
    if k == "l":
        an.limit_on = not an.limit_on
        return "limit line " + ("water level" if an.limit_on else "off")
    if k == "R":
        if an.reference is None:
            if an.last is None:
                return "no sweep to store yet"
            an.store_reference()
            for t in an.traces:
                t.clear()
            return "reference stored: directions frozen; m -> delta shows the change"
        an.clear_reference()
        for t in an.traces:
            t.clear()
        return "reference cleared: back to each sweep's own directions"
    if k == "w":
        an.waterfall_source = SOURCES[
            (SOURCES.index(an.waterfall_source) + 1) % len(SOURCES)
        ]
        an.waterfall.clear()
        return f"waterfall = {LABEL[an.waterfall_source]}"
    return ""

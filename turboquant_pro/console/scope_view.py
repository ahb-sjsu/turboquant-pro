"""The oscilloscope screen of the terminal console (DESIGN_console 5.1).

Laid out like a scope's front panel: a status line (RUN/STOP, trigger state, time base,
trigger source, slope and level, mode, acquisition), a 10 x 8 division graticule with
the trigger point on the top edge and the trigger level on the right edge, waveforms in
braille (2 x 4 dots per cell, as btop draws) over a dim persistence phosphor, channel
and trigger readouts, a measurement bar with statistics, the mask line and a softkey
row. :func:`render` draws into a :class:`~turboquant_pro.console.tui.Canvas`;
:func:`key` applies a front-panel key to the scope. Both are pure.
"""

from __future__ import annotations

import numpy as np

from .scope import COLORS, HDIV, SIGNALS, VDIV, Scope, step

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
SOFTKEYS = (
    "Spc Run/Stop",
    "s Single",
    "f Force",
    "a Autoset",
    "1-4 CH",
    "t Trig",
    "m Mode",
    "p Acq",
    "d Persist",
    "h History",
    "v View",
    "? Keys",
)
HELP = [
    ("Space", "run / stop acquisition"),
    ("s", "single: arm, catch one trigger, stop"),
    ("f", "force a trigger now"),
    ("a", "autoset scales, time base and trigger level"),
    ("1-4", "select a channel; again to turn it off"),
    ("Up / Down", "volts-per-div of the selected channel (1-2-5 steps)"),
    ("[ / ]", "vertical position of the selected channel"),
    ("Left / Right", "time base, seconds per division"),
    (", / .", "horizontal position"),
    ("c", "cycle the selected channel's signal"),
    ("t", "trigger on the selected channel"),
    ("+ / -", "trigger level"),
    ("/", "trigger slope: rising, falling, either"),
    ("m", "trigger mode: auto, normal, single"),
    ("p", "acquisition: peak detect, sample, average"),
    ("d", "persistence: decaying, infinite, off"),
    ("x", "clear persistence"),
    ("h", "history: step segments with Left/Right, Enter inspects the trigger query"),
    ("F", "FFT of the selected channel (Lomb-Scargle: arrivals are irregular)"),
    ("S", "save this setup to a .tqs file (recall with tqp console --setup FILE)"),
    ("v", "view: scope / spectrum / overview"),
    ("q", "quit"),
]


def _fmt(v, unit=""):
    if v is None:
        return "-"
    a = abs(v)
    s = (
        f"{v:.0f}"
        if a >= 1000
        else f"{v:.1f}" if a >= 10 else (f"{v:.2f}" if a >= 1 else f"{v:.3f}")
    )
    return f"{s} {unit}".rstrip()


def _per_div(v, unit):
    return f"{_fmt(v, unit)}/div"


def render(cv, st: dict, g: dict, now: float, top: int = 0) -> None:
    sc: Scope = st["scope"]
    w, h = cv.w, cv.h
    side = 26 if w >= 110 else 0
    gw, gh = w - side - 2, h - top - 7  # graticule interior size in cells
    y0, x0 = top + 1, 0

    # status line (the scope's top bar) --------------------------------------
    tg = sc.trigger
    run = ("RUN", "green") if sc.running else ("STOP", "red")
    stat = {
        "trig'd": ("Trig'd", "green"),
        "auto": ("Auto", "amber"),
        "ready": ("Ready", "amber"),
        "armed": ("Armed", "cyan"),
        "stop": ("Stop", "red"),
    }[sc.status]
    slope = {"rising": "↑", "falling": "↓", "either": "↕"}[tg.slope]
    if g.get("ascii"):
        slope = {"rising": "/", "falling": "\\", "either": "X"}[tg.slope]
    src_unit = SIGNALS[tg.source].unit
    parts = [
        (f" {run[0]} ", run[1]),
        (f"{stat[0]} ", stat[1]),
        (f"H {_per_div(sc.s_per_div, 's')}  ", None),
        (f"T {tg.source} {slope} {_fmt(tg.level, src_unit)} ", "cyan"),
        (f"[{tg.kind} {tg.mode}] ", None),
        (f"[{sc.acquire}] ", None),
        (
            f"[persist {'inf' if sc.decay >= 1 else 'off' if sc.decay <= 0 else 'on'}]",
            "dim",
        ),
    ]
    x = 0
    for text, col in parts:
        cv.put(top, x, text, col)
        x += len(text)
    if st.get("history") is not None:
        n = len(sc.segments)
        cv.put(top, x + 1, f"HISTORY {st['history'] + 1}/{n}", "purple")

    # graticule --------------------------------------------------------------
    cv.box(y0, x0, gh + 2, gw + 2, "", g, color="dim")
    for j in range(gh):
        for i in range(gw):
            on_v = (i * HDIV) % gw < HDIV
            on_h = (j * VDIV) % gh < VDIV
            if on_v and on_h:
                cv.put(y0 + 1 + j, x0 + 1 + i, "+" if g.get("ascii") else "┼", "grid")
            elif on_v and j % 2 == 0 or on_h and i % 2 == 0:
                cv.put(y0 + 1 + j, x0 + 1 + i, "." if g.get("ascii") else "·", "grid")

    sel = st.get("sel_ch", 0)
    if st.get("fft"):
        _render_fft(cv, st, g, now, y0, x0, gw, gh, top)
        _render_softkeys(cv, w, h)
        return

    # persistence phosphor, then live waveforms -------------------------------
    for ci, ch in enumerate(sc.channels):
        if not ch.on:
            continue
        col = COLORS[ci]
        if sc.decay > 0:
            grid = sc.persistence(ch, gw, gh, now)
            mx = grid.max() if grid.size else 0
            if mx > 0:
                for r in range(gh):
                    for c in range(gw):
                        v = grid[r, c] / mx
                        if v > 0.02:
                            cv.put(
                                y0 + gh - r,
                                x0 + 1 + c,
                                "░" if not g.get("ascii") else ":",
                                f"{col}_dim" if v < 0.5 else col,
                            )
        cols = sc.columns(ch.signal, gw * 2, now)
        if g.get("ascii"):
            for i, cc in enumerate(cols[::2]):
                if cc is None:
                    continue
                lo = int(ch.to_div(cc[0]) / VDIV * gh)
                hi = int(ch.to_div(cc[1]) / VDIV * gh)
                for r in range(max(lo, 0), min(hi, gh - 1) + 1):
                    cv.put(y0 + gh - r, x0 + 1 + i, "*", col)
            continue
        cells: dict = {}
        sub_h = gh * 4
        for sx, cc in enumerate(cols):
            if cc is None:
                continue
            lo = int(ch.to_div(cc[0]) / VDIV * sub_h)
            hi = int(ch.to_div(cc[1]) / VDIV * sub_h)
            lo, hi = max(lo, 0), min(hi, sub_h - 1)
            for sy in range(lo, hi + 1):
                cy, cx = gh - 1 - sy // 4, sx // 2
                dot = _DOT[(sx % 2, 3 - sy % 4)]
                cells[(cy, cx)] = cells.get((cy, cx), 0) | dot
            if hi < lo:  # off screen: mark the edge it left by
                edge = 0 if lo >= sub_h else gh - 1
                cells[(edge, sx // 2)] = cells.get((edge, sx // 2), 0) | 0x09
        for (cy, cx), bits in cells.items():
            cv.put(
                y0 + 1 + cy,
                x0 + 1 + cx,
                chr(BRAILLE + bits),
                f"{col}_bold" if ci == sel else col,
            )
        # the channel's ground marker on the left edge
        gy = gh - 1 - int(ch.to_div(0.0) / VDIV * gh)
        if 0 <= gy < gh:
            cv.put(y0 + 1 + gy, x0, str(ci + 1), col)

    # reference lines (drawn, never judged) ---------------------------------------
    for ch in sc.channels:
        ref = sc.references.get(ch.signal)
        if not ch.on or ref is None:
            continue
        ry = gh - 1 - int(ch.to_div(ref[0]) / VDIV * gh)
        if 0 <= ry < gh:
            for i in range(1, gw, 3):
                cv.put(
                    y0 + 1 + ry, x0 + 1 + i, "-" if g.get("ascii") else "╌", "purple"
                )
            lab = f" {ref[1]} "
            cv.put(y0 + 1 + ry, x0 + max(1, gw - len(lab)), lab, "purple")

    # trigger markers -----------------------------------------------------------
    t0, t1 = sc.window(now)
    tsrc = next((c for c in sc.channels if c.signal == tg.source), None)
    if tg.kind != "logic" and tsrc is not None:
        ty = gh - 1 - int(tsrc.to_div(tg.level) / VDIV * gh)
        if 0 <= ty < gh:
            cv.put(
                y0 + 1 + ty,
                x0 + gw + 1,
                "<" if g.get("ascii") else "◀",
                COLORS[sc.channels.index(tsrc)],
            )
    if (
        sc.record is not None
        and sc.record.trigger_t is not None
        and (t0, t1) == (sc.record.t0, sc.record.t1)
    ):
        tx = int((sc.record.trigger_t - t0) / (t1 - t0) * gw)
        cv.put(
            y0, x0 + 1 + max(0, min(gw - 1, tx)), "T" if g.get("ascii") else "▼", "cyan"
        )

    # side panel: channels and trigger -----------------------------------------
    if side:
        sx = w - side
        cv.box(y0, sx, gh + 2, side, "channels", g, color="dim")
        yy = y0 + 1
        for ci, ch in enumerate(sc.channels):
            spec = SIGNALS[ch.signal]
            mark = ">" if ci == sel else " "
            cv.put(
                yy,
                sx + 1,
                f"{mark}CH{ci + 1} {ch.signal}",
                COLORS[ci] if ch.on else "dim",
            )
            cv.put(
                yy + 1,
                sx + 3,
                (
                    f"{_per_div(ch.scale, spec.unit)}  pos {ch.position:+.1f}"
                    if ch.on
                    else "off"
                )[: side - 4],
                None if ch.on else "dim",
            )
            yy += 2
        yy += 1
        cv.put(yy, sx + 1, "trigger", "bold")
        rows = [
            f"{tg.kind}  {tg.mode}",
            f"src {tg.source}  {tg.slope}",
            f"level {_fmt(tg.level, src_unit)}",
            f"holdoff {tg.holdoff_s:g}s  pre {tg.position:.0%}",
        ]
        if tg.kind == "logic":
            rows[1:3] = [" AND ".join(f"{a}{o}{b}" for a, o, b in tg.conditions)]
        for i, r_ in enumerate(rows):
            cv.put(yy + 1 + i, sx + 2, r_[: side - 3], None)
        cv.put(yy + 6, sx + 1, f"segments {len(sc.segments)}", "purple")

    # measurement bar --------------------------------------------------------
    my = y0 + gh + 2
    for ci, ch in enumerate([c for c in sc.channels if c.on][:2]):
        m = sc.measure(ch.signal, now)
        unit = SIGNALS[ch.signal].unit
        idx = sc.channels.index(ch)
        if m.get("n"):
            txt = (
                f"CH{idx + 1} {ch.signal}: mean {_fmt(m['mean'], unit)}  "
                f"min {_fmt(m['min'])}  max {_fmt(m['max'])}  "
                f"pk-pk {_fmt(m['pk-pk'])}  σ {_fmt(m['std'])}  "
                f"p99 {_fmt(m['p99'])}  n {m['n']}"
            )
            stt = sc.statistics(ch.signal, "max")
            if stt["count"]:
                txt += (
                    f"   | max over {stt['count']} acq: μ {_fmt(stt['mean'])} "
                    f"σ {_fmt(stt['std'])}"
                )
        else:
            txt = f"CH{idx + 1} {ch.signal}: no samples in the window"
        cv.put(my + ci, 1, txt[: w - 2], COLORS[idx])
    ms = sc.mask_summary()
    if ms:

        def lim(x):
            return "" if x is None else f"{x:g}"

        txt = "  ".join(
            f"mask {sig} {lim(d['limits'][0])}..{lim(d['limits'][1])}: "
            f"{d['violations']} fail / {d['of']}"
            for sig, d in ms.items()
        )
        cv.put(
            my + 2,
            1,
            txt[: w - 2],
            "red" if any(d["violations"] for d in ms.values()) else "green",
        )

    _render_softkeys(cv, w, h)


def _render_softkeys(cv, w: int, h: int) -> None:
    x = 0
    for k in SOFTKEYS:
        if x + len(k) + 3 > w:
            break
        cv.put(h - 1, x, f"[{k}]", "dim")
        x += len(k) + 3


FFT_DB_SPAN = 50.0  # the display's dynamic range below the peak


def _render_fft(cv, st, g, now, y0, x0, gw, gh, top) -> None:
    """The selected channel in the frequency domain: Lomb-Scargle over the screen's
    time window (arrivals are irregular, so a plain FFT would be the wrong
    transform); x from 0 to the mean Nyquist rate, y in dB below the peak."""
    sc: Scope = st["scope"]
    ch = sc.channels[st.get("sel_ch", 0)]
    ci = st.get("sel_ch", 0)
    cv.put(top, 0, " " * (cv.w - 36))
    cv.put(
        top,
        0,
        f" FFT CH{ci + 1} {ch.signal}  Lomb-Scargle (irregular arrivals)  "
        f"window {_per_div(sc.s_per_div, 's')} x {HDIV}  {FFT_DB_SPAN:g} dB range",
        "cyan",
    )
    res = sc.periodogram(ch.signal, now, nfreq=gw * 2)
    if res is None:
        cv.put(y0 + 2, 3, "fewer than 8 samples in the window", "amber")
        return
    f, p = res
    pdb = 10 * np.log10(np.maximum(p, 1e-300))
    top_db = float(pdb.max())
    sub = gh * 4
    cells: dict = {}
    fmax = float(f[-1])
    prev = None
    for sx in range(gw * 2):
        i = min(len(f) - 1, int(sx / (gw * 2) * len(f)))
        v = (pdb[i] - (top_db - FFT_DB_SPAN)) / FFT_DB_SPAN
        ry = int(max(0.0, min(0.999, v)) * sub)
        lo, hi = (ry, ry) if prev is None else (min(ry, prev), max(ry, prev))
        prev = ry
        for sy in range(lo, hi + 1):
            cy, cx = gh - 1 - sy // 4, sx // 2
            cells[(cy, cx)] = cells.get((cy, cx), 0) | _DOT[(sx % 2, 3 - sy % 4)]
    for (cy, cx), bits in cells.items():
        glyph = "*" if g.get("ascii") else chr(BRAILLE + bits)
        cv.put(y0 + 1 + cy, x0 + 1 + cx, glyph, COLORS[ci])
    k = int(np.argmax(p))
    pk = float(f[k])
    msg = (
        f"peak {pk:.3g} Hz (period {1 / pk:.3g} s), {pdb[k] - np.median(pdb):+.1f} dB "
        f"over the median   |   0 .. {fmax:.3g} Hz"
    )
    cv.put(y0 + gh + 2, 1, msg[: cv.w - 2], COLORS[ci])


# ----------------------------------------------------------------------- keys
def key(st: dict, k: str, now: float) -> str:
    """Apply one front-panel key (a name such as 'space', 'up', '1', 'a') to the scope
    in ``st``; returns a short message for the status area ('' when none)."""
    sc: Scope = st["scope"]
    sel = st.setdefault("sel_ch", 0)
    ch = sc.channels[sel]
    tg = sc.trigger
    if k == "space":
        sc.run_stop()
        return "RUN" if sc.running else "STOP"
    if k == "s":
        sc.single()
        return "single: armed"
    if k == "f":
        sc.force(now)
        return "trigger forced"
    if k == "a":
        sc.autoset(now)
        return "autoset"
    if k in "1234" and len(k) == 1:
        i = int(k) - 1
        if st["sel_ch"] == i:
            sc.channels[i].on = not sc.channels[i].on
        else:
            st["sel_ch"], sc.channels[i].on = i, True
        return f"CH{i + 1} {'on' if sc.channels[i].on else 'off'}"
    if k in ("up", "down"):
        ch.scale = step(ch.scale, up=(k == "down"))  # down = zoom out, as the knob
        return f"CH{sel + 1} {ch.scale:g}/div"
    if k in ("[", "]"):
        ch.position += 0.5 if k == "]" else -0.5
        return f"CH{sel + 1} position {ch.position:+.1f} div"
    if k in ("left", "right"):
        if st.get("history") is not None and sc.segments:
            n = len(sc.segments)
            st["history"] = (st["history"] + (1 if k == "right" else -1)) % n
            sc.record = list(sc.segments)[st["history"]]
            return ""
        sc.s_per_div = step(sc.s_per_div, up=(k == "right"))
        return f"H {sc.s_per_div:g} s/div"
    if k in (",", "."):
        sc.h_position += sc.s_per_div if k == "." else -sc.s_per_div
        return f"H position {sc.h_position:+g} s"
    if k == "c":
        names = list(SIGNALS)
        ch.signal = names[(names.index(ch.signal) + 1) % len(names)]
        return f"CH{sel + 1} = {ch.signal}"
    if k == "t":
        tg.source, tg.kind = ch.signal, "edge" if tg.kind == "logic" else tg.kind
        return f"trigger source {ch.signal}"
    if k in ("+", "-"):
        src = next((c for c in sc.channels if c.signal == tg.source), ch)
        tg.level += (0.1 if k == "+" else -0.1) * src.scale
        return f"trigger level {tg.level:g}"
    if k == "/":
        order = ["rising", "falling", "either"]
        tg.slope = order[(order.index(tg.slope) + 1) % 3]
        return f"slope {tg.slope}"
    if k == "m":
        order = ["auto", "normal", "single"]
        tg.mode = order[(order.index(tg.mode) + 1) % 3]
        if tg.mode == "single":
            sc.single()
        return f"mode {tg.mode}"
    if k == "p":
        order = ["peak", "sample", "average"]
        sc.acquire = order[(order.index(sc.acquire) + 1) % 3]
        return f"acquisition {sc.acquire}"
    if k == "d":
        sc.decay = {0.85: 1.0, 1.0: 0.0}.get(sc.decay, 0.85)
        sc.persist = {}
        return "persistence " + {0.85: "on", 1.0: "infinite", 0.0: "off"}[sc.decay]
    if k == "x":
        sc.persist = {}
        return "persistence cleared"
    if k == "F":
        st["fft"] = not st.get("fft")
        return "FFT " + ("on" if st["fft"] else "off")
    if k == "h":
        if st.get("history") is None and sc.segments:
            st["history"] = len(sc.segments) - 1
            sc.record = sc.segments[-1]
            sc.running = False
            sc.status = "stop"
            return "history: Left/Right step, Enter inspects, h leaves"
        st["history"] = None
        return "history closed"
    return ""

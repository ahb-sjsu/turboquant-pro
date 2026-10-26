# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""The console drawn as vectors (``tqp console --style vector``): the same grid,
the same panels and the same numbers as the character display, in the manner of
an air-traffic-control screen. Dark field, thin phosphor strokes, and each trace
named by a data block on a leader line to the point it labels.

:func:`draw` is pure (a matplotlib Figure and the UI state in, the figure
drawn), so it is tested and saved as SVG without a display; :func:`run` is the
interactive window around it. The graphs (2 throughput, 3 pipeline, 7 scope, 8
spectrum) are drawn from the instruments' data as lines. The text panels
(1 system, 4 readscope, 5 index, 6 query stream), the zoomed fabric and the
overlays are the character display's own panels set in type, so the two styles
cannot disagree about a number.
"""

from __future__ import annotations

import time

from . import tui
from .scope import COLORS as SCOPE_COLORS
from .scope import SIGNALS, VDIV
from .spectrum_view import COLORS as SPEC_COLORS
from .spectrum_view import LABEL, MEANS

BG = "#02080a"
PANEL_BG = "#040d10"
EDGE = "#1d4a3a"
FOCUS = "#4fd1ff"
GRID = "#113326"
FONT = "DejaVu Sans Mono"
INK = {  # the character display's colour roles, as phosphor
    None: "#b8f5c8",
    "cyan": "#4fd1ff",
    "teal": "#3fe0c5",
    "purple": "#c792ff",
    "magenta": "#ff7ae0",
    "green": "#39ff88",
    "amber": "#ffbf3c",
    "yellow": "#f5f06b",
    "red": "#ff5c5c",
    "dim": "#5f8f74",
    "bold": "#e8ffe8",
    "grid": "#123824",
    "sel": "#4fd1ff",
}
TEXT_PT = 8.0  # type size of the text panels
CHAR_W = 0.6  # monospace advance, in ems


def ink(role) -> str:
    if role in INK:
        return INK[role]
    base = str(role).rsplit("_", 1)[0]
    return INK.get(base, INK[None])


# ------------------------------------------------------------------ layout
# (row, col) spans on a 4 x 6 grid; heights favour the two instruments.
LAYOUT = {
    1: ((0, 1), (0, 2)),
    2: ((0, 1), (2, 4)),
    3: ((0, 1), (4, 6)),
    7: ((1, 2), (0, 3)),
    8: ((1, 2), (3, 6)),
    4: ((2, 3), (0, 2)),
    5: ((2, 3), (2, 4)),
    9: ((2, 3), (4, 6)),
    6: ((3, 4), (0, 6)),
}
HEIGHTS = (1.0, 2.1, 1.2, 1.05)
TITLES = {
    1: "1 system",
    2: "2 throughput / latency",
    3: "3 pipeline  ms/query",
    4: "4 readscope  observer / certificate / provenance",
    5: "5 index",
    6: "6 query stream",
    7: "7 scope  query signals in time",
    8: "spectrum  what the observer reads, per direction",
    9: "9 NATS fabric",
}


def draw(fig, st: dict) -> None:
    """Draw the whole screen for ``st`` on ``fig`` (cleared first)."""
    fig.clear()
    fig.set_facecolor(BG)
    zoom = tui.zoom_of(st)
    if st.get("overlay") in ("help", "inspect"):
        _as_type(fig, st, tui.frame(st, *_cells(fig, fig.add_axes((0, 0, 1, 1)))))
        return
    if zoom in ("scope", "spectrum"):
        ax = fig.add_axes((0.05, 0.08, 0.92, 0.84))
        _panel(ax, 7 if zoom == "scope" else 8, st, zoomed=True)
        _header(fig, st, zoomed=True)
        return
    if zoom == "fabric":
        _as_type(fig, st, tui.frame(st, *_cells(fig, fig.add_axes((0, 0, 1, 1)))))
        return
    gs = fig.add_gridspec(
        4,
        6,
        height_ratios=HEIGHTS,
        left=0.035,
        right=0.985,
        top=0.955,
        bottom=0.03,
        hspace=0.32,
        wspace=0.42,
    )
    for n, ((r0, r1), (c0, c1)) in LAYOUT.items():
        _panel(fig.add_subplot(gs[r0:r1, c0:c1]), n, st)
    _header(fig, st)


def _header(fig, st: dict, zoomed: bool = False) -> None:
    cv = tui.Canvas(200, 1)
    tui._header(cv, st, st.get("snap") or {})
    line = cv.text()[0].rstrip()
    hint = "z or Esc: back to the grid  i notes  q quit" if zoomed else ""
    fig.text(
        0.01,
        0.985,
        line.split("q quit")[0].rstrip(),
        color=INK["bold"],
        family=FONT,
        fontsize=9,
        va="top",
    )
    fig.text(
        0.99,
        0.985,
        hint or "q quit  ? keys  z zoom  i notes",
        color=INK["dim"],
        family=FONT,
        fontsize=9,
        va="top",
        ha="right",
    )
    if st.get("message"):
        fig.text(
            0.5,
            0.005,
            st["message"],
            color=INK["amber"],
            family=FONT,
            fontsize=9,
            ha="center",
            va="bottom",
        )


def _frame(ax, n: int, st: dict, title: str | None = None) -> None:
    focus = st.get("focus") == n
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_color(FOCUS if focus else EDGE)
        sp.set_linewidth(1.2 if focus else 0.8)
    ax.tick_params(colors=INK["dim"], labelsize=7, length=2, width=0.5)
    ax.set_title(
        title or TITLES[n],
        loc="left",
        color=FOCUS if focus else INK["bold"],
        family=FONT,
        fontsize=8.5,
        pad=3,
    )


def _panel(ax, n: int, st: dict, zoomed: bool = False) -> None:
    _frame(ax, n, st)
    if n == 2:
        _throughput(ax, st)
    elif n == 3:
        _pipeline(ax, st)
    elif n == 7:
        _scope(ax, st, zoomed)
    elif n == 8:
        _spectrum(ax, st, zoomed)
    else:
        _text_panel(ax, n, st)


# ------------------------------------------------------------------ type
def _cells(fig, ax) -> tuple:
    """How many monospace cells of TEXT_PT fit in ``ax``: (columns, rows)."""
    bb = ax.get_position()
    w_pt = bb.width * fig.get_figwidth() * 72
    h_pt = bb.height * fig.get_figheight() * 72
    return max(4, int(w_pt / (TEXT_PT * CHAR_W))), max(2, int(h_pt / (TEXT_PT * 1.3)))


def _set(ax, cv, top: int = 0, left: int = 0, rows=None, cols=None) -> None:
    """A Canvas (or part of it) as type in ``ax``, one text run per colour."""
    rows = rows if rows is not None else range(top, cv.h)
    cols = cols if cols is not None else cv.w - left
    nrow = len(rows)
    for i, y in enumerate(rows):
        row = cv.cells[y][left : left + cols]
        x = 0
        while x < len(row):
            col = row[x][1]
            j = x
            while j < len(row) and row[j][1] == col:
                j += 1
            chunk = "".join(c for c, _ in row[x:j])
            if chunk.strip():
                ax.text(
                    x / cols,
                    1 - (i + 0.5) / nrow,
                    chunk,
                    color=ink(col),
                    family=FONT,
                    fontsize=TEXT_PT,
                    va="center",
                    ha="left",
                    transform=ax.transAxes,
                    fontweight="bold" if col in ("bold", "sel") else "normal",
                )
            x = j
    ax.set_xticks([])
    ax.set_yticks([])


def _as_type(fig, st: dict, cv) -> None:
    ax = fig.axes[0]
    ax.set_facecolor(BG)
    ax.set_axis_off()
    _set(ax, cv)


def _text_panel(ax, n: int, st: dict) -> None:
    """The character display's panel ``n``, drawn into a Canvas the size of
    ``ax`` and set in type (borders dropped: the axes frame is the border)."""
    fig = ax.figure
    w, h = _cells(fig, ax)
    cv = tui.Canvas(w + 2, h + 2)
    g, snap = tui.UNICODE, st.get("snap") or {}
    if n == 1:
        tui._p_system(cv, st, snap, 0, 0, h + 2, w + 2, g)
    elif n == 4:
        tui._p_readscope(cv, st, 0, 0, h + 2, w + 2, g)
    elif n == 5:
        tui._p_index(cv, st, snap, 0, 0, h + 2, w + 2, g)
    elif n == 6:
        tui._p_queries(cv, st, 0, 0, h + 2, w + 2, g)
    elif n == 9:
        tui._p_nats(cv, st, 0, 0, h + 2, w + 2, g)
    _set(ax, cv, rows=range(1, h + 1), left=1, cols=w)


# ------------------------------------------------------------------ graphs
def _block(ax, text, xy, color, rank: int = 0, notes: bool = True) -> None:
    """An ATC-style data block: the label off to one side, a thin leader line
    back to the point it names. It goes toward the open side of the panel (away
    from the nearer edges) and ``rank`` staggers blocks that share a panel."""
    if not notes:
        return
    ax.autoscale_view()
    fx, fy = ax.transLimits.transform(xy)
    sx = -1 if fx > 0.5 else 1
    sy = -1 if fy > 0.5 else 1
    ax.annotate(
        text,
        xy=xy,
        xytext=(sx * (28 + 14 * rank), sy * (12 + 16 * rank)),
        textcoords="offset points",
        color=color,
        family=FONT,
        fontsize=7.5,
        ha="left" if sx > 0 else "right",
        va="bottom" if sy > 0 else "top",
        arrowprops={"arrowstyle": "-", "color": color, "lw": 0.6, "shrinkA": 0},
        bbox={"facecolor": PANEL_BG, "edgecolor": "none", "alpha": 0.85, "pad": 1},
        annotation_clip=False,
    )


def _legend(ax, handles=None, labels=None) -> None:
    """A legend of the traces with their units, in the panel's corner."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    leg = ax.legend(
        handles,
        labels,
        loc="upper right",
        fontsize=7,
        frameon=True,
        facecolor=PANEL_BG,
        edgecolor=EDGE,
        labelcolor="linecolor",
        prop={"family": FONT, "size": 7},
    )
    leg.get_frame().set_alpha(0.9)


def _style_axes(ax) -> None:
    ax.grid(True, color=GRID, lw=0.5, ls=(0, (1, 3)))
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_family(FONT)


def _series(hist) -> tuple:
    vals = list(hist)
    xs = [i - len(vals) + 1 for i in range(len(vals))]  # seconds ago, newest 0
    pts = [(x, v) for x, v in zip(xs, vals) if v is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def _throughput(ax, st: dict) -> None:
    notes = st.get("annotate", True)
    qx, qy = _series(st.get("qps_hist", []))
    px, py = _series(st.get("p95_hist", []))
    ax.plot(qx, qy, color=INK["cyan"], lw=1.0)
    ax.set_ylabel("q/s", color=INK["cyan"], fontsize=7, family=FONT)
    ax2 = ax.twinx()
    ax2.plot(px, py, color=INK["purple"], lw=1.0)
    ax2.set_ylabel("p95 ms", color=INK["purple"], fontsize=7, family=FONT)
    ax2.tick_params(colors=INK["dim"], labelsize=7, length=2, width=0.5)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax.set_xlim(-239, 0)
    ax.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    _style_axes(ax)
    if notes:
        ax.set_xlabel(
            "seconds ago (last 4 min, newest right)",
            color=INK["dim"],
            fontsize=7,
            family=FONT,
        )
    if qx:
        _block(
            ax,
            f"QPS {tui.fmt(qy[-1], 1)}",
            (qx[-1], qy[-1]),
            INK["cyan"],
            notes=notes,
        )
    if px:
        _block(
            ax2,
            f"p95 {tui.fmt(py[-1], 2)} ms",
            (px[-1], py[-1]),
            INK["purple"],
            rank=1,
            notes=notes,
        )
    if not (qx or px):
        ax.text(
            0.5,
            0.5,
            "collecting",
            color=INK["dim"],
            family=FONT,
            fontsize=8,
            ha="center",
            transform=ax.transAxes,
        )


def _pipeline(ax, st: dict) -> None:
    r = tui._readings(st.get("snap") or {})
    names = ("encode", "scan", "rerank")
    vals = [r.get(f"search.stage_ms.{s}", {}).get("value") for s in names]
    ys = list(range(len(names)))[::-1]
    ax.barh(
        ys,
        [v or 0 for v in vals],
        height=0.45,
        color="none",
        edgecolor=INK["teal"],
        lw=1.0,
        hatch="////",
    )
    ax.set_yticks([])
    for y, name in zip(ys, names):
        ax.text(
            0,
            y + 0.28,
            f" {name}",
            color=INK[None],
            family=FONT,
            fontsize=7.5,
            va="bottom",
            ha="left",
        )
    mx = max([v for v in vals if v is not None] or [1.0])
    ax.set_xlim(0, mx * 1.35)
    ax.set_ylim(-0.6, len(names) - 0.1)
    for y, v in zip(ys, vals):
        ax.text(
            mx * 1.33,
            y,
            tui.fmt(v, 3),
            color=INK[None],
            family=FONT,
            fontsize=7.5,
            ha="right",
            va="center",
        )
    _style_axes(ax)
    wl = (st.get("snap") or {}).get("workload") or {}
    k, rr = wl.get("k", "-"), wl.get("rerank", 0)
    note = (
        "no index attached"
        if not wl
        else f"top-{k} from {k * rr} reranked" if rr else f"top-{k}, approximate"
    )
    ax.set_xlabel(
        f"ms per query, mean   {note}", color=INK["dim"], fontsize=7, family=FONT
    )


def _scope(ax, st: dict, zoomed: bool) -> None:
    notes = st.get("annotate", True)
    sc = st.get("scope")
    if sc is None:
        ax.text(
            0.5,
            0.5,
            "instrument not running",
            color=INK["dim"],
            transform=ax.transAxes,
            ha="center",
            family=FONT,
        )
        return
    now = st.get("now") or time.time()
    t0, t1 = sc.window(now)
    n = 600 if zoomed else 300
    ax.set_xlim(t0 - t1, 0)
    ax.set_ylim(0, VDIV)
    ax.set_xticks([(t0 - t1) * (1 - i / 10) for i in range(11)])
    ax.set_xticklabels(
        [f"{(t0 - t1) * (1 - i / 10):.0f}" if i % 2 == 0 else "" for i in range(11)]
    )
    on = [(i, c) for i, c in enumerate(sc.channels) if c.on]
    sel = st.get("sel_ch", 0)
    ci0, ch0 = next(((i, c) for i, c in on if i == sel), on[0] if on else (0, None))
    if ch0 is not None:  # the selected channel's calibration on the y axis
        ax.set_yticks(
            range(VDIV + 1),
            [
                f"{(k - VDIV / 2 - ch0.position) * ch0.scale:.4g}"
                for k in range(VDIV + 1)
            ],
        )
        ax.tick_params(axis="y", colors=ink(SCOPE_COLORS[ci0]))
    else:
        ax.set_yticks(range(VDIV + 1), [""] * (VDIV + 1))
    _style_axes(ax)
    status = "RUN" if sc.running else "STOP"
    ax.text(
        0.0,
        0.995,
        f" {status} {sc.status}  {sc.s_per_div:g} s/div  "
        f"trigger {sc.trigger.source} {sc.trigger.slope} {sc.trigger.level:g}  "
        f"[{sc.acquire}]",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color=INK["green"] if sc.running else INK["red"],
        family=FONT,
        fontsize=7.5,
    )
    for ci, ch in enumerate(sc.channels):
        if not ch.on:
            continue
        col = ink(SCOPE_COLORS[ci])
        cols = sc.columns(ch.signal, n, now)
        xs, lo, hi, mean = [], [], [], []
        for i, cc in enumerate(cols):
            if cc is None:
                continue
            xs.append((t0 - t1) + (i + 0.5) / n * (t1 - t0))
            lo.append(ch.to_div(cc[0]))
            hi.append(ch.to_div(cc[1]))
            mean.append(ch.to_div(cc[2]))
        g0 = ch.to_div(0.0)
        ax.annotate(
            str(ci + 1),
            xy=(0, g0),
            xycoords=("axes fraction", "data"),
            xytext=(-8, 0),
            textcoords="offset points",
            color=col,
            fontsize=7,
            family=FONT,
            va="center",
            ha="right",
            annotation_clip=False,
        )
        if not xs:
            continue
        ax.fill_between(xs, lo, hi, color=col, alpha=0.18, lw=0, step="mid")
        u = SIGNALS[ch.signal].unit or "ratio"
        ax.plot(
            xs,
            mean,
            color=col,
            lw=0.9,
            drawstyle="steps-mid",
            label=f"{ci + 1} {ch.signal}  {ch.scale:.4g} {u}/div",
        )
        spec = SIGNALS[ch.signal]
        last = cols[max(i for i, c in enumerate(cols) if c is not None)]
        label = f"{ci + 1} {ch.signal} {tui.fmt(last[1], 2)} {spec.unit}".rstrip()
        if zoomed:
            label += f"\n  {spec.description}; {ch.scale:g} {spec.unit or 'ratio'}/div"
        _block(
            ax,
            label,
            (xs[-1], min(max(hi[-1], 0.2), VDIV - 0.2)),
            col,
            rank=ci,
            notes=notes,
        )
    for ch in sc.channels:
        ref = sc.references.get(ch.signal)
        if ch.on and ref is not None:
            ax.axhline(ch.to_div(ref[0]), color=INK["purple"], lw=0.7, ls="--")
            ax.text(
                0,
                ch.to_div(ref[0]),
                f" {ref[1]}",
                color=INK["purple"],
                fontsize=7,
                family=FONT,
                va="bottom",
                ha="right",
            )
    _legend(ax)
    rec = sc.record
    if rec is not None and rec.trigger_t is not None and (t0, t1) == (rec.t0, rec.t1):
        ax.axvline(rec.trigger_t - t1, color=INK["cyan"], lw=0.6, ls=":")
        _block(
            ax,
            "trigger point",
            (rec.trigger_t - t1, VDIV),
            INK["cyan"],
            notes=notes,
        )
    if notes:
        ax.set_xlabel(
            f"seconds, newest at the right edge; {sc.s_per_div:g} s/div",
            color=INK["dim"],
            fontsize=7,
            family=FONT,
        )
        ax.set_ylabel(
            (
                f"CH{ci0 + 1} {ch0.signal} ({SIGNALS[ch0.signal].unit or 'ratio'}); "
                "other channels: legend scale, zero at their number"
                if ch0 is not None
                else "divisions"
            ),
            color=INK["dim"],
            fontsize=7,
            family=FONT,
        )


def _spectrum(ax, st: dict, zoomed: bool) -> None:
    notes = st.get("annotate", True)
    an = st.get("analyzer")
    if an is None or an.last is None:
        why = st.get("spectrum_reason") or "waiting for the first sweep"
        ax.text(
            0.5,
            0.5,
            why,
            color=INK["amber"],
            family=FONT,
            fontsize=8,
            ha="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return
    a, b = an.span()
    bottom = an.ref_db - an.db_div * VDIV
    ax.set_xlim(a, max(b - 1, a + 1))
    ax.set_ylim(bottom, an.ref_db)
    ax.set_yticks([bottom + an.db_div * i for i in range(VDIV + 1)])
    _style_axes(ax)
    s = an.last
    ax.text(
        0.0,
        0.995,
        f" {'RUN' if an.running else 'STOP'}  sweeps {an.sweeps}  "
        f"ref {an.ref_db:g} dB  {an.db_div:g} dB/div  [{an.detector}]",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color=INK["purple"],
        family=FONT,
        fontsize=7.5,
    )
    ax2 = None
    xs = list(range(a, b))
    for ti, tr in enumerate(an.traces):
        if tr.data is None or tr.mode == "blank":
            continue
        col = ink(SPEC_COLORS[ti])
        ys = tr.data[a:b]
        target = ax
        if tr.mode == "delta":  # a difference, not a level: its own axis
            if ax2 is None:
                ax2 = ax.twinx()
                half = an.db_div * VDIV / 2
                ax2.set_ylim(-half, half)
                ax2.axhline(0, color=INK["purple"], lw=0.5)
                ax2.set_ylabel("delta dB", color=INK["purple"], fontsize=7)
                ax2.tick_params(colors=INK["dim"], labelsize=7)
            target = ax2
        unit = "dB rel. reference" if tr.mode == "delta" else "dB"
        target.plot(
            xs,
            ys,
            color=col,
            lw=0.9,
            label=f"T{ti + 1} {LABEL.get(tr.source, tr.source)} ({unit}) [{tr.mode}]",
        )
        k = int(ys.argmax()) if ys.size else 0
        label = f"T{ti + 1} {LABEL.get(tr.source, tr.source)} [{tr.mode}]"
        if zoomed:
            label += f"\n  {MEANS.get(tr.source, '')}"
        _block(
            target,
            label,
            (xs[k], float(ys[k])),
            col,
            rank=ti,
            notes=notes,
        )
    lim = an.limit_db()
    if lim is not None:
        ax.axhline(lim, color=INK["red"], lw=0.7, ls="--", label="water level (dB)")
    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    _legend(ax, handles, labels)
    if lim is not None:
        _block(
            ax,
            "water level: above it a direction is worth bits",
            (b - 1, lim),
            INK["red"],
            notes=notes,
        )
    ro = [f"eff rank {s.effective_rank:.1f}/{s.sens.size}"]
    if s.predicted_total is not None:
        ro.append(f"predicted D {s.predicted_total:.3g}")
    if s.realised_total is not None:
        ro.append(f"realised D {s.realised_total:.3g}")
    ax.text(
        0.005,
        0.01,
        "   ".join(ro),
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        color=INK[None],
        family=FONT,
        fontsize=7.5,
    )
    if notes:
        ax.set_xlabel(
            "eigendirection of the observer's read operator, strongest first;"
            " y: power, dB",
            color=INK["dim"],
            fontsize=7,
            family=FONT,
        )


# ------------------------------------------------------------------ window
NON_INTERACTIVE = {"agg", "pdf", "ps", "svg", "pgf", "cairo", "template"}


def run(srv, setup: dict | None = None) -> None:  # pragma: no cover - needs a display
    """The vector console in a window: the state updates and redraws twice a
    second; the keys are the character display's."""
    import matplotlib

    for k in [k for k in matplotlib.rcParams if k.startswith("keymap.")]:
        matplotlib.rcParams[k] = []  # the console's keys, not matplotlib's
    import matplotlib.pyplot as plt

    if plt.get_backend().lower() in NON_INTERACTIVE:
        raise RuntimeError(
            f"the vector display needs a graphical session; matplotlib's backend "
            f"here is {plt.get_backend()!r} (no display). Run it where a display "
            "is available (or ssh -X), or use the character display"
        )
    st = tui.new_state(srv, setup)
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    try:
        fig.canvas.manager.set_window_title("TurboQuant Pro console (vector)")
    except AttributeError:
        pass

    def on_key(ev):
        key = ev.key or ""
        if key == "q":
            plt.close(fig)
            return
        st["message"] = ""
        if key == "i":
            st["annotate"] = not st.get("annotate", True)
        elif key == "?":
            st["overlay"] = None if st["overlay"] == "help" else "help"
        elif key == "escape":
            if st["overlay"] is None and st["zoom"] is not None:
                st["zoom"] = None
            st["overlay"], st["replay"] = None, None
        elif key == "z" and st["overlay"] is None:
            if st["zoom"] is not None:
                st["zoom"] = None
            elif st["focus"] in tui.ZOOMABLE:
                st["zoom"] = tui.ZOOMABLE[st["focus"]]
            else:
                st["message"] = "z opens panels 7 (scope), 8 (spectrum), 9 (NATS)"
        elif key == "tab":
            st["focus"] = st["focus"] % len(tui.PANELS) + 1
        elif key in "123456789" and len(key) == 1:
            st["focus"] = int(key)
        elif key == "p":
            st["paused"] = not st["paused"]
        elif key in ("down", "j"):
            st["sel"] = min(st["sel"] + 1, max(len(st["traces"]) - 1, 0))
        elif key in ("up", "k"):
            st["sel"] = max(st["sel"] - 1, 0)
        elif key == "enter" and st["traces"]:
            visible = list(reversed(st["traces"]))
            st["inspected"], st["overlay"] = visible[st["sel"]], "inspect"
        elif key == "P":
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            path = f"tqp-console-{stamp}.svg"
            try:
                fig.savefig(path, facecolor=BG)
                txt = tui.snapshot_txt(tui.frame(dict(st, message=""), 160, 48))
                st["message"] = f"snapshot written: {path}; {txt.split(': ', 1)[-1]}"
            except OSError as e:
                st["message"] = f"snapshot failed: {e}"
        elif key == "r" and st.get("inspected"):
            st["replay"] = srv.replay(st["inspected"]["id"])
        redraw()

    def redraw():
        draw(fig, st)
        fig.canvas.draw_idle()

    def tick():
        tui.update(st, srv, time.time())
        redraw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    timer = fig.canvas.new_timer(interval=500)
    timer.add_callback(tick)
    timer.start()
    tick()
    plt.show()

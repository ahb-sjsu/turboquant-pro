"""Render the console screens this branch changed, as SVG screenshots.

Two screens changed. The Overview's ReadScope panel now shows whether the loaded
certificate still applies: this renders the terminal UI's real frame
(``tui.frame``, the same function curses draws) for a demo session with a
certificate, over the data it was issued on (VALID) and over shifted data
(STALE). ``tqp fabric`` is new: with ``--nats URL`` it also renders
``fabric_view.frame`` from three live polls of that NATS monitoring port, with
IP addresses redacted. Each screen is written as an SVG in the terminal's
colours and as plain text.

    python docs/console-v2/screens.py [OUT_DIR] [--nats http://127.0.0.1:8222]

Run it where compute is allowed (Atlas), not on a laptop: it builds a 20k-row
demo index and runs the workload for a few seconds.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from html import escape
from pathlib import Path

import numpy as np

from turboquant_pro.cli import main as tqp
from turboquant_pro.console import tui
from turboquant_pro.console.server import ConsoleServer, demo_index

W, H = 160, 48
CELL_W, CELL_H = 8.4, 17  # px per terminal cell at 14 px monospace
BG = "#0b1020"
COLOURS = {  # the curses roles (tui._loop) in the requirements' Appendix B palette
    None: "#d1d5db",
    "cyan": "#22d3ee",
    "teal": "#2dd4bf",
    "purple": "#c084fc",
    "magenta": "#e879f9",
    "green": "#4ade80",
    "amber": "#fbbf24",
    "yellow": "#fde047",
    "red": "#f87171",
    "dim": "#6b7280",
    "bold": "#f3f4f6",
    "grid": "#1e3a8a",
}


def svg(canvas, title: str) -> str:
    """The canvas as SVG: one <text> per run of same-coloured cells."""
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W * CELL_W:.0f}" '
        f'height="{H * CELL_H + 8:.0f}" font-family="DejaVu Sans Mono, Menlo, '
        f'Consolas, monospace" font-size="14">',
        f"<title>{escape(title)}</title>",
        f'<rect width="100%" height="100%" fill="{BG}"/>',
    ]
    for y, row in enumerate(canvas.cells):
        x = 0
        while x < len(row):
            col = row[x][1]
            run = x
            while run < len(row) and row[run][1] == col:
                run += 1
            text = "".join(ch for ch, _ in row[x:run])
            if text.strip():
                base = (col or "").split("_")[0] or None
                fill = COLOURS.get(base, COLOURS[None])
                weight = ' font-weight="bold"' if col and "bold" in col else ""
                out.append(
                    f'<text x="{x * CELL_W:.1f}" y="{(y + 1) * CELL_H:.0f}" '
                    f'fill="{fill}"{weight} xml:space="preserve">{escape(text)}</text>'
                )
            x = run
    out.append("</svg>")
    return "\n".join(out)


def session(cert: dict, originals: np.ndarray, index, Q, source):
    s = ConsoleServer(
        index, Q, qps=40, k=10, rerank=4, originals=originals, certificate=cert,
        source=source, http=False,
    ).start()  # fmt: skip
    deadline = time.time() + 15
    while len(s.tracer.traces()) < 60 and time.time() < deadline:
        time.sleep(0.1)
    return s


def frame(s) -> object:
    st = {
        "snap": s.snapshot(),
        "traces": s.tracer.traces(200),
        "readscope": s.readscope(),
        "qps_hist": [20.0, 31.0, 38.5, 40.2, 39.6, 40.1],
        "p95_hist": [1.4, 1.2, 1.3, 1.1, 1.2],
        "sel": 0,
        "focus": 4,
        "paused": False,
        "overlay": None,
        "inspected": None,
        "replay": None,
        "message": "",
    }
    return tui.frame(st, W, H)


def main(out_dir: str = "docs/console-v2/screens") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index, Q, X, source, codec = demo_index()
    with tempfile.TemporaryDirectory() as tmp:
        o, r, c = (str(Path(tmp) / n) for n in ("o.npy", "r.npy", "cert.json"))
        sample = X[:4000]
        np.save(o, sample)
        np.save(r, codec(sample))
        tqp(["certify", "--original", o, "--reconstructed", r, "--validity",
             "--out", c])  # fmt: skip
        cert = json.loads(Path(c).read_text(encoding="utf-8"))
    shifted = X + np.float32(0.5)  # every row moved off the certified moments
    for name, data in (("valid", X), ("stale", shifted)):
        s = session(cert, data, index, Q, source)
        try:
            cv = frame(s)
            status = s.validity()["status"]
        finally:
            s.stop()
        title = f"tqp console --demo --certificate: validity {status}"
        (out / f"overview-validity-{name}.svg").write_text(
            svg(cv, title), encoding="utf-8"
        )
        (out / f"overview-validity-{name}.txt").write_text(
            "\n".join(cv.text()) + "\n", encoding="utf-8"
        )
        print(f"{name}: {status}")


def fabric(out_dir: str, url: str, polls: int = 3, interval: float = 2.0) -> None:
    """``tqp fabric`` on a live server, redacted, after ``polls`` polls."""
    from turboquant_pro.console import fabric_view
    from turboquant_pro.console.fabric import FabricMonitor

    mon, hist = FabricMonitor(url, redact=True), fabric_view.History()
    doc = None
    for i in range(polls):
        if i:
            time.sleep(interval)
        doc = mon.poll()
        hist.add(doc)
    cv = fabric_view.frame(doc, hist, W, H)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fabric.svg").write_text(svg(cv, "tqp fabric (IPs redacted)"), "utf-8")
    (out / "fabric.txt").write_text("\n".join(cv.text()) + "\n", "utf-8")
    print(f"fabric: reachable={doc['reachable']} leafs={len(doc['leafs'])}")


if __name__ == "__main__":
    args = sys.argv[1:]
    nats = None
    if "--nats" in args:
        i = args.index("--nats")
        nats = args[i + 1]
        del args[i : i + 2]
    target = args[0] if args else "docs/console-v2/screens"
    main(target)
    if nats:
        fabric(target, nats)

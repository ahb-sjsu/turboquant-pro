# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""The console engine: the session and its instruments in a process of their own.

``tqp console`` in a terminal runs two processes (the local agent and UI client of
the requirements' section 8):

* **this engine** runs the index workload, the traces, the spectrum sweeps, the
  NATS polling and the scope and analyzer state, and serves the terminal client
  the view model (:mod:`.viewmodel`) as JSON lines over a Unix socket in a private
  directory (mode 0700). It holds no terminal, so it can be paused (by a
  thermal guard, say) without touching anyone's screen: the client then shows
  its data as stale.
* **the terminal client** (``go/tqp-console``) owns the terminal and draws.

The engine lives as long as its lifeline: it exits when its standard input
reaches end of file, which happens when the client (which holds the other end)
exits for any reason, including being killed.

Protocol (one JSON object per line each way):

``{"op": "hello"}`` -> keys, titles, initial zoom, engine pid.
``{"op": "view", "scope": [gw, gh], "scope_zoom": bool, "spectrum": [gw, gh, wf],
"spectrum_zoom": bool, "fabric": [w, h]}`` -> the data for one frame.
``{"op": "key", "zoom": "scope" | "spectrum", "name": ...}`` -> instrument key.
``{"op": "action", "name": "pause" | "export" | "setup" | "hold" | "release"}``.
``{"op": "inspect", "id": ..., "replay": bool, "w": W, "h": H}`` -> overlay.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import socketserver
import sys
import threading
import time

from . import scope_view, spectrum_view, tui, viewmodel


def _cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


class Engine:
    """The instruments and their session, safe to call from the socket threads."""

    def __init__(self, srv, setup: dict | None = None, export_dir: str = "."):
        self.srv = srv
        self.export_dir = export_dir
        self.st = tui.new_state(srv, setup)
        self.lock = threading.Lock()
        self.halt = threading.Event()
        self.cpu_percent: float | None = None
        self._cpu_mark = (time.monotonic(), _cpu_seconds())
        self.blas = "1 thread"

    # ---- the instruments' clock (4 Hz), separate from any client
    def run(self) -> None:
        while not self.halt.is_set():
            with self.lock:
                tui.update(self.st, self.srv, time.time())
            t, c = time.monotonic(), _cpu_seconds()
            if t - self._cpu_mark[0] >= 2.0:
                self.cpu_percent = (
                    100.0 * (c - self._cpu_mark[1]) / (t - self._cpu_mark[0])
                )
                self._cpu_mark = (t, c)
            self.halt.wait(0.25)

    # ---- requests
    def handle(self, req: dict) -> dict:
        op = req.get("op")
        with self.lock:
            if op == "hello":
                out = viewmodel.hello(self.srv.sources())
                out["zoom"] = self.st.get("zoom")
                out["engine"] = {"pid": os.getpid()}
                return out
            if op == "view":
                return self._view(req)
            if op == "key":
                return self._key(req)
            if op == "action":
                return self._action(req)
            if op == "inspect":
                return self._inspect(req)
        return {"error": f"unknown op {op!r}"}

    def _view(self, req: dict) -> dict:
        st, now = self.st, time.time()
        st["now"] = now
        out = {
            "header": viewmodel.header(st),
            "engine": {
                "pid": os.getpid(),
                "cpu": self.cpu_percent,
                "blas": self.blas,
                "paused": bool(st.get("paused")),
            },
            "message": st.pop("engine_message", ""),
        }
        pages = [p["name"] for p in viewmodel.pages(self.srv.sources())]
        page = req.get("page") or (pages[0] if pages else "index")
        if page == "nats":
            ist = self._inst_state(page)
            if req.get("scope"):
                gw, gh = req["scope"]
                out["p7"] = viewmodel.scope(
                    ist, now, gw, gh, bool(req.get("scope_zoom"))
                )
            out["p9"] = viewmodel.nats(st)  # the panel; z opens the instrument
            if req.get("fabric"):
                w, h = req["fabric"]
                out["fabric"] = viewmodel.fabric_screen(st, w, h)
            return out
        if page in ("machine", "dht"):  # pages of PanelViews, one shape
            from . import dht_view, machine_view

            view = machine_view if page == "machine" else dht_view
            out["panels"] = view.panels(st)
            return out
        if page == "index" and req.get("grid", True):
            out.update(
                p1=viewmodel.system(st),
                p2=viewmodel.throughput(st),
                p3=viewmodel.pipeline(st),
                p4=viewmodel.readscope(st),
                p5=viewmodel.index(st),
                p6=viewmodel.queries(st),
                strip=viewmodel.strip(st),
            )
            if self.srv.fabric is not None:
                out["p9"] = viewmodel.nats(st)
        if req.get("scope"):
            gw, gh = req["scope"]
            out["p7"] = viewmodel.scope(st, now, gw, gh, bool(req.get("scope_zoom")))
        if req.get("spectrum"):
            gw, gh, wf = (list(req["spectrum"]) + [0])[:3]
            out["p8"] = viewmodel.spectrum(
                st, gw, gh, bool(req.get("spectrum_zoom")), wf
            )
        if req.get("fabric"):
            w, h = req["fabric"]
            out["fabric"] = viewmodel.fabric_screen(st, w, h)
        return out

    # ---- the instrument on a page: the index grid's scope, or the NATS page's
    INST_KEYS = ("scope", "sel_ch", "fft", "history")

    def _inst_state(self, page) -> dict:
        """The state the scope code reads for ``page``: the session's own for the
        index grid; for the NATS page, the same with that page's scope and its
        instrument keys in place."""
        fi = self.st.get("fabric_inst")
        if page != "nats" or fi is None:
            return self.st
        return {**self.st, **{k: fi[k] for k in self.INST_KEYS if k in fi}}

    def _inst_commit(self, page, ist: dict) -> None:
        """Keep what a key changed in the NATS page's instrument state."""
        fi = self.st.get("fabric_inst")
        if page == "nats" and fi is not None:
            for k in self.INST_KEYS:
                if k in ist:
                    fi[k] = ist[k]

    def _key(self, req: dict) -> dict:
        st, name = self.st, req.get("name", "")
        page = req.get("page") or "index"
        if page == "nats" and req.get("zoom") == "scope":
            if name == "enter":
                return {"message": "a NATS sample is a poll: no query to inspect"}
            ist = self._inst_state(page)
            msg = scope_view.key(ist, name, time.time())
            self._inst_commit(page, ist)
            return {"message": msg}
        if req.get("zoom") == "spectrum":
            return {"message": spectrum_view.key(st, name)}
        if req.get("zoom") == "scope":
            if name == "enter":  # inspect the query that fired the trigger
                rec = st["scope"].record
                if rec and rec.trigger_id and self.srv.tracer.get(rec.trigger_id):
                    return {"inspect": rec.trigger_id}
                return {"message": "no trigger query to inspect (or it was evicted)"}
            return {"message": scope_view.key(st, name, time.time())}
        return {"message": ""}

    def _action(self, req: dict) -> dict:
        st, name = self.st, req.get("name")
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        if name == "pause":
            st["paused"] = not st["paused"]
            return {"message": "display paused" if st["paused"] else "live"}
        if name in ("hold", "release"):  # the client is suspended / back
            wl = self.srv.workload
            if wl is not None:
                (wl._paused.set if name == "hold" else wl._paused.clear)()
            return {"message": ""}
        if name == "export":
            path = os.path.join(self.export_dir, f"tqp-console-{stamp}.json")
            try:
                from .server import dumps

                with open(path, "w", encoding="utf-8") as f:
                    f.write(dumps(self.srv.export()))
                return {"message": f"exported {path}"}
            except OSError as e:
                return {"message": f"export failed: {e}"}
        if name == "setup":
            if req.get("page") == "nats":
                return {
                    "message": "setups (S) save the index page's scope and analyzer"
                }
            from . import setup as SU

            path = os.path.join(self.export_dir, f"tqp-console-{stamp}.tqs")
            zoom = req.get("zoom")
            view = zoom if zoom in ("scope", "spectrum") else "overview"
            try:
                SU.save(
                    path,
                    SU.to_dict(
                        st["scope"], st["analyzer"], view, tui._observer_sha(self.srv)
                    ),
                )
                return {"message": f"setup saved: {path}"}
            except (OSError, SU.SetupError) as e:
                return {"message": f"setup not saved: {e}"}
        return {"message": f"unknown action {name!r}"}

    def _inspect(self, req: dict) -> dict:
        st = self.st
        t = self.srv.tracer.get(req.get("id", ""))
        if t is None:
            return {"message": "that query is no longer in the trace ring"}
        st["inspected"] = t
        st["replay"] = self.srv.replay(t["id"]) if req.get("replay") else None
        return viewmodel.inspect_sheet(
            st, int(req.get("w", 120)), int(req.get("h", 40))
        )


def serve(engine: Engine, path: str) -> socketserver.BaseServer:
    """Serve ``engine`` on the Unix socket ``path`` (one JSON object per line)."""

    class H(socketserver.StreamRequestHandler):
        def handle(self):
            for line in self.rfile:
                try:
                    req = json.loads(line)
                    resp = engine.handle(req)
                except Exception as e:  # a bad request answers, it never kills
                    resp = {"error": f"{type(e).__name__}: {e}"}
                data = viewmodel_dumps(resp)
                try:
                    self.wfile.write(data.encode() + b"\n")
                    self.wfile.flush()
                except OSError:
                    return

    class S(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

    old = os.umask(0o177)  # the socket itself: owner read/write only
    try:
        srv = S(path, H)
    finally:
        os.umask(old)
    threading.Thread(target=srv.serve_forever, daemon=True, name="engine-sock").start()
    return srv


def viewmodel_dumps(obj) -> str:
    from .server import dumps

    return dumps(obj)


def main(argv: list[str] | None = None) -> int:
    """``python -m turboquant_pro.console.engine --socket PATH --argv JSON``:
    build the session ``tqp console`` would, serve it, and live until standard
    input closes (the client's end of the lifeline) or a SIGTERM."""
    p = argparse.ArgumentParser(prog="tqp-console-engine")
    p.add_argument("--socket", required=True)
    p.add_argument("--argv", required=True, help="the `tqp console` arguments, JSON")
    p.add_argument("--export-dir", default=".")
    p.add_argument("--log", help="the engine's own log file, removed on a clean exit")
    a = p.parse_args(argv)

    from ..cli import build_console_session, build_parser
    from .threads import limit_blas_threads

    cargs = build_parser().parse_args(["console", *json.loads(a.argv)])
    # measured on Atlas: the console's kernels (one query, one sweep) are
    # milliseconds long and run fastest on one BLAS thread; more only spin
    limit_blas_threads(cargs.threads)
    try:
        os.nice(10)  # a monitor's engine yields to interactive work
    except OSError:
        pass
    halt = threading.Event()
    for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(s, lambda *_: halt.set())

    session, setup = build_console_session(cargs, http=False)
    engine = Engine(session, setup, a.export_dir)
    engine.blas = f"{cargs.threads} BLAS thread{'s' if cargs.threads != 1 else ''}"
    threading.Thread(target=engine.run, daemon=True, name="engine-clock").start()
    sock = serve(engine, a.socket)

    def lifeline():
        try:
            while sys.stdin.buffer.read(4096):
                pass
        except (OSError, ValueError):
            pass
        halt.set()

    threading.Thread(target=lifeline, daemon=True, name="engine-lifeline").start()
    sys.stdout.write("ready\n")
    sys.stdout.flush()
    try:
        while not halt.wait(0.5):
            pass
    finally:
        engine.halt.set()
        sock.shutdown()
        session.stop()
        _remove_own_files(a.socket, a.log)
    return 0


def _remove_own_files(socket_path: str, log: str | None) -> None:
    """Remove the socket (and the log, on a clean exit) and then their directory
    only if that leaves it empty: never a tree removal from a path argument."""
    for f in (socket_path, log):
        if f:
            try:
                os.unlink(f)
            except OSError:
                pass
    try:
        os.rmdir(os.path.dirname(socket_path))  # fails, harmlessly, unless empty
    except OSError:
        pass


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

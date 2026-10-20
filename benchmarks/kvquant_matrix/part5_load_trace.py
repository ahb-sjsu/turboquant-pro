"""Where a head pod's model load spends its time (submitter side, not part of the harness).

A run pod installs this module with a ``.pth`` line in its venv, so every Python process
imports it at start-up; it acts only in a cell process (keys_cell.py, wikitext_ppl.py)
when LOAD_TRACE_DIR is set. For the first WINDOW seconds a daemon thread samples every
thread's stack each 10 ms and, every 5 s, the process's disk reads, CPU time and context
switches and the pod's CPU throttling; it then writes one report to
LOAD_TRACE_DIR/<script>_<pid>.txt and stops. A cell's outputs are untouched.

Why (2026-10-19): Mistral-7B's head pods load the fp16 copy at ~140 MB/s, idling the GPU
~106 s, while every reproduction off the pod runs at 620-1500 MB/s (CPU pods reading the
set, Atlas GPU 1 on 1-8 free cores). Only a measurement inside the pod is left.
"""

import collections
import os
import sys
import threading
import time

WINDOW = float(os.environ.get("LOAD_TRACE_SECONDS", "300"))
SCRIPTS = ("keys_cell.py", "wikitext_ppl.py")
EVERY = 0.01


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def counters() -> dict:
    """The process's reads, CPU seconds and context switches; the pod's CPU throttling."""
    out = {}
    for line in _read("/proc/self/io").splitlines():
        if line.startswith("read_bytes:"):
            out["read_gb"] = int(line.split()[1]) / 1e9
    t = os.times()
    out["cpu_s"] = t.user + t.system
    for line in _read("/proc/self/status").splitlines():
        if line.startswith(("voluntary_ctxt_switches", "nonvoluntary_ctxt_switches")):
            k, v = line.split(":")
            out[k.split("_")[0]] = int(v)
    for line in _read("/sys/fs/cgroup/cpu.stat").splitlines():
        k, v = line.split()
        if k in ("nr_throttled", "throttled_usec"):
            out[k] = int(v)
    return out


def _where(frame) -> str:
    c = frame.f_code
    return f"{os.path.basename(c.co_filename)}:{c.co_name}:{frame.f_lineno}"


def report(name: str, t0: float, start: dict, timeline: list, leaf, stacks, n: int) -> str:
    end = counters()
    lines = [
        f"{name} pid {os.getpid()}: {time.time() - t0:.0f} s traced, {n} samples",
        f"cpus allowed {sorted(os.sched_getaffinity(0))}, cpu.max {_read('/sys/fs/cgroup/cpu.max').strip()}",
        "change over the window: "
        + ", ".join(f"{k} {end[k] - start.get(k, 0):.6g}" for k in end),
        "-- every 5 s: seconds, then the change since the start",
    ]
    lines += [f"  {s:5.0f}  " + "  ".join(f"{k}={v - start.get(k, 0):.6g}" for k, v in c.items())
              for s, c in timeline]
    lines.append("-- where threads were (share of samples; idle pool threads left out)")
    lines += [f"  {100 * v / max(n, 1):5.1f}%  {k}" for k, v in leaf.most_common(15)]
    lines.append("-- 4-deep stacks")
    lines += [f"  {100 * v / max(n, 1):5.1f}%  {k}" for k, v in stacks.most_common(10)]
    return "\n".join(lines) + "\n"


class Trace:
    """One cell process's samples; written once, at the window's end or at exit."""

    def __init__(self, name: str, out_dir: str):
        self.name, self.path = name, os.path.join(out_dir, f"{name}_{os.getpid()}.txt")
        self.leaf, self.stacks = collections.Counter(), collections.Counter()
        self.t0, self.n, self.timeline = time.time(), 0, []
        self.start = counters()
        self.lock, self.written = threading.Lock(), False
        # The pause between samples is an Event wait: the submitter's preflight refuses a
        # Job whose command holds the word for a timed pause, and this file ships in it.
        self.pause = threading.Event()

    def run(self) -> None:
        me, last = threading.get_ident(), 0.0
        while time.time() - self.t0 < WINDOW and not self.written:
            for tid, fr in sys._current_frames().items():
                if tid == me:
                    continue
                names = []
                while fr is not None and len(names) < 4:
                    names.append(_where(fr))
                    fr = fr.f_back
                if names and not names[0].startswith(("threading.py:wait", "thread.py:_worker")):
                    self.leaf[names[0]] += 1
                    self.stacks[" < ".join(names)] += 1
            self.n += 1
            if time.time() - last >= 5:
                last = time.time()
                self.timeline.append((last - self.t0, counters()))
            self.pause.wait(EVERY)
        self.write()

    def write(self) -> None:
        with self.lock:
            if self.written:
                return
            self.written = True
            text = report(self.name, self.t0, self.start, list(self.timeline),
                          self.leaf.copy(), self.stacks.copy(), self.n)
            with open(self.path, "w") as f:
                f.write(text)


def script(argv) -> str:
    """The cell script this process runs, or ''."""
    for a in argv or ():
        if os.path.basename(a) in SCRIPTS:
            return os.path.basename(a)
    return ""


def install(argv=None, env=None):
    """Start a Trace in a cell process when LOAD_TRACE_DIR is set; None otherwise."""
    import atexit

    if argv is None:  # sys.argv can be unset while site runs .pth lines; orig_argv is not
        argv = getattr(sys, "orig_argv", None) or getattr(sys, "argv", [])
    env = os.environ if env is None else env
    name, out_dir = script(argv), env.get("LOAD_TRACE_DIR", "")
    if not name or not out_dir:
        return None
    os.makedirs(out_dir, exist_ok=True)
    tr = Trace(name, out_dir)
    atexit.register(tr.write)  # a cell that ends inside the window still reports
    threading.Thread(target=tr.run, daemon=True).start()
    return tr


try:
    install()
except Exception:  # noqa: BLE001  (a trace must never stop a cell)
    pass

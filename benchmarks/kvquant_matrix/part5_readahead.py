"""Read a cell's fp16 copy ahead of its model load (submitter side, not part of the harness).

A run pod installs this module with a ``.pth`` line in its venv, so every Python process
imports it at start-up; it acts only in a cell process (keys_cell.py, wikitext_ppl.py)
when READAHEAD_DIR names the fp16 copy the cell loads. Four daemon threads then read the
copy's tensors in the loader's order (transformers' dot_natural_key over the index), in
16 MiB pieces into reused buffers, so the loader's page faults find the bytes in the page
cache. They stay at most READAHEAD_GB (default 1) ahead of the load, measured as the GPU
memory held by tensors (torch.cuda.memory_allocated: the load goes straight to the GPU,
and transformers' warm-up reservation is freed before the load starts), so the pod's
memory holds at most that much read but not yet loaded. One report goes to
LOAD_TRACE_DIR/readahead_<script>_<pid>.txt when the copy is read. A cell's outputs are
untouched: the bytes read are discarded.

Why (2026-10-20): the head pod's load trace (part5_load_trace.py) showed Mistral-7B's
14.85 GB load reading the volumes at ~120 MB/s for ~125 s at 7% CPU, its threads waiting
in transformers' per-tensor copy to the GPU for pages of the memory-mapped file. A CPU
pod reading the same set with four readers at once gets 620+ MB/s.
"""

import json
import os
import re
import struct
import sys
import threading
import time

SCRIPTS = ("keys_cell.py", "wikitext_ppl.py")
INDEX = "model.safetensors.index.json"
PIECE = 16 << 20
READERS = 4
POLL = 0.05


def natural_key(name: str):
    """transformers' load order (core_model_loading.dot_natural_key), as part5_preload."""
    return [
        (0, int(p), "") if p.isdigit() else (1, 0, p) for p in re.split(r"\.", name)
    ]


def pieces(model_dir: str) -> list:
    """[(path, begin, end)] of the copy's tensor bytes in the loader's order, cut into
    pieces of at most PIECE bytes."""
    with open(os.path.join(model_dir, INDEX)) as f:
        files = sorted(set(json.load(f)["weight_map"].values()))
    ts = []
    for name in files:
        path = os.path.join(model_dir, name)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(n))
        for t, e in head.items():
            if t != "__metadata__":
                b, en = e["data_offsets"]
                ts.append((t, path, 8 + n + b, 8 + n + en))
    out = []
    for _, path, b, e in sorted(ts, key=lambda t: natural_key(t[0])):
        out += [(path, s, min(s + PIECE, e)) for s in range(b, e, PIECE)]
    return out


def gpu_loaded() -> int:
    """Bytes held by tensors on the GPU, 0 before the cell has started CUDA."""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_initialized():
            return int(torch.cuda.memory_allocated(0))
    except Exception:  # noqa: BLE001
        pass
    return 0


class ReadAhead:
    """One cell process's read-ahead; reports once, when the copy is read or at exit."""

    def __init__(self, name, model_dir, out_dir, window, loaded=gpu_loaded):
        self.name, self.out_dir, self.window, self.loaded = (
            name,
            out_dir,
            window,
            loaded,
        )
        self.todo = pieces(model_dir)
        self.total = sum(e - b for _, b, e in self.todo)
        self.t0, self.next, self.read, self.waits = time.time(), 0, 0, 0
        self.lock, self.pause = threading.Lock(), threading.Event()
        self.done, self.timeline, self.written = 0, [], False
        self.fds = {}

    def take(self):
        """The next piece, once it is inside the window of the load; None at the end."""
        while True:
            with self.lock:
                if self.next >= len(self.todo):
                    return None
                p = self.todo[self.next]
                if self.read - self.loaded() < self.window:
                    self.next += 1
                    self.read += p[2] - p[1]
                    return p
                self.waits += 1
            self.pause.wait(POLL)

    def fd(self, path):
        with self.lock:
            if path not in self.fds:
                self.fds[path] = os.open(path, os.O_RDONLY)
            return self.fds[path]

    def reader(self):
        buf = bytearray(PIECE)
        view = memoryview(buf)
        try:
            while True:
                p = self.take()
                if p is None:
                    break
                path, b, e = p
                got, fd = 0, self.fd(path)
                while got < e - b:
                    n = os.preadv(fd, [view[: e - b - got]], b + got)
                    if n <= 0:
                        break
                    got += n
        except (
            Exception
        ):  # noqa: BLE001  (a failed read leaves the load to fault it in)
            pass
        with self.lock:
            self.done += 1
            last = self.done == READERS
        if last:
            for fd in self.fds.values():
                os.close(fd)
            self.write()

    def watch(self):
        """Every 5 s until the copy is read: seconds, GB read ahead, GB on the GPU."""
        while not self.written:
            self.timeline.append(
                (time.time() - self.t0, self.read / 1e9, self.loaded() / 1e9)
            )
            self.pause.wait(5)

    def start(self):
        for _ in range(READERS):
            threading.Thread(target=self.reader, daemon=True).start()
        threading.Thread(target=self.watch, daemon=True).start()
        return self

    def write(self):
        with self.lock:
            if self.written:
                return
            self.written = True
        s = time.time() - self.t0
        lines = [
            f"{self.name} pid {os.getpid()}: read {self.read / 1e9:.2f} of "
            f"{self.total / 1e9:.2f} GB in {s:.0f} s ({self.read / 1e6 / max(s, 1e-9):.0f} "
            f"MB/s), window {self.window / 1e9:.2f} GB, {self.waits} waits on the load",
            "-- every 5 s: seconds, GB read ahead, GB on the GPU",
        ]
        lines += [f"  {t:5.0f}  {r:6.2f}  {g:6.2f}" for t, r, g in self.timeline]
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            path = os.path.join(
                self.out_dir, f"readahead_{self.name}_{os.getpid()}.txt"
            )
            with open(path, "w") as f:
                f.write("\n".join(lines) + "\n")


def script(argv) -> str:
    """The cell script this process runs, or ''."""
    for a in argv or ():
        if os.path.basename(a) in SCRIPTS:
            return os.path.basename(a)
    return ""


def install(argv=None, env=None):
    """Start a ReadAhead in a cell process when READAHEAD_DIR holds a copy; None otherwise."""
    import atexit

    if (
        argv is None
    ):  # sys.argv can be unset while site runs .pth lines; orig_argv is not
        argv = getattr(sys, "orig_argv", None) or getattr(sys, "argv", [])
    env = os.environ if env is None else env
    name, model_dir = script(argv), env.get("READAHEAD_DIR", "")
    if not name or not model_dir or not os.path.exists(os.path.join(model_dir, INDEX)):
        return None
    window = int(float(env.get("READAHEAD_GB", "1")) * 1e9)
    ra = ReadAhead(name, model_dir, env.get("LOAD_TRACE_DIR", ""), window)
    atexit.register(ra.write)
    return ra.start()


try:
    install()
except Exception:  # noqa: BLE001  (a read-ahead must never stop a cell)
    pass

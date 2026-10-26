"""The host memory a job actually holds, recorded by the job itself.

NRP sizes a pod by what the kernel can take back from it: anonymous memory (``RssAnon``)
cannot be reclaimed, the page cache of the memory-mapped checkpoint (``RssFile``) can. A
job's peak RSS mixes the two, so ``HostMem`` samples both from ``/proc/self/status`` and
appends their peaks, with ``ru_maxrss``, to a JSON-lines file when the block ends. Off
Linux there is no ``/proc`` and the peaks are recorded as ``None``.
"""

from __future__ import annotations

import json
import threading
import time

FIELDS = {"RssAnon": "peak_anon_gib", "RssFile": "peak_file_gib"}


def _status() -> dict | None:
    try:
        with open("/proc/self/status") as f:
            return {
                ln.split(":")[0]: int(ln.split()[1])
                for ln in f
                if ln.split(":")[0] in FIELDS
            }
    except OSError:
        return None


class HostMem:
    def __init__(self, path: str, phase: str, interval: float = 0.02):
        self.path, self.phase, self.interval = path, phase, interval
        self.peak = {k: 0 for k in FIELDS} if _status() is not None else None
        self._stop = threading.Event()

    def _sample(self) -> None:
        while not self._stop.is_set():
            s = _status() or {}
            for k, v in s.items():
                self.peak[k] = max(self.peak[k], v)
            self._stop.wait(self.interval)

    def __enter__(self):
        self.t0 = time.time()
        if self.peak is not None:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self.peak is not None:
            self._stop.set()
            self._thread.join()
        rec = {"phase": self.phase, "seconds": round(time.time() - self.t0, 1)}
        for k, name in FIELDS.items():
            rec[name] = None if self.peak is None else round(self.peak[k] / 2**20, 3)
        try:
            import resource

            maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rec["maxrss_gib"] = round(maxrss / 2**20, 3)  # KiB on Linux
        except ImportError:
            rec["maxrss_gib"] = None
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return False

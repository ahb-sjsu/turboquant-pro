# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""Cap the BLAS thread pool of the running process.

The console is a monitor: its own workload (the replayed queries, the spectrum
sweeps) must not take the machine. numpy's OpenBLAS otherwise starts one thread
per CPU (48 on a two-socket workstation), and a 20 query/s demo then ran at
~700% CPU. numpy is already imported by the time a command runs, so the
OPENBLAS_NUM_THREADS environment variable is too late; the pool is resized
through the library's own setter instead (threadpoolctl when installed, else
OpenBLAS's C entry point found among the loaded libraries).
"""

from __future__ import annotations

import ctypes
import sys

_SETTERS = (
    "scipy_openblas_set_num_threads64_",
    "scipy_openblas_set_num_threads",
    "openblas_set_num_threads64_",
    "openblas_set_num_threads",
)


def _loaded_openblas() -> list:
    """Paths of the OpenBLAS libraries mapped into this process (Linux)."""
    if not sys.platform.startswith("linux"):
        return []
    try:
        with open("/proc/self/maps", encoding="utf-8") as f:
            paths = {line.split()[-1] for line in f if "openblas" in line.lower()}
    except OSError:
        return []
    return sorted(p for p in paths if p.startswith("/"))


def limit_blas_threads(n: int) -> str:
    """Set the BLAS pool to ``n`` threads; returns what was done, for a log line
    or a test ("threadpoolctl", "openblas <path>", or "unchanged: <why>")."""
    import numpy  # noqa: F401  (make sure the library is loaded)

    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=n, user_api="blas")
        return "threadpoolctl"
    except ImportError:
        pass
    done = []  # every copy: numpy, faiss and scipy each bundle their own
    for path in _loaded_openblas():
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for name in _SETTERS:
            fn = getattr(lib, name, None)
            if fn is not None:
                fn.argtypes = [ctypes.c_int]
                fn(int(n))
                done.append(path.rsplit("/", 1)[-1])
                break
    if done:
        return "openblas " + ", ".join(done)
    return "unchanged: no BLAS thread setter found"

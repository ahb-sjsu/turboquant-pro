# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""How an artifact was produced: the ``invocation`` block.

Every JSON document the ``tqp`` CLI emits carries one, so the document says how
to reproduce it rather than leaving a reader to guess (requirements BM-005; the
console's "reproduce with" is *recorded* when this block is present and only
*derived* otherwise)::

    "invocation": {
      "argv": ["tqp", "certify", "--original", "o.npy", ...],
      "cwd": "/data/run",
      "tool_version": "2.0.0",
      "git_commit": "8e252ec...",  # the turboquant_pro source; null if installed
      "git_dirty": false,          # uncommitted changes there; null if unknown
      "python": "3.12.3",
      "created_utc": "2026-09-26T08:00:00Z"
    }

``git_commit`` names the commit of the **turboquant_pro source that ran**, found
from this package's own directory, not of whatever repository the shell happens
to be in. An installed wheel has no repository, so it is ``null``: the version
then identifies the code. ``git_dirty`` is ``true`` when that source differs from
the commit, so a hash is never presented as the code that ran when it was not.

The block records paths (``cwd`` and any path in ``argv``); it records no vector
data and no environment variables.
"""

from __future__ import annotations

import functools
import os
import platform
import subprocess
import time
from pathlib import Path

__all__ = ["invocation", "source_commit"]

PROGRAM = "tqp"


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
    except Exception:  # noqa: BLE001 - no git, not a repository, or a timeout
        return None


@functools.lru_cache(maxsize=1)
def source_commit() -> tuple[str | None, bool | None]:
    """``(commit, dirty)`` of the git checkout this package runs from, or
    ``(None, None)`` when it is not one (an installed wheel)."""
    pkg = Path(__file__).resolve().parent
    top = _git(["rev-parse", "--show-toplevel"], pkg)
    if top is None or not (Path(top) / "turboquant_pro").resolve() == pkg:
        return None, None  # not the source tree this package was imported from
    commit = _git(["rev-parse", "HEAD"], pkg)
    if commit is None:
        return None, None
    status = _git(["status", "--porcelain", "--untracked-files=no", "--", "."], pkg)
    return commit, None if status is None else bool(status)


def invocation(argv: list[str] | None) -> dict:
    """The ``invocation`` block for a document produced by ``tqp argv...``."""
    from turboquant_pro import __version__

    commit, dirty = source_commit()
    return {
        "argv": [PROGRAM, *(str(a) for a in (argv or []))],
        "cwd": os.getcwd(),
        "tool_version": __version__,
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

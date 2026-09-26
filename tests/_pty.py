"""Driving a curses program through a pseudo-terminal in tests."""

from __future__ import annotations

import os
import select
import time


def quit_and_drain(master: int, proc, key: bytes = b"q", timeout: float = 20.0) -> int:
    """Send ``key`` and keep reading the pty until ``proc`` exits; return its code.

    A pty nobody reads fills, and a program that redraws keeps writing: once the
    buffer is full it blocks in ``write`` and never gets to read the key or to
    restore the terminal on the way out. So the reader must keep draining until the
    process is gone. An ``OSError`` on read means the child has closed its end.
    """
    os.write(master, key)
    deadline = time.time() + timeout
    while proc.poll() is None and time.time() < deadline:
        ready, _, _ = select.select([master], [], [], 0.1)
        if ready:
            try:
                os.read(master, 65536)
            except OSError:
                time.sleep(0.05)
    return proc.wait(timeout=max(0.1, deadline - time.time()))

# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""The model source: a live model's consumer telemetry, read from the readings that
``tqp model watch --readings FILE`` writes (one JSON reading per line).

The console never loads a model. A watcher runs the model under its monitor and
appends readings; this source tails the file, read-only, and keeps the newest window
it has seen in full. A window is taken as complete when a reading of a later window
arrives, or when the file has not grown for ``settle_s`` seconds (the watcher writes a
window's readings at once). A truncated or replaced file is read again from the start.
"""

from __future__ import annotations

import json
import os
import time

__all__ = ["ModelReadingsMonitor"]


class ModelReadingsMonitor:
    def __init__(self, path: str, settle_s: float = 2.0):
        self.path = path
        self.settle_s = float(settle_s)
        self._pos = 0
        self._ino = None
        self._partial = ""
        self._grew = 0.0
        self._windows: dict = {}  # (run, model, window) -> list of readings
        self._order: list = []
        self.bad_lines = 0

    def _read_new(self) -> None:
        try:
            st = os.stat(self.path)
        except OSError:
            return
        ino = (st.st_dev, st.st_ino)
        if ino != self._ino or st.st_size < self._pos:  # replaced or truncated
            self._ino, self._pos, self._partial = ino, 0, ""
            self._windows, self._order = {}, []
        if st.st_size == self._pos:
            return
        with open(self.path, encoding="utf-8") as f:
            f.seek(self._pos)
            chunk = f.read()
            self._pos = f.tell()
        self._grew = time.time()
        text = self._partial + chunk
        lines = text.split("\n")
        self._partial = lines.pop()  # an unfinished last line waits for its end
        for line in lines:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                lab = r["labels"]
                key = (lab.get("run"), lab.get("model"), int(lab["window"]))
            except (ValueError, KeyError, TypeError):
                self.bad_lines += 1
                continue
            if key not in self._windows:
                self._windows[key] = []
                self._order.append(key)
            self._windows[key].append(r)
        while len(self._order) > 8:  # keep a few windows, not the whole file
            self._windows.pop(self._order.pop(0), None)

    def poll(self) -> dict:
        """The newest complete window, per key head, with what the file holds."""
        self._read_new()
        now = time.time()
        doc = {
            "t": now,
            "path": self.path,
            "exists": os.path.exists(self.path),
            "windows_seen": len(self._order),
            "bad_lines": self.bad_lines,
            "window": None,
        }
        settled = now - self._grew >= self.settle_s
        done = self._order if settled else self._order[:-1]
        if not done:
            return doc
        key = done[-1]
        heads: dict = {}
        for r in self._windows[key]:
            lab = r["labels"]
            h = heads.setdefault(
                (int(lab["layer"]), int(lab["kv_head"])), {"values": {}, "why": {}}
            )
            h["values"][r["name"]] = r["value"]
            if r["value"] is None and r.get("unavailable_reason"):
                h["why"][r["name"]] = r["unavailable_reason"]
            if r.get("status"):
                h["status"], h["reason"] = r["status"], r.get("status_reason", "")
        first = self._windows[key][0]["labels"]
        doc["window"] = {
            "run": key[0],
            "model": key[1],
            "index": key[2],
            "tokens": first.get("tokens"),
            "consumer": first.get("consumer"),
            "as_of": max(r["as_of"] for r in self._windows[key]),
            "heads": [
                dict(layer=li, kv_head=h, **v) for (li, h), v in sorted(heads.items())
            ],
        }
        return doc

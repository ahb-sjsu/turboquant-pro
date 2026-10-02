# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Read a TQPW file (``tqp.packed_weights/1``) with numpy and the standard library.

A single-file, dependency-free reader written from ``docs/PACKED_WEIGHTS_SPEC.md``
alone: it shares no code with turboquant-pro, so a test that it decodes the golden
corpus is a test of the spec, not of the implementation. Copy it anywhere.

    meta, weights = read("weights.tqpw")   # weights: {name: float32 (out, in)}
"""

from __future__ import annotations

import json
import struct
import zlib

import numpy as np

_HEADER = struct.Struct("<4sHHI")  # magic, version, n_sections, reserved
_ENTRY = struct.Struct("<32sQQII")  # name, offset, length, crc32, flags
GROUP = 128


def _sections(blob: bytes) -> dict:
    magic, version, n, _ = _HEADER.unpack_from(blob, 0)
    if magic != b"TQPW" or version != 1:
        raise ValueError(f"not a TQPW v1 file ({magic!r}, version {version})")
    out = {}
    for i in range(n):
        name, off, length, crc, _ = _ENTRY.unpack_from(
            blob, _HEADER.size + i * _ENTRY.size
        )
        name = name.rstrip(b"\0").decode("utf-8")
        data = blob[off : off + length]
        if len(data) != length or zlib.crc32(data) & 0xFFFFFFFF != crc:
            raise ValueError(f"section {name!r} is corrupt or truncated")
        out[name] = data
    return out


def _codes(data: bytes, n: int, bits: int) -> np.ndarray:
    """Value j is stream bits [j*bits, (j+1)*bits), least significant first; stream
    bit p is bit p % 8 of byte p // 8."""
    stream = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")
    b = stream[: n * bits].reshape(n, bits).astype(np.uint16)
    return (b << np.arange(bits, dtype=np.uint16)).sum(1)


def read(path: str) -> tuple:
    """``(meta, {name: float32 weights (out, in)})``."""
    sec = _sections(open(path, "rb").read())
    meta = json.loads(sec["meta"])
    weights = {}
    for i, m in enumerate(meta["matrices"]):
        rows, cols = m["shape"]
        ng = cols // GROUP
        r = _codes(sec[f"c{i}"], rows * cols, m["bits"]).reshape(rows, ng, GROUP)
        grid = np.frombuffer(sec[f"g{i}"], dtype="<f2").reshape(rows, ng, 2)
        lo = grid[..., 0:1].astype(np.float32)
        step = grid[..., 1:2].astype(np.float32)
        weights[m["name"]] = (r.astype(np.float32) * step + lo).reshape(rows, cols)
    return meta, weights

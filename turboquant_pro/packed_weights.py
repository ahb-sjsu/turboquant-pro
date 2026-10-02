# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""TQPW: the stored form of a planned weight encoding (``tqp.packed_weights/1``).

A ``tqp.weight_plan/1`` counts ``numel * bits + (numel / 128) * 32`` stored bits per
matrix (:func:`turboquant_pro.weight_plan.stored_bits`). A TQPW file holds exactly that
payload: each matrix's codes packed at its own width by the canonical LSB-first stream
packer (:mod:`turboquant_pro.packed_codes`), and for every group of 128 input columns
of every row the grid that decodes them, a float16 minimum and a float16 step (32 bits).
Specified in ``docs/PACKED_WEIGHTS_SPEC.md``.

Decoding a code ``r`` of a group with grid ``(lo, step)`` gives ``r * step + lo`` in
float32. The product ``r * step`` is exact in float32 (at most 8 + 11 significant bits),
so the result is one rounding of the exact value, with or without a fused multiply-add,
and every conforming reader decodes the same bits.

The container is TQIX's (:mod:`turboquant_pro.index_file`) under the magic ``TQPW``:
a JSON ``meta`` section, then a codes section ``c<i>`` and a grid section ``g<i>`` per
matrix, each CRC32-checked. Corruption and truncation raise
:class:`~turboquant_pro.index_file.IndexCorruptionError`, never decode silently.

The grid is stored in float16 because the plan's byte count allows 32 bits per group.
The codec (:mod:`turboquant_pro.weight_codec`) computes it in float32, so a stored
matrix decodes to the codec's output up to the rounding of ``lo`` and ``step`` to
float16; :func:`grid_rounding` measures that difference. The codes are the codec's own.

numpy only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .index_file import IndexCorruptionError, read_container, write_container
from .packed_codes import pack_bits, packed_nbytes, unpack_bits

__all__ = [
    "FORMAT",
    "GROUP",
    "MAGIC",
    "VERSION",
    "IndexCorruptionError",
    "PackedMatrix",
    "decode",
    "grid_rounding",
    "pack_matrix",
    "read",
    "write",
]

FORMAT = "tqp.packed_weights/1"
MAGIC = b"TQPW"
VERSION = 1
GROUP = 128


@dataclass(frozen=True)
class PackedMatrix:
    """One matrix in stored form: ``codes``, the packed uint8 stream of its
    ``out * in`` codes in row-major order, and ``grid``, float16 of shape
    ``(out, in // 128, 2)`` holding ``[lo, step]`` per group."""

    name: str
    shape: tuple
    bits: int
    codes: np.ndarray
    grid: np.ndarray

    @property
    def payload_bits(self) -> int:
        """Stored bits: the codes and the grid, as the plan counts them."""
        return 8 * (self.codes.nbytes + self.grid.nbytes)


def pack_matrix(name: str, codes, lo, step, bits: int) -> PackedMatrix:
    """Stored form of integer ``codes`` (out, in) on the per-group grid ``lo``,
    ``step`` (each (out, in // 128), any float dtype; rounded to float16 here).

    Refuses a width outside 1..8, a code that does not fit it, a grid that is not
    finite in float16, and a step that underflows to 0 in a group whose codes are
    not all 0 (that group would decode to a constant)."""
    codes = np.asarray(codes)
    if codes.ndim != 2:
        raise ValueError(f"{name}: codes must be 2-D (out, in), got {codes.shape}")
    out, inp = codes.shape
    if inp % GROUP:
        raise ValueError(f"{name}: in-dimension {inp} is not a multiple of {GROUP}")
    if not 1 <= int(bits) <= 8:
        raise ValueError(f"{name}: bits must be in 1..8, got {bits}")
    ng = inp // GROUP
    lo = np.asarray(lo, dtype=np.float32).reshape(out, ng)
    step = np.asarray(step, dtype=np.float32).reshape(out, ng)
    if codes.size and (codes.min() < 0 or codes.max() >= 2 ** int(bits)):
        raise ValueError(f"{name}: a code does not fit in {bits} bits")
    with np.errstate(over="ignore"):
        grid = np.stack([lo, step], -1).astype(np.float16)
    if not np.isfinite(grid).all():
        raise ValueError(f"{name}: grid is not finite in float16")
    used = codes.reshape(out, ng, GROUP).max(-1) > 0
    if (used & (grid[..., 1] == 0)).any():
        raise ValueError(f"{name}: a step underflows to 0 in float16")
    return PackedMatrix(
        name, (out, inp), int(bits), pack_bits(codes.astype(np.uint8), bits), grid
    )


def decode(pm: PackedMatrix) -> np.ndarray:
    """The float32 weights (out, in): ``r * step + lo`` per group."""
    out, inp = pm.shape
    r = unpack_bits(pm.codes, out * inp, pm.bits).reshape(out, inp // GROUP, GROUP)
    g = pm.grid.astype(np.float32)
    w = r.astype(np.float32) * g[..., 1:2] + g[..., 0:1]
    return w.reshape(out, inp)


def grid_rounding(pm: PackedMatrix, reference: np.ndarray, step) -> dict:
    """How far the stored matrix decodes from ``reference`` (the codec's float32
    output), as the largest absolute difference and the largest in units of each
    group's float32 ``step`` (out, in // 128)."""
    out, inp = pm.shape
    d = np.abs(decode(pm) - np.asarray(reference, dtype=np.float32))
    st = np.asarray(step, dtype=np.float32).reshape(out, inp // GROUP, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(st > 0, d.reshape(out, inp // GROUP, GROUP) / st, 0.0)
    return {
        "max_abs": float(d.max(initial=0.0)),
        "max_steps": float(rel.max(initial=0.0)),
    }


def _meta_bytes(matrices: list, meta: dict) -> bytes:
    doc = {
        **meta,
        "format": FORMAT,
        "group": GROUP,
        "grid": "float16 [lo, step] per row and group of 128 input columns",
        "decode": "float32 r * step + lo",
        "matrices": [
            {"name": m.name, "shape": list(m.shape), "bits": m.bits} for m in matrices
        ],
    }
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write(path: str, matrices: list, meta: dict | None = None) -> int:
    """Write ``matrices`` (in order) and ``meta`` (extra JSON fields, e.g. the codec
    and the plan's hash) to ``path`` atomically; returns the file's size in bytes."""
    names = [m.name for m in matrices]
    if len(set(names)) != len(names):
        raise ValueError("duplicate matrix names")
    sections = [("meta", _meta_bytes(matrices, dict(meta or {})))]
    for i, m in enumerate(matrices):
        sections.append((f"c{i}", np.ascontiguousarray(m.codes).tobytes()))
        sections.append((f"g{i}", np.ascontiguousarray(m.grid, dtype="<f2").tobytes()))
    write_container(path, VERSION, sections, magic=MAGIC)
    return os.path.getsize(path)


def read(path: str) -> tuple:
    """``(meta, [PackedMatrix, ...])`` from a TQPW file, every section CRC-checked
    and every size checked against the shapes and widths ``meta`` declares."""
    version, sec = read_container(path, magic=MAGIC)
    if version != VERSION:
        raise IndexCorruptionError(f"TQPW version {version} is not supported")
    meta = json.loads(sec["meta"].decode("utf-8"))
    if meta.get("format") != FORMAT or meta.get("group") != GROUP:
        raise IndexCorruptionError(f"meta declares {meta.get('format')!r}")
    out = []
    for i, m in enumerate(meta["matrices"]):
        rows, cols = (int(x) for x in m["shape"])
        bits = int(m["bits"])
        codes = np.frombuffer(sec[f"c{i}"], dtype=np.uint8)
        if codes.size != packed_nbytes(rows * cols, bits):
            raise IndexCorruptionError(f"{m['name']}: codes are {codes.size} bytes")
        g = np.frombuffer(sec[f"g{i}"], dtype="<f2")
        if g.size != rows * (cols // GROUP) * 2:
            raise IndexCorruptionError(f"{m['name']}: grid is {g.size} values")
        grid = g.astype(np.float16).reshape(rows, cols // GROUP, 2)
        out.append(PackedMatrix(m["name"], (rows, cols), bits, codes, grid))
    return meta, out

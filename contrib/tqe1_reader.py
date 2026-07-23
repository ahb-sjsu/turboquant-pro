#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""tqe1_reader — a single-file, dependency-free reference reader for TQE1.

This file is the "safetensors-style" portability proof for the TQE1 record
format (``docs/FORMAT_SPEC.md``): it decodes any TQE1 v1/v2 record or batch
file using **only the Python standard library and numpy** — no turboquant-pro
import, no shared code. Copy this one file into your project (or vendor it)
and you can read TQE1 forever; it is validated byte-for-byte against the
in-tree implementation by the golden-corpus conformance suite
(``tests/golden/tqe1/`` + ``tests/test_tqe1_golden.py``).

Every constant below is normative and mirrors a section of FORMAT_SPEC.md;
section references are given inline. The decode contract:

    reconstruction = norm * (codebook(bits)[indices] @ R(seed, rotation))

with indices bit-unpacked LSB-first and R the seed-deterministic orthogonal
rotation. Records are self-describing: no out-of-band metadata is required.

CLI usage::

    python tqe1_reader.py file.tqe            # summary of each record
    python tqe1_reader.py file.tqe --out x.npy  # decode all records -> (n, dim)

Library usage::

    from tqe1_reader import read_records, decode
    for rec in read_records(open("file.tqe", "rb").read()):
        vec = decode(rec)
"""

from __future__ import annotations

import argparse
import math
import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"TQE1"
_HEADER_V1 = struct.Struct("<4sBBHIfI")  # SPEC "Record layout", v1 (20 bytes)
_HEADER_V2 = struct.Struct("<4sBBHIfBI")  # SPEC "Record layout", v2 (21 bytes)
_ROTATIONS = {0: "qr", 1: "hadamard"}  # SPEC v2 `rotation` byte
_VALID_BITS = (2, 3, 4)

# SPEC "Decode algorithm" step 2: fixed Lloyd-Max centroids for a unit-Gaussian
# coordinate; scaled by 1/sqrt(dim) at decode time.
_CODEBOOKS = {
    2: np.array([-1.510, -0.453, 0.453, 1.510]),
    3: np.array([-1.748, -1.050, -0.500, -0.069, 0.069, 0.500, 1.050, 1.748]),
    4: np.array(
        [
            -2.401,
            -1.844,
            -1.437,
            -1.099,
            -0.800,
            -0.524,
            -0.262,
            -0.066,
            0.066,
            0.262,
            0.524,
            0.800,
            1.099,
            1.437,
            1.844,
            2.401,
        ]
    ),
}


@dataclass
class TQERecord:
    """One parsed (not yet decoded) TQE1 record."""

    version: int
    bits: int
    dim: int
    seed: int
    norm: float
    rotation: str
    codes: bytes


# ----------------------------------------------------------------- parsing
def parse_record(buf: bytes, offset: int = 0) -> tuple[TQERecord, int]:
    """Parse one record at ``offset``; returns ``(record, next_offset)``.

    Raises ``ValueError`` on bad magic, unknown version/rotation/bits, or
    truncation — the SPEC "Conformance" error cases.
    """
    if len(buf) < offset + 20:
        raise ValueError("buffer too small for a TQE header")
    if buf[offset : offset + 4] != MAGIC:
        raise ValueError(f"bad magic {buf[offset:offset + 4]!r}; expected {MAGIC!r}")
    version = buf[offset + 4]
    if version == 1:
        _, _, bits, dim, seed, norm, codelen = _HEADER_V1.unpack_from(buf, offset)
        rotation, hsize = "qr", _HEADER_V1.size
    elif version == 2:
        if len(buf) < offset + _HEADER_V2.size:
            raise ValueError("buffer too small for a TQE v2 header")
        _, _, bits, dim, seed, norm, rot, codelen = _HEADER_V2.unpack_from(buf, offset)
        if rot not in _ROTATIONS:
            raise ValueError(f"unknown rotation code {rot}")
        rotation, hsize = _ROTATIONS[rot], _HEADER_V2.size
    else:
        raise ValueError(f"unsupported TQE format version {version}")
    if bits not in _VALID_BITS:
        raise ValueError(f"unsupported bits {bits}; expected one of {_VALID_BITS}")
    end = offset + hsize + codelen
    if len(buf) < end:
        raise ValueError("truncated TQE record (codes shorter than codelen)")
    rec = TQERecord(
        version=version,
        bits=bits,
        dim=dim,
        seed=seed,
        norm=norm,
        rotation=rotation,
        codes=buf[offset + hsize : end],
    )
    return rec, end


def read_records(buf: bytes) -> list[TQERecord]:
    """Parse a batch file: back-to-back records of either version."""
    out, off = [], 0
    while off < len(buf):
        rec, off = parse_record(buf, off)
        out.append(rec)
    return out


# ----------------------------------------------------------------- decoding
def _unpack_indices(codes: bytes, bits: int, dim: int) -> np.ndarray:
    """SPEC "Bit-packing": LSB-first indices, `dim` of them, `bits` each."""
    b = np.frombuffer(codes, dtype=np.uint8).astype(np.uint32)
    if bits == 2:
        vals = np.column_stack([(b >> s) & 0x3 for s in (0, 2, 4, 6)]).ravel()
    elif bits == 3:
        trip = b.reshape(-1, 3)
        w = trip[:, 0] | (trip[:, 1] << 8) | (trip[:, 2] << 16)
        vals = np.column_stack([(w >> s) & 0x7 for s in range(0, 24, 3)]).ravel()
    else:  # bits == 4
        vals = np.column_stack([b & 0xF, (b >> 4) & 0xF]).ravel()
    return vals[:dim].astype(np.uint8)


def _fwht(a: np.ndarray) -> np.ndarray:
    """Unnormalized Fast Walsh-Hadamard Transform along the last axis."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    d = a.shape[-1]
    out = a.reshape(-1, d).copy()
    n, h = out.shape[0], 1
    while h < d:
        out = out.reshape(n, d // (2 * h), 2, h)
        x, y = out[:, :, 0, :], out[:, :, 1, :]
        out = np.stack((x + y, x - y), axis=2).reshape(n, d)
        h *= 2
    return out.reshape(a.shape)


def _unrotate(y: np.ndarray, dim: int, seed: int, rotation: str) -> np.ndarray:
    """SPEC "Decode algorithm" step 3: apply R(seed)^T.

    The rotation is fully determined by ``(dim, seed, rotation)`` with numpy's
    ``default_rng`` — the draws below happen in exactly the writer's order.
    """
    rng = np.random.default_rng(seed)
    if rotation == "hadamard":
        sign = rng.choice([-1.0, 1.0], size=dim).astype(np.float32)
        scale = np.float32(1.0 / math.sqrt(dim))
        return _fwht(y) * scale * sign
    if dim <= 4096:  # full QR of a seeded float32 Gaussian matrix
        g = rng.standard_normal((dim, dim)).astype(np.float32)
        q, _ = np.linalg.qr(g)
        return y @ q
    # structured rotation for very large dim: sign flip + permutation
    sign = rng.choice([-1.0, 1.0], size=dim).astype(np.float32)
    perm = rng.permutation(dim)
    inv_perm = np.argsort(perm)
    return y[..., inv_perm] / sign


def decode(rec: TQERecord) -> np.ndarray:
    """SPEC "Decode algorithm" steps 1-4: record -> float32 vector."""
    indices = _unpack_indices(rec.codes, rec.bits, rec.dim)
    centroids = (_CODEBOOKS[rec.bits] / math.sqrt(rec.dim)).astype(np.float32)
    y_hat = centroids[indices]
    x_hat = _unrotate(y_hat, rec.dim, rec.seed, rec.rotation)
    return (x_hat * np.float32(rec.norm)).astype(np.float32)


def decode_file(path: str) -> np.ndarray:
    """Decode every record in a batch file -> array of shape (n, dim)."""
    with open(path, "rb") as f:
        recs = read_records(f.read())
    return np.stack([decode(r) for r in recs])


# ---------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="a .tqe record/batch file")
    ap.add_argument("--out", help="write decoded vectors to this .npy")
    a = ap.parse_args(argv)
    with open(a.path, "rb") as f:
        recs = read_records(f.read())
    for i, r in enumerate(recs):
        print(
            f"[{i}] v{r.version} bits={r.bits} dim={r.dim} seed={r.seed} "
            f"rotation={r.rotation} norm={r.norm:.6g} codes={len(r.codes)}B"
        )
    if a.out:
        np.save(a.out, np.stack([decode(r) for r in recs]))
        print(f"decoded {len(recs)} vectors -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

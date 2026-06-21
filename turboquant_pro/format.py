"""Versioned, self-describing on-disk/wire format for compressed embeddings.

A compressed embedding is only portable if it carries the parameters needed to
decode it. This module defines **TQE1**, a small self-describing container so a
reader can reconstruct a vector with no out-of-band metadata, and so the format is
stable across versions (a prerequisite for an industry-standard tool).

Layout (little-endian)::

    offset size field
    0      4    magic   b"TQE1"
    4      1    version uint8  (== 1)
    5      1    bits    uint8  (2, 3, or 4)
    6      2    dim     uint16 (quantized dimension d')
    8      4    seed    uint32 (rotation seed -> reproduces codebook + rotation)
    12     4    norm    float32 (per-vector L2 norm)
    16     4    codelen uint32 (length of packed code bytes)
    20     ..   codes   `codelen` bytes (bit-packed indices)

Total header = 20 bytes. ``pack_batch`` concatenates records length-prefixed.
Forward compatibility: readers must reject an unknown ``version`` and may ignore
trailing bytes after ``codes`` within a record.
"""

from __future__ import annotations

import struct

from .pgvector import CompressedEmbedding

MAGIC = b"TQE1"
VERSION = 1
_HEADER = struct.Struct("<4sBBHIfI")  # magic, version, bits, dim, seed, norm, codelen
HEADER_SIZE = _HEADER.size  # 20


def pack(ce: CompressedEmbedding, seed: int = 42) -> bytes:
    """Serialize a :class:`CompressedEmbedding` to a self-describing TQE1 record."""
    codes = bytes(ce.packed_bytes)
    return (
        _HEADER.pack(
            MAGIC,
            VERSION,
            int(ce.bits),
            int(ce.dim),
            int(seed) & 0xFFFFFFFF,
            float(ce.norm),
            len(codes),
        )
        + codes
    )


def unpack(buf: bytes) -> tuple[CompressedEmbedding, int]:
    """Parse one TQE1 record. Returns ``(CompressedEmbedding, seed)``.

    Raises ``ValueError`` on bad magic, unsupported version, or truncation.
    """
    if len(buf) < HEADER_SIZE:
        raise ValueError("buffer too small for a TQE1 header")
    magic, version, bits, dim, seed, norm, codelen = _HEADER.unpack(buf[:HEADER_SIZE])
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}; expected {MAGIC!r}")
    if version != VERSION:
        raise ValueError(f"unsupported TQE format version {version}")
    end = HEADER_SIZE + codelen
    if len(buf) < end:
        raise ValueError("truncated TQE1 record (codes shorter than codelen)")
    ce = CompressedEmbedding(
        packed_bytes=buf[HEADER_SIZE:end], norm=norm, dim=dim, bits=bits
    )
    return ce, seed


def record_size(buf: bytes, offset: int = 0) -> int:
    """Total byte length of the TQE1 record beginning at ``offset``."""
    if len(buf) < offset + HEADER_SIZE:
        raise ValueError("buffer too small for a TQE1 header")
    codelen = _HEADER.unpack(buf[offset : offset + HEADER_SIZE])[6]
    return HEADER_SIZE + codelen


def pack_batch(embeddings: list[CompressedEmbedding], seed: int = 42) -> bytes:
    """Serialize many embeddings as back-to-back TQE1 records."""
    return b"".join(pack(ce, seed) for ce in embeddings)


def unpack_batch(buf: bytes) -> list[CompressedEmbedding]:
    """Parse a concatenation of TQE1 records produced by :func:`pack_batch`."""
    out: list[CompressedEmbedding] = []
    off = 0
    n = len(buf)
    while off < n:
        size = record_size(buf, off)
        ce, _ = unpack(buf[off : off + size])
        out.append(ce)
        off += size
    return out

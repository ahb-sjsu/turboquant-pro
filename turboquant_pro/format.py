"""Versioned, self-describing on-disk/wire format for compressed embeddings.

A compressed embedding is only portable if it carries the parameters needed to
decode it. This module defines **TQE1**, a small self-describing container so a
reader can reconstruct a vector with no out-of-band metadata, and so the format is
stable across versions (a prerequisite for an industry-standard tool).

Layout (little-endian). ``version == 1`` (20-byte header) is the original, used
for the default ``"qr"`` rotation and is byte-identical to prior releases::

    offset size field
    0      4    magic   b"TQE1"
    4      1    version uint8  (== 1)
    5      1    bits    uint8  (2, 3, or 4)
    6      2    dim     uint16 (quantized dimension d')
    8      4    seed    uint32 (rotation seed -> reproduces codebook + rotation)
    12     4    norm    float32 (per-vector L2 norm)
    16     4    codelen uint32 (length of packed code bytes)
    20     ..   codes   `codelen` bytes (bit-packed indices)

``version == 2`` (21-byte header) adds a ``rotation`` byte after ``norm`` so a
reader reconstructs the exact rotation family (e.g. the opt-in Hadamard rotation)
with no out-of-band metadata. It is written only when the rotation is not the
default ``"qr"``, keeping v1 the common on-disk case::

    offset size field
    0      4    magic   b"TQE1"
    4      1    version uint8  (== 2)
    5      1    bits    uint8
    6      2    dim     uint16
    8      4    seed    uint32
    12     4    norm    float32
    16     1    rot     uint8  (0 = qr, 1 = hadamard)
    17     4    codelen uint32
    21     ..   codes   `codelen` bytes

``pack_batch`` concatenates records length-prefixed. Forward compatibility:
readers must reject an unknown ``version`` and may ignore trailing bytes after
``codes`` within a record.
"""

from __future__ import annotations

import struct

from .pgvector import CompressedEmbedding

MAGIC = b"TQE1"
VERSION = 1  # default "qr" rotation; byte-identical to prior releases
VERSION_ROT = 2  # adds a rotation byte (for non-default rotations)

# Rotation family <-> on-disk code. Keep in sync with pgvector._ROTATIONS.
_ROTATION_CODE = {"qr": 0, "hadamard": 1}
_ROTATION_NAME = {v: k for k, v in _ROTATION_CODE.items()}

# v1: magic, version, bits, dim, seed, norm, codelen
_HEADER = struct.Struct("<4sBBHIfI")
# v2: magic, version, bits, dim, seed, norm, rotation, codelen
_HEADER_V2 = struct.Struct("<4sBBHIfBI")
HEADER_SIZE = _HEADER.size  # 20 (v1)
HEADER_SIZE_V2 = _HEADER_V2.size  # 21 (v2)

# Smallest header we must read before we can learn the version.
_MIN_HEADER = HEADER_SIZE


def _header_size(version: int) -> int:
    if version == VERSION:
        return HEADER_SIZE
    if version == VERSION_ROT:
        return HEADER_SIZE_V2
    raise ValueError(f"unsupported TQE format version {version}")


_VALID_BITS = (2, 3, 4)


def pack(ce: CompressedEmbedding, seed: int | None = None) -> bytes:
    """Serialize a :class:`CompressedEmbedding` to a self-describing TQE record.

    The decode seed is taken from ``ce.seed`` by default so the stored seed can
    never drift from the record it describes; a wrong seed silently decodes to a
    different vector, so it must travel *with* the object, not out-of-band. Pass
    ``seed`` only to deliberately override the record's own seed.

    Writes the 20-byte v1 header for the default ``"qr"`` rotation (byte-identical
    to prior releases) and the 21-byte v2 header (with a rotation byte) otherwise.
    """
    seed = getattr(ce, "seed", 42) if seed is None else seed
    if int(ce.bits) not in _VALID_BITS:
        raise ValueError(f"unsupported bits {ce.bits!r}; expected one of {_VALID_BITS}")
    codes = bytes(ce.packed_bytes)
    rotation = getattr(ce, "rotation", "qr")
    if rotation not in _ROTATION_CODE:
        raise ValueError(f"unknown rotation {rotation!r}")
    if rotation == "qr":
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
    return (
        _HEADER_V2.pack(
            MAGIC,
            VERSION_ROT,
            int(ce.bits),
            int(ce.dim),
            int(seed) & 0xFFFFFFFF,
            float(ce.norm),
            _ROTATION_CODE[rotation],
            len(codes),
        )
        + codes
    )


def unpack(buf: bytes) -> tuple[CompressedEmbedding, int]:
    """Parse one TQE record. Returns ``(CompressedEmbedding, seed)``.

    The returned embedding's ``rotation`` field reflects the stored rotation
    family (``"qr"`` for v1 records), so decode is fully self-describing.
    Raises ``ValueError`` on bad magic, unsupported version, or truncation.
    """
    if len(buf) < _MIN_HEADER:
        raise ValueError("buffer too small for a TQE header")
    magic = buf[:4]
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}; expected {MAGIC!r}")
    version = buf[4]
    if version == VERSION:
        _, _, bits, dim, seed, norm, codelen = _HEADER.unpack(buf[:HEADER_SIZE])
        rotation = "qr"
        hsize = HEADER_SIZE
    elif version == VERSION_ROT:
        if len(buf) < HEADER_SIZE_V2:
            raise ValueError("buffer too small for a TQE v2 header")
        _, _, bits, dim, seed, norm, rot, codelen = _HEADER_V2.unpack(
            buf[:HEADER_SIZE_V2]
        )
        if rot not in _ROTATION_NAME:
            raise ValueError(f"unknown rotation code {rot}")
        rotation = _ROTATION_NAME[rot]
        hsize = HEADER_SIZE_V2
    else:
        raise ValueError(f"unsupported TQE format version {version}")
    if bits not in _VALID_BITS:
        raise ValueError(f"unsupported bits {bits}; expected one of {_VALID_BITS}")
    end = hsize + codelen
    if len(buf) < end:
        raise ValueError("truncated TQE record (codes shorter than codelen)")
    ce = CompressedEmbedding(
        packed_bytes=buf[hsize:end],
        norm=norm,
        dim=dim,
        bits=bits,
        rotation=rotation,
        seed=seed,
    )
    return ce, seed


def record_size(buf: bytes, offset: int = 0) -> int:
    """Total byte length of the TQE record beginning at ``offset``."""
    if len(buf) < offset + _MIN_HEADER:
        raise ValueError("buffer too small for a TQE header")
    version = buf[offset + 4]
    hsize = _header_size(version)
    if len(buf) < offset + hsize:
        raise ValueError("buffer too small for the TQE header of this version")
    codelen = struct.unpack_from("<I", buf, offset + hsize - 4)[0]
    return hsize + codelen


def pack_batch(embeddings: list[CompressedEmbedding], seed: int | None = None) -> bytes:
    """Serialize many embeddings as back-to-back TQE records.

    Each record's seed is taken from its own ``ce.seed`` unless ``seed`` overrides.
    """
    return b"".join(pack(ce, seed) for ce in embeddings)


def unpack_batch(buf: bytes) -> list[CompressedEmbedding]:
    """Parse a concatenation of TQE records produced by :func:`pack_batch`."""
    out: list[CompressedEmbedding] = []
    off = 0
    n = len(buf)
    while off < n:
        size = record_size(buf, off)
        ce, _ = unpack(buf[off : off + size])
        out.append(ce)
        off += size
    return out

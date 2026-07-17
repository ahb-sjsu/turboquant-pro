# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""TQIX — the versioned, corruption-checkable section container for a persisted index.

A production index must survive on disk, grow, and be trusted. TQIX is the
low-level substrate for that: a self-describing container of named byte
*sections*, each independently CRC32-checked, behind a fixed magic + version
header. It knows nothing about embeddings — :mod:`turboquant_pro.index` layers
the PCA basis, ADC codes, ids, and tombstones on top as sections.

Layout (little-endian)::

    offset size field
    0      4    magic       b"TQIX"
    4      2    version     uint16   (container format version)
    6      2    n_sections  uint16
    8      4    reserved    uint32   (0)
    12     ..   directory   n_sections x 56-byte entries:
                  32  name    utf-8, null-padded
                  8   offset  uint64 (from file start)
                  8   length  uint64
                  4   crc32   uint32 (of the section bytes)
                  4   flags   uint32 (0)
    ..     ..   section payloads, back to back

Guarantees:

- **Corruption detection.** Every section carries a CRC32; :func:`read_container`
  raises :class:`IndexCorruptionError` on any mismatched or truncated section, so
  a flipped bit is a clean error, never silent bad data.
- **Atomic writes.** :func:`write_container` writes a sibling temp file and
  ``os.replace``s it into place, so a crash mid-write never leaves a torn index.
- **Cheap inspection.** :func:`read_directory` returns the header + section
  table without loading payloads (for ``tqp index info`` and memmap).
"""

from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass

MAGIC = b"TQIX"

# magic, version, n_sections, reserved
_FIXED = struct.Struct("<4sHHI")
# name(32), offset(u64), length(u64), crc32(u32), flags(u32)
_ENTRY = struct.Struct("<32sQQII")
_NAME_LEN = 32


class IndexCorruptionError(ValueError):
    """A TQIX file failed magic, structural, or CRC validation."""


@dataclass(frozen=True)
class SectionRef:
    """Where a section lives in the file, without its bytes."""

    name: str
    offset: int
    length: int
    crc32: int


def _encode_name(name: str) -> bytes:
    b = name.encode("utf-8")
    if len(b) > _NAME_LEN:
        raise ValueError(f"section name {name!r} exceeds {_NAME_LEN} bytes")
    return b.ljust(_NAME_LEN, b"\x00")


def write_container(path: str, version: int, sections: list[tuple[str, bytes]]) -> None:
    """Write ``sections`` (ordered ``(name, bytes)``) to ``path`` atomically.

    Duplicate section names are rejected. The write goes to ``path + ".tmp"``
    and is ``os.replace``d into place so readers never observe a partial file.
    """
    if not 0 <= version <= 0xFFFF:
        raise ValueError(f"version {version} out of uint16 range")
    seen: set[str] = set()
    for name, _ in sections:
        if name in seen:
            raise ValueError(f"duplicate section name {name!r}")
        seen.add(name)

    n = len(sections)
    header_size = _FIXED.size + n * _ENTRY.size
    directory = bytearray()
    payload = bytearray()
    offset = header_size
    for name, data in sections:
        crc = zlib.crc32(data) & 0xFFFFFFFF
        directory += _ENTRY.pack(_encode_name(name), offset, len(data), crc, 0)
        payload += data
        offset += len(data)

    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(_FIXED.pack(MAGIC, version, n, 0))
        f.write(directory)
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_directory(path: str) -> tuple[int, list[SectionRef]]:
    """Return ``(version, [SectionRef, ...])`` without loading section payloads."""
    with open(path, "rb") as f:
        head = f.read(_FIXED.size)
        if len(head) < _FIXED.size:
            raise IndexCorruptionError("file too small for a TQIX header")
        magic, version, n, _ = _FIXED.unpack(head)
        if magic != MAGIC:
            raise IndexCorruptionError(f"bad magic {magic!r}; expected {MAGIC!r}")
        dir_bytes = f.read(n * _ENTRY.size)
        if len(dir_bytes) < n * _ENTRY.size:
            raise IndexCorruptionError("truncated section directory")
    refs = []
    for i in range(n):
        entry = _ENTRY.unpack_from(dir_bytes, i * _ENTRY.size)
        name_b, off, length, crc, _flags = entry
        name = name_b.rstrip(b"\x00").decode("utf-8")
        refs.append(SectionRef(name, off, length, crc))
    return version, refs


def read_container(path: str) -> tuple[int, dict[str, bytes]]:
    """Return ``(version, {name: bytes})``, verifying every section's CRC32.

    Raises :class:`IndexCorruptionError` on bad magic, a truncated file, or any
    section whose bytes do not match its stored CRC.
    """
    version, refs = read_directory(path)
    size = os.path.getsize(path)
    sections: dict[str, bytes] = {}
    with open(path, "rb") as f:
        for ref in refs:
            end = ref.offset + ref.length
            if end > size:
                raise IndexCorruptionError(
                    f"section {ref.name!r} runs past end of file "
                    f"({end} > {size}) — truncated"
                )
            f.seek(ref.offset)
            blob = f.read(ref.length)
            if (zlib.crc32(blob) & 0xFFFFFFFF) != ref.crc32:
                raise IndexCorruptionError(
                    f"CRC mismatch in section {ref.name!r} — the index is corrupt"
                )
            sections[ref.name] = blob
    return version, sections

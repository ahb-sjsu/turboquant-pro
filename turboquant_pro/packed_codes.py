# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Row-aligned bit-packing for stored ADC codes (index format v3).

The in-memory ADC scan (and the AVX2 kernel) wants one ``uint8`` per code, but
storing a 4-bit code in a full byte doubles the dominant section of an
``--no-originals`` index. Format v3 packs codes on disk at *slot* granularity —
2 codes/byte for 3-4 bit, 4/byte for 2-bit, 8/byte for 1-bit — with every row
starting on a byte boundary, so a row gather reads whole bytes and unpacks with
two vectorized shift/mask passes. This is deliberately not the transport
packing in :mod:`turboquant_pro.core` (tight 3-bit stream packing): that layout
lets codes straddle bytes across rows, which would make random row access —
the whole point of the IVF/memmap search path — a bit-arithmetic exercise.

:class:`PackedCodes` wraps a memory-mapped packed array behind the small part
of the ``ndarray`` interface the search paths use (``len``/``shape``/row-gather
``[]``/``__array__``), unpacking only the rows a probe touches. Packing is
lossless re-encoding of the quantizer's level indices, so every ranking is
bit-identical to the unpacked format.
"""

from __future__ import annotations

import numpy as np

_SLOT_BITS = {1: 1, 2: 2, 3: 4, 4: 4}


def slot_bits_for(bits: int) -> int:
    """Stored bits per code for a quantizer of ``bits`` (8 = not packed)."""
    return _SLOT_BITS.get(int(bits), 8)


def packed_cols(dim: int, slot_bits: int) -> int:
    """Bytes per stored row for ``dim`` codes at ``slot_bits`` per code."""
    if slot_bits >= 8:
        return int(dim)
    per = 8 // slot_bits
    return -(-int(dim) // per)


def pack_rows(codes: np.ndarray, slot_bits: int) -> np.ndarray:
    """Pack ``(n, dim)`` uint8 level indices into ``(n, packed_cols)`` bytes.

    Slot 0 occupies the least-significant bits of each byte. Values must fit in
    ``slot_bits`` (guaranteed for level indices of a ``bits <= slot_bits``
    quantizer).
    """
    codes = np.asarray(codes, dtype=np.uint8)
    if slot_bits >= 8:
        return np.ascontiguousarray(codes)
    per = 8 // slot_bits
    n, dim = codes.shape
    cols = packed_cols(dim, slot_bits)
    pad = cols * per - dim
    if pad:
        codes = np.concatenate([codes, np.zeros((n, pad), dtype=np.uint8)], axis=1)
    slots = codes.reshape(n, cols, per)
    packed = np.zeros((n, cols), dtype=np.uint8)
    for j in range(per):
        packed |= slots[:, :, j] << np.uint8(slot_bits * j)
    return packed


def unpack_rows(packed: np.ndarray, dim: int, slot_bits: int) -> np.ndarray:
    """Inverse of :func:`pack_rows`: ``(n, packed_cols)`` -> ``(n, dim)`` uint8."""
    packed = np.asarray(packed, dtype=np.uint8)
    if slot_bits >= 8:
        return np.ascontiguousarray(packed)
    per = 8 // slot_bits
    mask = np.uint8((1 << slot_bits) - 1)
    n, cols = packed.shape
    out = np.empty((n, cols, per), dtype=np.uint8)
    for j in range(per):
        out[:, :, j] = (packed >> np.uint8(slot_bits * j)) & mask
    return np.ascontiguousarray(out.reshape(n, cols * per)[:, :dim])


class PackedCodes:
    """Read-only packed code store presenting unpacked rows on access.

    Wraps the packed (usually memory-mapped) ``(n, packed_cols)`` byte array of
    a v3 index. Any indexing — a row gather (``codes[rows]``), a block slice
    (``codes[s:e]``), or a whole-array conversion (``np.asarray``, as
    ``cent[codes]`` triggers) — reads only the touched packed bytes and returns
    ordinary unpacked ``uint8`` codes, so every existing scoring path works
    unchanged while disk reads shrink by the packing factor.
    """

    def __init__(self, packed: np.ndarray, dim: int, slot_bits: int):
        self._packed = packed
        self._dim = int(dim)
        self._slot = int(slot_bits)
        self.shape = (len(packed), self._dim)
        self.dtype = np.dtype(np.uint8)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key) -> np.ndarray:
        rows = np.asarray(self._packed[key])
        if rows.ndim == 1:  # a single row
            return unpack_rows(rows[None], self._dim, self._slot)[0]
        return unpack_rows(rows, self._dim, self._slot)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        out = unpack_rows(np.asarray(self._packed), self._dim, self._slot)
        return out if dtype is None else out.astype(dtype)

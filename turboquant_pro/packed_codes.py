# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Bit-packing of quantizer level indices — the single implementation.

Every sub-byte code the package *stores* goes through :func:`pack_bits` /
:func:`unpack_bits` here: the TQE1 record and pgvector ``bytea``
(:mod:`turboquant_pro.pgvector`), ``CompressedKV.indices`` when ``packed=True``
(:mod:`turboquant_pro.core`), and the index format v3 ``codes`` section
(:func:`pack_rows`). The layout is the **LSB-first stream** promised by
``docs/FORMAT_SPEC.md``: value ``j`` occupies stream bits
``[j*bits, (j+1)*bits)``, stream bit ``p`` lives in byte ``p // 8`` at bit
position ``p % 8``, and a ragged tail is zero-padded to a whole *group* — the
smallest run of values whose bits fill whole bytes (4 values/1 byte at 2-bit,
8 values/3 bytes at 3-bit, 2 values/1 byte at 4-bit). Changing any of this
silently corrupts data already on disk; ``tests/test_bitpack_identity.py`` pins
the bytes.

Index format v3 needs random row access, so it packs at *slot* granularity —
2 codes/byte for 3-4 bit, 4/byte for 2-bit, 8/byte for 1-bit — with every row
starting on a byte boundary. That is the same stream packer applied to
byte-aligned rows: a 3-bit quantizer's codes are stored in 4-bit slots
precisely because the tight 3-bit stream lets codes straddle bytes, which would
make a row gather a bit-arithmetic exercise. A v3 row at 2 or 4 bits is
byte-identical to the transport stream of the same values.

:class:`PackedCodes` wraps a memory-mapped packed array behind the small part
of the ``ndarray`` interface the search paths use (``len``/``shape``/row-gather
``[]``/``__array__``), unpacking only the rows a probe touches. Packing is
lossless re-encoding of the quantizer's level indices, so every ranking is
bit-identical to the unpacked format.

The one layout that is *not* this one is ``per_channel_kv._pack_indices``
(``CompressedPerChannelKV.indices``): same value order, but MSB-first within
each byte and padded to ``ceil(n*bits/8)`` bytes. It is a separate stored
format with GPU kernels written against it, so it stays separate.
"""

from __future__ import annotations

import math

import numpy as np

_SLOT_BITS = {1: 1, 2: 2, 3: 4, 4: 4}


# ---------------------------------------------------------------------- #
# Canonical LSB-first stream packer                                        #
# ---------------------------------------------------------------------- #


def stream_group(bits: int) -> int:
    """Values per padding group: the fewest whose ``bits`` fill whole bytes."""
    return 8 // math.gcd(8, int(bits))


def packed_nbytes(n_values: int, bits: int) -> int:
    """Bytes :func:`pack_bits` emits for ``n_values`` codes of ``bits`` each."""
    group = stream_group(bits)
    return -(-int(n_values) // group) * (group * int(bits) // 8)


def _stream_geometry(bits: int) -> tuple[int, int, np.dtype, np.dtype]:
    """``(group, bytes_per_group, native word dtype, little-endian word dtype)``.

    A group is packed through one unsigned word wide enough to hold it: a
    byte when ``bits`` divides 8 (no cross-byte straddling), a ``uint32`` for
    the 24-bit 3-bit/6-bit groups, a ``uint64`` for 5-bit/7-bit.
    """
    bits = int(bits)
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in 1..8, got {bits}")
    group = stream_group(bits)
    nbytes = group * bits // 8
    if nbytes == 1:
        return group, nbytes, np.dtype(np.uint8), np.dtype(np.uint8)
    if nbytes <= 4:
        return group, nbytes, np.dtype(np.uint32), np.dtype("<u4")
    return group, nbytes, np.dtype(np.uint64), np.dtype("<u8")


def pack_bits(values: np.ndarray, bits: int) -> np.ndarray:
    """Pack integer codes (``0 <= v < 2**bits``) into a flat ``uint8`` stream.

    ``values`` may have any shape and any integer dtype; it is flattened in C
    order. The tail is zero-padded to a whole group (see :func:`stream_group`),
    so the result has :func:`packed_nbytes` bytes. Inverse: :func:`unpack_bits`.
    """
    group, nbytes, word, word_le = _stream_geometry(bits)
    flat = np.asarray(values).reshape(-1)
    n = flat.size
    m = -(-n // group)
    slots = np.zeros(m * group, dtype=word)
    slots[:n] = flat
    slots = slots.reshape(m, group)
    packed = np.zeros(m, dtype=word)
    for j in range(group):
        packed |= slots[:, j] << word.type(bits * j)
    if nbytes == 1:
        return packed
    raw = packed.astype(word_le, copy=False).view(np.uint8).reshape(m, word.itemsize)
    return np.ascontiguousarray(raw[:, :nbytes]).reshape(-1)


def unpack_bits(packed: np.ndarray, n_values: int, bits: int) -> np.ndarray:
    """Inverse of :func:`pack_bits`: the first ``n_values`` codes as ``uint8``.

    ``packed`` is flattened; its length must be a whole number of groups (it
    always is for :func:`pack_bits` output).
    """
    group, nbytes, word, word_le = _stream_geometry(bits)
    raw = np.ascontiguousarray(np.asarray(packed, dtype=np.uint8).reshape(-1))
    if raw.size % nbytes:
        raise ValueError(
            f"packed length {raw.size} is not a multiple of {nbytes} bytes "
            f"(the {group}-value group at {bits} bits)"
        )
    m = raw.size // nbytes
    if nbytes == 1:
        words = raw
    else:
        buf = np.zeros((m, word.itemsize), dtype=np.uint8)
        buf[:, :nbytes] = raw.reshape(m, nbytes)
        words = buf.view(word_le).reshape(-1).astype(word, copy=False)
    mask = word.type((1 << int(bits)) - 1)
    out = np.empty((m, group), dtype=np.uint8)
    for j in range(group):
        out[:, j] = (words >> word.type(bits * j)) & mask
    return out.reshape(-1)[: int(n_values)]


# ---------------------------------------------------------------------- #
# Index format v3: byte-aligned rows                                       #
# ---------------------------------------------------------------------- #


def slot_bits_for(bits: int) -> int:
    """Stored bits per code for a quantizer of ``bits`` (8 = not packed)."""
    return _SLOT_BITS.get(int(bits), 8)


def packed_cols(dim: int, slot_bits: int) -> int:
    """Bytes per stored row for ``dim`` codes at ``slot_bits`` per code."""
    if slot_bits >= 8:
        return int(dim)
    per = 8 // slot_bits
    return -(-int(dim) // per)


def _slots_per_byte(slot_bits: int) -> int:
    """Codes per byte for a v3 slot width; slots never straddle bytes."""
    slot_bits = int(slot_bits)
    if slot_bits not in (1, 2, 4):
        raise ValueError(
            f"slot_bits must be one of 1, 2, 4 (or >= 8 for unpacked), got {slot_bits}"
        )
    return 8 // slot_bits


def pack_rows(codes: np.ndarray, slot_bits: int) -> np.ndarray:
    """Pack ``(n, dim)`` uint8 level indices into ``(n, packed_cols)`` bytes.

    Each row is :func:`pack_bits` of that row (slot 0 in the least-significant
    bits of each byte), padded to whole bytes so rows stay byte-aligned. Values
    must fit in ``slot_bits`` (guaranteed for level indices of a
    ``bits <= slot_bits`` quantizer).
    """
    codes = np.asarray(codes, dtype=np.uint8)
    if slot_bits >= 8:
        return np.ascontiguousarray(codes)
    per = _slots_per_byte(slot_bits)
    n, dim = codes.shape
    cols = packed_cols(dim, slot_bits)
    pad = cols * per - dim
    if pad:
        codes = np.concatenate([codes, np.zeros((n, pad), dtype=np.uint8)], axis=1)
    # every row is a whole number of groups, so the flat stream never straddles
    return pack_bits(codes, slot_bits).reshape(n, cols)


def unpack_rows(packed: np.ndarray, dim: int, slot_bits: int) -> np.ndarray:
    """Inverse of :func:`pack_rows`: ``(n, packed_cols)`` -> ``(n, dim)`` uint8."""
    packed = np.asarray(packed, dtype=np.uint8)
    if slot_bits >= 8:
        return np.ascontiguousarray(packed)
    per = _slots_per_byte(slot_bits)
    n, cols = packed.shape
    out = unpack_bits(packed, n * cols * per, slot_bits).reshape(n, cols * per)
    return np.ascontiguousarray(out[:, :dim])


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

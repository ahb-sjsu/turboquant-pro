"""Byte-level pins for every bit-packing layout the package stores.

Bit-packing is stored-format code (TQE1 ``codes``, pgvector ``bytea``, index
format v3 ``codes``, ``CompressedPerChannelKV.indices``): a silent change in bit
order or padding corrupts data that is already on disk. This file pins the
exact bytes each packer emits against pure-Python bit-by-bit oracles and a few
hand-checked golden byte strings, and asserts the cross-implementation
identities the code base relies on. It is the gate for any refactor of the
packers: it must pass before *and* after.

Two layouts exist, and they are **not** interchangeable:

``lsb_stream`` (``core``, ``pgvector``, ``packed_codes``, TQE1, index v3)
    Value ``j`` occupies stream bits ``[j*bits, (j+1)*bits)``; stream bit ``p``
    lives in byte ``p // 8`` at bit position ``p % 8`` (LSB-first within the
    byte). Values are zero-padded to a whole *group* (the smallest count whose
    bits fill whole bytes: 4 at 2-bit, 8 at 3-bit, 2 at 4-bit), so a ragged
    3-bit stream is padded to a multiple of 3 bytes, not to ``ceil(n*3/8)``.

``msb_bytes`` (``per_channel_kv``, ``volta_kernels`` CPU fallback)
    Same value-to-stream mapping (low bit of a value first), but stream bit
    ``p`` lives in byte ``p // 8`` at bit position ``7 - p % 8`` (MSB-first
    within the byte, ``np.packbits`` default order), and the stream is padded to
    ``ceil(n*bits/8)`` bytes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from turboquant_pro.core import TurboQuantKV
from turboquant_pro.packed_codes import pack_rows, packed_cols, unpack_rows
from turboquant_pro.per_channel_kv import _pack_indices, _unpack_indices
from turboquant_pro.pgvector import TurboQuantPGVector
from turboquant_pro.volta_kernels import _unpack_ref

LENGTHS = [0, 1, 7, 8, 9, 63, 64, 65, 1000]
SEEDS = [0, 1, 2]
STREAM_BITS = [2, 3, 4]  # widths core / pgvector accept
SLOT_BITS = [1, 2, 4, 8]  # widths slot_bits_for() produces for index v3


# ------------------------------------------------------------------ #
# Pure-Python oracles                                                 #
# ------------------------------------------------------------------ #


def stream_group(bits: int) -> int:
    """Values per padding group for the LSB-first stream layout."""
    return 8 // math.gcd(8, bits)


def ref_lsb_stream(values, bits: int) -> bytes:
    vals = [int(v) for v in values]
    group = stream_group(bits)
    pad = (-len(vals)) % group
    vals += [0] * pad
    out = bytearray(len(vals) * bits // 8)
    for j, v in enumerate(vals):
        for k in range(bits):
            if (v >> k) & 1:
                p = j * bits + k
                out[p // 8] |= 1 << (p % 8)
    return bytes(out)


def ref_msb_bytes(values, bits: int) -> bytes:
    vals = [int(v) for v in values]
    out = bytearray(-(-len(vals) * bits // 8))
    for j, v in enumerate(vals):
        for k in range(bits):
            if (v >> k) & 1:
                p = j * bits + k
                out[p // 8] |= 1 << (7 - p % 8)
    return bytes(out)


def rand_values(n: int, bits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed * 1000 + bits * 10_000 + n)
    return rng.integers(0, 2**bits, size=n, dtype=np.uint8)


def _tq(bits: int) -> TurboQuantKV:
    return TurboQuantKV(head_dim=8, n_heads=1, bits=bits, use_gpu=False, seed=0)


def _pg(bits: int) -> TurboQuantPGVector:
    return TurboQuantPGVector(dim=8, bits=bits, seed=0)


# ------------------------------------------------------------------ #
# Golden bytes (hand-checked; computed before any refactor)           #
# ------------------------------------------------------------------ #

# 3-bit [1,2,3,4,5,6,7,0]: 1 | 2<<3 | 3<<6 | 4<<9 | 5<<12 | 6<<15 | 7<<18
# = 0x1F58D1 -> little-endian bytes d1 58 1f; the ragged tail [5] pads to a
# whole 3-byte group.
GOLDEN_STREAM = {
    2: ([0, 1, 2, 3, 3, 2, 1, 0, 1], "e41b01"),
    3: ([1, 2, 3, 4, 5, 6, 7, 0, 5], "d1581f050000"),
    4: ([1, 2, 15, 8, 3], "218f03"),
}
GOLDEN_MSB = {
    1: ([1, 0, 1, 1, 0, 0, 1, 0, 1], "b280"),
    2: ([0, 1, 2, 3, 3, 2, 1, 0, 1], "27d880"),
    3: ([1, 2, 3, 4, 5, 6, 7, 0, 5], "8b1af8a0"),
    4: ([1, 2, 15, 8, 3], "84f1c0"),
}


@pytest.mark.parametrize("bits", STREAM_BITS)
def test_golden_stream_bytes(bits):
    vals, hexstr = GOLDEN_STREAM[bits]
    x = np.array(vals, dtype=np.uint8)
    assert ref_lsb_stream(x, bits).hex() == hexstr
    assert _tq(bits)._pack_bits_cpu(x).tobytes().hex() == hexstr
    assert _pg(bits)._pack_bits_cpu(x).tobytes().hex() == hexstr
    if bits != 3:
        assert pack_rows(x[None], bits)[0].tobytes().hex() == hexstr


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_golden_msb_bytes(bits):
    vals, hexstr = GOLDEN_MSB[bits]
    x = np.array(vals, dtype=np.uint8)
    assert ref_msb_bytes(x, bits).hex() == hexstr
    assert _pack_indices(x, bits).tobytes().hex() == hexstr


# ------------------------------------------------------------------ #
# core.TurboQuantKV  (TQE1 / CompressedKV.indices when packed=True)  #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("bits", STREAM_BITS)
def test_core_cpu_pack_matches_oracle(bits, n, seed):
    x = rand_values(n, bits, seed)
    tq = _tq(bits)
    packed = tq._pack_bits_cpu(x)
    assert packed.dtype == np.uint8 and packed.ndim == 1
    assert packed.tobytes() == ref_lsb_stream(x, bits)
    back = tq._unpack_bits_cpu(packed, n)
    assert back.dtype == np.uint8
    np.testing.assert_array_equal(back, x)
    # explicit-width override (asymmetric K/V path) is the same function
    other = 2 if bits != 2 else 4
    y = rand_values(n, other, seed)
    assert _tq(bits)._pack_bits_cpu(y, bits=other).tobytes() == ref_lsb_stream(y, other)
    np.testing.assert_array_equal(
        _tq(bits)._unpack_bits_cpu(
            _tq(bits)._pack_bits_cpu(y, bits=other), n, bits=other
        ),
        y,
    )


@pytest.mark.parametrize("bits", STREAM_BITS)
def test_core_pack_accepts_any_shape_and_int_dtype(bits):
    x = rand_values(96, bits, 7)
    tq = _tq(bits)
    ref = tq._pack_bits_cpu(x).tobytes()
    assert tq._pack_bits_cpu(x.reshape(2, 3, 16)).tobytes() == ref
    assert tq._pack_bits_cpu(x.astype(np.int64)).tobytes() == ref
    assert tq._pack_bits_cpu(x.astype(np.uint32)).tobytes() == ref
    # unpack accepts a 2-D packed array too (ravel semantics)
    packed = tq._pack_bits_cpu(x)
    np.testing.assert_array_equal(tq._unpack_bits_cpu(packed.reshape(-1, 1), 96), x)


@pytest.mark.parametrize("bits", STREAM_BITS)
def test_core_public_compress_uses_stream_layout(bits):
    tq = TurboQuantKV(head_dim=24, n_heads=2, bits=bits, use_gpu=False, seed=3)
    x = np.random.default_rng(bits).standard_normal((1, 2, 5, 24)).astype(np.float32)
    unpacked = tq.compress(x, packed=False)
    packed = tq.compress(x, packed=True)
    assert packed.indices.tobytes() == ref_lsb_stream(unpacked.indices.ravel(), bits)
    np.testing.assert_array_equal(tq.decompress(packed), tq.decompress(unpacked))


@pytest.mark.parametrize("bits", [1, 5, 8])
def test_core_rejects_unsupported_widths(bits):
    tq = _tq(3)
    with pytest.raises(ValueError):
        tq._pack_bits_cpu(np.zeros(8, dtype=np.uint8), bits=bits)
    with pytest.raises(ValueError):
        tq._unpack_bits_cpu(np.zeros(8, dtype=np.uint8), 8, bits=bits)


# ------------------------------------------------------------------ #
# pgvector.TurboQuantPGVector  (bytea / TQE1 codes)                   #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("bits", STREAM_BITS)
def test_pgvector_cpu_pack_matches_oracle(bits, n, seed):
    x = rand_values(n, bits, seed)
    pg = _pg(bits)
    packed = pg._pack_bits_cpu(x)
    assert packed.dtype == np.uint8 and packed.ndim == 1
    assert packed.tobytes() == ref_lsb_stream(x, bits)
    back = pg._unpack_bits_cpu(packed, n)
    assert back.dtype == np.uint8
    np.testing.assert_array_equal(back, x)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("bits", STREAM_BITS)
def test_core_and_pgvector_are_byte_identical(bits, n, seed):
    x = rand_values(n, bits, seed)
    a = _tq(bits)._pack_bits_cpu(x)
    b = _pg(bits)._pack_bits_cpu(x)
    assert a.tobytes() == b.tobytes()
    np.testing.assert_array_equal(
        _tq(bits)._unpack_bits_cpu(b, n), _pg(bits)._unpack_bits_cpu(a, n)
    )


@pytest.mark.parametrize("bits", STREAM_BITS)
@pytest.mark.parametrize("dim", [7, 24, 64])
def test_pgvector_public_bytea_uses_stream_layout(bits, dim):
    pg = TurboQuantPGVector(dim=dim, bits=bits, seed=1)
    emb = np.random.default_rng(dim + bits).standard_normal(dim).astype(np.float32)
    ce = pg.compress_embedding(emb)
    idx = pg._unpack_bits_cpu(np.frombuffer(ce.packed_bytes, dtype=np.uint8), dim)
    assert ce.packed_bytes == ref_lsb_stream(idx, bits)
    assert len(ce.packed_bytes) == len(ref_lsb_stream(np.zeros(dim), bits))
    batch = pg.compress_batch(np.stack([emb, -emb]))
    assert batch[0].packed_bytes == ce.packed_bytes


# ------------------------------------------------------------------ #
# packed_codes  (index format v3 row layout)                          #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("dim", LENGTHS)
@pytest.mark.parametrize("slot", SLOT_BITS)
def test_pack_rows_matches_oracle_per_row(slot, dim, seed):
    n_rows = 5
    rng = np.random.default_rng(seed * 7 + slot * 100 + dim)
    codes = rng.integers(0, 2**slot, size=(n_rows, dim), dtype=np.uint8)
    packed = pack_rows(codes, slot)
    assert packed.dtype == np.uint8
    assert packed.shape == (n_rows, packed_cols(dim, slot))
    for r in range(n_rows):
        assert packed[r].tobytes() == ref_lsb_stream(codes[r], slot)
    out = unpack_rows(packed, dim, slot)
    assert out.dtype == np.uint8 and out.shape == (n_rows, dim)
    np.testing.assert_array_equal(out, codes)
    # single-row and empty-row-count inputs
    np.testing.assert_array_equal(pack_rows(codes[:1], slot), packed[:1])
    assert pack_rows(codes[:0], slot).shape == (0, packed_cols(dim, slot))
    assert unpack_rows(packed[:0], dim, slot).shape == (0, dim)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("bits", [2, 4])
def test_pack_rows_row_equals_core_stream(bits, n, seed):
    """At 2 and 4 bits a v3 row is byte-identical to the core/pgvector stream."""
    x = rand_values(n, bits, seed)
    assert (
        pack_rows(x[None], bits)[0].tobytes() == _tq(bits)._pack_bits_cpu(x).tobytes()
    )
    np.testing.assert_array_equal(
        unpack_rows(_tq(bits)._pack_bits_cpu(x)[None], n, bits)[0], x
    )


def test_pack_rows_3bit_uses_4bit_slots_not_core_stream():
    """Index v3 stores 3-bit codes in 4-bit slots; the core 3-bit stream differs."""
    x = rand_values(64, 3, 0)
    assert pack_rows(x[None], 4)[0].tobytes() == ref_lsb_stream(x, 4)
    assert pack_rows(x[None], 4)[0].tobytes() != _tq(3)._pack_bits_cpu(x).tobytes()


# ------------------------------------------------------------------ #
# per_channel_kv / volta_kernels  (MSB-first byte layout)             #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
def test_per_channel_pack_matches_oracle(bits, n, seed):
    x = rand_values(n, bits, seed)
    packed = _pack_indices(x, bits)
    assert packed.dtype == np.uint8 and packed.ndim == 1
    assert packed.tobytes() == ref_msb_bytes(x, bits)
    back = _unpack_indices(packed, n, bits)
    assert back.dtype == np.uint8
    np.testing.assert_array_equal(back, x)
    # volta CPU fallback is the same unpack, reshaped to (H, S, D)
    np.testing.assert_array_equal(_unpack_ref(packed, 1, 1, n, bits).ravel(), x)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_volta_unpack_ref_equals_per_channel_unpack(bits):
    H, S, D = 3, 5, 16
    x = rand_values(H * S * D, bits, 11).reshape(H, S, D)
    packed = _pack_indices(x, bits)
    np.testing.assert_array_equal(
        _unpack_ref(packed, H, S, D, bits),
        _unpack_indices(packed, H * S * D, bits).reshape(H, S, D),
    )


@pytest.mark.parametrize("bits", STREAM_BITS)
def test_per_channel_layout_differs_from_stream_layout(bits):
    """Documented divergence: per-channel bytes are the bit-reversal of the stream."""
    x = rand_values(48, bits, 5)  # 48 values fill whole bytes at 2/3/4 bits
    stream = _tq(bits)._pack_bits_cpu(x)
    msb = _pack_indices(x, bits)
    assert len(stream) == len(msb)
    assert stream.tobytes() != msb.tobytes()
    reversed_bytes = np.packbits(np.unpackbits(stream, bitorder="little"))
    assert reversed_bytes.tobytes() == msb.tobytes()

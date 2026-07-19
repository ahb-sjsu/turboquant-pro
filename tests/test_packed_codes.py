"""Row-aligned code packing (index format v3): lossless round-trips, the
PackedCodes view, and the on-disk savings it exists for.

Pure-numpy; runs in CI. Packing is a re-encoding of quantizer level indices,
so every test asserts exact equality — never a tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro.packed_codes import (
    PackedCodes,
    pack_rows,
    packed_cols,
    slot_bits_for,
    unpack_rows,
)


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
@pytest.mark.parametrize("dim", [1, 7, 24, 33])
def test_pack_unpack_roundtrip_exact(bits, dim):
    rng = np.random.default_rng(bits * 100 + dim)
    codes = rng.integers(0, 2**bits, size=(257, dim), dtype=np.uint8)
    slot = slot_bits_for(bits)
    packed = pack_rows(codes, slot)
    assert packed.shape == (257, packed_cols(dim, slot))
    np.testing.assert_array_equal(unpack_rows(packed, dim, slot), codes)


def test_slot_widths():
    # 3-bit codes ride in 4-bit slots (row-aligned beats tight 3-bit packing
    # for random row access); >= 5 bits are stored unpacked.
    assert [slot_bits_for(b) for b in (1, 2, 3, 4, 5, 8)] == [1, 2, 4, 4, 8, 8]
    assert packed_cols(24, 4) == 12
    assert packed_cols(24, 2) == 6
    assert packed_cols(24, 8) == 24


def test_pack_noop_at_8_bits():
    codes = np.arange(64, dtype=np.uint8).reshape(4, 16)
    np.testing.assert_array_equal(pack_rows(codes, 8), codes)
    np.testing.assert_array_equal(unpack_rows(codes, 16, 8), codes)


def test_empty_rows():
    codes = np.zeros((0, 24), dtype=np.uint8)
    packed = pack_rows(codes, 4)
    assert packed.shape == (0, 12)
    assert unpack_rows(packed, 24, 4).shape == (0, 24)


def test_packed_codes_view_matches_plain_indexing():
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 16, size=(500, 24), dtype=np.uint8)
    view = PackedCodes(pack_rows(codes, 4), dim=24, slot_bits=4)
    assert len(view) == 500
    assert view.shape == (500, 24)
    # Row gather (the IVF scan path), slice (the blocked scan path), single row,
    # and whole-array conversion (cent[codes] via __array__).
    rows = np.array([3, 499, 0, 77], dtype=np.uint32)
    np.testing.assert_array_equal(view[rows], codes[rows])
    np.testing.assert_array_equal(view[10:20], codes[10:20])
    np.testing.assert_array_equal(view[42], codes[42])
    np.testing.assert_array_equal(np.asarray(view), codes)


def test_packed_codes_view_fancy_indexes_a_table():
    # cent[codes] — an ndarray fancy-indexed BY the view — must materialize it.
    rng = np.random.default_rng(11)
    codes = rng.integers(0, 16, size=(50, 8), dtype=np.uint8)
    cent = rng.standard_normal(16).astype(np.float32)
    view = PackedCodes(pack_rows(codes, 4), dim=8, slot_bits=4)
    np.testing.assert_array_equal(cent[np.asarray(view)], cent[codes])

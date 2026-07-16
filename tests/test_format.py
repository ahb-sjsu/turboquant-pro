"""Conformance tests for the TQE1 compressed-embedding format."""

import numpy as np
import pytest

from turboquant_pro import TurboQuantPGVector
from turboquant_pro.format import (
    HEADER_SIZE,
    HEADER_SIZE_V2,
    MAGIC,
    VERSION,
    VERSION_ROT,
    pack,
    pack_batch,
    record_size,
    unpack,
    unpack_batch,
)


def _make(dim=64, bits=3, seed=42):
    tq = TurboQuantPGVector(dim=dim, bits=bits, seed=seed)
    rng = np.random.default_rng(0)
    return tq, tq.compress_embedding(rng.standard_normal(dim).astype(np.float32))


def test_roundtrip_exact():
    tq, ce = _make()
    ce2, seed = unpack(pack(ce, seed=tq.seed))
    assert seed == tq.seed
    assert ce2.bits == ce.bits and ce2.dim == ce.dim
    assert abs(ce2.norm - ce.norm) < 1e-4
    assert bytes(ce2.packed_bytes) == bytes(ce.packed_bytes)
    np.testing.assert_allclose(
        tq.decompress_embedding(ce2), tq.decompress_embedding(ce), atol=1e-5
    )


def test_header_self_describing():
    _, ce = _make(dim=128, bits=4)
    blob = pack(ce)
    assert blob[:4] == MAGIC
    assert record_size(blob) == len(blob)
    assert len(blob) == HEADER_SIZE + len(ce.packed_bytes)


def test_batch_roundtrip():
    tq, _ = _make()
    rng = np.random.default_rng(1)
    ces = tq.compress_batch(rng.standard_normal((10, 64)).astype(np.float32))
    out = unpack_batch(pack_batch(ces))
    assert len(out) == 10
    for a, b in zip(ces, out):
        assert bytes(a.packed_bytes) == bytes(b.packed_bytes) and a.bits == b.bits


def test_bad_magic():
    with pytest.raises(ValueError):
        unpack(b"XXXX" + b"\x00" * 16)


def test_unsupported_version():
    _, ce = _make()
    blob = bytearray(pack(ce))
    blob[4] = 99
    with pytest.raises(ValueError):
        unpack(bytes(blob))


def test_truncated_record():
    _, ce = _make()
    with pytest.raises(ValueError):
        unpack(pack(ce)[:-1])


# ------------------------------------------------------------------ #
# Version 2: rotation-aware, self-describing                          #
# ------------------------------------------------------------------ #


def _make_hadamard(dim=64, bits=3, seed=42):
    tq = TurboQuantPGVector(dim=dim, bits=bits, seed=seed, rotation="hadamard")
    rng = np.random.default_rng(0)
    return tq, tq.compress_embedding(rng.standard_normal(dim).astype(np.float32))


def test_qr_stays_v1_byte_identical():
    # The default rotation must still serialize as the 20-byte v1 header.
    _, ce = _make(dim=64, bits=3)
    blob = pack(ce, seed=42)
    assert blob[4] == VERSION
    assert len(blob) == HEADER_SIZE + len(ce.packed_bytes)


def test_hadamard_roundtrip_self_describing():
    tq, ce = _make_hadamard()
    blob = pack(ce, seed=tq.seed)
    assert blob[4] == VERSION_ROT
    assert record_size(blob) == len(blob) == HEADER_SIZE_V2 + len(ce.packed_bytes)
    ce2, seed = unpack(blob)
    assert seed == tq.seed
    assert ce2.rotation == "hadamard"
    assert bytes(ce2.packed_bytes) == bytes(ce.packed_bytes)
    # decode is exact once rotation is carried through
    np.testing.assert_allclose(
        tq.decompress_embedding(ce2), tq.decompress_embedding(ce), atol=1e-5
    )


def test_v1_unpack_defaults_qr():
    _, ce = _make()
    ce2, _ = unpack(pack(ce))
    assert ce2.rotation == "qr"


def test_pack_uses_records_own_seed():
    # pack(ce) with no explicit seed must stamp ce.seed, not a hardcoded 42, so
    # the stored seed can never drift from the record it decodes.
    tq, ce = _make(seed=7)
    assert ce.seed == 7
    ce2, seed = unpack(pack(ce))
    assert seed == 7
    assert ce2.seed == 7  # seed travels back on the object too


def test_explicit_seed_overrides():
    _, ce = _make(seed=7)
    _, seed = unpack(pack(ce, seed=123))
    assert seed == 123


def test_pack_rejects_bad_bits():
    from turboquant_pro.pgvector import CompressedEmbedding

    bad = CompressedEmbedding(
        packed_bytes=b"\x00", norm=1.0, dim=8, bits=5, rotation="qr", seed=42
    )
    with pytest.raises(ValueError, match="bits"):
        pack(bad)


def test_unpack_rejects_bad_bits():
    _, ce = _make()
    blob = bytearray(pack(ce))
    blob[5] = 5  # bits byte in the v1 header
    with pytest.raises(ValueError, match="bits"):
        unpack(bytes(blob))


def test_mixed_batch_roundtrip():
    # A batch mixing qr (v1) and hadamard (v2) records must parse each correctly.
    rng = np.random.default_rng(1)
    tq_q = TurboQuantPGVector(dim=64, bits=3, seed=42, rotation="qr")
    tq_h = TurboQuantPGVector(dim=64, bits=3, seed=42, rotation="hadamard")
    ces = [
        tq_q.compress_embedding(rng.standard_normal(64).astype(np.float32)),
        tq_h.compress_embedding(rng.standard_normal(64).astype(np.float32)),
        tq_q.compress_embedding(rng.standard_normal(64).astype(np.float32)),
    ]
    out = unpack_batch(pack_batch(ces))
    assert [c.rotation for c in out] == ["qr", "hadamard", "qr"]
    for a, b in zip(ces, out):
        assert bytes(a.packed_bytes) == bytes(b.packed_bytes)


def test_unknown_rotation_code_rejected():
    tq, ce = _make_hadamard()
    blob = bytearray(pack(ce, seed=tq.seed))
    blob[16] = 99  # rotation byte in the v2 header
    with pytest.raises(ValueError):
        unpack(bytes(blob))

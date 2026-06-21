"""Conformance tests for the TQE1 compressed-embedding format."""

import numpy as np
import pytest

from turboquant_pro import TurboQuantPGVector
from turboquant_pro.format import (
    HEADER_SIZE,
    MAGIC,
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

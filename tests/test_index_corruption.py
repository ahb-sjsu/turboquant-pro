"""The index container is trustworthy: corruption is detected, never silent.

Every section carries a CRC32, so a flipped byte is a clean error rather than
wrong-but-plausible vectors. Includes a deterministic single-byte-flip fuzzer
whose invariant is: opening a corrupted file either raises, or returns an index
that searches *identically* to the pristine one — it never silently returns
corrupted data.
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro import TQEIndex
from turboquant_pro.index_file import (
    IndexCorruptionError,
    read_container,
    write_container,
)


def _pristine(tmp_path):
    rng = np.random.default_rng(0)
    corpus = rng.standard_normal((400, 48)).astype(np.float32)
    idx = TQEIndex.create(corpus, output_dim=24, bits=4, seed=1)
    p = tmp_path / "idx.tqe"
    idx.save(str(p))
    return p, idx, corpus


# --------------------------------------------------------------------------- #
# Container-level                                                             #
# --------------------------------------------------------------------------- #


def test_bad_magic_raises(tmp_path):
    p = tmp_path / "x.tqe"
    write_container(str(p), 1, [("meta", b"hello")])
    data = bytearray(p.read_bytes())
    data[0] ^= 0xFF  # corrupt magic
    p.write_bytes(data)
    with pytest.raises(IndexCorruptionError):
        read_container(str(p))


def test_crc_mismatch_raises(tmp_path):
    p = tmp_path / "x.tqe"
    write_container(str(p), 1, [("data", b"abcdefghij")])
    data = bytearray(p.read_bytes())
    data[-1] ^= 0x01  # corrupt a payload byte
    p.write_bytes(data)
    with pytest.raises(IndexCorruptionError):
        read_container(str(p))


def test_truncation_raises(tmp_path):
    p = tmp_path / "x.tqe"
    write_container(str(p), 1, [("data", b"abcdefghij" * 10)])
    data = p.read_bytes()
    p.write_bytes(data[: len(data) - 20])  # chop the tail
    with pytest.raises(IndexCorruptionError):
        read_container(str(p))


def test_roundtrip_sections_preserved(tmp_path):
    p = tmp_path / "x.tqe"
    secs = [("meta", b'{"k":1}'), ("codes", bytes(range(256)))]
    write_container(str(p), 2, secs)
    version, got = read_container(str(p))
    assert version == 2
    assert got["meta"] == b'{"k":1}'
    assert got["codes"] == bytes(range(256))


def test_duplicate_section_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_container(str(tmp_path / "x.tqe"), 1, [("a", b"1"), ("a", b"2")])


# --------------------------------------------------------------------------- #
# Index-level corruption is detected                                          #
# --------------------------------------------------------------------------- #


def test_corrupt_codes_section_detected(tmp_path):
    p, _, _ = _pristine(tmp_path)
    data = bytearray(p.read_bytes())
    data[len(data) // 2] ^= 0xFF  # somewhere in the payload region
    p.write_bytes(data)
    with pytest.raises(IndexCorruptionError):
        TQEIndex.open(str(p))


# --------------------------------------------------------------------------- #
# Single-byte-flip fuzzer                                                     #
# --------------------------------------------------------------------------- #


def test_single_byte_flip_fuzz(tmp_path):
    p, pristine, corpus = _pristine(tmp_path)
    original = p.read_bytes()
    q = corpus[:20]
    base_ids, _ = pristine.search(q, k=10)

    rng = np.random.default_rng(1234)
    detected = intact = 0
    for _ in range(200):
        pos = int(rng.integers(0, len(original)))
        flip = int(rng.integers(1, 256))
        data = bytearray(original)
        data[pos] ^= flip
        p.write_bytes(data)
        try:
            idx = TQEIndex.open(str(p))
        except (IndexCorruptionError, ValueError):
            detected += 1  # corruption caught cleanly
            continue
        # Opened without error: the sections were intact (CRC passed), so the
        # index MUST behave identically — never silently corrupted.
        got, _ = idx.search(q, k=10)
        np.testing.assert_array_equal(got, base_ids)
        intact += 1

    # Sanity: the vast majority of flips land in CRC-covered payload/header and
    # are caught; a few hit unused/degenerate bytes and open cleanly (asserted
    # identical above). Neither branch may crash with an unexpected error.
    assert detected > 0 and detected + intact == 200

"""TQPW, the packed-weights format (turboquant_pro.packed_weights). numpy only.

Pins: the payload is exactly the plan's stored bits; decoding is one float32 rounding
of the exact ``r * step + lo``; a round trip is lossless; corruption, truncation, a
foreign container and an unknown version are refused; the writer's guards hold.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from turboquant_pro import packed_weights as PW
from turboquant_pro import weight_plan as W
from turboquant_pro.index_file import write_container


def _matrix(seed, out=6, inp=256, bits=3, name="m"):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 2**bits, size=(out, inp))
    lo = rng.normal(0, 0.05, size=(out, inp // 128)).astype(np.float32)
    step = rng.uniform(1e-3, 2e-2, size=(out, inp // 128)).astype(np.float32)
    return codes, lo, step, PW.pack_matrix(name, codes, lo, step, bits)


@pytest.mark.parametrize("bits", range(1, 9))
def test_the_payload_is_exactly_the_plans_stored_bits(bits):
    _, _, _, pm = _matrix(bits, bits=bits)
    assert pm.payload_bits == W.stored_bits(6 * 256, bits)


@pytest.mark.parametrize("bits", [2, 3, 5, 8])
def test_decode_is_one_rounding_of_the_exact_value(bits):
    codes, _, _, pm = _matrix(10 + bits, bits=bits)
    g = pm.grid.astype(np.float64)
    r = codes.reshape(6, 2, 128).astype(np.float64)
    exact = (r * g[..., 1:2] + g[..., 0:1]).reshape(6, 256)  # exact in float64
    assert np.array_equal(PW.decode(pm), exact.astype(np.float32))


def test_round_trip_is_lossless_and_the_file_is_payload_plus_container(tmp_path):
    ms = [
        _matrix(s, out=4 + s, inp=128 * (1 + s % 3), bits=2 + s, name=f"m{s}")[3]
        for s in range(5)
    ]
    path = tmp_path / "w.tqpw"
    size = PW.write(str(path), ms, {"codec": "gptq", "plan_sha256": "ab" * 32})
    meta, back = PW.read(str(path))
    assert meta["format"] == PW.FORMAT and meta["codec"] == "gptq"
    for a, b in zip(ms, back):
        assert (a.name, a.shape, a.bits) == (b.name, b.shape, b.bits)
        assert np.array_equal(a.codes, b.codes)
        assert np.array_equal(PW.decode(a), PW.decode(b))
    payload = sum(m.payload_bits for m in ms) // 8
    meta_len = len(json.dumps(meta, sort_keys=True, separators=(",", ":")))
    assert size == payload + 12 + 56 * (1 + 2 * len(ms)) + meta_len
    assert path.read_bytes()[:4] == b"TQPW"


def test_corruption_truncation_foreign_and_unknown_versions_are_refused(tmp_path):
    path = tmp_path / "w.tqpw"
    PW.write(str(path), [_matrix(1)[3]])
    blob = path.read_bytes()
    for i in (len(blob) - 1, len(blob) - 200):  # a grid byte, a code byte
        bad = bytearray(blob)
        bad[i] ^= 0x01
        (tmp_path / "bad.tqpw").write_bytes(bytes(bad))
        with pytest.raises(PW.IndexCorruptionError, match="CRC"):
            PW.read(str(tmp_path / "bad.tqpw"))
    (tmp_path / "short.tqpw").write_bytes(blob[:-10])
    with pytest.raises(PW.IndexCorruptionError, match="truncated"):
        PW.read(str(tmp_path / "short.tqpw"))
    write_container(str(tmp_path / "ix"), 1, [("meta", b"{}")])  # a TQIX file
    with pytest.raises(PW.IndexCorruptionError, match="magic"):
        PW.read(str(tmp_path / "ix"))
    v2 = bytearray(blob)
    v2[4:6] = struct.pack("<H", 2)
    (tmp_path / "v2.tqpw").write_bytes(bytes(v2))
    with pytest.raises(PW.IndexCorruptionError, match="version 2"):
        PW.read(str(tmp_path / "v2.tqpw"))


def test_the_writer_refuses_what_it_cannot_store():
    codes, lo, step, _ = _matrix(2)
    with pytest.raises(ValueError, match="fit"):
        PW.pack_matrix("m", codes + 8, lo, step, 3)
    with pytest.raises(ValueError, match="multiple"):
        PW.pack_matrix("m", codes[:, :200], lo, step, 3)
    with pytest.raises(ValueError, match="1..8"):
        PW.pack_matrix("m", codes, lo, step, 9)
    with pytest.raises(ValueError, match="finite"):
        PW.pack_matrix("m", codes, lo + 1e6, step, 3)
    with pytest.raises(ValueError, match="underflows"):
        PW.pack_matrix("m", codes, lo, step * 0 + 1e-9, 3)
    with pytest.raises(ValueError, match="duplicate"):
        m = PW.pack_matrix("m", codes, lo, step, 3)
        PW.write("unused.tqpw", [m, m])


def test_grid_rounding_is_zero_on_a_float16_grid_and_bounded_otherwise():
    codes, lo, step, pm = _matrix(3, bits=4)
    lo16, st16 = lo.astype(np.float16), step.astype(np.float16)
    exact = PW.pack_matrix("m", codes, lo16, st16, 4)
    rebuilt = PW.decode(exact)
    assert PW.grid_rounding(exact, rebuilt, st16)["max_abs"] == 0.0
    # The codec's float32 output against the stored float16 grid: each weight moves by
    # at most the rounding of lo plus r times the rounding of step (half an ulp each).
    r = codes.reshape(6, 2, 128).astype(np.float32)
    ref = (r * step[..., None] + lo[..., None]).reshape(6, 256)
    d = np.abs(PW.decode(pm) - ref).reshape(6, 2, 128)
    # (plus half the float16 subnormal spacing, where a small lo's error is absolute)
    bound = 2.0**-11 * (np.abs(lo)[..., None] + r * step[..., None]) * 1.001
    bound += 2.0**-24
    assert (d <= bound).all()
    got = PW.grid_rounding(pm, ref, step)
    assert got["max_abs"] == pytest.approx(float(d.max()))
    assert got["max_steps"] == pytest.approx(float((d / step[..., None]).max()))

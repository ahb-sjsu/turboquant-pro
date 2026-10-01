"""The scalar codebooks: the legacy table is frozen, the Lloyd-Max table is what it
claims, and a non-legacy codebook is always declared where the codes are stored."""

import importlib.util
import json
import math
import pathlib
import sys

import numpy as np
import pytest

from turboquant_pro import codebooks as cbm
from turboquant_pro.format import HEADER_SIZE, HEADER_SIZE_V3, pack, unpack
from turboquant_pro.index import CODEBOOK_VERSION, TQEIndex, _codebook_from_meta
from turboquant_pro.index_file import read_container
from turboquant_pro.pgvector import TurboQuantPGVector

# ------------------------------------------------------------- Gaussian helpers


def _pdf(x: float) -> float:
    return 0.0 if math.isinf(x) else math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _cells(c):
    edges = [-math.inf, *((a + b) / 2 for a, b in zip(c[:-1], c[1:])), math.inf]
    return list(zip(edges[:-1], edges[1:]))


def _lloyd_step(c):
    """One Lloyd iteration for N(0, 1): each centroid -> its cell's mean."""
    return [(_pdf(lo) - _pdf(hi)) / (_cdf(hi) - _cdf(lo)) for lo, hi in _cells(c)]


def _mse(c):
    """Exact N(0, 1) mean-square error of nearest-centroid quantization."""
    total = 0.0
    for (lo, hi), m in zip(_cells(c), c):
        p = _cdf(hi) - _cdf(lo)
        # E[X^2 1{lo<X<hi}] = p + lo*pdf(lo) - hi*pdf(hi), E[X 1{...}] = pdf(lo)-pdf(hi)
        ex2 = p + (0.0 if math.isinf(lo) else lo * _pdf(lo))
        ex2 -= 0.0 if math.isinf(hi) else hi * _pdf(hi)
        ex1 = _pdf(lo) - _pdf(hi)
        total += ex2 - 2 * m * ex1 + m * m * p
    return total


# ------------------------------------------------------------------ the tables


def test_legacy_tables_are_frozen():
    # What every stored legacy code decodes to. Never edit these values.
    assert cbm.codebook(2).tolist() == [-1.510, -0.453, 0.453, 1.510]
    assert cbm.codebook(3).tolist() == [
        -1.748, -1.050, -0.500, -0.069, 0.069, 0.500, 1.050, 1.748,
    ]  # fmt: skip
    assert cbm.codebook(4).tolist() == [
        -2.401, -1.844, -1.437, -1.099, -0.800, -0.524, -0.262, -0.066,
        0.066, 0.262, 0.524, 0.800, 1.099, 1.437, 1.844, 2.401,
    ]  # fmt: skip


def test_every_copy_of_the_legacy_table_agrees():
    from turboquant_pro import core, pgvector

    path = pathlib.Path(__file__).parents[1] / "contrib" / "tqe1_reader.py"
    spec = importlib.util.spec_from_file_location("tqe1_reader_cb", path)
    reader = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reader
    try:
        spec.loader.exec_module(reader)
    finally:
        sys.modules.pop(spec.name, None)
    for b in (2, 3, 4):
        legacy = cbm.codebook(b, "legacy")
        np.testing.assert_array_equal(core._CODEBOOKS[b], legacy)
        np.testing.assert_array_equal(pgvector._CODEBOOKS[b], legacy)
        np.testing.assert_array_equal(reader._CODEBOOKS["legacy"][b], legacy)
        np.testing.assert_array_equal(
            reader._CODEBOOKS["lloyd-max"][b], cbm.codebook(b, "lloyd-max")
        )


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_lloyd_max_table_is_a_fixed_point_of_lloyds_iteration(bits):
    c = cbm.codebook(bits, "lloyd-max").tolist()
    # Six-decimal rounding: one step may move a level by at most a few 1e-7.
    assert max(abs(a - b) for a, b in zip(_lloyd_step(c), c)) < 2e-6


@pytest.mark.parametrize(
    "bits,legacy_mse,lloyd_mse",
    [(2, 0.1174819, 0.1174818), (3, 0.0455557, 0.0345478), (4, 0.0108027, 0.0095010)],
)
def test_documented_mse_values(bits, legacy_mse, lloyd_mse):
    assert _mse(cbm.codebook(bits, "legacy").tolist()) == pytest.approx(
        legacy_mse, abs=2e-7
    )
    assert _mse(cbm.codebook(bits, "lloyd-max").tolist()) == pytest.approx(
        lloyd_mse, abs=2e-7
    )


def test_legacy_table_is_not_lloyd_max_at_3_and_4_bits():
    for b in (3, 4):
        legacy = cbm.codebook(b, "legacy").tolist()
        assert _mse(legacy) > 1.1 * _mse(cbm.codebook(b, "lloyd-max").tolist())
        assert max(abs(a - b) for a, b in zip(_lloyd_step(legacy), legacy)) > 0.05


def test_unknown_codebook_is_rejected():
    with pytest.raises(ValueError, match="codebook"):
        cbm.codebook(3, "nf4")
    with pytest.raises(ValueError, match="codebook"):
        TurboQuantPGVector(dim=32, bits=3, codebook="nf4")


def test_lloyd_max_lowers_reconstruction_error_on_rotated_unit_vectors():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 128)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    for b in (3, 4):
        err = {}
        for name in ("legacy", "lloyd-max"):
            q = TurboQuantPGVector(dim=128, bits=b, seed=5, codebook=name)
            xh = q.decompress_batch(q.compress_batch(x))
            err[name] = float(np.mean(np.sum((x - xh) ** 2, axis=1)))
        assert err["lloyd-max"] < 0.92 * err["legacy"], (b, err)


# ------------------------------------------------------- declared where stored


def test_tqe_record_declares_a_non_legacy_codebook():
    v = np.random.default_rng(1).standard_normal(32).astype(np.float32)
    legacy = TurboQuantPGVector(dim=32, bits=3, seed=3)
    lm = TurboQuantPGVector(dim=32, bits=3, seed=3, codebook="lloyd-max")

    rec_legacy = pack(legacy.compress_embedding(v))
    assert rec_legacy[4] == 1 and len(rec_legacy) == HEADER_SIZE + 12  # v1, unchanged

    rec = pack(lm.compress_embedding(v))
    assert rec[4] == 3 and rec[17] == 1 and len(rec) == HEADER_SIZE_V3 + 12
    ce, _ = unpack(rec)
    assert ce.codebook == "lloyd-max"
    np.testing.assert_array_equal(
        lm.decompress_embedding(ce),
        lm.decompress_embedding(lm.compress_embedding(v)),
    )


def test_decode_against_the_wrong_codebook_is_refused():
    v = np.random.default_rng(2).standard_normal(32).astype(np.float32)
    lm = TurboQuantPGVector(dim=32, bits=4, seed=3, codebook="lloyd-max")
    legacy = TurboQuantPGVector(dim=32, bits=4, seed=3)
    ce, _ = unpack(pack(lm.compress_embedding(v)))
    with pytest.raises(ValueError, match="codebook mismatch"):
        legacy.decompress_embedding(ce)
    ce_legacy, _ = unpack(pack(legacy.compress_embedding(v)))
    with pytest.raises(ValueError, match="codebook mismatch"):
        lm.decompress_embedding(ce_legacy)


def test_v3_record_with_unknown_codebook_byte_is_rejected():
    v = np.random.default_rng(3).standard_normal(32).astype(np.float32)
    lm = TurboQuantPGVector(dim=32, bits=3, seed=3, codebook="lloyd-max")
    rec = bytearray(pack(lm.compress_embedding(v)))
    rec[17] = 9
    with pytest.raises(ValueError, match="codebook"):
        unpack(bytes(rec))


def test_index_with_lloyd_max_is_written_as_v4_and_round_trips(tmp_path):
    corpus = np.random.default_rng(4).standard_normal((300, 48)).astype(np.float32)
    idx = TQEIndex.create(corpus, output_dim=32, bits=4, seed=1, codebook="lloyd-max")
    path = str(tmp_path / "lm.tqe")
    idx.save(path)
    version, sections = read_container(path)
    assert version == CODEBOOK_VERSION
    assert json.loads(sections["meta"])["quant"]["codebook"] == "lloyd-max"

    back = TQEIndex.open(path)
    assert back.stats()["codebook"] == "lloyd-max"
    ids_a, _ = idx.search(corpus[:5], k=5)
    ids_b, _ = back.search(corpus[:5], k=5)
    np.testing.assert_array_equal(ids_a, ids_b)
    mm = TQEIndex.open(path, mmap=True)
    ids_c, _ = mm.search(corpus[:5], k=5)
    np.testing.assert_array_equal(ids_a, ids_c)


def test_legacy_index_stays_v3_with_unchanged_metadata(tmp_path):
    corpus = np.random.default_rng(5).standard_normal((200, 48)).astype(np.float32)
    idx = TQEIndex.create(corpus, output_dim=32, bits=3, seed=1)
    path = str(tmp_path / "legacy.tqe")
    idx.save(path)
    version, sections = read_container(path)
    assert version == 3
    assert "codebook" not in json.loads(sections["meta"])["quant"]
    assert TQEIndex.open(path).stats()["codebook"] == "legacy"


def test_codebook_in_a_pre_v4_index_is_refused():
    meta = {"format_version": 3, "quant": {"codebook": "lloyd-max"}}
    with pytest.raises(ValueError, match="requires version 4"):
        _codebook_from_meta(meta)
    assert _codebook_from_meta({"format_version": 3, "quant": {}}) == "legacy"

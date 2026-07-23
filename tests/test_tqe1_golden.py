"""TQE1 golden-corpus conformance (ROADMAP_2.0 Pillar 2).

Three legs: (1) the committed golden files are immutable (sha256 pinned in the
manifest — a writer change that alters bytes is a format break and must fail
here); (2) the in-tree implementation decodes them to the committed expected
tensors; (3) the dependency-free single-file reader (``contrib/tqe1_reader.py``,
imported WITHOUT turboquant-pro) reproduces the same decode from the bytes
alone — the portability claim, tested rather than asserted.
"""

import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

GOLD = pathlib.Path(__file__).parent / "golden" / "tqe1"
MANIFEST = json.loads((GOLD / "manifest.json").read_text())


def _load_standalone_reader():
    path = pathlib.Path(__file__).parents[1] / "contrib" / "tqe1_reader.py"
    src = path.read_text(encoding="utf-8")
    assert (
        "import turboquant" not in src and "from turboquant" not in src
    ), "the standalone reader must not import turboquant-pro"
    spec = importlib.util.spec_from_file_location("tqe1_reader_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules[__name__];
    # register before exec or @dataclass fails under `from __future__ import
    # annotations`.
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    return mod


@pytest.mark.parametrize("name", sorted(MANIFEST["files"]))
def test_golden_files_are_immutable(name):
    meta = MANIFEST["files"][name]
    blob = (GOLD / f"{name}.tqe").read_bytes()
    assert len(blob) == meta["bytes"]
    assert hashlib.sha256(blob).hexdigest() == meta["sha256"], (
        f"{name}.tqe changed on disk — either corruption or a writer change "
        "that breaks format stability; both are release blockers"
    )


@pytest.mark.parametrize("name", sorted(MANIFEST["files"]))
def test_in_tree_decode_matches_expected(name):
    from turboquant_pro.format import unpack_batch
    from turboquant_pro.pgvector import TurboQuantPGVector

    meta = MANIFEST["files"][name]
    ces = unpack_batch((GOLD / f"{name}.tqe").read_bytes())
    assert len(ces) == MANIFEST["n"]
    q = TurboQuantPGVector(
        dim=MANIFEST["dim"],
        bits=meta["bits"],
        seed=meta["seed"],
        rotation=meta["rotation"],
    )
    got = np.stack([q.decompress_embedding(ce) for ce in ces])
    want = np.load(GOLD / "expected.npz")[name]
    # Cross-platform BLAS may differ in the last float ulps of the QR; the
    # decode contract is numerical, not bitwise, across platforms.
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", sorted(MANIFEST["files"]))
def test_standalone_reader_matches_expected(name):
    reader = _load_standalone_reader()
    meta = MANIFEST["files"][name]
    recs = reader.read_records((GOLD / f"{name}.tqe").read_bytes())
    assert len(recs) == MANIFEST["n"]
    for r in recs:
        assert r.version == meta["version"]
        assert r.bits == meta["bits"]
        assert r.dim == MANIFEST["dim"]
        assert r.seed == meta["seed"]
        assert r.rotation == meta["rotation"]
    got = np.stack([reader.decode(r) for r in recs])
    want = np.load(GOLD / "expected.npz")[name]
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_standalone_reader_rejects_malformed():
    reader = _load_standalone_reader()
    blob = (GOLD / "qr_b3.tqe").read_bytes()
    with pytest.raises(ValueError, match="magic"):
        reader.parse_record(b"NOPE" + blob[4:])
    bad_version = blob[:4] + bytes([99]) + blob[5:]
    with pytest.raises(ValueError, match="version"):
        reader.parse_record(bad_version)
    with pytest.raises(ValueError, match="truncated|too small"):
        reader.parse_record(blob[:24])


def test_standalone_decode_file_roundtrip(tmp_path):
    reader = _load_standalone_reader()
    out = reader.decode_file(str(GOLD / "hadamard_b3.tqe"))
    assert out.shape == (MANIFEST["n"], MANIFEST["dim"])
    want = np.load(GOLD / "expected.npz")["hadamard_b3"]
    np.testing.assert_allclose(out, want, rtol=1e-4, atol=1e-5)

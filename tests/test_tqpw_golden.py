"""TQPW golden-corpus conformance.

Three legs, as for TQE1: the committed file is immutable (sha256 in the manifest), the
in-tree reader decodes it to the committed tensors and the in-tree writer reproduces
its bytes, and the dependency-free ``contrib/tqpw_reader.py`` (imported without
turboquant-pro) decodes it to the same tensors from the spec alone.
"""

import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np

from turboquant_pro import packed_weights as PW

GOLD = pathlib.Path(__file__).parent / "golden" / "tqpw"
MANIFEST = json.loads((GOLD / "manifest.json").read_text())


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    return mod


def test_the_golden_file_is_immutable():
    meta = MANIFEST["files"]["golden"]
    blob = (GOLD / "golden.tqpw").read_bytes()
    assert len(blob) == meta["bytes"]
    assert (
        hashlib.sha256(blob).hexdigest() == meta["sha256"]
    ), "golden.tqpw changed on disk: corruption or a format break"


def test_the_in_tree_reader_decodes_and_the_writer_reproduces_it(tmp_path):
    expected = np.load(GOLD / "expected.npz")
    meta, ms = PW.read(str(GOLD / "golden.tqpw"))
    assert meta["format"] == PW.FORMAT
    assert sorted(m.name for m in ms) == sorted(expected.files)
    for m in ms:
        assert np.array_equal(PW.decode(m), expected[m.name]), m.name
    gen = _load(GOLD / "generate.py", "tqpw_generate")
    out = tmp_path / "again.tqpw"
    PW.write(str(out), gen.build(), gen.META)
    assert out.read_bytes() == (GOLD / "golden.tqpw").read_bytes()


def test_the_standalone_reader_decodes_it_from_the_spec_alone():
    path = pathlib.Path(__file__).parents[1] / "contrib" / "tqpw_reader.py"
    src = path.read_text(encoding="utf-8")
    assert "import turboquant" not in src and "from turboquant" not in src
    reader = _load(path, "tqpw_reader_standalone")
    expected = np.load(GOLD / "expected.npz")
    meta, weights = reader.read(str(GOLD / "golden.tqpw"))
    assert meta["format"] == "tqp.packed_weights/1"
    assert sorted(weights) == sorted(expected.files)
    for name, w in weights.items():
        assert np.array_equal(w, expected[name]), name

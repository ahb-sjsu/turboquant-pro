"""part5_preload and part5_cache_fast: the verified fp16 copy kept on a model's set.

The preload round trip runs on temporary directories standing in for the set's volumes.
The fast cache runs end to end on CPU with the smoke's model, as test_part5_cache_fp16
does, and skips without the checkpoint in the local cache.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "kvquant_matrix")
)
import part5_preload as PL  # noqa: E402

KEY = "m"


def _model16(tmp_path):
    src = tmp_path / "model16"
    src.mkdir()
    (src / "model-00001-of-00002.safetensors").write_bytes(os.urandom(3000))
    (src / "model-00002-of-00002.safetensors").write_bytes(os.urandom(2000))
    (src / "config.json").write_text("{}")
    (src / "tokenizer.json").write_text("{}")
    vols = [tmp_path / f"w{i}" for i in range(2)]
    for v in vols:
        v.mkdir()
    return src, ",".join(str(v) for v in vols)


def test_place_puts_the_largest_file_on_the_roomiest_volume():
    where = PL.place({"a": 5, "b": 4, "c": 1}, {"/w0": 6, "/w1": 5})
    assert where == {"a": "/w0", "b": "/w1", "c": "/w0"}
    with pytest.raises(SystemExit):
        PL.place({"a": 9}, {"/w0": 6, "/w1": 5})


def test_a_preload_links_back_byte_for_byte(tmp_path):
    src, vols = _model16(tmp_path)
    dest = tmp_path / "link"
    assert PL.main(["--link", str(dest), "--vols", vols, "--key", KEY]) == 3  # none yet
    assert (
        PL.main(["--src", str(src), "--vols", vols, "--key", KEY, "--reserve", "0"])
        == 0
    )
    rec = json.loads((tmp_path / "w0" / "model16" / KEY / PL.READY).read_text())[
        "files"
    ]
    assert set(rec) == set(os.listdir(src))
    spread = {e["vol"] for e in rec.values()}  # the two weight files on both volumes
    assert spread == set(vols.split(","))
    assert rec["config.json"]["vol"].endswith("w0")
    assert PL.main(["--link", str(dest), "--vols", vols, "--key", KEY]) == 0
    for n in os.listdir(src):
        assert (dest / n).is_symlink()
        assert (dest / n).read_bytes() == (src / n).read_bytes()


def test_a_short_copy_on_the_set_is_rebuilt_not_linked(tmp_path):
    src, vols = _model16(tmp_path)
    PL.main(["--src", str(src), "--vols", vols, "--key", KEY, "--reserve", "0"])
    files = PL.ready_files(vols.split(","), KEY)
    victim = files["model-00001-of-00002.safetensors"]
    with open(victim, "r+b") as f:
        f.truncate(10)
    assert PL.main(["--link", str(tmp_path / "x"), "--vols", vols, "--key", KEY]) == 3


def test_no_ready_marker_without_a_complete_copy(tmp_path, monkeypatch):
    src, vols = _model16(tmp_path)
    real = PL.sha256
    calls = []

    def bad(path):  # the copy on the set reads back wrong
        calls.append(path)
        return "0" * 64 if len(calls) == 2 else real(path)

    monkeypatch.setattr(PL, "sha256", bad)
    with pytest.raises(SystemExit):
        PL.main(["--src", str(src), "--vols", vols, "--key", KEY, "--reserve", "0"])
    assert PL.ready_files(vols.split(","), KEY) is None


def test_fast_cache_matches_the_loaded_model_on_cpu(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import part5_cache_fast as F

    try:
        from huggingface_hub import snapshot_download

        src = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    except Exception as e:  # not cached here
        pytest.skip(f"checkpoint not cached: {e!r}")
    dest = tmp_path / "fp16"
    assert F.main(["--src", src, "--dest", str(dest), "--device", "cpu"]) == 0
    model = F.C.load(str(dest), "cpu")
    assert F.check(model, str(dest), "cpu") is None
    name = next(iter(model.state_dict()))
    with __import__("torch").no_grad():
        model.state_dict()[name].view(-1)[0] += 1  # one changed element is caught
    assert F.check(model, str(dest), "cpu") is not None

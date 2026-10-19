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
    """A two-file fp16 copy as save_pretrained writes it: tensors in key order, the
    first layers in one file and the rest in the other."""
    np = pytest.importorskip("numpy")
    st = pytest.importorskip("safetensors.numpy")
    src = tmp_path / "model16"
    src.mkdir()
    rng = np.random.default_rng(0)
    names = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"] + [
        f"model.layers.{i}.{p}.weight" for i in range(12) for p in ("q", "k", "v", "o")
    ]
    ts = {
        n: rng.standard_normal((8, 16 + i)).astype(np.float16)
        for i, n in enumerate(names)
    }
    keys = sorted(ts, key=PL.natural_key)
    first, second = keys[: len(keys) // 2], keys[len(keys) // 2 :]
    st.save_file(
        {k: ts[k] for k in first},
        str(src / "model-00001-of-00002.safetensors"),
        metadata={"format": "pt"},
    )
    st.save_file(
        {k: ts[k] for k in second},
        str(src / "model-00002-of-00002.safetensors"),
        metadata={"format": "pt"},
    )
    (src / PL.INDEX).write_text(
        json.dumps(
            {
                "weight_map": {
                    **{k: "model-00001-of-00002.safetensors" for k in first},
                    **{k: "model-00002-of-00002.safetensors" for k in second},
                }
            }
        )
    )
    (src / "config.json").write_text("{}")
    (src / "tokenizer.json").write_text("{}")
    vols = [tmp_path / f"w{i}" for i in range(3)]
    for v in vols:
        v.mkdir()
    return src, ",".join(str(v) for v in vols), ts


def _load(d):
    import safetensors.numpy as st

    idx = json.loads((d / PL.INDEX).read_text())["weight_map"]
    out = {}
    for f in set(idx.values()):
        out.update(st.load_file(str(d / f)))
    assert set(out) == set(idx) and all(idx[k] in os.listdir(d) for k in out)
    return out


def test_stripe_deals_neighbouring_tensors_onto_different_volumes():
    ts = [(f"t{i}", "f", "F16", [4], 10 * i, 10 * i + 10) for i in range(9)]
    parts = PL.stripe(ts, [100, 100, 100])
    assert [[t[0] for t in p] for p in parts] == [
        ["t0", "t3", "t6"],
        ["t1", "t4", "t7"],
        ["t2", "t5", "t8"],
    ]
    # each volume fills in proportion to its room
    parts = PL.stripe(ts, [200, 100])
    assert [len(p) for p in parts] == [6, 3]
    assert PL.natural_key("model.layers.10.q") > PL.natural_key("model.layers.9.q")


def test_a_striped_copy_links_back_tensor_for_tensor(tmp_path):
    src, vols, ts = _model16(tmp_path)
    dest = tmp_path / "link"
    assert PL.main(["--link", str(dest), "--vols", vols, "--key", KEY]) == 3  # none yet
    assert PL.main(["--src", str(src), "--vols", vols, "--key", KEY]) == 0
    rec = json.loads((tmp_path / "w0" / PL.DIR / KEY / PL.READY).read_text())
    assert rec["layout"] == "striped"
    weights = [n for n in rec["files"] if n.endswith(".safetensors")]
    assert len(weights) == 3 and {rec["files"][n]["vol"] for n in weights} == set(
        vols.split(",")
    )
    assert rec["files"]["config.json"]["vol"].endswith("w0")
    assert PL.main(["--link", str(dest), "--vols", vols, "--key", KEY]) == 0
    got = _load(dest)
    assert set(got) == set(ts)
    for k, v in ts.items():
        assert got[k].dtype == v.dtype and got[k].tobytes() == v.tobytes()
    # in load order, neighbouring tensors are in different files
    idx = json.loads((dest / PL.INDEX).read_text())["weight_map"]
    order = sorted(idx, key=PL.natural_key)
    assert all(idx[a] != idx[b] for a, b in zip(order, order[1:]))


def test_bench_reads_the_whole_copy_both_ways(tmp_path, capsys):
    src, vols, ts = _model16(tmp_path)
    assert PL.main(["--src", str(src), "--vols", vols, "--key", KEY]) == 0
    capsys.readouterr()
    assert PL.main(["--bench", "--vols", vols, "--key", KEY]) == 0
    out = capsys.readouterr().out
    total = sum(v.nbytes for v in ts.values()) / 1e9
    assert "(model16s, 3 files, 4 threads) mmap" in out and ") pread" in out
    assert out.count(f"{total:.2f} GB") == 2


def test_restripe_builds_from_the_sets_whole_file_copy_and_link_prefers_it(tmp_path):
    src, vols, ts = _model16(tmp_path)
    v = vols.split(",")
    old = tmp_path / "w0" / PL.OLD / KEY
    old.mkdir(parents=True)
    files = {}
    for n in os.listdir(src):
        (old / n).write_bytes((src / n).read_bytes())
        files[n] = {"vol": v[0], "bytes": (src / n).stat().st_size, "sha256": "-"}
    (old / PL.READY).write_text(json.dumps({"key": KEY, "files": files}))
    assert PL.main(["--link", str(tmp_path / "a"), "--vols", vols, "--key", KEY]) == 0
    assert (tmp_path / "a" / "model-00001-of-00002.safetensors").exists()  # the old one
    assert PL.main(["--restripe", "--vols", vols, "--key", KEY]) == 0
    assert PL.main(["--link", str(tmp_path / "b"), "--vols", vols, "--key", KEY]) == 0
    assert not (tmp_path / "b" / "model-00001-of-00002.safetensors").exists()
    got = _load(tmp_path / "b")
    assert all(got[k].tobytes() == ts[k].tobytes() for k in ts)


def test_a_short_copy_on_the_set_is_rebuilt_not_linked(tmp_path):
    src, vols, _ = _model16(tmp_path)
    PL.main(["--src", str(src), "--vols", vols, "--key", KEY])
    files = PL.ready_files(vols.split(","), KEY)
    victim = files["model-s01-of-03.safetensors"]
    with open(victim, "r+b") as f:
        f.truncate(10)
    assert PL.main(["--link", str(tmp_path / "x"), "--vols", vols, "--key", KEY]) == 3


def test_no_ready_marker_without_a_complete_copy(tmp_path, monkeypatch):
    src, vols, _ = _model16(tmp_path)
    monkeypatch.setattr(PL, "sha256", lambda path: "0" * 64)  # every read-back is wrong
    with pytest.raises(SystemExit):
        PL.main(["--src", str(src), "--vols", vols, "--key", KEY])
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

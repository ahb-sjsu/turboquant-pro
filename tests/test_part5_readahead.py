"""The run pods' read-ahead (part5_readahead.py): it reads the copy's tensor bytes in
the order part5_preload records as the loader's, stays inside its window of the load,
acts only in a cell process given a copy, and a venv's .pth starts it in a cell."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

KVDIR = Path(__file__).resolve().parents[1] / "benchmarks" / "kvquant_matrix"
sys.path.insert(0, str(KVDIR))

import part5_preload as PL  # noqa: E402
import part5_readahead as RA  # noqa: E402


def write_copy(d: Path, files: dict) -> None:
    """A safetensors copy: {file: [(tensor, nbytes)]} and its index."""
    d.mkdir(parents=True, exist_ok=True)
    wmap = {}
    for fname, ts in files.items():
        head, off = {"__metadata__": {"format": "pt"}}, 0
        for t, n in ts:
            head[t] = {
                "dtype": "F16",
                "shape": [n // 2],
                "data_offsets": [off, off + n],
            }
            off += n
            wmap[t] = fname
        raw = json.dumps(head).encode()
        raw += b" " * (-len(raw) % 8)
        with open(d / fname, "wb") as f:
            f.write(struct.pack("<Q", len(raw)) + raw + os.urandom(off))
    (d / RA.INDEX).write_text(json.dumps({"metadata": {}, "weight_map": wmap}))


LAYERS = {
    "a.safetensors": [(f"model.layers.{i}.mlp.weight", 1000) for i in (0, 2, 10)],
    "b.safetensors": [(f"model.layers.{i}.attn.weight", 1000) for i in (1, 9, 11)]
    + [("model.embed_tokens.weight", 3000)],
}


def test_pieces_follow_the_loaders_order_and_cover_every_tensor(tmp_path):
    write_copy(tmp_path, LAYERS)
    got = RA.pieces(str(tmp_path))
    files = sorted(str(tmp_path / f) for f in LAYERS)
    want = [(t[1], t[4], t[5]) for t in PL.tensors(files)]  # part5_preload's order
    assert got == want
    assert sum(e - b for _, b, e in got) == 9000
    # numbers compare as numbers: layer 2 before layer 10
    names = [n for n, *_ in PL.tensors(files)]
    assert names.index("model.layers.2.mlp.weight") < names.index(
        "model.layers.10.mlp.weight"
    )


def test_a_large_tensor_is_cut_into_pieces(tmp_path, monkeypatch):
    monkeypatch.setattr(RA, "PIECE", 1024)
    write_copy(tmp_path, {"a.safetensors": [("w", 3000)]})
    p1, p2, p3 = RA.pieces(str(tmp_path))
    assert p1[2] - p1[1] == 1024 and p3[2] == p1[1] + 3000 and p2[1] == p1[2]


def test_a_one_file_model_without_an_index_is_read_too(tmp_path):
    write_copy(tmp_path, {"model.safetensors": LAYERS["b.safetensors"]})
    (tmp_path / RA.INDEX).unlink()
    f = str(tmp_path / "model.safetensors")
    assert RA.pieces(str(tmp_path)) == [(t[1], t[4], t[5]) for t in PL.tensors([f])]
    assert sum(e - b for _, b, e in RA.pieces(str(tmp_path))) == 6000


def test_the_read_ahead_stays_inside_its_window_of_the_load(tmp_path):
    write_copy(tmp_path, LAYERS)
    loaded = [0]
    out = tmp_path / "out"
    ra = RA.ReadAhead("keys_cell.py", str(tmp_path), str(out), 2500, lambda: loaded[0])
    ra.start()
    deadline = time.time() + 10
    while ra.read < 3000 and time.time() < deadline:
        threading.Event().wait(0.01)
    threading.Event().wait(0.3)
    # the embedding comes first in the loader's order and fills the window: it waits
    assert ra.read == 3000 and not ra.written
    loaded[0] = 4000  # the load moves on: four 1000-byte tensors, to 3000 ahead of it
    while ra.read < 7000 and time.time() < deadline:
        threading.Event().wait(0.01)
    threading.Event().wait(0.3)
    assert ra.read == 7000 and not ra.written
    loaded[0] = 10**9
    while not ra.written and time.time() < deadline:
        threading.Event().wait(0.01)
    assert ra.written and ra.read == 9000 and ra.waits > 0
    (f,) = list(out.iterdir())
    text = f.read_text()
    assert f.name == f"readahead_keys_cell.py_{os.getpid()}.txt"
    assert text.startswith(f"keys_cell.py pid {os.getpid()}: read 0.00 of 0.00 GB")
    ra.write()  # written once
    assert f.read_text() == text


def test_only_cell_processes_given_a_copy_read_ahead(tmp_path):
    write_copy(tmp_path / "m", LAYERS)
    m = str(tmp_path / "m")
    assert RA.install(["python", "keys_run.py"], {"READAHEAD_DIR": m}) is None
    assert RA.install(["python", "/x/keys_cell.py"], {}) is None
    assert (
        RA.install(["python", "/x/keys_cell.py"], {"READAHEAD_DIR": str(tmp_path)})
        is None
    )
    ra = RA.install(
        ["/venv/bin/python", "/x/wikitext_ppl.py"],
        {"READAHEAD_DIR": m, "READAHEAD_GB": "0.5"},
    )
    assert ra is not None and ra.name == "wikitext_ppl.py" and ra.window == 5 * 10**8
    deadline = time.time() + 10
    while not ra.written and time.time() < deadline:
        threading.Event().wait(0.01)
    assert ra.read == 9000  # no GPU here: the whole copy fits inside the window


def test_a_venv_reads_ahead_in_a_cell_from_interpreter_start_up(tmp_path):
    """The pod's path: the .pth in the venv's site-packages runs while site starts the
    interpreter, before the cell script, which keys_run starts as python <script>."""
    venv = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True
    )
    py = str(venv / "bin" / "python")
    sp = subprocess.check_output(
        [py, "-c", "import site; print(site.getsitepackages()[0])"], text=True
    ).strip()
    Path(sp, "part5_readahead.py").write_text(
        (KVDIR / "part5_readahead.py").read_text()
    )
    Path(sp, "part5_readahead.pth").write_text("import part5_readahead\n")
    write_copy(tmp_path / "m", LAYERS)
    cell = tmp_path / "keys_cell.py"
    cell.write_text(
        "import threading\nthreading.Event().wait(1.0)\nprint('cell done')\n"
    )
    out = tmp_path / "trace"
    env = {
        **os.environ,
        "READAHEAD_DIR": str(tmp_path / "m"),
        "LOAD_TRACE_DIR": str(out),
    }
    r = subprocess.run(
        [py, str(cell)], env=env, capture_output=True, text=True, timeout=60
    )
    assert r.returncode == 0 and r.stdout == "cell done\n" and r.stderr == "", r.stderr
    (f,) = list(out.iterdir())
    assert (
        f.name.startswith("readahead_keys_cell.py_")
        and " read 0.00 of 0.00 GB" in f.read_text()
    )

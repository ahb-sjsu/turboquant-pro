"""The head pods' load trace (part5_load_trace.py): it acts only in a cell process with
LOAD_TRACE_DIR set, reports at exit inside its window, and a real .pth install traces a
cell script started the way keys_run starts one."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

KVDIR = Path(__file__).resolve().parents[1] / "benchmarks" / "kvquant_matrix"
sys.path.insert(0, str(KVDIR))

import part5_load_trace as T  # noqa: E402


def test_only_cell_processes_with_a_trace_dir_are_traced(tmp_path):
    d = str(tmp_path / "t")
    assert T.install(["python", "keys_run.py"], {"LOAD_TRACE_DIR": d}) is None
    assert T.install(["python", "/x/keys_cell.py"], {}) is None
    assert not os.path.exists(d)
    tr = T.install(["/venv/bin/python", "/x/wikitext_ppl.py"], {"LOAD_TRACE_DIR": d})
    assert tr is not None and tr.name == "wikitext_ppl.py"
    tr.write()
    text = open(tr.path).read()
    assert text.startswith(f"wikitext_ppl.py pid {os.getpid()}:")
    assert (
        "cpus allowed" in text
        and "change over the window" in text
        and "read_gb" in text
    )
    tr.write()  # written once
    assert open(tr.path).read() == text


def test_a_pth_install_traces_a_cell_script_started_like_keys_run(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "part5_load_trace.py").write_text(
        (KVDIR / "part5_load_trace.py").read_text()
    )
    (site / "part5_load_trace.pth").write_text("import part5_load_trace\n")
    cell = tmp_path / "keys_cell.py"
    cell.write_text(
        "import threading, time\n"
        "def work():\n"
        "    t = time.time()\n"
        "    while time.time() - t < 0.6:\n"
        "        sum(range(1000))\n"
        "th = threading.Thread(target=work); th.start(); th.join()\n"
        "print('cell done')\n"
    )
    out = tmp_path / "trace"
    boot = (
        f"import site; site.addsitedir({str(site)!r}); exec(open({str(cell)!r}).read())"
    )
    # addsitedir processes the .pth as site does at start-up; argv names the cell script
    env = {**os.environ, "LOAD_TRACE_DIR": str(out)}
    r = subprocess.run(
        [sys.executable, "-c", boot, str(cell)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0 and "cell done" in r.stdout, r.stderr
    (f,) = list(out.iterdir())
    text = f.read_text()
    assert text.startswith("keys_cell.py pid ") and " samples" in text
    assert "keys_cell.py:work:" in text  # the worker thread was sampled
    # a process that is not a cell writes nothing
    out2 = tmp_path / "trace2"
    env2 = {**env, "LOAD_TRACE_DIR": str(out2)}
    r = subprocess.run(
        [sys.executable, "-c", f"import site; site.addsitedir({str(site)!r})"],
        env=env2,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0 and not out2.exists()


def test_a_venv_traces_a_cell_from_interpreter_start_up(tmp_path):
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
    Path(sp, "part5_load_trace.py").write_text(
        (KVDIR / "part5_load_trace.py").read_text()
    )
    Path(sp, "part5_load_trace.pth").write_text("import part5_load_trace\n")
    cell = tmp_path / "wikitext_ppl.py"
    cell.write_text("import time\ntime.sleep(0.3)\nprint('cell done')\n")
    out = tmp_path / "trace"
    r = subprocess.run(
        [py, str(cell)],
        env={**os.environ, "LOAD_TRACE_DIR": str(out)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0 and "cell done" in r.stdout, r.stderr
    (f,) = list(out.iterdir())
    assert f.name.startswith("wikitext_ppl.py_") and "samples" in f.read_text()

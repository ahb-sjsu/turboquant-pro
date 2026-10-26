"""pod_guard (benchmarks/kvquant_matrix/pod_guard.py): bounds Part V's GPU jobs.

A pod ends itself, so nothing outside deletes an active Job: the child's exit code
passes through, the deadline stops it (124), a GPU idle after warm-up stops it (125),
and every sample is written to the heartbeat file. The GPU reading is stubbed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "kvquant_matrix")
)
import pod_guard  # noqa: E402

PY = sys.executable


def _run(tmp_path, monkeypatch, util, child, **kw):
    monkeypatch.setattr(pod_guard, "gpu_sample", lambda: (util, 1234))
    hb = tmp_path / "hb.json"
    args = [f"--{k.replace('_', '-')}={v}" for k, v in kw.items()]
    rc = pod_guard.main([*args, f"--heartbeat={hb}", "--", PY, "-c", child])
    return rc, json.loads(hb.read_text())


def test_child_exit_code_passes_through(tmp_path, monkeypatch):
    rc, hb = _run(
        tmp_path,
        monkeypatch,
        90,
        "import sys; print('work'); sys.exit(3)",
        deadline=60,
        interval=1,
    )
    assert rc == 3
    assert hb["state"] == "exited rc=3" and hb["last"] == "work"


def test_deadline_stops_the_child(tmp_path, monkeypatch):
    rc, hb = _run(
        tmp_path,
        monkeypatch,
        90,
        "import time; print('busy'); time.sleep(60)",
        deadline=2,
        interval=1,
    )
    assert rc == 124
    assert "deadline" in hb["state"]


def test_idle_gpu_after_warmup_stops_the_child(tmp_path, monkeypatch):
    rc, hb = _run(
        tmp_path,
        monkeypatch,
        5,
        "import time; time.sleep(60)",
        deadline=60,
        interval=1,
        warmup=0,
        samples=2,
    )
    assert rc == 125
    assert hb["window_n"] == 2 and hb["window_mean"] < 40


def test_busy_gpu_is_never_stopped_for_utilization(tmp_path, monkeypatch):
    rc, _ = _run(
        tmp_path,
        monkeypatch,
        95,
        "import time; time.sleep(3)",
        deadline=60,
        interval=1,
        warmup=0,
        samples=1,
    )
    assert rc == 0


@pytest.mark.parametrize("util", [None])
def test_unreadable_gpu_never_counts_as_idle(tmp_path, monkeypatch, util):
    rc, hb = _run(
        tmp_path,
        monkeypatch,
        util,
        "import time; time.sleep(3)",
        deadline=60,
        interval=1,
        warmup=0,
        samples=1,
    )
    assert rc == 0 and hb["window_n"] == 0


def test_dips_between_cells_do_not_stop_a_busy_pod(tmp_path, monkeypatch):
    """A runner that starts one process per cell reads 0% between cells: alternating
    0 and 90 averages 45, above the floor, so the pod runs on (the third smoke was
    stopped on such dips by consecutive single readings)."""
    seq = iter([0, 90] * 50)
    monkeypatch.setattr(pod_guard, "gpu_sample", lambda: (next(seq), 1234))
    hb = tmp_path / "hb.json"
    rc = pod_guard.main(
        [
            "--deadline=60",
            "--interval=1",
            "--warmup=0",
            "--samples=4",
            f"--heartbeat={hb}",
            "--",
            PY,
            "-c",
            "import time; time.sleep(6)",
        ]
    )
    assert rc == 0
    assert 40 <= json.loads(hb.read_text())["window_mean"] <= 50

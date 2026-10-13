"""Part V's grid drives Part II's runner (keys_run.py --grid part5).

The first wiring smoke (NRP, 2026-10-13) failed in seconds because part5_grid lacked an
attribute keys_run reads (G0_TASKS). These tests read the runner's source for every
``KG.<NAME>`` it uses and require the grid to define each, and walk the runner's cell
loop for every arm without starting a process.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

KVDIR = Path(__file__).resolve().parents[1] / "benchmarks" / "kvquant_matrix"
sys.path.insert(0, str(KVDIR))

import part5_grid  # noqa: E402


def test_grid_defines_every_name_the_runner_reads():
    used = set(re.findall(r"KG\.([A-Z_0-9]+)", (KVDIR / "keys_run.py").read_text()))
    assert used, "the runner reads no grid names: the pattern is stale"
    missing = sorted(n for n in used if not hasattr(part5_grid, n))
    assert not missing, f"part5_grid lacks {missing}"


def test_every_runner_arm_has_an_env_line_and_tasks():
    sys.argv = ["keys_run.py", "--grid", "part5"]
    import importlib

    import keys_run

    keys_run = importlib.reload(keys_run)
    kg = keys_run.KG
    assert kg is part5_grid or kg.__name__ == "part5_grid"
    for arm in keys_run.all_arms():
        line = {**kg.ARMS, **kg.G0_ARMS}[arm]
        tasks = kg.G0_TASKS if arm in kg.G0_ARMS else kg.TASKS
        env = dict(kv.split("=", 1) for kv in line.split())
        assert tasks and ("NOQUANT" in env)
        if env.get("KEY_ALLOC") in ("head", "fisher"):
            assert env.get("KEY_BASIS", "native") == "native"

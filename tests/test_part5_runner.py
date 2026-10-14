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


def test_run_pods_cover_every_arm_at_full_length():
    """A registered wave's pod runs every document and chunk (no smoke limits), each arm
    of its wave, under a deadline 1.5 x its estimate, and keeps its telemetry on the
    volume; its shape is inside NRP's usage windows at the smoke's measured usage."""
    import pytest

    pytest.importorskip("nats_bursting")
    sys.path.insert(0, str(KVDIR.parent))
    import part5_nrp as N

    assert set(N.WAVE_HOURS) == set(part5_grid.WAVES)
    commit = "0" * 40
    for wave, arms in part5_grid.WAVES.items():
        s = N.run_script(commit, "qwen2.5-7b-instruct", wave)
        assert "--docs" not in s and "--ppl-chunks" not in s
        assert f"--arms {','.join(arms)} " in s
        assert f"--deadline {int(N.WAVE_HOURS[wave] * 1.5 * 3600)} " in s
        assert f"pod_{wave}.log" in s and "set -euo pipefail" in s
        bs = chr(92)  # a backslash
        assert bs + chr(10) in s and bs + bs not in s  # shell continuations survive
    mean_cpu, mean_mem, peak_mem = 0.894, 4.637, 5.632  # 7B set-smoke, guard record
    shape = N.RUN_SHAPE
    assert 0.2 <= mean_cpu / shape["cpu"] <= 2.0
    assert 0.2 <= mean_mem / shape["mem_gib"] <= 1.5 and peak_mem < shape["mem_gib"]


def test_tiers_are_placed_beside_their_volumes():
    """Tier A sets are linstor-unl volumes read by A10 pods in zone unl; Tier B sets
    are Fullerton RBD volumes read by L4 pods in zone fullerton (owner, 2026-10-13).
    Every pod of a set, GPU or CPU, is pinned to the set's zone; every model has a set.
    """
    import pytest

    pytest.importorskip("nats_bursting")
    sys.path.insert(0, str(KVDIR.parent))
    import part5_nrp as N

    assert set(N.PLACEMENT) == set(N.SHARDS_GB) == set(part5_grid.MODELS)
    for key, (cls, zone, product) in N.PLACEMENT.items():
        tier_a = key in part5_grid.TIER_A
        assert cls == ("linstor-unl" if tier_a else "rook-ceph-block-fullerton")
        assert zone["topology.kubernetes.io/zone"] == ("unl" if tier_a else "fullerton")
        assert product == ("NVIDIA-A10" if tier_a else "NVIDIA-L4")
        assert f"storageClassName: {cls}" in N.pvc_yaml(key)
        assert len(N.set_volumes(key)) == len(N.SHARDS_GB[key])
    assert N.node_selector(1, N.FULLERTON, "NVIDIA-L4") == {
        "topology.kubernetes.io/zone": "fullerton",
        "nvidia.com/gpu.product": "NVIDIA-L4",
    }
    assert N.node_selector(0, N.FULLERTON) == N.FULLERTON  # a CPU pod of the set


def test_a_lengthened_set_smoke_keeps_its_own_directory():
    """The default set-smoke runs 4 documents and 8 chunks into set-smoke/; a lengthened
    one (a small model whose arms must outlast their process start) passes its limits
    through and writes elsewhere, so the two never mix."""
    import pytest

    pytest.importorskip("nats_bursting")
    sys.path.insert(0, str(KVDIR.parent))
    import part5_nrp as N

    c = "0" * 40
    s = N.set_smoke_script(c, "qwen2.5-1.5b-instruct")
    assert "--docs 4 --ppl-chunks 8 " in s and "/set-smoke/heartbeat.json" in s
    s = N.set_smoke_script(c, "qwen2.5-1.5b-instruct", 20, 40)
    assert "--docs 20 --ppl-chunks 40 " in s
    assert "/set-smoke-d20c40/heartbeat.json" in s
    assert "--root /w0/runs/000000000000/set-smoke-d20c40 " in s
    bs = chr(92)
    assert bs + chr(10) in s and bs + bs not in s


def test_tier_b_chain_runs_every_wave_but_fidelity_and_stops_on_failure(monkeypatch):
    """Tier B runs every arm: every wave but Qwen2.5-7B's fidelity check, gate first
    (the baselines read its u3) and g0 before any scored arm (it fits the calibration
    tables). A failed or vanished Job stops the chain; nothing is deleted or retried."""
    import json
    import types

    import pytest

    pytest.importorskip("nats_bursting")
    sys.path.insert(0, str(KVDIR.parent))
    import part5_chain as C

    order = C.ORDERS["tierb"]
    assert set(order) == set(part5_grid.WAVES) - {"fidelity"} and len(
        set(order)
    ) == len(order)
    assert order.index("gate") < order.index("baselines")
    assert order[:2] == ["gate", "g0"]

    def fake(status):
        def kubectl(*args):
            body = json.dumps({"status": status}) if status is not None else ""
            err = "" if status is not None else 'jobs.batch "x" not found (NotFound)'
            return types.SimpleNamespace(
                returncode=0 if status is not None else 1, stdout=body, stderr=err
            )

        return kubectl

    cases = [
        ({"conditions": [{"type": "Complete", "status": "True"}]}, "complete"),
        ({"conditions": [{"type": "Failed", "status": "True"}]}, "failed"),
        ({"active": 1}, "active"),
        (None, "missing"),
    ]
    for status, want in cases:
        monkeypatch.setattr(C.N.W, "kubectl", fake(status))
        assert C.job_state("x") == want


def test_tier_a_verdict_waves_wait_for_registration():
    """Before registration only the gate and the fidelity check run on Tier A (owner,
    2026-10-14); every other Tier A wave waits until the prereg on master is no longer
    a DRAFT. Tier B and smokes are not held."""
    import pytest

    pytest.importorskip("nats_bursting")
    sys.path.insert(0, str(KVDIR.parent))
    import part5_nrp as N

    for w in part5_grid.WAVES:
        held = w not in ("gate", "fidelity")
        assert N.needs_registration("mistral-7b-instruct", w) is held
        assert not N.needs_registration("mistral-7b-instruct", w, smoke=True)
        assert not N.needs_registration("qwen2.5-1.5b-instruct", w)
    draft = "# Pre-registration\n\n**Status: DRAFT for review. Not registered.** ..."
    done = "# Pre-registration\n\n**Status: Registered 2026-10-20 at abc1234.** ..."
    assert not N.registered(lambda: None)  # not on master
    assert not N.registered(lambda: draft)
    assert not N.registered(lambda: "# no status line")
    assert N.registered(lambda: done)


def test_waves_run_the_registered_harness_only(monkeypatch):
    """Every wave's cells run the registered harness commit (owner, 2026-10-14), so a
    model's arms share one run root; a smoke may run any commit."""
    import pytest

    pytest.importorskip("nats_bursting")
    sys.path.insert(0, str(KVDIR.parent))
    import part5_nrp as N

    monkeypatch.setattr(N.W, "guard_alive", lambda *a, **k: True)

    assert N.HARNESS.startswith("2e8956a") and len(N.HARNESS) == 40
    with pytest.raises(SystemExit, match="registered harness"):
        N.main(
            [
                "set-run",
                "--commit",
                "1" * 40,
                "--model",
                "qwen2.5-1.5b-instruct",
                "--wave",
                "gate",
                "--dry-run",
            ]
        )

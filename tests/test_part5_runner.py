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

    assert N.HARNESS.startswith("d388fc3") and len(N.HARNESS) == 40
    assert N.GATE_ONLY.startswith("2e8956a") and N.RUN_KEY == "2e8956abe483"
    # every wave's cells in one root, whatever commit a pod runs; a smoke keeps its own
    assert (
        N.run_root(N.HARNESS)
        == N.run_root(N.GATE_ONLY)
        == "/w0/runs/2e8956abe483/part5"
    )
    assert N.run_root(N.HARNESS, smoke=True) == "/w0/runs/d388fc31db6e/smoke"
    assert "cd /w0/runs/2e8956abe483/part5/x" in N.fetch_set_script(
        N.HARNESS, "x", "part5"
    )
    # calibration tables stay on the volume (a fetch log keeps only its tail)
    assert "--exclude='*.pt'" in N.fetch_set_script(N.HARNESS, "x", "part5")
    import types

    monkeypatch.setattr(  # the commit rule only: no nats_bursting descriptor
        N,
        "descriptor",
        lambda name, *a, **k: types.SimpleNamespace(
            name=name, resources=None, node_selector=None, command=["x"]
        ),
    )
    monkeypatch.setattr(N.W, "preflight", lambda *a, **k: [])
    for wave, commit, ok in (
        ("gate", N.GATE_ONLY, True),
        ("gate", N.HARNESS, True),
        ("g0", N.GATE_ONLY, False),
        ("baselines", N.HARNESS, True),
    ):
        argv = [
            "set-run",
            "--commit",
            commit,
            "--model",
            "qwen2.5-1.5b-instruct",
            "--wave",
            wave,
            "--dry-run",
        ]
        if ok:
            assert N.main(argv) == 0
        else:
            with pytest.raises(SystemExit, match="registered harness"):
                N.main(argv)
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


def test_chain_adopts_only_its_commit_and_retries_a_lost_wave(monkeypatch):
    """NRP pods can die at any time: the chain resubmits a failed or vanished wave
    unchanged (its cells resume), up to MAX_TRIES submissions, adopts a running Job only
    if it runs the chain's commit, and runs the fidelity wave on Qwen2.5-7B alone."""
    import json
    import types

    import pytest

    pytest.importorskip("nats_bursting")
    sys.path.insert(0, str(KVDIR.parent))
    import part5_chain as C

    def job(cmd, status):
        body = {
            "spec": {"template": {"spec": {"containers": [{"command": cmd}]}}},
            "status": status,
        }
        return lambda *a: types.SimpleNamespace(
            returncode=0, stdout=json.dumps(body), stderr=""
        )

    monkeypatch.setattr(
        C.N.W, "kubectl", job(["bash", "-lc", "HARNESS_COMMIT=abc x"], {"active": 1})
    )
    assert C.job_state("x", "abc") == "active"
    assert C.job_state("x", "def") == "other"
    assert C.MAX_TRIES >= 3
    with pytest.raises(SystemExit, match="not chainable"):
        C.main(
            ["--commit", "abc", "--model", "mistral-7b-instruct", "--waves", "fidelity"]
        )

    # a wave lost twice, then completed: two resubmissions, then the next wave
    states = iter(["missing", "failed", "missing", "complete"])
    subs = []
    monkeypatch.setattr(C, "job_state", lambda name, commit=None: next(states))
    monkeypatch.setattr(C, "submit", lambda log, c, m, w: subs.append(w) or True)
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    monkeypatch.setattr(C.N, "STATE", str(pytest.importorskip("tempfile").mkdtemp()))
    assert (
        C.main(
            ["--commit", "abc", "--model", "qwen2.5-1.5b-instruct", "--waves", "gate"]
        )
        == 0
    )
    assert subs == ["gate", "gate", "gate"]  # the first submission and two retries


def test_tier_a_chain_holds_for_registration(monkeypatch):
    """Tier A's verdict chain (g0, verdict, head, baselines) waits for registration
    instead of stopping: set-run's refusal holds it until the prereg is merged."""
    import pytest

    pytest.importorskip("nats_bursting")
    sys.path.insert(0, str(KVDIR.parent))
    import part5_chain as C

    assert C.ORDERS["tiera"] == ["g0", "verdict", "head", "baselines"]
    calls = []

    def fake_main(argv):
        calls.append(argv)
        if len(calls) < 3:
            raise SystemExit("g0 on mistral-7b-instruct waits for registration: ...")
        return 0

    monkeypatch.setattr(C.N, "main", fake_main)
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    monkeypatch.setattr(C, "say", lambda log, msg: None)
    assert C.submit("log", "abc", "mistral-7b-instruct", "g0") is True
    assert len(calls) == 3


def test_a_wave_pod_fails_when_an_arm_is_short(monkeypatch, tmp_path):
    """keys_run.py logs a failed cell and exits 0 (Mistral-7B's u3_head, 2026-10-15), so
    the pod's own check, run here on fake cells, fails the pod and names the short arm.
    """
    import subprocess

    import pytest

    pytest.importorskip("nats_bursting")
    sys.path.insert(0, str(KVDIR.parent))
    import part5_nrp as N

    monkeypatch.setattr(N, "run_root", lambda commit, smoke=False: str(tmp_path))
    key, wave = "mistral-7b-instruct", "verdict"
    s = N.run_script("0" * 40, key, wave)
    check = s.split("<<'CHECK'", 1)[1].split("\n", 1)[1].split("\nCHECK\n", 1)[0]

    def fill(arm, complete):
        cell = tmp_path / key / arm
        cell.mkdir(parents=True)
        for t in part5_grid.TASKS:
            (cell / f"{t}.0.jsonl").write_text("{}\n" * 200)
        if complete:
            (cell / "ppl.done").write_text("{}")

    def run():
        return subprocess.run(
            [sys.executable, "-c", check], cwd=KVDIR, capture_output=True, text=True
        )

    arms = part5_grid.WAVES[wave]
    for arm in arms[:-1]:
        fill(arm, True)
    fill(arms[-1], False)  # its perplexity failed: no ppl.done
    r = run()
    assert r.returncode != 0
    assert f"{N.INCOMPLETE} {key} {wave}: {arms[-1]}" in r.stderr
    (tmp_path / key / arms[-1] / "ppl.done").write_text("{}")
    assert run().returncode == 0


def test_chain_stops_on_a_short_wave_instead_of_resubmitting(monkeypatch):
    """A failed Job whose pod printed the INCOMPLETE line stops the chain at once; a
    failed Job without it (a lost pod) is resubmitted as before."""
    import types

    import pytest

    pytest.importorskip("nats_bursting")
    sys.path.insert(0, str(KVDIR.parent))
    import part5_chain as C

    log = f"[run] x\n{C.N.INCOMPLETE} mistral-7b-instruct head: u3_head\n"
    monkeypatch.setattr(
        C.N.W,
        "kubectl",
        lambda *a: types.SimpleNamespace(returncode=0, stdout=log, stderr=""),
    )
    subs = []
    monkeypatch.setattr(C, "job_state", lambda name, commit=None: "failed")
    monkeypatch.setattr(C, "submit", lambda lg, c, m, w: subs.append(w) or True)
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    monkeypatch.setattr(C.N, "STATE", str(pytest.importorskip("tempfile").mkdtemp()))
    argv = ["--commit", "abc", "--model", "mistral-7b-instruct", "--waves", "head"]
    assert C.main(argv) == 1
    assert subs == ["head"]  # submitted once, never resubmitted
    assert (
        "STOP head"
        in open(f"{C.N.STATE}/chain_{C.N.short('mistral-7b-instruct')}.log").read()
    )


def test_a_kubectl_stall_holds_the_chain_instead_of_killing_it(monkeypatch):
    """2026-10-15 18:03Z: a two-minute API stall raised TimeoutExpired out of kubectl()
    and killed all five chains. A stalled call now returns 124; a Job's state reads as
    unknown (the chain polls again), a submission is held, and submit() will not go on
    without clearing the old Job."""
    import subprocess
    import types

    import pytest

    pytest.importorskip("nats_bursting")
    sys.path.insert(0, str(KVDIR.parent))
    import part5_chain as C

    def hang(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))

    monkeypatch.setattr(C.N.W.subprocess, "run", hang)
    r = C.N.W.kubectl("get", "jobs")
    assert r.returncode == 124 and "timed out" in r.stderr
    assert C.job_state("kv5-run-head-x") == "unknown"
    with pytest.raises(SystemExit, match="cannot clear the old Job"):
        C.N.W.submit(types.SimpleNamespace(name="kv5-run-head-x"))

    calls = []

    def main(argv):
        calls.append(argv)
        if len(calls) < 3:
            raise SystemExit("cannot list jobs: kubectl timed out after 120 s")

    monkeypatch.setattr(C.N, "main", main)
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    assert C.submit("/dev/null", "abc", "mistral-7b-instruct", "head") is True
    assert len(calls) == 3

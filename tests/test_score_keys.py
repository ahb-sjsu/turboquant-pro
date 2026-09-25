"""Registered scorer of observer-advantage Part II, on synthetic cells.

Pins the parts of ``benchmarks/kvquant_matrix/score_keys.py`` a verdict depends on:
cell verification against the registered env line, the paired bootstrap, and the
materiality rule in units of the run-to-run floor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "kvquant_matrix")
)
import keys_grid as KG  # noqa: E402
import score_keys as SK  # noqa: E402


def _cell(tmp_path, arm, **override):
    """A cell whose sidecar is what the harness writes for ``arm``."""
    env = SK._env(KG.ARMS[arm])
    cfg = {"noquant": int(env.get("NOQUANT", 0)), "shard": 0}
    if not cfg["noquant"]:
        cfg.update(
            codebook=env["CODEBOOK"],
            key_bits=int(env["KEY_BITS"]),
            sink=int(env["SINK"]),
            outlier_frac=float(env["OUTLIER_FRAC"]),
            hot=int(env["HOT"]),
            group=int(env["GROUP"]),
            prerope=int(env["PREROPE"]),
        )
        staged = {
            k: env[k]
            for k in ("KEY_BASIS", "BASIS_FIT", "KEY_ALLOC", "BYTE_MATCH", "KEY_JITTER")
            if k in env
        }
        if staged:
            cfg["key_coding"] = {
                "key_basis": env.get("KEY_BASIS", "native"),
                "basis_fit": env.get("BASIS_FIT", "prefill"),
                "key_alloc": env.get("KEY_ALLOC", "uniform"),
                "byte_match": int(env.get("BYTE_MATCH", 0)),
                "key_jitter": int(env.get("KEY_JITTER", 0)),
            }
    cfg.update(override)
    d = tmp_path / arm
    d.mkdir()
    (d / "config.0.json").write_text(json.dumps(cfg))
    return str(d), env


@pytest.mark.parametrize(
    "arm", ["fp16", "nf4a", "nf4a_jit", "nf4a_bm", "nf4a_O", "u3_read", "u2_O_read"]
)
def test_registered_cells_verify(tmp_path, arm):
    cell, env = _cell(tmp_path, arm)
    assert SK.verify(cell, env) is None


def test_a_mislabeled_cell_is_refused(tmp_path):
    cell, _ = _cell(tmp_path, "nf4a")  # ran the shipped arm
    assert "key-coding record missing" in SK.verify(cell, SK._env(KG.ARMS["nf4a_O"]))


def test_a_wrong_basis_is_refused(tmp_path):
    cell, env = _cell(tmp_path, "nf4a_O")
    bad = json.loads(Path(cell, "config.0.json").read_text())
    bad["key_coding"]["key_basis"] = "P"
    Path(cell, "config.0.json").write_text(json.dumps(bad))
    assert "key_basis" in SK.verify(cell, env)


def test_disagreeing_shards_are_refused(tmp_path):
    cell, env = _cell(tmp_path, "nf4a")
    other = json.loads(Path(cell, "config.0.json").read_text())
    other.update(shard=1, sink=0)
    Path(cell, "config.1.json").write_text(json.dumps(other))
    assert SK.verify(cell, env) == "sidecars disagree across shards"


def test_paired_bootstrap_sign_and_direction():
    x = {i: 1.0 + 0.01 * i for i in range(50)}
    y = {i: 0.5 + 0.01 * i for i in range(50)}
    up = SK.paired(x, y)
    assert up["mean"] == pytest.approx(0.5) and up["lo"] > 0
    down = SK.paired(x, y, higher_better=False)  # e.g. NLL: larger is worse
    assert down["mean"] == pytest.approx(-0.5) and down["hi"] < 0


def test_materiality_is_in_floor_units():
    delta = {"mean": 0.3, "lo": 0.1, "hi": 0.5}
    assert SK.judge(delta, floor=0.1) == "better"
    assert SK.judge(delta, floor=0.2) == "neither"  # inside twice the floor
    assert SK.judge({"mean": -0.3, "lo": -0.5, "hi": -0.1}, 0.0) == "worse"
    assert SK.judge({"mean": 0.3, "lo": -0.1, "hi": 0.7}, 0.0) == "neither"
    assert SK.judge(None, 0.1) == "missing"


def test_every_comparison_names_registered_arms():
    for pairs in KG.COMPARISONS.values():
        for arm, ref, rep in pairs:
            assert {arm, ref, rep} <= set(KG.ARMS)
    for arm, ref, rep in KG.REPORTED:
        assert {arm, ref, rep} <= set(KG.ARMS)


def test_environment_is_recorded_by_the_runner(tmp_path):
    import keys_run as KR

    cell = tmp_path / "c"
    cell.mkdir()
    a = {"gpu": "A100", "torch": "2.10.0"}
    KR.record_env(str(cell), a)
    KR.record_env(str(cell), a)  # same place: fine
    with pytest.raises(SystemExit, match="must finish where it began"):
        KR.record_env(str(cell), {"gpu": "Quadro GV100", "torch": "2.10.0"})


def test_k3_waits_for_every_control_on_every_model():
    """Amendment 3: missing control cells leave K3 INCOMPLETE, never DOES NOT HOLD."""
    full = [{"models_better": 2, "scored": 3}, {"models_better": 3, "scored": 3}]
    assert SK.k3_verdict(full, 3) == "HOLDS"
    one_short = [{"models_better": 2, "scored": 3}, {"models_better": 1, "scored": 3}]
    assert SK.k3_verdict(one_short, 3) == "DOES NOT HOLD"
    unrun = [{"models_better": 0, "scored": 0}, {"models_better": 2, "scored": 3}]
    assert SK.k3_verdict(unrun, 3) == "INCOMPLETE"
    partial = [{"models_better": 1, "scored": 2}, {"models_better": 3, "scored": 3}]
    assert SK.k3_verdict(partial, 3) == "INCOMPLETE"

"""Part V scorer (benchmarks/kvquant_matrix/score_part5.py) on synthetic cells.

Pins, before any registered cell exists:

  a complete grid with a planted effect scores the verdict it was built to give;
  a partial cell, a cell run with limits, a calibrated cell at unregistered windows or
      seed, and calibrated cells coded with different tables are refused, not scored;
  G1 passes inside its noise-unit tolerance and withholds every verdict outside it;
  V1-V4 and V5 follow their registered rules, V5 is not scored unless V2 holds, and
      AATC is dropped from V5 when its fidelity check failed;
  Part II's recorded u3 is read from the hash-pinned scorer output.

Usage:
    python -m pytest tests/test_score_part5.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "kvquant_matrix"))

import part5_grid as KG  # noqa: E402
import score_part5 as S  # noqa: E402


def _exact(pred, gt, all_classes=None):
    return float(pred == gt)


METRICS = {"trec": _exact, "triviaqa": _exact, "qasper": _exact}
ENV = {"gpu": "NVIDIA A10", "torch": "2.10.0", "transformers": "5.5.0"}
N_CHUNKS = 20


def _sidecar(line: str) -> dict:
    env = dict(kv.split("=", 1) for kv in line.split())
    if env.get("NOQUANT") == "1":
        return {"noquant": 1}
    c = {
        "noquant": 0,
        "codebook": env["CODEBOOK"],
        "key_bits": int(env["KEY_BITS"]),
        "sink": int(env["SINK"]),
        "outlier_frac": float(env["OUTLIER_FRAC"]),
        "hot": int(env["HOT"]),
        "group": int(env["GROUP"]),
        "prerope": int(env["PREROPE"]),
    }
    stages = {
        "KEY_BASIS": "native",
        "BASIS_FIT": "prefill",
        "KEY_ALLOC": "uniform",
        "BYTE_MATCH": "0",
        "KEY_JITTER": "0",
    }
    if any(env.get(k, d) != d for k, d in stages.items()):
        c["key_coding"] = {k.lower(): env.get(k, d) for k, d in stages.items()}
        cal = S.calibrated(env)
        c["key_coding"].update(
            part5_calib_seq=KG.CALIB_SEQ if cal else 0,
            part5_calib_n=KG.CALIB_N if cal else 0,
            fisher_seed=KG.FISHER_SEED if env.get("KEY_ALLOC") == "fisher" else 0,
        )
    return c


def make_cell(
    root,
    mk,
    arm,
    p=0.5,
    nll=2.0,
    docs=200,
    chunks=N_CHUNKS,
    limits=None,
    sha="a" * 64,
    env=ENV,
):
    """One runner cell: a document is answered correctly when its shared draw u_i < p
    (so arms differ on the same documents), and a chunk's NLL is ``nll`` plus a shared
    per-chunk offset."""
    line = {**KG.ARMS, **KG.G0_ARMS}.get(arm)
    cell = Path(root) / mk / arm
    cell.mkdir(parents=True)
    if line is None:  # a baseline
        (cell / "baseline.json").write_text(
            json.dumps(
                {
                    "baseline": arm,
                    "matched": {
                        "reference": "u3",
                        "arm_bits": 8.6,
                        "reference_bits": 8.6,
                    },
                }
            )
        )
    else:
        (cell / "config.0.json").write_text(json.dumps(_sidecar(line)))
        if S.calibrated(dict(kv.split("=", 1) for kv in line.split())):
            calfile = (
                "alloc_calib_fisher.pt" if "fisher" in arm else "alloc_calib_head.pt"
            )
            (cell / "calib.json").write_text(
                json.dumps({"file": calfile, "sha256": sha})
            )
    (cell / "env.json").write_text(json.dumps(env))
    u = np.random.default_rng(1).random(200)
    tasks = KG.G0_TASKS if arm.startswith("g0_") else KG.TASKS
    for t in tasks:
        with open(cell / f"{t}.0.jsonl", "w") as f:
            for i in range(docs):
                f.write(
                    json.dumps(
                        {
                            "idx": i,
                            "pred": "a" if u[i] < p else "b",
                            "answers": ["a"],
                            "all_classes": None,
                        }
                    )
                    + "\n"
                )
    off = np.random.default_rng(2).random(chunks)
    with open(cell / "ppl_chunks.jsonl", "w") as f:
        for c in range(chunks):
            f.write(
                json.dumps(
                    {
                        "chunk": c,
                        "start": c * 2048,
                        "nll": (nll + off[c]) * 2047,
                        "tokens": 2047,
                    }
                )
                + "\n"
            )
    (cell / "ppl.done").write_text(
        json.dumps(limits or {"harness": "x", "maxchunks": 0, "docs": 0})
    )
    return cell


def grid(root, models=tuple(KG.TIER_A), effect=None):
    """Every verdict cell on ``models``; ``effect`` maps an arm to (p, nll)."""
    effect = effect or {}
    arms = [a for w in KG.VERDICT_WAVES for a in KG.WAVES[w]]
    for mk in models:
        for arm in arms + ["aatc", "ea"]:
            p, nll = effect.get(arm, (0.5, 2.0))
            make_cell(root, mk, arm, p=p, nll=nll)


def recorded(root, models=tuple(KG.TIER_A)):
    """Part II's u3 set to the synthetic u3, so G1's difference is zero."""
    out = {}
    for mk in models:
        d, ent = S.read_cell(str(Path(root) / mk / "u3"), METRICS)
        out[mk] = {"qasper": ent["qasper"], "ppl": ent["ppl"]}
    return out


def run(root, fidelity=True, rec=None):
    return S.score(str(root), METRICS, [], fidelity, rec or recorded(root))


def test_planted_effect_gives_the_verdicts_it_was_built_for(tmp_path):
    grid(tmp_path, effect={"u3_head": (0.9, 1.9)})
    r = run(tmp_path)
    assert r["verdict_status"] == "FINAL"
    assert r["verdicts"]["V1"]["verdict"] == "HOLDS"
    assert r["verdicts"]["V2"]["verdict"] == "HOLDS"
    assert r["verdicts"]["V3"]["verdict"] == "FAILS"  # fisher_cal = head_cal
    assert r["verdicts"]["V4"]["verdict"] == "HOLDS"
    assert r["verdicts"]["V5"]["verdict"] == "HOLDS"
    for mk in KG.TIER_A:
        assert all(g["pass"] for g in r["models"][mk]["g0"].values())


def test_reversed_effect_and_v5_not_scored_without_v2(tmp_path):
    grid(tmp_path, effect={"u3_head": (0.1, 2.1)})
    r = run(tmp_path)
    assert r["verdicts"]["V1"]["verdict"] == "FAILS (reversed)"
    assert r["verdicts"]["V5"]["verdict"].startswith("NOT SCORED")


@pytest.mark.parametrize("fidelity,want", [(None, "INCOMPLETE"), (False, "HOLDS")])
def test_aatc_fidelity_decides_v5s_baselines(tmp_path, fidelity, want):
    grid(
        tmp_path, effect={"u3_head": (0.9, 1.9), "aatc": (1.0, 1.0)}
    )  # aatc unbeatable
    r = run(tmp_path, fidelity=fidelity)
    assert r["verdicts"]["V5"]["verdict"] == want
    if fidelity is False:
        assert set(r["verdicts"]["V5"]["rows"]) == {"ea"}


def test_v5_fails_when_a_baseline_wins_on_two_models(tmp_path):
    grid(tmp_path, effect={"u3_head": (0.9, 1.9), "ea": (1.0, 1.0)})
    assert run(tmp_path)["verdicts"]["V5"]["verdict"] == "FAILS"


@pytest.mark.parametrize(
    "kw,reason",
    [
        ({"docs": 150}, "incomplete: trec"),
        (
            {"limits": {"harness": "x", "maxchunks": 8, "docs": 4}},
            "incomplete: ran with limits",
        ),
        (
            {"limits": {"harness": "x"}},
            "incomplete: ran with limits",
        ),  # pre-Part V runner
    ],
)
def test_partial_cells_are_refused(tmp_path, kw, reason):
    grid(tmp_path)
    mk = next(iter(KG.TIER_A))
    import shutil

    shutil.rmtree(tmp_path / mk / "u3_head")
    make_cell(tmp_path, mk, "u3_head", **kw)
    r = run(tmp_path)
    cell = r["models"][mk]["cells"]["u3_head"]
    assert not cell["verified"] and cell["reason"].startswith(reason)
    assert r["verdicts"]["V1"]["verdict"] == "INCOMPLETE"


def test_calibrated_cell_at_unregistered_seed_is_refused(tmp_path):
    mk = next(iter(KG.TIER_A))
    cell = make_cell(tmp_path, mk, "u3_fisher_cal")
    c = json.loads((cell / "config.0.json").read_text())
    c["key_coding"]["fisher_seed"] = 7
    (cell / "config.0.json").write_text(json.dumps(c))
    why = S.verify_part5(
        str(cell), dict(kv.split("=", 1) for kv in KG.ARMS["u3_fisher_cal"].split())
    )
    assert why.startswith("fisher_seed=7")


def test_cells_coded_with_different_tables_are_refused(tmp_path):
    grid(tmp_path)
    mk = next(iter(KG.TIER_A))
    import shutil

    shutil.rmtree(tmp_path / mk / "u3_head_cal")
    make_cell(tmp_path, mk, "u3_head_cal", sha="b" * 64)
    r = run(tmp_path)
    for arm in ("u3_head_cal", "g0_head_cal"):
        assert "different tables" in r["models"][mk]["cells"][arm]["reason"]
    assert r["models"][mk]["cells"]["u3_fisher_cal"]["verified"]  # the other weight


def test_g1_outside_tolerance_withholds_every_verdict(tmp_path):
    grid(tmp_path, effect={"u3_head": (0.9, 1.9)})
    rec = recorded(tmp_path)
    mk = next(iter(KG.TIER_A))
    rec[mk]["qasper"] += 5.0  # far outside 3 x max(repeat 0, jitter 0, cross-GPU 0.09)
    r = run(tmp_path, rec=rec)
    assert r["gates"]["G1"][mk]["status"] == "FAIL_UNEXPLAINED"
    assert r["verdict_status"] == "WITHHELD"
    assert set(r["verdicts"].values()) == {"WITHHELD"}


def test_g1_tolerance_is_in_noise_units():
    got, rep = {"qasper": 41.0, "ppl": 7.53}, {"qasper": 41.0, "ppl": 7.53}
    jit = {"qasper": 40.0, "ppl": 7.53}  # the jitter moves qasper one point
    ok, d = S.g1_check(got, rep, jit, {"qasper": 43.9, "ppl": 7.53})
    assert ok and d["qasper"]["tolerance"] == pytest.approx(3.0)
    ok, d = S.g1_check(got, rep, jit, {"qasper": 44.1, "ppl": 7.53})
    assert not ok
    ok, d = S.g1_check(got, rep, jit, {"qasper": 41.0, "ppl": 7.53 * 1.004})
    assert not ok and d["ppl"]["tolerance"] == pytest.approx(
        0.003
    )  # cross-GPU 0.1% x 3
    assert S.g1_check(got, {}, jit, {"qasper": 41.0, "ppl": 7.53})[0] is None


def test_g0_failure_removes_the_model_from_the_verdict(tmp_path):
    grid(tmp_path, effect={"u3_head": (0.9, 1.9)})
    mk = next(iter(KG.TIER_A))
    import shutil

    shutil.rmtree(tmp_path / mk / "g0_head")
    make_cell(tmp_path, mk, "g0_head", p=0.4)  # differs from g0_native
    r = run(tmp_path)
    assert not r["models"][mk]["g0"]["g0_head"]["pass"]
    assert r["verdicts"]["V1"]["verdict"] == "INCOMPLETE"


def test_mixed_environments_are_not_compared(tmp_path):
    grid(tmp_path, effect={"u3_head": (0.9, 1.9)})
    mk = next(iter(KG.TIER_A))
    (tmp_path / mk / "u3_read" / "env.json").write_text(
        json.dumps({**ENV, "gpu": "NVIDIA L40"})
    )
    r = run(tmp_path)
    assert not r["comparisons"][f"{mk}|u3_head|u3_read"]["g0_pass"]
    assert r["verdicts"]["V1"]["verdict"] == "INCOMPLETE"


@pytest.mark.parametrize(
    "rows,want",
    [
        ([("better", "better")] * 2 + [("neither", "neither")], "HOLDS"),
        ([("better", "better")] * 2 + [("neither", "worse")], "INCONCLUSIVE"),
        ([("worse", "neither")] * 2 + [("neither", "neither")], "FAILS (reversed)"),
        ([("better", "neither")] * 3, "FAILS"),
        ([("better", "better")] * 2, "INCOMPLETE"),
    ],
)
def test_v_rule(rows, want):
    assert S.v_rule([{"qasper": q, "ppl": p} for q, p in rows]) == want


def test_recorded_u3_reads_the_pinned_part2_output():
    rec = S.recorded_u3(str(ROOT / "benchmarks" / "kvquant_matrix" / KG.G1_SOURCE))
    assert rec["qwen2.5-7b-instruct"]["qasper"] == pytest.approx(41.4231, abs=1e-4)
    assert rec["llama2-7b-chat-4k"]["ppl"] == pytest.approx(6.9801, abs=1e-4)


def test_every_harness_arm_is_in_exactly_one_wave():
    waves = [a for w in KG.WAVES.values() for a in w]
    assert len(waves) == len(set(waves))
    assert set(waves) == set(KG.ARMS) | set(KG.G0_ARMS)
    # a calibrated table is fitted by its G0 arm before any scored arm reads it
    order = [a for w in KG.VERDICT_WAVES for a in KG.WAVES[w]]
    assert order.index("g0_head_cal") < order.index("u3_head_cal")
    assert order.index("g0_fisher_cal") < order.index("u3_fisher_cal")


def test_floor_is_the_jitter_against_u3_not_against_the_reference(tmp_path):
    """V1's reference is u3_read; its floor must be |u3_jit - u3| (section 4), not
    |u3_jit - u3_read|, which would carry u3_read's own effect into the bar."""
    grid(tmp_path, effect={"u3_read": (0.8, 1.95), "u3_head": (0.9, 1.9)})
    r = run(tmp_path)
    mk = next(iter(KG.TIER_A))
    c = r["comparisons"][f"{mk}|u3_head|u3_read"]
    assert c["qasper"]["floor"] == 0.0 and c["ppl"]["floor"] == 0.0
    assert r["verdicts"]["V1"]["verdict"] == "HOLDS"

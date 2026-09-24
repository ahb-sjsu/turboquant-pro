"""The P0 exit test's runner and scorer (docs/PREREG_planner_exit.md), synthetic data.

Pins, before any real run: the registered budgets, the pinned design space equal to
the reachable grid, the mapping from a planner choice to a grid config key, the
manifest seal, and every verdict rule of section 4, on hand-made grid cells.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

from planner_exit import run as R  # noqa: E402
from planner_exit import score as S  # noqa: E402
from rabitq_public import grid as G  # noqa: E402

ARM = "glove-100-angular"


def test_registered_budgets():
    b = R.runs(100)
    assert [b[f"B{i}"]["max_bytes"] for i in (1, 2, 3, 4)] == [17, 29, 42, 54]
    assert {k: v["floor"] for k, v in b.items() if k.startswith("F")} == {
        "F90": 0.90,
        "F95": 0.95,
        "F99": 0.99,
    }
    assert len(b) == 7


@pytest.mark.parametrize("arm", sorted(G.DIMS))
def test_the_pinned_space_is_exactly_the_reachable_grid(arm):
    pinned = R.pinned_configs(arm)
    reach = [c for c in G.configs(arm) if c["method"] in R.METHOD_CODEC]
    assert sum(len(v) for v in pinned.values()) == len(reach)
    keys = {R.grid_key(arm, codec, cfg) for codec, cs in pinned.items() for cfg in cs}
    want = {G.cell_id(arm, c, 0).rsplit("-s", 1)[0] for c in reach}
    assert keys == want


def test_grid_key_matches_the_campaigns_cell_ids():
    assert R.grid_key(ARM, "tq_embedding", {"bits": 3, "out_dim": 100}) == (
        "glove-100-angular-tq-d100-b3"
    )
    assert (
        R.grid_key(ARM, "faiss_rabitq", {"bits": 2})
        == "glove-100-angular-rabitq_flat-b2"
    )
    assert R.grid_key(ARM, "faiss_opq", {"m": 25}) == "glove-100-angular-opq-m25"


# ------------------------------------------------------------------ #
# Scorer, on a fabricated grid                                        #
# ------------------------------------------------------------------ #

NQ = 400


def _cell(root, method, cfg, seed, recall, stored, cores=1.0, wall=100.0):
    rng = np.random.default_rng(hash((method, str(cfg), seed)) % 2**32)
    hits = np.clip(np.round(recall * 10 + rng.normal(0, 0.3, NQ)), 0, 10).astype(int)
    c = dict(dataset=ARM, method=method, seed=seed, **cfg)
    c["cell_id"] = G.cell_id(ARM, dict(method=method, **cfg), seed)
    rec = dict(
        cell=c,
        stored_bytes_per_vec=stored,
        hits_rr5=hits.tolist(),
        hits_single=hits.tolist(),
        hits_rr2=hits.tolist(),
        build_s=1,
        search_s=1,
        usage=dict(mean_cpu_cores=cores, wall_s=wall),
    )
    with open(os.path.join(root, f"{c['cell_id']}.json"), "w") as f:
        json.dump(rec, f)


@pytest.fixture
def grid(tmp_path):
    root = tmp_path / "grid"
    root.mkdir()
    table = [  # method, cfg, recall, bytes
        ("tq", {"out_dim": 100, "bits": 2}, 0.80, 29),
        ("tq", {"out_dim": 100, "bits": 3}, 0.93, 42),
        ("tq", {"out_dim": 100, "bits": 4}, 0.975, 54),
        ("pq", {"m": 20}, 0.85, 20),
        ("opq", {"m": 50}, 0.96, 50),
        ("rabitq_flat", {"bits": 1}, 0.60, 21),
        ("rabitq_ivf", {"bits": 4, "nlist": 4096}, 0.999, 50),  # unreachable
    ]
    for method, cfg, rec, b in table:
        for s in G.SEEDS:
            _cell(str(root), method, cfg, s, rec, b)
    return str(root)


def _record(run, codec, cfg, *, bound=0.5, fresh=0.9, cpu=10.0, reg=None):
    reg = reg or R.runs(100)[run]
    rec = dict(arm=ARM, run=run, registered=reg, cpu_seconds=cpu, plan={})
    if codec is None:
        rec["choice"] = None
    else:
        rec["choice"] = dict(
            codec=codec,
            config=cfg,
            grid_key=R.grid_key(ARM, codec, cfg),
            holdout_bound=bound,
            holdout_mean=bound,
        )
        rec["fresh"] = dict(mean=fresh)
    return rec


def test_byte_budget_regret_against_the_reachable_best(grid):
    ok = _record("B4", "tq_embedding", {"bits": 4, "out_dim": 100})  # 54 B, best
    bad = _record("B4", "tq_embedding", {"bits": 2, "out_dim": 100})
    res = S.score([ok, bad], grid)
    a, b = res["runs"]
    assert a["no_regret"] and a["reach"]["best"].endswith("tq-d100-b4")
    assert not b["no_regret"] and b["reach"]["verdict"] == "LOSES"
    assert b["loses_by"] == pytest.approx(0.175, abs=0.02)
    # the unreachable IVF config is the full-grid best, not the reachable one
    assert a["full"]["best"].endswith("rabitq_ivf-b4-n4096")


def test_floor_runs_score_bytes_and_correct_abstention(grid):
    cheap = _record("F95", "faiss_opq", {"m": 50})  # 50 B, clears 0.95
    wasteful = _record("F95", "tq_embedding", {"bits": 4, "out_dim": 100})  # 54 B
    abstain = _record("F99", None, None)  # nothing reachable clears 0.99
    res = S.score([cheap, wasteful, abstain], grid)
    c, w, ab = res["runs"]
    assert c["no_regret"] and c["meets_floor"]
    assert w["meets_floor"] and not w["no_regret"]  # 54 > 1.05 * 50
    assert ab["no_regret"] and ab["cheapest"] is None


def test_verdicts_and_the_manifest_seal(grid, tmp_path):
    recs = [
        _record(
            "B4", "tq_embedding", {"bits": 4, "out_dim": 100}, bound=0.95, fresh=0.97
        )
        for _ in range(5)
    ]
    res = S.score(recs, grid)
    assert res["R"]["verdict"] == "PASS" and res["C"]["violations"] == 0
    # 5 runs x 10 cpu-s against 6 reachable configs x 3 seeds x 100 core-s
    assert res["K"]["ratio_by_arm"][ARM] == pytest.approx(50 / 1800)

    root = tmp_path / "records"
    (root / ARM).mkdir(parents=True)
    lines = []
    for i, r in enumerate(recs):
        p = root / ARM / f"B{i}.json"
        p.write_text(json.dumps(r))
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()} {ARM}/B{i}.json")
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    assert len(S.verified_records(str(root))) == 5
    (root / ARM / "B0.json").write_text(json.dumps({**recs[0], "cpu_seconds": 1}))
    with pytest.raises(SystemExit, match="does not match"):
        S.verified_records(str(root))


def test_binomial_cap():
    assert S.binomial_cap(42) == 5  # P(Bin(42, .05) > 5) < .05 <= P(> 4)

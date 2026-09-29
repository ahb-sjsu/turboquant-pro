"""The systems harness (benchmarks/systems, docs/PREREG_systems.md) on a small
synthetic arm: every system it can import runs the same protocol, bytes per row
are measured, stages are reported against their own references, and strata
split the recalls when a strata file is present."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

from rabitq_public import gt as GT  # noqa: E402
from rabitq_public.datasets import SPECS, Dataset, Spec  # noqa: E402
from systems import cell as C  # noqa: E402
from systems import grid as G  # noqa: E402

N, D, NQ = 3000, 32, 40


@pytest.fixture
def arm(tmp_path):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((N, D)) * np.geomspace(2.0, 0.3, D)).astype(np.float32)
    d = tmp_path / "smoke-sys"
    d.mkdir()
    np.save(d / "part_000.npy", x[: N // 2])
    np.save(d / "part_001.npy", x[N // 2 :])
    np.save(d / "queries.npy", rng.standard_normal((NQ, D)).astype(np.float32))
    SPECS["smoke-systems"] = Spec("smoke-systems", "npy", "smoke-sys", NQ, parts=2)
    ds = Dataset("smoke-systems", str(tmp_path))
    (tmp_path / "gt").mkdir()
    np.save(tmp_path / "gt" / "smoke-systems.npy", GT.exact_topk(ds, ds.queries))
    (tmp_path / "strata").mkdir()
    half = list(range(NQ // 2))
    rest = list(range(NQ // 2, NQ))
    (tmp_path / "strata" / "smoke-systems.json").write_text(
        json.dumps({"strata": {"easy": half, "hard": rest}})
    )
    yield tmp_path
    SPECS.pop("smoke-systems", None)


def _run(root, cfg):
    cell = dict(cfg, dataset="smoke-systems", seed=0, repeats=1, n_latency=5)
    cell["cell_id"] = G.cell_id("smoke-systems", cell)
    out = root / "out"
    with open(C.run(cell, str(root), str(out), threads=1), encoding="utf-8") as f:
        return json.load(f)


def test_tq_reports_both_scorers_and_the_scoring_stage(arm):
    rec = _run(arm, dict(method="tq", out_dim=16, bits=4))
    assert set(rec["variants"]) == {"exact", "fast"}
    v = rec["variants"]["exact"]
    assert v["throughput"]["qps"] > 0 and v["latency"]["n"] == 5
    assert v["latency"]["p50_ms"] <= v["latency"]["p99_ms"]
    assert [r["depth"] for r in v["rerank"]] == list(C.RERANK_DEPTHS)
    assert v["rerank"][-1]["recall10"] >= v["recall10_single"] - 1e-9
    assert set(v["recall10_single_by_stratum"]) == {"easy", "hard"}
    s = rec["stages"]["scoring_fast_vs_exact"]
    assert 0 <= s["min"] <= s["mean"] <= 1
    # packed 4-bit codes of 16 dims (8 B) plus the per-row norms the kernel reads
    assert rec["bytes_per_row"] >= 8 + 8
    assert rec["provenance"]["scorers"]["reference"] == "exact-float"
    assert rec["machine"]["threads"] == 1 and rec["rerank_tier_bytes_per_row"] == D * 4


def test_tq_ivf_reports_the_routing_stage(arm):
    rec = _run(arm, dict(method="tq_ivf", out_dim=16, bits=4, nlist=16, nprobe=4))
    r = rec["stages"]["routing_nprobe4_vs_nprobe_all"]
    assert 0 <= r["mean"] <= 1


def test_faiss_systems_measure_bytes_that_grow_with_n(arm):
    pytest.importorskip("faiss")
    pq = _run(arm, dict(method="pq", m=8))
    assert pq["bytes_per_row"] == pytest.approx(8, abs=0.5)  # 8 one-byte codes
    rb = _run(arm, dict(method="rabitq_ivf", bits=1, nlist=16, nprobe=4))
    assert "routing_nprobe4_vs_nprobe_all" in rb["stages"]
    assert rb["bytes_per_row"] > 8  # codes, factors and the stored 8-byte id


def test_scann_runs_the_same_protocol(arm):
    pytest.importorskip("scann")
    rec = _run(
        arm,
        dict(
            method="scann",
            num_leaves=16,
            leaves_to_search=4,
            dims_per_block=2,
            aq_threshold=0.2,
            reorder=20,
        ),
    )
    assert set(rec["variants"]) == {"ah", "reorder", "all_leaves"}
    assert "routing_ah_vs_all_leaves" in rec["stages"]


def test_the_draft_grid_names_every_system_on_every_arm():
    cs = G.cells()
    assert {c["method"] for c in cs} == {
        "tq",
        "tq_ivf",
        "pq",
        "opq",
        "rabitq_ivf",
        "scann",
    }
    assert len({c["cell_id"] for c in cs}) == len(cs)
    assert {c["dataset"] for c in cs} == set(G.DIMS)

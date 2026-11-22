# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond. MIT License.
"""Feasibility before the sweep (issue #176, phase 1): the omission floor
measured, the dimensions a consumer never reads, the attainable region of the
available widths, and the two verdicts that are not a codec choice."""

from __future__ import annotations

import json

import numpy as np
import pytest

from turboquant_pro.cli import main
from turboquant_pro.feasibility import (
    ABSTAIN,
    INFEASIBLE,
    MAX_WIDTH,
    PASS,
    feasibility,
    min_bytes_for_distortion,
)
from turboquant_pro.observer import ObserverContract
from turboquant_pro.spectrum import DISTORTION

D = 32


def _projector(lo, hi):
    P = np.zeros((D, D))
    P[lo:hi, lo:hi] = np.eye(hi - lo)
    return P


def _corpus(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, D)) * np.linspace(3.0, 0.5, D)).astype(np.float32)


# ---- the attainable region ---------------------------------------------------


def test_min_bytes_falls_as_the_target_loosens():
    w = np.linspace(1.0, 0.01, D)
    b10, _ = min_bytes_for_distortion(w, 0.10)
    b05, _ = min_bytes_for_distortion(w, 0.05)
    b01, _ = min_bytes_for_distortion(w, 0.01)
    assert b10 <= b05 <= b01
    assert all(b is not None for b in (b10, b05, b01))


def test_a_target_below_the_widest_width_is_unreachable():
    w = np.ones(D)
    # the widest width leaves exactly D(MAX_WIDTH) of the signal
    assert min_bytes_for_distortion(w, DISTORTION[MAX_WIDTH] * 0.5) == (None, None)
    b, bits = min_bytes_for_distortion(w, DISTORTION[MAX_WIDTH] * 1.01)
    assert b is not None and int(bits.max()) == MAX_WIDTH


def test_the_returned_allocation_actually_meets_the_target():
    w = np.linspace(1.0, 0.01, D)
    _, bits = min_bytes_for_distortion(w, 0.05)
    left = float(np.sum(w * np.array([DISTORTION[int(b)] for b in bits])))
    assert left <= 0.05 * w.sum() + 1e-12


# ---- the report --------------------------------------------------------------


def test_healthy_corpus_passes_and_counts_the_unread_dimensions():
    r = feasibility(_corpus(), _projector(0, 8), max_distortion=0.05)
    assert r.result == PASS and r.action is None
    assert 7.0 < r.observable["observable_rank"] < 9.0
    assert r.observable["unread_dimensions"] > 10
    assert r.observable["omitted_sensitivity_fraction"] == pytest.approx(0.0, abs=1e-9)
    assert r.attainable["min_bytes"] is not None
    assert r.source["rank"] == D
    json.dumps(r.as_dict())
    text = r.explain()
    assert "Observable rank" in text and "Result: PASS" in text


def test_a_consumer_reading_where_the_corpus_is_flat_is_infeasible():
    x = _corpus()
    x[:, 16:] = 0.0  # the encoder produced nothing along these directions
    r = feasibility(x, _projector(20, 28), max_distortion=0.05)
    assert r.result == INFEASIBLE
    assert r.observable["omitted_sensitivity_fraction"] == pytest.approx(1.0, abs=1e-6)
    assert "carries no row-to-row information" in r.reason
    assert "re-embed" in r.action
    assert r.source["rank"] == 16
    assert r.warnings == []


def test_a_target_the_widths_cannot_reach_is_infeasible():
    r = feasibility(_corpus(), _projector(0, 8), max_distortion=1e-6)
    assert r.result == INFEASIBLE
    assert "no allocation of these widths reaches it" in r.reason
    assert r.attainable["min_bytes"] is None
    assert r.attainable["floor_distortion_fraction"] > 1e-6


def test_a_rank_floor_no_compressed_code_certifies_is_infeasible():
    r = feasibility(_corpus(), _projector(0, 8), min_tau=0.999, max_distortion=0.05)
    assert r.result == INFEASIBLE
    assert r.rank["near_exact"] is True
    assert "validate or rerank" in r.action


def test_a_reachable_rank_floor_passes_and_reports_its_kappa():
    r = feasibility(_corpus(), _projector(0, 8), min_tau=0.5, max_distortion=0.05)
    assert r.result == PASS
    assert r.rank["max_certifiable_kappa"] > 1.01 and not r.rank["near_exact"]


def test_an_estimated_operator_abstains_when_it_cannot_identify_the_rest():
    r = feasibility(
        _corpus(), _projector(0, 8), max_distortion=0.05, exact_operator=False
    )
    assert r.result == ABSTAIN
    assert r.identification["unidentified_variance_fraction"] > 0.05
    assert "unread or merely unidentified" in r.reason
    assert "full dimension" in r.action


def test_an_estimated_operator_that_reads_everything_does_not_abstain():
    r = feasibility(_corpus(), np.eye(D), max_distortion=0.05, exact_operator=False)
    assert r.result == PASS
    assert r.identification["unidentified_variance_fraction"] < 0.05


def test_omission_is_checked_before_the_width_floor():
    # both conditions hold; the upstream one is the one to act on
    x = _corpus()
    x[:, 16:] = 0.0
    r = feasibility(x, _projector(20, 28), max_distortion=1e-9)
    assert r.result == INFEASIBLE and "row-to-row information here" in r.reason


def test_a_minority_of_flat_directions_is_a_warning_not_a_verdict():
    # the identity reads every direction; half of them are constant here, which
    # costs a reconstruction consumer nothing and is reported as such
    x = _corpus()
    x[:, 16:] = 0.0
    r = feasibility(x, np.eye(D), max_distortion=0.05)
    assert r.result == PASS
    assert r.observable["omitted_sensitivity_fraction"] == pytest.approx(0.5, abs=1e-6)
    assert r.warnings and "no row-to-row information" in r.warnings[0]
    assert "WARNING" in r.explain()


def test_operator_shape_is_checked():
    with pytest.raises(ValueError):
        feasibility(_corpus(), np.eye(D - 1))


# ---- CLI ----------------------------------------------------------------------


def _contract(consumers):
    return ObserverContract.from_dict(
        {
            "schema": "turboquant-pro/observer-contract",
            "profile": "tqp-observer/1",
            "observer": "svc",
            "target": "embedding",
            "consumers": consumers,
        }
    )


def test_cli_reads_a_contract_and_exits_zero_on_pass(tmp_path, capsys):
    art = tmp_path / "x.npy"
    np.save(art, _corpus())
    c = tmp_path / "c.json"
    c.write_text(
        json.dumps(
            _contract(
                [{"metric": "read_operator", "config": {"provider": "identity"}}]
            ).as_dict()
        ),
        encoding="utf-8",
    )
    out = tmp_path / "f.json"
    rc = main(
        [
            "feasibility",
            "--artifact",
            str(art),
            "--observer",
            str(c),
            "--max-distortion",
            "0.05",
            "--out",
            str(out),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["result"] == PASS
    assert doc["observer"]["observer"] == "svc"
    assert doc["artifact"]["rows_used"] == 2000
    capsys.readouterr()


def test_cli_exits_one_on_infeasible_and_two_without_an_observer(tmp_path, capsys):
    x = _corpus()
    x[:, 16:] = 0.0
    art = tmp_path / "x.npy"
    np.save(art, x)
    # a declared operator reading where the corpus is flat
    P = _projector(20, 28)
    pm = tmp_path / "p.npy"
    np.save(pm, P)
    assert main(["feasibility", "--artifact", str(art), "--reference", "identity"]) == 0
    capsys.readouterr()
    assert main(["feasibility", "--artifact", str(art)]) == 2
    assert "feasible for whom" in capsys.readouterr().err
    assert (
        main(
            [
                "feasibility",
                "--artifact",
                str(art),
                "--reference",
                "identity",
                "--max-distortion",
                "5",
            ]
        )
        == 2
    )
    assert "fraction in (0, 1)" in capsys.readouterr().err


def test_cli_retrieval_consumer_without_queries_is_an_estimate(tmp_path, capsys):
    art = tmp_path / "x.npy"
    np.save(art, _corpus())
    c = tmp_path / "c.json"
    c.write_text(
        json.dumps(
            _contract([{"metric": "topk_cosine", "config": {"k": 10}}]).as_dict()
        ),
        encoding="utf-8",
    )
    rc = main(
        [
            "feasibility",
            "--artifact",
            str(art),
            "--observer",
            str(c),
            "--max-distortion",
            "0.05",
            "--format",
            "json",
        ]
    )
    doc = json.loads(capsys.readouterr().out)
    assert doc["identification"]["checked"] is True
    assert doc["identification"]["exact"] is False
    assert rc in (0, 1)

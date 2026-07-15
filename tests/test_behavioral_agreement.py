# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""
Tests for decision-level behavioral metrics (behavioral_agreement).

The headline acceptance test is the *illusion* scenario: construct two models
with identical accuracy and identical Correctness Agreement, where one is a
faithful copy and the other has churned half its answers. The paper's CA cannot
tell them apart; flip_rate / behavioral_agreement / the noise-floor excess must.

    pytest tests/test_behavioral_agreement.py -v
"""

from __future__ import annotations

import numpy as np

from turboquant_pro.behavioral_agreement import (
    BehavioralReport,
    FlipResult,
    NoiseFloor,
    behavioral_agreement,
    correctness_agreement,
    evaluate,
    flip_rate,
    noise_floor,
)


# ── the paper's metric, exactly ──────────────────────────────────────────
def test_correctness_agreement_matches_definition():
    base = [1, 1, 1, 0]
    quant = [1, 0, 1, 1]
    # both-correct on items 0 and 2 -> 2/4
    assert correctness_agreement(base, quant) == 0.5


def test_ca_bounded_by_min_accuracy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        bc = rng.integers(0, 2, size=200)
        qc = rng.integers(0, 2, size=200)
        ca = correctness_agreement(bc, qc)
        assert ca <= min(bc.mean(), qc.mean()) + 1e-12


# ── symmetric flips ──────────────────────────────────────────────────────
def test_flip_rate_directions():
    # item0: right->right, item1: right->wrong (regression),
    # item2: wrong->right (recovery), item3: wrong->wrong
    base = [1, 1, 0, 0]
    quant = [1, 0, 1, 0]
    r = flip_rate(base, quant)
    assert isinstance(r, FlipResult)
    assert r.regressions == 0.25
    assert r.recoveries == 0.25
    assert r.churn == 0.5
    assert r.net_delta == 0.0  # balanced -> accuracy unchanged
    assert r.acc_base == r.acc_quant == 0.5


def test_accuracy_delta_hides_churn():
    # 100 items: 25 regress, 25 recover -> net accuracy delta 0, churn 0.5
    base = np.array([1] * 50 + [0] * 50)
    quant = np.array([1] * 25 + [0] * 25 + [1] * 25 + [0] * 25)
    r = flip_rate(base, quant)
    assert abs(r.net_delta) < 1e-9  # accuracy says "nothing changed"
    assert r.churn == 0.5  # but half the correctness decisions moved


def test_mcnemar_flags_systematic_regression():
    # heavy one-directional regression: base right, quant wrong on many
    base = np.ones(100, dtype=int)
    quant = np.array([0] * 30 + [1] * 70)
    r = flip_rate(base, quant)
    assert r.regressions == 0.30 and r.recoveries == 0.0
    assert r.mcnemar_p < 0.001  # not a symmetric wash


def test_mcnemar_symmetric_is_not_significant():
    rng = np.random.default_rng(1)
    # balanced discordant pairs -> McNemar should not fire
    base = rng.integers(0, 2, size=400)
    quant = base.copy()
    idx = rng.choice(400, size=40, replace=False)
    # flip 20 each direction among a controlled subset
    for k, i in enumerate(idx):
        quant[i] = 1 - base[i] if k < 20 else base[i]
    r = flip_rate(base, quant)
    assert r.mcnemar_p > 0.05


# ── prediction-level behavioral agreement ────────────────────────────────
def test_behavioral_agreement_label_independent():
    # multiclass predictions; agreement ignores whether they're correct
    base = ["A", "B", "C", "D"]
    quant = ["A", "B", "X", "Y"]
    assert behavioral_agreement(base, quant) == 0.5


def test_identical_models_full_agreement():
    p = np.array([0, 1, 2, 3, 2, 1])
    assert behavioral_agreement(p, p) == 1.0


# ── the illusion scenario: CA identical, behavior different ───────────────
def test_illusion_ca_blind_to_churn():
    n = 200
    rng = np.random.default_rng(7)
    gold = rng.integers(0, 4, size=n)  # 4-way MCQ
    base_pred = gold.copy()
    # base gets 60% right
    wrong = rng.choice(n, size=int(0.4 * n), replace=False)
    base_pred[wrong] = (gold[wrong] + 1) % 4

    # Model 1 ("faithful"): identical predictions -> CA = base acc, no churn
    faithful = base_pred.copy()

    # Model 2 ("churned"): SAME accuracy and SAME correctness set size, but
    # swaps which items are right — take the correct set, make some wrong, and
    # fix an equal number of previously-wrong items.
    churned = base_pred.copy()
    base_correct_idx = np.where(base_pred == gold)[0]
    base_wrong_idx = np.where(base_pred != gold)[0]
    k = 30
    churned[base_correct_idx[:k]] = (gold[base_correct_idx[:k]] + 2) % 4  # now wrong
    churned[base_wrong_idx[:k]] = gold[base_wrong_idx[:k]]  # now right

    ca_faithful = correctness_agreement(base_pred == gold, faithful == gold)
    ca_churned = correctness_agreement(base_pred == gold, churned == gold)
    acc_faithful = np.mean(faithful == gold)
    acc_churned = np.mean(churned == gold)

    # Same accuracy (equal swap) ...
    assert abs(acc_faithful - acc_churned) < 1e-9
    # ... and CA differs by only the k swapped-correct items — accuracy-level
    # metrics look nearly identical.
    assert ca_faithful > ca_churned  # CA sees the k lost, but not the recoveries

    # The behavioral metrics separate them sharply:
    assert behavioral_agreement(base_pred, faithful) == 1.0
    ba_churned = behavioral_agreement(base_pred, churned)
    assert ba_churned < 0.8  # 2k/n = 60/200 = 0.30 of predictions moved

    r_faithful = flip_rate(base_pred == gold, faithful == gold)
    r_churned = flip_rate(base_pred == gold, churned == gold)
    assert r_faithful.churn == 0.0
    assert r_churned.churn > 0.25  # 2k/n churn the faithful copy has none of


# ── noise floor + excess ─────────────────────────────────────────────────
def test_noise_floor_and_excess_z():
    n = 500
    rng = np.random.default_rng(3)
    gold = rng.integers(0, 4, size=n)
    base_pred = gold.copy()
    base_pred[rng.choice(n, 150, replace=False)] = rng.integers(0, 4, size=150)

    # two near-lossless reference runs: each flips ~3% of predictions
    ref_a = base_pred.copy()
    ref_b = base_pred.copy()
    ref_a[rng.choice(n, 15, replace=False)] = rng.integers(0, 4, size=15)
    ref_b[rng.choice(n, 15, replace=False)] = rng.integers(0, 4, size=15)
    floor = noise_floor(ref_a, ref_b, gold=gold)
    assert isinstance(floor, NoiseFloor)
    assert 0.0 < floor.floor_disagreement < 0.12  # small free churn

    # genuine quantization: flips ~25% of predictions -> well above floor
    quant_pred = base_pred.copy()
    quant_pred[rng.choice(n, 125, replace=False)] = rng.integers(0, 4, size=125)
    rep = evaluate(base_pred, quant_pred, gold, floor=floor)
    assert isinstance(rep, BehavioralReport)
    assert rep.disagreement > floor.floor_disagreement
    assert rep.excess_disagreement > 0.10
    assert rep.excess_z > 3.0  # clears the noise floor decisively

    # a run that only matches the floor should NOT clear it
    rep_floor = evaluate(ref_a, ref_b, gold, floor=floor)
    assert abs(rep_floor.excess_disagreement) < 0.05
    assert rep_floor.summary()  # smoke


def test_evaluate_smoke_no_floor():
    gold = np.array([0, 1, 2, 3])
    base = np.array([0, 1, 2, 0])
    quant = np.array([0, 1, 0, 0])
    rep = evaluate(base, quant, gold)
    assert rep.floor is None
    assert np.isnan(rep.excess_z)
    assert "acc_base" in rep.summary()


def test_shape_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        flip_rate([1, 0, 1], [1, 0])
    with pytest.raises(ValueError):
        correctness_agreement([], [])

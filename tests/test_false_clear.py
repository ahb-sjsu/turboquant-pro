# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Empirical false-clear rate: nominal clears while the consumer fails."""

import numpy as np
import pytest

from turboquant_pro.false_clear import (
    FALSE_CLEAR_FAIL,
    FalseClearReport,
    check_false_clear,
    false_clear,
    false_clear_from_scores,
)


def test_perfect_agreement_is_ok():
    accept = np.array([True, False, True, False])
    ok = np.array([True, False, True, False])
    r = false_clear(accept, ok)
    assert r.false_clear_rate == 0.0
    assert r.false_clear_given_cleared == 0.0
    assert r.conservative_miss_rate == 0.0
    assert r.agreement == 1.0
    assert r.verdict == "ok"


def test_kv_keys_scenario_is_a_full_false_clear():
    # cosine cleared everything; the consumer failed everything.
    accept = np.ones(100, dtype=bool)
    ok = np.zeros(100, dtype=bool)
    r = false_clear(accept, ok)
    assert r.false_clear_rate == 1.0
    assert r.false_clear_given_cleared == 1.0
    assert r.verdict == "fail"
    assert r.n_cleared == 100
    assert r.n_consumer_fail == 100


def test_cleared_and_consumer_ok_is_ok():
    accept = np.ones(50, dtype=bool)
    ok = np.ones(50, dtype=bool)
    r = false_clear(accept, ok)
    assert r.false_clear_rate == 0.0
    assert r.false_clear_given_cleared == 0.0
    assert r.verdict == "ok"


def test_conservative_miss_is_not_a_false_clear():
    # nominal rejects everything, consumer would have been fine: harmless.
    accept = np.zeros(40, dtype=bool)
    ok = np.ones(40, dtype=bool)
    r = false_clear(accept, ok)
    assert r.false_clear_rate == 0.0
    assert r.false_clear_given_cleared == 0.0
    assert r.conservative_miss_rate == 1.0
    assert r.verdict == "ok"


def test_nothing_cleared_gives_zero_conditional_and_ok():
    accept = np.zeros(10, dtype=bool)
    ok = np.array([True, False] * 5)
    r = false_clear(accept, ok)
    assert r.n_cleared == 0
    assert r.false_clear_given_cleared == 0.0
    assert r.false_clear_rate == 0.0
    assert r.verdict == "ok"


def test_conditional_rate_recovers_consumer_fail_probability_under_independence():
    rng = np.random.default_rng(0)
    n = 200_000
    accept = rng.random(n) < 0.5
    ok = rng.random(n) < 0.7          # 30% consumer failure, independent of accept
    r = false_clear(accept, ok)
    # P(fail | cleared) -> P(fail) = 0.30 when independent
    assert r.false_clear_given_cleared == pytest.approx(0.30, abs=0.01)
    # joint P(cleared and fail) -> 0.5 * 0.3 = 0.15
    assert r.false_clear_rate == pytest.approx(0.15, abs=0.01)
    assert r.verdict == "fail"


def test_rates_are_bounded_and_consistent():
    rng = np.random.default_rng(3)
    accept = rng.random(1000) < 0.6
    ok = rng.random(1000) < 0.6
    r = false_clear(accept, ok)
    for v in (r.false_clear_rate, r.false_clear_given_cleared,
              r.conservative_miss_rate, r.agreement):
        assert 0.0 <= v <= 1.0
    # joint false-clear never exceeds the conditional (n_cleared <= n)
    assert r.false_clear_rate <= r.false_clear_given_cleared + 1e-12


def test_verdict_thresholds_warn_and_fail():
    # 10% of cleared items fail -> warn (>=0.05, <0.20)
    accept = np.ones(100, dtype=bool)
    ok = np.array([False] * 10 + [True] * 90)
    assert false_clear(accept, ok).verdict == "warn"
    # 25% fail -> fail
    ok2 = np.array([False] * 25 + [True] * 75)
    assert false_clear(accept, ok2).verdict == "fail"


def test_custom_thresholds_are_honored():
    accept = np.ones(100, dtype=bool)
    ok = np.array([False] * 10 + [True] * 90)     # 10% conditional
    assert false_clear(accept, ok, warn=0.15, fail=0.30).verdict == "ok"
    assert false_clear(accept, ok, warn=0.05, fail=0.08).verdict == "fail"


def test_from_scores_higher_is_better_both():
    nominal = np.array([0.99, 0.98, 0.10, 0.05])   # cosine-like
    consumer = np.array([0.95, 0.10, 0.90, 0.20])  # recall-like
    # accept cosine >= 0.5 -> [T,T,F,F]; ok recall >= 0.5 -> [T,F,T,F]
    r = false_clear_from_scores(
        nominal, consumer, nominal_threshold=0.5, consumer_threshold=0.5)
    # cleared={0,1}; among them item 1 fails -> conditional 0.5
    assert r.n_cleared == 2
    assert r.false_clear_given_cleared == pytest.approx(0.5)
    assert r.false_clear_rate == pytest.approx(0.25)


def test_from_scores_distance_lower_is_better():
    # nominal is a distance/MSE (lower better); consumer is a loss (lower better)
    nominal = np.array([0.01, 0.02, 0.90])         # accept dist <= 0.5 -> [T,T,F]
    consumer = np.array([5.0, 9000.0, 3.0])        # ok loss <= 100 -> [T,F,T]
    r = false_clear_from_scores(
        nominal, consumer,
        nominal_threshold=0.5, consumer_threshold=100.0,
        nominal_higher_is_better=False, consumer_higher_is_better=False)
    # cleared={0,1}; item 1 is the KV-keys case (tiny dist, huge loss)
    assert r.n_cleared == 2
    assert r.false_clear_given_cleared == pytest.approx(0.5)
    assert r.verdict == "fail"


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        false_clear(np.ones(3, dtype=bool), np.ones(4, dtype=bool))


def test_empty_raises():
    with pytest.raises(ValueError, match="at least one item"):
        false_clear(np.array([], dtype=bool), np.array([], dtype=bool))


def test_accepts_plain_lists():
    r = false_clear([True, True, False], [False, True, False])
    assert isinstance(r, FalseClearReport)
    assert r.n == 3
    assert r.n_cleared == 2


def test_to_dict_round_trips_all_fields():
    r = false_clear(np.ones(4, dtype=bool), np.array([True, True, False, True]))
    d = r.to_dict()
    for key in ("false_clear_rate", "false_clear_given_cleared",
                "conservative_miss_rate", "agreement", "verdict",
                "n", "n_cleared", "n_consumer_fail", "thresholds"):
        assert key in d
    assert d["thresholds"] == {"warn": pytest.approx(0.05), "fail": pytest.approx(FALSE_CLEAR_FAIL)}


def test_check_false_clear_warns_then_raises():
    accept = np.ones(100, dtype=bool)
    fail_ok = np.array([False] * 30 + [True] * 70)
    with pytest.warns(UserWarning, match="false clear"):
        check_false_clear(accept, fail_ok)
    with pytest.raises(ValueError, match="false clear"):
        check_false_clear(accept, fail_ok, strict=True)
    # a clean case neither warns nor raises
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_false_clear(np.ones(10, dtype=bool), np.ones(10, dtype=bool)).verdict == "ok"

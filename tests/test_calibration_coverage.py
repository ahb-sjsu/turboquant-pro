# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Calibration-set coverage guard (backported from readscope.loading)."""

import numpy as np
import pytest

from turboquant_pro.calibration_coverage import (
    COVERAGE_FAIL,
    calibration_coverage,
    check_calibration_coverage,
)


def test_identical_distributions_score_zero():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4000, 8))
    r = calibration_coverage(x, x)
    assert r.jeffreys == pytest.approx(0.0, abs=1e-9)
    assert r.mean_shift == pytest.approx(0.0, abs=1e-9)
    assert r.spectral_ratio == pytest.approx(1.0, abs=1e-6)
    assert r.verdict == "ok"


def test_a_drawn_sample_of_the_same_distribution_passes():
    rng = np.random.default_rng(1)
    serving = rng.standard_normal((6000, 8))
    calib = rng.standard_normal((600, 8))
    assert calibration_coverage(calib, serving).verdict == "ok"


def test_mean_shift_is_detected():
    rng = np.random.default_rng(2)
    serving = rng.standard_normal((4000, 6))
    shifted = rng.standard_normal((4000, 6)) + 3.0
    r = calibration_coverage(shifted, serving)
    assert r.mean_shift > 2.0
    assert r.jeffreys > calibration_coverage(serving, serving).jeffreys


def test_variance_mismatch_is_detected():
    rng = np.random.default_rng(3)
    serving = rng.standard_normal((4000, 6))
    squashed = rng.standard_normal((4000, 6)) * np.array([8.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    r = calibration_coverage(squashed, serving)
    assert r.spectral_ratio > 5.0
    assert r.verdict in ("warn", "fail")


def test_divergence_grows_with_the_shift():
    rng = np.random.default_rng(4)
    serving = rng.standard_normal((4000, 5))
    near = calibration_coverage(serving + 0.2, serving).jeffreys
    far = calibration_coverage(serving + 4.0, serving).jeffreys
    assert far > near


def test_shape_pooling_matches_calibrate():
    """(B, H, S, D) pools to (N, D) exactly as calibrate() does."""
    rng = np.random.default_rng(5)
    four_d = rng.standard_normal((2, 4, 50, 16))
    flat = four_d.reshape(-1, 16)
    a = calibration_coverage(four_d, flat)
    assert a.dim == 16
    assert a.n_calibration == 2 * 4 * 50
    assert a.jeffreys == pytest.approx(0.0, abs=1e-9)


def test_channel_mismatch_is_an_error():
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError, match="channel counts differ"):
        calibration_coverage(
            rng.standard_normal((100, 8)), rng.standard_normal((100, 16))
        )


def test_strict_mode_raises_on_a_bad_calibration_set():
    rng = np.random.default_rng(7)
    serving = rng.standard_normal((3000, 4))
    terrible = rng.standard_normal((3000, 4)) * 0.05 + 25.0
    assert calibration_coverage(terrible, serving).jeffreys >= COVERAGE_FAIL
    with pytest.raises(ValueError, match="wrong measure"):
        check_calibration_coverage(terrible, serving, strict=True)


def test_non_strict_mode_warns_instead_of_raising():
    rng = np.random.default_rng(8)
    serving = rng.standard_normal((3000, 4))
    terrible = rng.standard_normal((3000, 4)) * 0.05 + 25.0
    with pytest.warns(RuntimeWarning, match="wrong measure"):
        r = check_calibration_coverage(terrible, serving)
    assert r.verdict == "fail"


def test_report_serializes_with_its_thresholds():
    rng = np.random.default_rng(9)
    x = rng.standard_normal((500, 4))
    d = calibration_coverage(x, x).to_dict()
    assert d["thresholds"]["fail"] == COVERAGE_FAIL
    assert d["verdict"] == "ok"

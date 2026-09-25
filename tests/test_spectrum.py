"""The console's spectrum analyzer (turboquant_pro.console.spectrum): its traces are the
read_allocation quantities, so the tests check the identities that tie them together."""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro.console import spectrum as SP
from turboquant_pro.read_allocation import allocate_bits, realised_distortion


def _data(seed=0, n=2000, d=24, qn=500, read_rank=None):
    rng = np.random.default_rng(seed)
    scales = np.geomspace(3.0, 0.05, d)  # a decaying source spectrum
    X = rng.standard_normal((n, d)) * scales
    Q = rng.standard_normal((qn, d))
    if read_rank is not None:  # queries that only read the first read_rank coordinates
        Q[:, read_rank:] = 0.0
    return X, Q


def _codec(X, step=0.25):
    """A uniform scalar quantizer: deterministic, with error in every direction."""
    return np.round(X / step) * step


def test_read_operator_is_the_query_second_moment():
    _, Q = _data()
    P = SP.read_operator_from_queries(Q)
    np.testing.assert_allclose(P, Q.T @ Q / len(Q))
    np.testing.assert_allclose(P, P.T)


def test_noise_sums_to_the_realised_distortion_and_predicted_to_the_model():
    X, Q = _data()
    P = SP.read_operator_from_queries(Q)
    Xh = _codec(X)
    budget = 3.0 * X.shape[1]
    s = SP.sweep(P, X, Xh, budget_bits=budget, t=0.0)
    assert s.realised_total == pytest.approx(realised_distortion(P, X, Xh), rel=1e-9)
    alloc = allocate_bits(P, budget_bits=budget, activations=X)
    assert s.predicted_total == pytest.approx(alloc.predicted_distortion, rel=1e-12)
    assert np.all(np.diff(s.sens) <= 1e-12)  # descending sensitivity
    np.testing.assert_allclose(s.weighted, s.sens * s.var)


def test_the_water_level_separates_funded_from_starved_directions():
    X, Q = _data(seed=1)
    P = SP.read_operator_from_queries(Q)
    # a tight budget (0.2 bits per direction), where water-filling must starve some
    budget = 0.2 * X.shape[1]
    s = SP.sweep(P, X, budget_bits=budget, t=0.0)
    funded = s.bits > 0
    assert funded.any() and (~funded).any()
    assert np.all(s.weighted[funded] > s.water_level)
    assert np.all(s.weighted[~funded] <= s.water_level * (1 + 1e-9))
    assert s.bits.sum() == pytest.approx(budget, rel=1e-6)


def test_error_the_reader_does_not_read_costs_nothing():
    """Queries read only the first 6 coordinates: the noise trace is zero beyond the
    read subspace however large the error there, and the sensitivity has rank 6."""
    X, Q = _data(read_rank=6)
    P = SP.read_operator_from_queries(Q)
    Xh = X.copy()
    Xh[:, 6:] += 5.0 * np.random.default_rng(9).standard_normal(
        (len(X), X.shape[1] - 6)
    )
    s = SP.sweep(P, X, Xh, t=0.0)
    assert np.sum(s.sens > 1e-9) == 6
    assert s.noise[6:].max() <= 1e-9 * s.sens[0]
    assert s.realised_total == pytest.approx(0.0, abs=1e-9)  # nothing read was changed


def test_trace_modes_hold_and_average_in_power():
    t = SP.Trace("sens", mode="maxhold")
    t.update(np.array([0.0, -10.0]))
    t.update(np.array([-5.0, 0.0]))
    np.testing.assert_allclose(t.data, [0.0, 0.0])
    t = SP.Trace("sens", mode="minhold")
    t.update(np.array([0.0, -10.0]))
    t.update(np.array([-5.0, 0.0]))
    np.testing.assert_allclose(t.data, [-5.0, -10.0])
    t = SP.Trace("sens", mode="average", avg_n=2)
    t.update(np.array([0.0]))
    t.update(np.array([10.0]))
    assert t.data[0] == pytest.approx(10 * np.log10((1 + 10) / 2))  # power, not dB
    t = SP.Trace("sens", mode="blank")
    t.update(np.array([1.0]))
    assert t.data is None


def _analyzer_with(values_db):
    an = SP.Analyzer()
    an.last = SP.Sweep(
        0.0,
        np.ones(len(values_db)),
        np.ones(len(values_db)),
        np.ones(len(values_db)),
        np.eye(len(values_db)),
    )
    an.traces[0].data = np.asarray(values_db, dtype=float)
    return an


def test_detectors_reduce_many_directions_per_column():
    an = _analyzer_with([0, -20, -20, -20, -40, -40, -10, -40])
    an.detector = "peak"
    assert an.columns(an.traces[0], 2) == [0.0, -10.0]
    an.detector = "average"
    col = an.columns(an.traces[0], 2)
    assert col[0] == pytest.approx(10 * np.log10((1 + 3 * 0.01) / 4))


def test_markers_peak_next_peak_and_delta():
    an = _analyzer_with([-30, -5, -30, -30, -12, -30, -20, -30])
    m1 = an.peak_search(0)
    assert m1 == 1
    m2 = an.peak_search(0, after=m1)
    assert m2 == 4  # the highest local maximum below the first
    an.markers = [m1, m2]
    ro = an.marker_readout(0)
    assert ro[1]["delta_db"] == pytest.approx(-7.0) and ro[1]["delta_dirs"] == 3


def test_limit_line_defaults_to_the_water_level_and_reports_failures():
    X, Q = _data(seed=2)
    P = SP.read_operator_from_queries(Q)
    an = SP.Analyzer()
    an.feed(SP.sweep(P, X, _codec(X, step=1.5), budget_bits=2.0 * X.shape[1], t=0.0))
    res = an.limit_check()
    assert res["limit_db"] == pytest.approx(float(SP.db(an.last.water_level)))
    lim = res["limit_db"]
    noise_db = an.traces[1].data
    assert res["fail"] == [i for i in range(len(noise_db)) if noise_db[i] > lim]
    an.limit = 1e6
    assert an.limit_check()["passed"] is True


def test_waterfall_is_bounded_and_stop_freezes_the_display():
    X, Q = _data(seed=3)
    P = SP.read_operator_from_queries(Q)
    an = SP.Analyzer(waterfall=5)
    for i in range(8):
        an.feed(SP.sweep(P, X, t=float(i)))
    assert len(an.waterfall) == 5 and an.sweeps == 8
    an.running = False
    an.feed(SP.sweep(P, X, t=99.0))
    assert an.sweeps == 8 and an.last.t == 7.0

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


# -------------------------------------------------------- the session and screen
from turboquant_pro.console import spectrum_view as SV  # noqa: E402
from turboquant_pro.console import tui  # noqa: E402
from turboquant_pro.console.server import ConsoleServer, demo_index  # noqa: E402


@pytest.fixture(scope="module")
def session():
    index, Q, X, source, codec = demo_index(n=1500, dim=48, out_dim=24)
    s = ConsoleServer(
        index,
        Q,
        qps=50,
        k=5,
        rerank=2,
        originals=X,
        source=source,
        http=False,
        codec=codec,
    )
    yield s, X, codec
    s.stop()


def test_a_session_sweep_is_the_real_codec_seen_by_the_real_traffic(session):
    s, X, codec = session
    sw, why = s.spectrum_sweep(n_queries=128, n_sample=400)
    assert why is None and sw.noise is not None and sw.predicted is not None
    sample, recon = s._spec_cache["sample"], s._spec_cache["recon"]
    q = s.workload.queries
    end = s.workload.row or len(q)
    rows = [(end - 1 - i) % len(q) for i in range(128)]
    P = SP.read_operator_from_queries(q[rows])
    assert sw.realised_total == pytest.approx(
        realised_distortion(P, sample, recon), rel=1e-9
    )
    bits = 8.0 * s.index.stored_bytes_per_row
    assert sw.bits.sum() == pytest.approx(bits, rel=1e-6)  # the index's own budget


def test_no_originals_means_no_spectrum_and_says_why():
    index, Q, _, source, _ = demo_index(n=400, dim=32, out_dim=16)
    s = ConsoleServer(index, Q, qps=10, http=False, source=source)
    try:
        sw, why = s.spectrum_sweep()
        assert sw is None and "originals" in why
    finally:
        s.stop()


def _screen_state(session):
    s, _, _ = session
    an = SP.Analyzer()
    for _ in range(3):
        an.feed(s.spectrum_sweep(n_queries=128, n_sample=400)[0])
    an.autoscale()
    return {"view": "spectrum", "analyzer": an, "sel_trace": 0}


@pytest.mark.parametrize("w,h", [(80, 24), (120, 40), (200, 60)])
def test_the_spectrum_screen_reads_like_an_analyzer(session, w, h):
    st = _screen_state(session)
    lines = tui.frame(st, w, h).text()
    assert len(lines) == h and all(len(x) == w for x in lines)
    screen = "\n".join(lines)
    assert "SPECTRUM sweeps 3" in lines[0] and "dB/div" in lines[0]
    assert any(0x2801 <= ord(c) <= 0x28FF for c in screen)
    assert "eff rank" in screen and "realised D" in screen and "predicted D" in screen
    if h >= 30:
        assert "waterfall:" in screen


def test_analyzer_keys_markers_delta_limit_and_span(session):
    st = _screen_state(session)
    an = st["analyzer"]
    assert SV.key(st, "k").startswith("M1")
    m1 = an.markers[0]
    SV.key(st, "d")
    assert an.markers == [m1, m1]
    msg = SV.key(st, "n")  # moves the active (delta) marker, not the reference
    if msg != "no lower peak":
        assert an.markers[0] == m1 and an.markers[1] != m1
        ro = an.marker_readout(0)
        assert ro[1]["delta_db"] <= 0  # a lower peak reads below the reference
    SV.key(st, "l")
    assert an.limit_check()["passed"] is None  # limit off: nothing to judge
    SV.key(st, "l")
    assert an.limit_check()["limit_db"] is not None
    n = an.last.sens.size
    SV.key(st, "[")
    assert an.span()[1] - an.span()[0] == max(4, n // 2)
    assert SV.key(st, "m") == "T1 maxhold" and an.traces[0].data is None
    assert SV.key(st, "c").startswith("T1 = ")


def test_optimal_distortion_is_min_of_weighted_and_the_water_level():
    """Reverse water-filling: a funded direction ends exactly at theta, a starved one
    keeps all of w. So the water level is the right limit line for realised noise."""
    for seed, per_dir in [(4, 0.2), (5, 1.0), (6, 3.0)]:
        X, Q = _data(seed=seed)
        s = SP.sweep(
            SP.read_operator_from_queries(Q), X, budget_bits=per_dir * X.shape[1], t=0.0
        )
        np.testing.assert_allclose(
            s.predicted, np.minimum(s.weighted, s.water_level), rtol=1e-6
        )


def test_the_waterfall_has_its_own_colour_scale():
    X, Q = _data(seed=7)
    an = SP.Analyzer()
    an.waterfall_source = "sens"
    an.feed(SP.sweep(SP.read_operator_from_queries(Q), X, t=0.0))
    lo, hi = an.waterfall_range()
    sens_db = SP.db(an.last.sens)
    assert hi == pytest.approx(sens_db.max()) and lo < hi
    an.ref_db = -200.0  # moving the graticule does not move the waterfall's scale
    assert an.waterfall_range() == (lo, hi)


# ------------------------------------------------------------ compare mode
def _rand_basis(d, seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    return q


def test_measurements_in_a_borrowed_basis_keep_the_traces_exact():
    """In any orthonormal basis, sensitivity sums to tr(P) and noise to the realised
    distortion; in the operator's own eigenbasis the borrowed sweep is the ordinary one.
    """
    X, Q = _data(seed=8)
    P = SP.read_operator_from_queries(Q)
    Xh = _codec(X)
    for seed in (1, 2):
        s = SP.sweep(P, X, Xh, basis=_rand_basis(X.shape[1], seed), t=0.0)
        assert s.in_reference_basis and s.predicted is None
        assert s.sens.sum() == pytest.approx(np.trace(P), rel=1e-10)
        assert s.realised_total == pytest.approx(
            realised_distortion(P, X, Xh), rel=1e-9
        )
    own = SP.sweep(P, X, Xh, t=0.0)
    same = SP.sweep(P, X, Xh, basis=own.basis, t=0.0)
    np.testing.assert_allclose(same.sens, own.sens, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(same.noise, own.noise, rtol=1e-7, atol=1e-14)


def test_drift_and_delta_traces_against_a_stored_reference():
    X, Q = _data(seed=9)
    P = SP.read_operator_from_queries(Q)
    an = SP.Analyzer()
    an.traces[0].mode = "delta"
    an.feed(SP.sweep(P, X, _codec(X), t=0.0))
    an.store_reference()
    an.feed(SP.sweep(P, X, _codec(X), basis=an.reference.basis, t=1.0))
    assert an.drift() == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(an.traces[0].data, 0.0, atol=1e-9)  # nothing changed
    Q2 = Q.copy()
    Q2[:, :6] *= 3.0  # the traffic now leans on the first six coordinates
    P2 = SP.read_operator_from_queries(Q2)
    an.feed(SP.sweep(P2, X, _codec(X), basis=an.reference.basis, t=2.0))
    assert an.drift() > 0.5
    assert np.abs(an.traces[0].data).max() > 3.0  # the delta trace shows where
    an.clear_reference()
    assert an.drift() is None


def test_the_reference_key_and_status(session):
    st = _screen_state(session)
    an = st["analyzer"]
    assert SV.key(st, "R").startswith("reference stored")
    s, _, _ = session
    an.feed(s.spectrum_sweep(n_queries=128, n_sample=400, basis=an.reference.basis)[0])
    line = tui.frame(st, 120, 40).text()[0]
    assert "REF drift" in line
    assert SV.key(st, "R").startswith("reference cleared") and an.reference is None


def test_decoration_never_overwrites_the_status_line(session):
    st = _screen_state(session)
    st["analyzer"].store_reference()
    for w in (80, 100, 120, 160):
        top = tui.frame(st, w, 30).text()[0]
        assert "REF" in top
        assert "TurboQuant console" not in top or top.index("TurboQuant console") > (
            top.index("REF")
        )

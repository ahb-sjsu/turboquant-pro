"""The console's oscilloscope engine (turboquant_pro.console.scope), on synthetic
signals, the way a scope is checked against a function generator."""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro.console import scope as S


def tr(t, latency, i=0, scan_path="kernel", agree=None, k=5):
    d = {
        "id": f"{i:016x}",
        "started_unix": t,
        "input": {"n_queries": 1},
        "total_ms": latency,
        "scan_path": scan_path,
        "params": {"workload_row": i},
        "stages": [{"name": "scan", "ms": latency * 0.8, "candidates": 20}],
        "results": None,
    }
    if agree is not None:
        d["results"] = {
            "k": k,
            "approximate": [],
            "final": [],
            "rerank_agreement": agree,
        }
    return d


def feed(sc, ts, ys, **kw):
    for i, (t, y) in enumerate(zip(ts, ys)):
        sc.feed(tr(t, y, i, **kw), now=t)


def test_1_2_5_steps():
    assert [S.step_125(x) for x in (0.7, 1, 1.3, 3, 7, 0.013)] == [1, 1, 2, 5, 10, 0.02]
    assert S.step(2, up=True) == 5 and S.step(5, up=True) == 10
    assert S.step(1, up=False) == 0.5 and S.step(0.2, up=False) == pytest.approx(0.1)


def test_edge_trigger_fires_on_the_crossing_with_slope_and_holdoff():
    sc = S.Scope()
    sc.trigger = S.Trigger(
        kind="edge", source="latency", level=10, slope="rising", mode="normal"
    )
    # 0.2 s records: each completes before the next crossing (a scope ignores triggers
    # while a record is still acquiring, as this one does)
    sc.s_per_div = 0.02
    ys = [5, 5, 12, 12, 5, 12, 5, 5] + [5] * 30
    feed(sc, np.arange(len(ys)) * 0.1, ys)
    trig = [seg.trigger_t for seg in sc.segments]
    assert trig == pytest.approx([0.2, 0.5])  # two rising crossings, none on falls
    sc2 = S.Scope()
    sc2.trigger = S.Trigger(level=10, slope="falling", mode="normal")
    sc2.s_per_div = 0.02
    feed(sc2, np.arange(len(ys)) * 0.1, ys)
    assert [s.trigger_t for s in sc2.segments] == pytest.approx([0.4, 0.6])
    sc3 = S.Scope()
    sc3.trigger = S.Trigger(level=10, mode="normal", holdoff_s=0.5)
    sc3.s_per_div = 0.01
    feed(sc3, np.arange(len(ys)) * 0.1, ys)
    assert [s.trigger_t for s in sc3.segments] == pytest.approx([0.2])  # 0.5 held off


def test_single_catches_one_shot_then_stops_with_the_pretrigger_share():
    sc = S.Scope()
    sc.single()
    sc.trigger.level, sc.trigger.position = 10, 0.25
    sc.s_per_div = 0.1  # 1 s record
    ys = [5] * 20 + [50] + [5] * 40
    ts = np.arange(len(ys)) * 0.05
    feed(sc, ts, ys)
    assert not sc.running and sc.status == "stop" and len(sc.segments) == 1
    rec = sc.record
    assert rec.trigger_t == pytest.approx(1.0)
    assert (rec.trigger_t - rec.t0) / (rec.t1 - rec.t0) == pytest.approx(0.25)
    assert any(s.values["latency"] == 50 for s in rec.samples)
    feed(sc, ts + 10, ys)  # stopped: a second one-shot is not captured
    assert len(sc.segments) == 1


def test_pulse_and_logic_triggers():
    sc = S.Scope()
    sc.trigger = S.Trigger(kind="pulse", level=10, width=3, mode="normal")
    sc.s_per_div = 0.01
    ys = [12, 12, 5, 12, 12, 12, 12, 5]
    feed(sc, np.arange(len(ys)) * 0.1, ys)
    assert [s.trigger_t for s in sc.segments] == pytest.approx([0.5])  # third in a row
    lg = S.Scope()
    lg.trigger = S.Trigger(
        kind="logic",
        mode="normal",
        conditions=[("scan_path", "==", "numpy"), ("agree", "<", 0.5)],
    )
    lg.s_per_div = 0.01
    lg.feed(tr(0.0, 5, 0, scan_path="numpy", agree=0.9), now=0.0)
    lg.feed(tr(0.1, 5, 1, scan_path="kernel", agree=0.2), now=0.1)
    lg.feed(tr(0.2, 5, 2, scan_path="numpy", agree=0.2), now=0.2)
    lg.tick(1.0)
    assert [s.trigger_id for s in lg.segments] == [f"{2:016x}"]


def test_peak_detect_keeps_a_one_query_spike_that_sampling_can_lose():
    """1000 queries across a 1 s/div screen of 20 columns: one 100 ms spike."""
    ts = np.linspace(0, 9.99, 1000)
    ys = np.full(1000, 2.0)
    ys[503] = 100.0
    peak = S.Scope()
    peak.s_per_div, peak.acquire = 1.0, "peak"
    feed(peak, ts, ys)
    cols = peak.columns("latency", 20, now=10.0)
    assert max(c[1] for c in cols if c) == 100.0
    samp = S.Scope()
    samp.s_per_div, samp.acquire = 1.0, "sample"
    feed(samp, ts, ys)
    assert max(c[1] for c in samp.columns("latency", 20, now=10.0) if c) == 2.0


def test_persistence_decays_and_infinite_accumulates():
    sc = S.Scope()
    sc.s_per_div = 1.0
    ch = sc.channels[0]
    ch.scale, ch.position = 1.0, -4.0  # 0..8 ms on screen
    feed(sc, np.linspace(0, 9.9, 100), [3.0] * 100)
    g1 = sc.persistence(ch, 20, 16, now=10.0).sum()
    g2 = sc.persistence(ch, 20, 16, now=10.0).sum()
    assert g2 == pytest.approx(g1 * sc.decay + g1)
    sc.decay, sc.persist = 1.0, {}
    a = sc.persistence(ch, 20, 16, now=10.0).sum()
    b = sc.persistence(ch, 20, 16, now=10.0).sum()
    assert b == pytest.approx(2 * a)


def test_measurements_and_statistics_across_acquisitions():
    sc = S.Scope()
    sc.trigger = S.Trigger(level=10, mode="normal")
    sc.s_per_div = 0.05
    ys = ([5, 12] + [5] * 20) * 3
    feed(sc, np.arange(len(ys)) * 0.1, ys)
    m = sc.measure("latency", now=len(ys) * 0.1)
    assert m["n"] > 0 and m["max"] >= m["p99"] >= m["p50"] >= m["min"]
    st = sc.statistics("latency", "max")
    assert st["count"] == len(sc.segments) == 3 and st["max"] == 12.0


def test_mask_counts_violations_and_stop_on_fail_captures_the_failure():
    sc = S.Scope()
    sc.masks = {"latency": (None, 20.0)}
    sc.stop_on_fail = True
    sc.s_per_div = 0.1
    ys = [5] * 10 + [30] + [5] * 30
    feed(sc, np.arange(len(ys)) * 0.1, ys)
    assert sc.mask_summary()["latency"]["violations"] == 1
    assert not sc.running and any(s.values["latency"] == 30 for s in sc.record.samples)


def test_autoset_picks_1_2_5_scales_around_the_data():
    sc = S.Scope()
    ts = np.linspace(0, 20, 400)
    feed(sc, ts, 4 + 2 * np.sin(ts))
    sc.autoset(now=20.0)
    ch = sc.channels[0]
    assert ch.scale in {0.5, 1.0, 2.0}
    lo, hi = ch.to_div(2.0), ch.to_div(6.0)
    assert 0 <= lo < hi <= S.VDIV  # the signal fits the screen
    assert sc.trigger.level == pytest.approx(4.0, abs=0.5)


def test_a_trigger_during_acquisition_is_ignored_like_a_scope():
    sc = S.Scope()
    sc.trigger = S.Trigger(level=10, mode="normal")
    sc.s_per_div = 0.1  # 1 s records: the crossing at 0.5 s falls inside the first
    ys = [5, 5, 12, 12, 5, 12, 5, 5] + [5] * 30
    feed(sc, np.arange(len(ys)) * 0.1, ys)
    assert [s.trigger_t for s in sc.segments] == pytest.approx([0.2])

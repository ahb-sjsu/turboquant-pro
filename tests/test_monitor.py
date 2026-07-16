"""
Tests for the production quality monitor.

Usage:
    pytest tests/test_monitor.py -v
"""

from __future__ import annotations

import numpy as np

from turboquant_pro.monitor import QualityMonitor

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

RNG = np.random.default_rng(42)


def _random_pair(dim: int = 128, noise: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Return an (original, reconstructed) pair with controllable noise."""
    original = RNG.standard_normal(dim).astype(np.float32)
    reconstructed = original + RNG.standard_normal(dim).astype(np.float32) * noise
    return original, reconstructed


def _high_quality_pairs(
    n: int = 50, dim: int = 128
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate pairs with very high cosine similarity (>0.99)."""
    return [_random_pair(dim, noise=0.01) for _ in range(n)]


def _low_quality_pairs(
    n: int = 50, dim: int = 128
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate pairs with low cosine similarity (<0.90)."""
    return [_random_pair(dim, noise=1.0) for _ in range(n)]


# ------------------------------------------------------------------ #
# TestQualityMonitor                                                   #
# ------------------------------------------------------------------ #


class TestQualityMonitor:
    """Tests for QualityMonitor."""

    def test_record_returns_cosine(self) -> None:
        """record() returns a float between -1 and 1."""
        mon = QualityMonitor()
        orig, recon = _random_pair()
        cos = mon.record(orig, recon)
        assert isinstance(cos, float)
        assert -1.0 <= cos <= 1.0

    def test_record_batch(self) -> None:
        """record_batch() processes multiple embeddings."""
        mon = QualityMonitor()
        originals = RNG.standard_normal((10, 128)).astype(np.float32)
        noise = RNG.standard_normal((10, 128)).astype(np.float32) * 0.05
        reconstructed = originals + noise
        sims = mon.record_batch(originals, reconstructed)
        assert sims.shape == (10,)
        assert all(-1.0 <= s <= 1.0 for s in sims)
        assert mon.stats()["n_total"] == 10

    def test_stats_keys(self) -> None:
        """stats() returns all expected keys."""
        mon = QualityMonitor()
        orig, recon = _random_pair()
        mon.record(orig, recon)
        s = mon.stats()
        expected_keys = {
            "n_total",
            "n_window",
            "mean_cosine",
            "min_cosine",
            "std_cosine",
            "p95_cosine",
            "quality_floor",
            "is_healthy",
            "n_alerts",
            "drift_detected",
            "median_tangential_fraction",
            "radial_drift_detected",
        }
        assert set(s.keys()) == expected_keys

    def test_healthy_when_above_floor(self) -> None:
        """is_healthy is True for high-quality data."""
        mon = QualityMonitor(quality_floor=0.90)
        for orig, recon in _high_quality_pairs(50):
            mon.record(orig, recon)
        assert mon.stats()["is_healthy"] is True

    def test_unhealthy_when_below_floor(self) -> None:
        """is_healthy is False for degraded data."""
        mon = QualityMonitor(quality_floor=0.99)
        for orig, recon in _low_quality_pairs(50):
            mon.record(orig, recon)
        assert mon.stats()["is_healthy"] is False

    def test_alert_callback_fires(self) -> None:
        """Alert callback is invoked when floor is breached."""
        alerts: list[dict] = []
        mon = QualityMonitor(
            quality_floor=0.999,
            alert_callback=lambda d: alerts.append(d),
        )
        # Record enough low-quality pairs to trigger an alert
        for orig, recon in _low_quality_pairs(20):
            mon.record(orig, recon)
        assert len(alerts) > 0

    def test_alert_callback_receives_details(self) -> None:
        """Alert callback dict contains mean_cosine and quality_floor."""
        alerts: list[dict] = []
        mon = QualityMonitor(
            quality_floor=0.999,
            alert_callback=lambda d: alerts.append(d),
        )
        for orig, recon in _low_quality_pairs(10):
            mon.record(orig, recon)
        assert len(alerts) > 0
        detail = alerts[0]
        assert "mean_cosine" in detail
        assert "quality_floor" in detail
        assert detail["quality_floor"] == 0.999

    def test_rolling_window(self) -> None:
        """Old data falls off after window_size."""
        mon = QualityMonitor(window_size=10)
        for orig, recon in _high_quality_pairs(20):
            mon.record(orig, recon)
        assert mon.stats()["n_total"] == 20
        assert mon.stats()["n_window"] == 10

    def test_drift_detection_stable(self) -> None:
        """No drift detected on data from a stable distribution."""
        rng = np.random.default_rng(99)
        mon = QualityMonitor(window_size=200)
        # Record 200 pairs with uniform noise level
        for _ in range(200):
            orig = rng.standard_normal(128).astype(np.float32)
            recon = orig + rng.standard_normal(128).astype(np.float32) * 0.05
            mon.record(orig, recon)
        assert mon.check_drift() is False

    def test_drift_detection_shift(self) -> None:
        """Drift is detected when quality changes mid-window."""
        rng = np.random.default_rng(77)
        mon = QualityMonitor(window_size=200)
        # First 100: very high quality
        for _ in range(100):
            orig = rng.standard_normal(128).astype(np.float32)
            recon = orig + rng.standard_normal(128).astype(np.float32) * 0.01
            mon.record(orig, recon)
        # Next 100: much lower quality
        for _ in range(100):
            orig = rng.standard_normal(128).astype(np.float32)
            recon = orig + rng.standard_normal(128).astype(np.float32) * 0.8
            mon.record(orig, recon)
        assert mon.check_drift() is True

    def test_metrics_dict_keys(self) -> None:
        """metrics_dict() returns all Prometheus-style keys."""
        mon = QualityMonitor()
        for orig, recon in _high_quality_pairs(10):
            mon.record(orig, recon)
        m = mon.metrics_dict()
        expected_keys = {
            "turboquant_quality_mean_cosine",
            "turboquant_quality_min_cosine",
            "turboquant_quality_std_cosine",
            "turboquant_quality_p95_cosine",
            "turboquant_quality_total_records",
            "turboquant_quality_alerts_total",
            "turboquant_quality_is_healthy",
            "turboquant_quality_drift_detected",
            "turboquant_quality_median_tangential_fraction",
            "turboquant_quality_radial_drift_detected",
        }
        assert set(m.keys()) == expected_keys

    def test_metrics_dict_values(self) -> None:
        """Numeric values in metrics_dict are reasonable."""
        mon = QualityMonitor()
        for orig, recon in _high_quality_pairs(20):
            mon.record(orig, recon)
        m = mon.metrics_dict()
        assert 0.0 <= m["turboquant_quality_mean_cosine"] <= 1.0
        assert 0.0 <= m["turboquant_quality_min_cosine"] <= 1.0
        assert m["turboquant_quality_std_cosine"] >= 0.0
        assert m["turboquant_quality_total_records"] == 20
        assert m["turboquant_quality_is_healthy"] in (0, 1)
        assert m["turboquant_quality_drift_detected"] in (0, 1)

    def test_reset(self) -> None:
        """reset() clears all state."""
        mon = QualityMonitor()
        for orig, recon in _high_quality_pairs(30):
            mon.record(orig, recon)
        assert mon.stats()["n_total"] == 30
        mon.reset()
        s = mon.stats()
        assert s["n_total"] == 0
        assert s["n_window"] == 0
        assert s["n_alerts"] == 0

    def test_default_no_callback(self) -> None:
        """Works without an alert callback even when quality is below floor."""
        mon = QualityMonitor(quality_floor=0.999)
        # Should not raise even when quality is poor
        for orig, recon in _low_quality_pairs(10):
            mon.record(orig, recon)
        assert mon.stats()["n_alerts"] > 0


class TestTangentialFraction:
    """The (A2) tangential-fraction stream: the KV-keys-class guardrail."""

    def test_tangential_stat_present_after_records(self) -> None:
        """median_tangential_fraction is populated once pairs accumulate."""
        mon = QualityMonitor()
        for orig, recon in _high_quality_pairs(30):
            mon.record(orig, recon)
        s = mon.stats()
        assert 0.0 <= s["median_tangential_fraction"] <= 1.0

    def test_direction_dominated_stream_reads_high(self) -> None:
        """Random gaussian vectors differ mostly in direction: fraction ~1."""
        mon = QualityMonitor()
        rng = np.random.default_rng(7)
        for _ in range(60):
            v = rng.standard_normal(128)
            mon.record(v, v)
        assert mon.stats()["median_tangential_fraction"] > 0.8

    def test_norm_dominated_stream_reads_low(self) -> None:
        """Vectors on a fixed ray differ only in norm: fraction ~0."""
        mon = QualityMonitor()
        rng = np.random.default_rng(7)
        direction = rng.standard_normal(128)
        direction /= np.linalg.norm(direction)
        for _ in range(60):
            v = direction * float(rng.uniform(0.5, 5.0))
            mon.record(v, v)
        assert mon.stats()["median_tangential_fraction"] < 0.2

    def test_radial_drift_detected_on_regime_change(self) -> None:
        """Direction-dominated -> norm-dominated flips check_radial_drift."""
        mon = QualityMonitor(window_size=400)
        rng = np.random.default_rng(7)
        for _ in range(150):
            v = rng.standard_normal(128)
            mon.record(v, v)
        assert not mon.check_radial_drift()
        direction = rng.standard_normal(128)
        direction /= np.linalg.norm(direction)
        for _ in range(150):
            v = direction * float(rng.uniform(0.5, 5.0))
            mon.record(v, v)
        assert mon.check_radial_drift()

    def test_reservoir_disabled(self) -> None:
        """tangential_reservoir=0 disables the statistic cleanly."""
        mon = QualityMonitor(tangential_reservoir=0)
        for orig, recon in _high_quality_pairs(20):
            mon.record(orig, recon)
        s = mon.stats()
        assert np.isnan(s["median_tangential_fraction"])
        assert s["radial_drift_detected"] is False

    def test_reset_clears_tangential_state(self) -> None:
        """reset() clears the tangential window and reservoir."""
        mon = QualityMonitor()
        for orig, recon in _high_quality_pairs(20):
            mon.record(orig, recon)
        mon.reset()
        assert np.isnan(mon.stats()["median_tangential_fraction"])


class TestA2Health:
    """is_healthy must incorporate the (A2) tangential signal, not cosine alone.

    A stream can read perfect cosine while sliding into the norm-dominated regime
    where angular quantization silently destroys ranking; the health verdict must
    catch that.
    """

    @staticmethod
    def _norm_dominated(mon: QualityMonitor, n: int = 80) -> None:
        """Record n pairs on a fixed ray (differ only in norm) — recon == orig, so
        cosine is a perfect 1.0 while the tangential fraction collapses to ~0."""
        rng = np.random.default_rng(7)
        direction = rng.standard_normal(128)
        direction /= np.linalg.norm(direction)
        for _ in range(n):
            v = (direction * float(rng.uniform(0.5, 5.0))).astype(np.float32)
            mon.record(v, v)

    def test_high_quality_stream_stays_healthy(self) -> None:
        """Regression: direction-dominated high-quality data must not false-trip."""
        mon = QualityMonitor(quality_floor=0.90)
        for orig, recon in _high_quality_pairs(60):
            mon.record(orig, recon)
        assert mon.stats()["is_healthy"] is True

    def test_norm_collapse_below_floor_is_unhealthy(self) -> None:
        """Perfect cosine but collapsed tangential fraction reads UNHEALTHY once
        the (A2) noncollapse floor is set."""
        mon = QualityMonitor(quality_floor=0.5, tangential_floor=0.3)
        self._norm_dominated(mon)
        s = mon.stats()
        assert s["mean_cosine"] > 0.99  # cosine is blind to the collapse
        assert s["median_tangential_fraction"] < 0.2
        assert s["is_healthy"] is False  # (A2) caught what cosine could not

    def test_floor_off_by_default(self) -> None:
        """Without an explicit floor, a *stable* norm-dominated stream (no drift)
        stays healthy — the level gate is opt-in, the drift gate needs a shift."""
        mon = QualityMonitor(quality_floor=0.5)  # tangential_floor defaults to 0
        self._norm_dominated(mon)
        assert mon.stats()["is_healthy"] is True

    def test_downward_drift_is_unhealthy(self) -> None:
        """A regime shift from direction- to norm-dominated (perfect cosine
        throughout) trips health via the self-calibrating drift guard, no floor."""
        mon = QualityMonitor(quality_floor=0.5, window_size=400)
        rng = np.random.default_rng(7)
        for _ in range(150):
            v = rng.standard_normal(128).astype(np.float32)
            mon.record(v, v)
        assert mon.stats()["is_healthy"] is True  # stable, healthy so far
        self._norm_dominated(mon, n=150)
        s = mon.stats()
        assert s["radial_drift_detected"] is True
        assert s["is_healthy"] is False  # downward (A2) drift caught

    def test_reservoir_disabled_leaves_a2_inert(self) -> None:
        """tangential_reservoir=0 disables the statistic — (A2) must not force an
        unhealthy verdict on good cosine when there is no signal to judge."""
        mon = QualityMonitor(quality_floor=0.90, tangential_reservoir=0)
        for orig, recon in _high_quality_pairs(30):
            mon.record(orig, recon)
        assert mon.stats()["is_healthy"] is True

    def test_alert_reason_names_a2(self) -> None:
        """An (A2)-driven alert reports its reason, even when cosine is perfect."""
        alerts: list[dict] = []
        mon = QualityMonitor(
            quality_floor=0.0,  # cosine can never alert
            tangential_floor=0.3,
            alert_callback=lambda d: alerts.append(d),
        )
        self._norm_dominated(mon)
        assert len(alerts) > 0
        assert any("(A2)" in r for r in alerts[-1]["reasons"])

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

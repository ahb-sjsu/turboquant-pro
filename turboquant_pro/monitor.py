# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""
Production quality monitor for TurboQuant-compressed embeddings.

Tracks cosine similarity between original and reconstructed embeddings over
time, detects distribution drift via a scipy-free Kolmogorov-Smirnov test,
and raises alerts when quality drops below a configurable floor.

Also tracks the **(A2) tangential fraction** of the incoming data stream --
the share of pairwise displacement that survives row-normalization,
``(|x - y|^2 - (|x| - |y|)^2) / |x - y|^2`` over sampled pairs of recorded
originals. Angular quantization is blind to the radial complement, so a
drift of this statistic toward 0 (norm-dominated variation, rising hubness)
predicts ranking damage that cosine similarity *cannot see* -- so it feeds the
health verdict directly: ``is_healthy`` and the alert require both the cosine
floor *and* (A2) noncollapse, never cosine on its own. Scope note:
this guards the radial-displacement failure class; the v1.2.0 KV-keys
incident was the *other* class (direction concentration under a shared
per-channel offset, where the tangential fraction reads ~1), which is
caught at calibration time by the end-to-end probe in
``turboquant_pro.a2_probe`` -- run both.

Designed for integration with Prometheus, Datadog, or any metrics pipeline
that polls a flat dict of gauge/counter values.

Usage::

    from turboquant_pro.monitor import QualityMonitor

    monitor = QualityMonitor(quality_floor=0.95, window_size=1000)

    # In your serving loop:
    cos = monitor.record(original_embedding, reconstructed_embedding)

    # Periodically export metrics:
    metrics = monitor.metrics_dict()
    # -> {"turboquant_quality_mean_cosine": 0.982, ...}
"""

from __future__ import annotations

import collections
import logging
import math
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["QualityMonitor"]


class QualityMonitor:
    """Production quality monitor for compressed embeddings.

    Tracks cosine similarity between original and reconstructed embeddings
    over time, detects distribution drift, and raises alerts when quality
    drops below a threshold.

    Args:
        quality_floor: Minimum acceptable mean cosine similarity (default 0.95).
        window_size: Rolling window size for statistics (default 1000).
        alert_callback: Optional callable invoked when quality drops below floor.
            Receives a dict with alert details.
        tangential_reservoir: Size of the reservoir of recent originals used
            to sample pairs for the (A2) tangential-fraction statistic
            (default 64; set 0 to disable the statistic entirely).
        tangential_floor: (A2) noncollapse floor. When > 0, ``is_healthy`` also
            requires the median tangential fraction to stay at or above this
            value — a hard gate against the norm-dominated regime where angular
            quantization damages ranking while cosine still reads fine. Default
            0.0 leaves this explicit level gate off; the *directional* (A2) guard
            (health drops on a significant downward drift of the tangential
            stream) is self-calibrating and always on when the statistic is
            enabled. See the module docstring and ``turboquant_pro.a2_probe``.
    """

    def __init__(
        self,
        quality_floor: float = 0.95,
        window_size: int = 1000,
        alert_callback: Callable | None = None,
        tangential_reservoir: int = 64,
        tangential_floor: float = 0.0,
    ) -> None:
        self._quality_floor = quality_floor
        self._window_size = window_size
        self._alert_callback = alert_callback
        self._reservoir_size = tangential_reservoir
        self._tangential_floor = tangential_floor

        self._window: collections.deque[float] = collections.deque(
            maxlen=window_size,
        )
        self._tang_window: collections.deque[float] = collections.deque(
            maxlen=window_size,
        )
        self._reservoir: list[np.ndarray] = []
        self._res_rng = np.random.default_rng(0)
        self._n_total: int = 0
        self._n_alerts: int = 0

    # ------------------------------------------------------------------ #
    # Core recording                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two 1-D vectors."""
        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _tangential_fraction(x: np.ndarray, y: np.ndarray) -> float:
        """Share of the displacement x - y that survives row-normalization.

        (|x - y|^2 - (|x| - |y|)^2) / |x - y|^2 in [0, 1]; NaN if x == y.
        This is condition (A2) of the angular-transfer theorem, measured
        pairwise (see ``turboquant_pro.a2_probe``).
        """
        d2 = float(((x - y) ** 2).sum())
        if d2 <= 0.0:
            return float("nan")
        dr = float(np.linalg.norm(x)) - float(np.linalg.norm(y))
        frac = (d2 - dr * dr) / d2
        return float(min(max(frac, 0.0), 1.0))

    def _record_tangential(self, original: np.ndarray) -> None:
        """Update the (A2) tangential window and the originals reservoir."""
        if self._reservoir_size <= 0:
            return
        if self._reservoir:
            partner = self._reservoir[
                int(self._res_rng.integers(0, len(self._reservoir)))
            ]
            frac = self._tangential_fraction(original, partner)
            if not math.isnan(frac):
                self._tang_window.append(frac)
        # Reservoir sampling keeps a uniform sample of all originals seen.
        if len(self._reservoir) < self._reservoir_size:
            self._reservoir.append(original.copy())
        else:
            slot = int(self._res_rng.integers(0, self._n_total + 1))
            if slot < self._reservoir_size:
                self._reservoir[slot] = original.copy()

    def record(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Record a single original/reconstructed pair.

        Computes cosine similarity, appends to the rolling window, updates
        the (A2) tangential-fraction stream, checks the alert threshold, and
        returns the similarity value.
        """
        original = original.ravel()
        cos = self._cosine_similarity(
            original,
            reconstructed.ravel(),
        )
        self._window.append(cos)
        self._record_tangential(original)
        self._n_total += 1
        self._maybe_alert()
        return cos

    def record_batch(
        self,
        originals: np.ndarray,
        reconstructed: np.ndarray,
    ) -> np.ndarray:
        """Record a batch of original/reconstructed pairs.

        Returns an array of cosine similarities, one per row.
        """
        if originals.ndim == 1:
            originals = originals.reshape(1, -1)
            reconstructed = reconstructed.reshape(1, -1)

        n = originals.shape[0]
        sims = np.empty(n, dtype=np.float64)
        for i in range(n):
            sims[i] = self.record(originals[i], reconstructed[i])
        return sims

    # ------------------------------------------------------------------ #
    # Alert logic                                                         #
    # ------------------------------------------------------------------ #

    def _a2_collapse(self) -> str | None:
        """(A2) health guard: a reason string when the tangential-fraction stream
        signals collapse toward norm-dominated variation, else ``None``.

        That regime is where angular quantization silently destroys ranking while
        cosine similarity still reads healthy — the failure class cosine *cannot*
        see — so it belongs in the health verdict, not only in the raw metrics. A
        no-op when the (A2) statistic is disabled or too sparse to judge.

        Two triggers: an explicit noncollapse floor (``tangential_floor``, off by
        default), and a self-calibrating downward drift of the tangential stream
        (significant per the KS test *and* trending down — an upward shift toward
        more angular variation is safe and does not fire).
        """
        tang = np.array(self._tang_window, dtype=np.float64)
        if len(tang) < 8:
            return None
        if self._tangential_floor > 0.0:
            median = float(np.median(tang))
            if median < self._tangential_floor:
                return (
                    f"tangential fraction {median:.3f} < floor "
                    f"{self._tangential_floor:.3f}"
                )
        if self.check_radial_drift():
            mid = len(tang) // 2
            if float(np.median(tang[mid:])) < float(np.median(tang[:mid])):
                return "tangential fraction drifting down (norm-dominated regime)"
        return None

    def _maybe_alert(self) -> None:
        """Fire the alert callback when quality drops: mean cosine below the floor
        or an (A2) tangential-collapse signal (both drive ``is_healthy``)."""
        if len(self._window) == 0:
            return
        mean_cos = float(np.mean(list(self._window)))
        reasons: list[str] = []
        if mean_cos < self._quality_floor:
            reasons.append(
                f"mean cosine {mean_cos:.4f} < floor {self._quality_floor:.4f}"
            )
        a2_reason = self._a2_collapse()
        if a2_reason is not None:
            reasons.append(f"(A2) {a2_reason}")
        if not reasons:
            return
        self._n_alerts += 1
        median_tang = (
            float(np.median(self._tang_window))
            if len(self._tang_window)
            else float("nan")
        )
        details = {
            "mean_cosine": mean_cos,
            "quality_floor": self._quality_floor,
            "n_total": self._n_total,
            "n_window": len(self._window),
            "median_tangential_fraction": median_tang,
            "reasons": reasons,
        }
        logger.warning("Quality alert: %s", "; ".join(reasons))
        if self._alert_callback is not None:
            self._alert_callback(details)

    # ------------------------------------------------------------------ #
    # Statistics                                                          #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        """Return a dict of rolling-window statistics.

        Returns:
            Dictionary with keys: n_total, n_window, mean_cosine,
            min_cosine, std_cosine, p95_cosine, quality_floor,
            is_healthy, n_alerts, drift_detected,
            median_tangential_fraction, radial_drift_detected.
        """
        arr = np.array(self._window, dtype=np.float64)
        tang = np.array(self._tang_window, dtype=np.float64)
        median_tang = float(np.median(tang)) if len(tang) else float("nan")
        n_window = len(arr)
        if n_window == 0:
            return {
                "n_total": self._n_total,
                "n_window": 0,
                "mean_cosine": float("nan"),
                "min_cosine": float("nan"),
                "std_cosine": float("nan"),
                "p95_cosine": float("nan"),
                "quality_floor": self._quality_floor,
                "is_healthy": True,
                "n_alerts": self._n_alerts,
                "drift_detected": False,
                "median_tangential_fraction": median_tang,
                "radial_drift_detected": False,
            }

        mean_cos = float(np.mean(arr))
        # Health requires BOTH the cosine floor and (A2) tangential-noncollapse:
        # a stream sliding into the norm-dominated regime damages ranking while
        # cosine still reads fine, so it must not be reported as healthy.
        is_healthy = mean_cos >= self._quality_floor and self._a2_collapse() is None
        return {
            "n_total": self._n_total,
            "n_window": n_window,
            "mean_cosine": mean_cos,
            "min_cosine": float(np.min(arr)),
            "std_cosine": float(np.std(arr)),
            "p95_cosine": float(np.percentile(arr, 5)),  # worst 5%
            "quality_floor": self._quality_floor,
            "is_healthy": is_healthy,
            "n_alerts": self._n_alerts,
            "drift_detected": self.check_drift(),
            "median_tangential_fraction": median_tang,
            "radial_drift_detected": self.check_radial_drift(),
        }

    # ------------------------------------------------------------------ #
    # Drift detection (scipy-free KS test)                                #
    # ------------------------------------------------------------------ #

    def check_drift(self, significance: float = 0.05) -> bool:
        """Check for distribution drift in the cosine rolling window.

        Splits the window into first and second halves and computes the
        two-sample Kolmogorov-Smirnov statistic.  Uses the asymptotic
        critical-value approximation so no scipy dependency is needed.

        Args:
            significance: Significance level (default 0.05).

        Returns:
            True if drift is detected (the quality distribution is changing).
        """
        return self._ks_half_split(
            np.array(self._window, dtype=np.float64), significance
        )

    def check_radial_drift(self, significance: float = 0.05) -> bool:
        """Check for drift in the (A2) tangential-fraction stream.

        A significant shift here -- especially downward -- means the data's
        pairwise variation is becoming norm-dominated: angular quantization
        will start destroying ranking while cosine similarity still reads
        fine. (Direction-concentration failures, the KV-keys class, need
        the calibration-time probe in ``turboquant_pro.a2_probe``.)
        """
        return self._ks_half_split(
            np.array(self._tang_window, dtype=np.float64), significance
        )

    @staticmethod
    def _ks_half_split(arr: np.ndarray, significance: float) -> bool:
        """Two-sample KS test between the halves of ``arr`` (scipy-free)."""
        n = len(arr)
        if n < 8:  # too few samples for a meaningful test
            return False

        mid = n // 2
        first_half = np.sort(arr[:mid])
        second_half = np.sort(arr[mid:])

        n1 = len(first_half)
        n2 = len(second_half)

        # Compute empirical CDFs and max absolute difference (KS stat).
        all_values = np.sort(np.concatenate([first_half, second_half]))
        cdf1 = np.searchsorted(first_half, all_values, side="right") / n1
        cdf2 = np.searchsorted(second_half, all_values, side="right") / n2
        ks_stat = float(np.max(np.abs(cdf1 - cdf2)))

        # KS critical value approximation:
        #   c(alpha) = sqrt(-0.5 * ln(alpha / 2))
        #   D_crit   = c(alpha) * sqrt((n1 + n2) / (n1 * n2))
        c_alpha = math.sqrt(-0.5 * math.log(significance / 2.0))
        d_crit = c_alpha * math.sqrt((n1 + n2) / (n1 * n2))

        return ks_stat > d_crit

    # ------------------------------------------------------------------ #
    # Prometheus-compatible metrics                                       #
    # ------------------------------------------------------------------ #

    def metrics_dict(self) -> dict:
        """Return a flat dict of Prometheus-compatible metric values.

        Suitable for direct export via a ``/metrics`` endpoint or a push
        gateway client.
        """
        s = self.stats()
        return {
            "turboquant_quality_mean_cosine": s["mean_cosine"],
            "turboquant_quality_min_cosine": s["min_cosine"],
            "turboquant_quality_std_cosine": s["std_cosine"],
            "turboquant_quality_p95_cosine": s["p95_cosine"],
            "turboquant_quality_total_records": s["n_total"],
            "turboquant_quality_alerts_total": s["n_alerts"],
            "turboquant_quality_is_healthy": 1 if s["is_healthy"] else 0,
            "turboquant_quality_drift_detected": 1 if s["drift_detected"] else 0,
            "turboquant_quality_median_tangential_fraction": s[
                "median_tangential_fraction"
            ],
            "turboquant_quality_radial_drift_detected": (
                1 if s["radial_drift_detected"] else 0
            ),
        }

    # ------------------------------------------------------------------ #
    # Reset                                                               #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Clear all recorded state, counters, and alerts."""
        self._window.clear()
        self._tang_window.clear()
        self._reservoir.clear()
        self._res_rng = np.random.default_rng(0)
        self._n_total = 0
        self._n_alerts = 0

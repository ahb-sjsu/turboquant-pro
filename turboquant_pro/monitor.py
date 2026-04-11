# TurboQuant Pro: Open-source PolarQuant+QJL for LLM KV cache compression
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""
Production quality monitor for TurboQuant-compressed embeddings.

Tracks cosine similarity between original and reconstructed embeddings over
time, detects distribution drift via a scipy-free Kolmogorov-Smirnov test,
and raises alerts when quality drops below a configurable floor.

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
    """

    def __init__(
        self,
        quality_floor: float = 0.95,
        window_size: int = 1000,
        alert_callback: Callable | None = None,
    ) -> None:
        self._quality_floor = quality_floor
        self._window_size = window_size
        self._alert_callback = alert_callback

        self._window: collections.deque[float] = collections.deque(
            maxlen=window_size,
        )
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

    def record(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Record a single original/reconstructed pair.

        Computes cosine similarity, appends to the rolling window, checks
        the alert threshold, and returns the similarity value.
        """
        cos = self._cosine_similarity(
            original.ravel(),
            reconstructed.ravel(),
        )
        self._window.append(cos)
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

    def _maybe_alert(self) -> None:
        """Fire the alert callback when mean cosine drops below the floor."""
        if len(self._window) == 0:
            return
        mean_cos = float(np.mean(list(self._window)))
        if mean_cos < self._quality_floor:
            self._n_alerts += 1
            details = {
                "mean_cosine": mean_cos,
                "quality_floor": self._quality_floor,
                "n_total": self._n_total,
                "n_window": len(self._window),
            }
            logger.warning(
                "Quality alert: mean cosine %.4f < floor %.4f",
                mean_cos,
                self._quality_floor,
            )
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
            is_healthy, n_alerts, drift_detected.
        """
        arr = np.array(self._window, dtype=np.float64)
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
            }

        mean_cos = float(np.mean(arr))
        return {
            "n_total": self._n_total,
            "n_window": n_window,
            "mean_cosine": mean_cos,
            "min_cosine": float(np.min(arr)),
            "std_cosine": float(np.std(arr)),
            "p95_cosine": float(np.percentile(arr, 5)),  # worst 5%
            "quality_floor": self._quality_floor,
            "is_healthy": mean_cos >= self._quality_floor,
            "n_alerts": self._n_alerts,
            "drift_detected": self.check_drift(),
        }

    # ------------------------------------------------------------------ #
    # Drift detection (scipy-free KS test)                                #
    # ------------------------------------------------------------------ #

    def check_drift(self, significance: float = 0.05) -> bool:
        """Check for distribution drift in the rolling window.

        Splits the window into first and second halves and computes the
        two-sample Kolmogorov-Smirnov statistic.  Uses the asymptotic
        critical-value approximation so no scipy dependency is needed.

        Args:
            significance: Significance level (default 0.05).

        Returns:
            True if drift is detected (the quality distribution is changing).
        """
        arr = np.array(self._window, dtype=np.float64)
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
        }

    # ------------------------------------------------------------------ #
    # Reset                                                               #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Clear all recorded state, counters, and alerts."""
        self._window.clear()
        self._n_total = 0
        self._n_alerts = 0

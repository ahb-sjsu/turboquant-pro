#!/usr/bin/env python3
"""QualityMonitor drift-detection benchmark.

Demonstrates turboquant-pro's production observability: when compression quality
regresses (e.g. a config change drops bit-width), does QualityMonitor's drift
detection catch it? We feed a clean (original, reconstructed) stream, then a
degraded one, and report whether drift is flagged and how quickly.

Needs turboquant-pro v1.0+ (QualityMonitor). Run:
  python benchmark_quality_monitor.py --npy /tmp/labse_bench.npy --n 8000
"""

import argparse

import numpy as np


def recon(pipe, V):
    return np.asarray(pipe.decompress_batch(pipe.compress_batch(V)), dtype=np.float32)


def mean_cos(V, R):
    a = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-30)
    b = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-30)
    return float((a * b).sum(1).mean())


def _is_drift(d):
    if d is True:
        return True
    if isinstance(d, dict):
        return bool(d.get("drift") or d.get("drift_detected") or d.get("is_drift"))
    return bool(getattr(d, "drift_detected", False) or getattr(d, "drift", False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True)
    ap.add_argument("--n", type=int, default=8000)
    a = ap.parse_args()

    from turboquant_pro import PCAMatryoshka, QualityMonitor

    X = np.load(a.npy, mmap_mode="r").astype(np.float32)[: 2 * a.n]
    pca = PCAMatryoshka(input_dim=X.shape[1], output_dim=256)
    pca.fit(X[: min(len(X), 50000)])
    good = pca.with_quantizer(bits=4)  # high quality
    bad = pca.with_quantizer(bits=2)  # regressed quality
    Vg, Vb = X[: a.n], X[a.n : 2 * a.n]
    Rg, Rb = recon(good, Vg), recon(bad, Vb)
    print(
        f"mean cosine: good(4-bit)={mean_cos(Vg, Rg):.4f}  "
        f"bad(2-bit)={mean_cos(Vb, Rb):.4f}",
        flush=True,
    )

    alerts = []
    qm = QualityMonitor(
        quality_floor=0.95, window_size=500, alert_callback=lambda *x: alerts.append(x)
    )

    # Phase 1: clean stream -> expect no drift.
    for o, r in zip(Vg, Rg):
        qm.record(o, r)
    print(f"[clean] drift={_is_drift(qm.check_drift())} stats={qm.stats()}", flush=True)
    clean_alerts = len(alerts)

    # Phase 2: regressed stream -> expect drift, measure detection latency.
    detected_at = None
    for i, (o, r) in enumerate(zip(Vb, Rb)):
        qm.record(o, r)
        if i % 50 == 0 and _is_drift(qm.check_drift()):
            detected_at = i
            break
    print(
        f"[regressed] drift detected after {detected_at} samples; "
        f"final drift={_is_drift(qm.check_drift())}",
        flush=True,
    )
    print(f"alerts: clean-phase={clean_alerts}, total={len(alerts)}", flush=True)
    try:
        print("metrics_dict (Prometheus-style):", qm.metrics_dict(), flush=True)
    except Exception as e:
        print("metrics_dict err:", str(e)[:80], flush=True)


if __name__ == "__main__":
    main()

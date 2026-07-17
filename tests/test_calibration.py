# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Opt-in online key calibration (`PerChannelKV.calibrate` / `calibrate_key_quantizer`).

Scope note (see benchmarks/RESULTS_calibration.md): the Lloyd-Max codebook
reliably lowers *reconstruction* error but does **not** beat the calibration-free
default on the attention consumer metric (softmax-KL) — the API is experimental.
These tests assert only the robust facts: calibration lowers reconstruction
error, the container round-trips, and the zero-calibration default is untouched.
They deliberately do NOT assert a softmax-KL win, which is not robust.
"""

from __future__ import annotations

import numpy as np

from turboquant_pro import PerChannelKV, calibrate_key_quantizer

H, D = 4, 32


def _keys(S, seed):
    """Realistic post-RoPE-like keys: per-channel DC offset + heavy tails."""
    r = np.random.default_rng(seed)
    dc = np.random.default_rng(0).uniform(-6, 6, (H, D)).astype(np.float32)
    sc = np.random.default_rng(1).uniform(0.2, 1.5, (H, D)).astype(np.float32)
    t = r.standard_t(3, (1, H, S, D)).astype(np.float32)
    return (dc[None, :, None, :] + sc[None, :, None, :] * t).astype(np.float32)


def _recon_err(kq, test):
    return float(np.abs(kq.decompress(kq.compress(test))[0] - test[0]).mean())


def test_calibration_lowers_reconstruction_error():
    # Lloyd-Max minimizes quantization MSE, so calibrated reconstruction is
    # reliably closer than the fixed NF4 grid (this is robust; the attention-KL
    # win is not -- see RESULTS_calibration.md).
    cal, test = _keys(1500, 1), _keys(256, 2)
    nf4 = PerChannelKV(head_dim=D, n_heads=H, nf4_asym=True, outlier_frac=0.02)
    calq = calibrate_key_quantizer(
        cal, bits=4, outlier_frac=0.02, head_dim=D, n_heads=H, iters=10
    )
    assert _recon_err(calq, test) < _recon_err(nf4, test)


def test_default_path_untouched():
    # calibration is strictly opt-in; a plain quantizer is never calibrated
    q = PerChannelKV(head_dim=D, n_heads=H)
    assert q.calibrated_levels is None
    x = _keys(32, 5)
    assert q.decompress(q.compress(x)).shape == x.shape


def test_calibrated_container_roundtrips():
    cal, test = _keys(1200, 1), _keys(64, 2)
    kq = calibrate_key_quantizer(cal, bits=4, head_dim=D, n_heads=H, iters=8)
    assert kq.calibrated_levels.shape == (H, D, 16)
    out = kq.decompress(kq.compress(test, packed=True))
    assert out.shape == test.shape
    assert np.isfinite(out).all()


def test_shared_codebook_from_2d_samples():
    # (N, D) samples -> one codebook shared across heads, broadcasts on compress
    cal2d = _keys(2000, 1)[0].transpose(1, 0, 2).reshape(-1, D)  # (H*S, D)
    kq = calibrate_key_quantizer(cal2d, bits=4, head_dim=D, n_heads=H, iters=8)
    assert kq.calibrated_levels.shape == (D, 16)  # shared across heads
    test = _keys(64, 2)
    assert kq.decompress(kq.compress(test)).shape == test.shape


def test_weighted_calibration_runs():
    cal = _keys(1200, 1)
    w = np.abs(np.random.default_rng(3).standard_normal(cal.shape[0] * cal.shape[2]))
    kq = calibrate_key_quantizer(
        cal, bits=4, outlier_frac=0.02, head_dim=D, n_heads=H, weights=w, iters=8
    )
    assert kq.calibrated_levels.shape == (H, D, 16)

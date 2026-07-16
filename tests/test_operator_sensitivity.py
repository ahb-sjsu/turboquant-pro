# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Tests for operator_sensitivity (gate routing + SSM state-decay)."""

from __future__ import annotations

import numpy as np

from turboquant_pro.operator_sensitivity import (
    decay_gain,
    decay_sensitivity,
    decay_time_constant,
    differential_fraction,
    predict_routing_flips,
    quantize_decay,
    routing_margins,
    routing_sensitivity,
    state_decay_sensitivity,
)


def _q_symmetric(W, bits):
    s = np.maximum(np.abs(W).max(axis=1, keepdims=True), 1e-30) / (2 ** (bits - 1) - 1)
    return np.round(W / s) * s


# --------------------------------------------------------------------------- #
# Gates / routing                                                              #
# --------------------------------------------------------------------------- #


def test_routing_margins_values():
    logits = np.array([[3.0, 1.0, 0.0], [2.0, 2.5, 1.0]])
    m = routing_margins(logits, k=1)
    assert np.allclose(m, [3.0 - 1.0, 2.5 - 2.0])
    m2 = routing_margins(logits, k=2)
    assert np.allclose(m2, [1.0 - 0.0, 2.0 - 1.0])


def test_differential_fraction_common_mode_is_free():
    # Pure common-mode perturbation -> differential fraction ~ 0 (routing-safe).
    d_common = np.tile(np.array([[0.7, 0.7, 0.7, 0.7]]), (5, 1))
    assert np.all(np.nan_to_num(differential_fraction(d_common)) < 1e-9)
    # Pure differential (zero-mean) perturbation -> fraction ~ 1 (dangerous).
    d_diff = np.tile(np.array([[1.0, -1.0, 1.0, -1.0]]), (5, 1))
    assert np.allclose(differential_fraction(d_diff), 1.0)


def test_predict_routing_flips_monotonic_in_noise():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((2000, 8))
    f_lo = predict_routing_flips(logits, 0.05)
    f_hi = predict_routing_flips(logits, 0.5)
    assert 0.0 <= f_lo <= f_hi <= 1.0
    assert f_hi > f_lo


def test_routing_flips_concentrate_at_low_margin():
    """The verified boundary: quantization flips low-margin tokens far more."""
    rng = np.random.default_rng(0)
    d, E, n = 64, 16, 6000
    W = rng.standard_normal((E, d)) / np.sqrt(d)
    X = rng.standard_normal((n, d))
    logits = X @ W.T
    top1 = np.argmax(logits, axis=1)
    margin = routing_margins(logits, k=1)
    Wq = _q_symmetric(W, 4)
    flipped = np.argmax(X @ Wq.T, axis=1) != top1
    med = np.median(margin)
    lo = flipped[margin <= med].mean()
    hi = flipped[margin > med].mean()
    assert lo > hi  # low-margin tokens are the fragile ones
    assert lo > 3.0 * hi  # and by a wide margin at 4-bit


def test_routing_sensitivity_report():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((500, 8))
    rep = routing_sensitivity(logits, logit_noise_std=0.1, k=1)
    assert rep.n_tokens == 500
    assert rep.margin_p10 <= rep.margin_p50
    assert 0.0 <= rep.predicted_flip_fraction <= 1.0
    rep2 = routing_sensitivity(logits, k=1)
    assert rep2.predicted_flip_fraction is None


# --------------------------------------------------------------------------- #
# State-space recurrences                                                      #
# --------------------------------------------------------------------------- #


def test_decay_gain_and_time_constant():
    a = np.array([0.0, 0.5, 0.9, 0.99])
    assert np.allclose(decay_gain(a), 1.0 / (1.0 - a))
    # time constant grows toward a -> 1
    tau = decay_time_constant(a)
    assert tau[1] < tau[2] < tau[3]


def test_decay_sensitivity_slow_channels_and_seq_len():
    slow = np.array([0.97])
    fast = np.array([0.3])
    # steady-state: slow >> fast
    assert decay_sensitivity(slow)[0] > 50.0 * decay_sensitivity(fast)[0]
    # truncated sensitivity grows with sequence length for a slow channel
    s_short = decay_sensitivity(slow, seq_len=8)[0]
    s_long = decay_sensitivity(slow, seq_len=256)[0]
    assert s_long > s_short


def test_quantize_decay_log_tau_beats_linear():
    """The verified fix: log-time-constant basis cuts state drift vs linear."""
    rng = np.random.default_rng(3)
    C = 64
    a = np.clip(1 - np.exp(rng.uniform(-7, -0.5, size=C)), 0, 0.9999)

    def run(decay, x):
        h = np.zeros(C)
        for t in range(x.shape[0]):
            h = decay * h + x[t]
        return h

    for bits in (3, 4):
        a_lin = quantize_decay(a, bits, basis="linear")
        a_warp = quantize_decay(a, bits, basis="log_tau")
        dl, dg = [], []
        for _ in range(30):
            x = rng.standard_normal((128, C))
            h0 = run(a, x)
            dl.append(np.abs(run(a_lin, x) - h0).mean())
            dg.append(np.abs(run(a_warp, x) - h0).mean())
        assert np.mean(dg) < np.mean(dl)  # warp wins at matched bits


def test_quantize_decay_stays_in_range():
    a = np.array([0.1, 0.5, 0.9, 0.99, 0.999])
    for basis in ("linear", "log_tau"):
        q = quantize_decay(a, 4, basis=basis)
        assert np.all(q >= 0.0) and np.all(q < 1.0)


def test_state_decay_sensitivity_report():
    rng = np.random.default_rng(2)
    a = np.clip(1 - np.exp(rng.uniform(-6, -0.3, size=48)), 0, 0.9999)
    rep = state_decay_sensitivity(a, seq_len=128)
    assert rep.n_channels == 48
    assert rep.max_gain >= rep.mean_gain
    assert 0.0 <= rep.slow_channel_fraction <= 1.0
    assert rep.recommended_basis == "log_tau"

# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Bit allocation against a measured read operator.

The middle third of the principle: measure the consumer, spend bits against
it, certify the result. These check the spending step against cases whose
answers are known in closed form, and against the baseline it has to beat.
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro.read_allocation import (
    allocate_bits,
    allocation_report,
    predicted_distortion,
    realised_distortion,
    uniform_bits,
)
from turboquant_pro.read_operators import create_read_operator

RNG = np.random.default_rng(17)
D = 12


# ------------------------------------------------------------- closed forms


def test_flat_spectrum_reproduces_uniform_allocation():
    """With no direction preferred, water-filling must not invent one."""
    alloc = allocate_bits(np.eye(D), budget_bits=4.0 * D)
    assert np.allclose(alloc.bits, uniform_bits(D, 4.0 * D), atol=1e-6)
    assert alloc.n_starved == 0
    assert alloc.effective_rank == pytest.approx(D)


def test_rank_one_operator_starves_everything_else():
    v = RNG.standard_normal(D)
    P = np.outer(v, v)
    alloc = allocate_bits(P, budget_bits=8.0)
    assert alloc.bits[0] == pytest.approx(8.0)
    assert np.allclose(alloc.bits[1:], 0.0)
    assert alloc.n_starved == D - 1
    assert alloc.effective_rank == pytest.approx(1.0)


def test_allocation_respects_its_budget():
    A = RNG.standard_normal((D, D))
    alloc = allocate_bits(A @ A.T, budget_bits=3.0 * D)
    assert alloc.bits.sum() == pytest.approx(3.0 * D, abs=1e-6)
    assert np.all(alloc.bits >= 0.0)


def test_max_bits_ceiling_is_respected():
    lam = np.zeros(D)
    lam[0] = 1e6
    alloc = allocate_bits(
        np.diag(lam) + 1e-6 * np.eye(D), budget_bits=6.0 * D, max_bits=4.0
    )
    assert np.all(alloc.bits <= 4.0 + 1e-9)


def test_zero_budget_spends_nothing():
    alloc = allocate_bits(np.eye(D), budget_bits=0.0)
    assert np.allclose(alloc.bits, 0.0)
    assert alloc.n_starved == D


def test_degenerate_operator_is_flagged_not_crashed():
    alloc = allocate_bits(np.zeros((D, D)), budget_bits=10.0)
    assert alloc.meta.get("degenerate") is True
    assert np.allclose(alloc.bits, 0.0)


# ------------------------------------------------------ beating the baseline


def test_allocation_beats_uniform_on_a_skewed_consumer():
    """The entire claim: at a fixed budget, spending against the consumer
    costs it less than spending evenly."""
    lam = np.array([100.0, 30.0, 5.0, 1.0] + [0.05] * (D - 4))
    rep = allocation_report(np.diag(lam), budget_bits=3.0 * D)
    assert rep["allocated_distortion"] < rep["uniform_distortion"]
    assert rep["gain_over_uniform"] > 1.5


def test_allocation_cannot_beat_uniform_on_a_flat_consumer():
    """And it must not claim a gain where none exists."""
    rep = allocation_report(np.eye(D), budget_bits=3.0 * D)
    assert rep["gain_over_uniform"] == pytest.approx(1.0, rel=1e-6)


def test_more_budget_never_costs_the_consumer_more():
    A = RNG.standard_normal((D, D))
    P = A @ A.T
    d = [
        allocate_bits(P, budget_bits=b).predicted_distortion
        for b in (D, 2 * D, 4 * D, 8 * D)
    ]
    assert all(d[i] > d[i + 1] for i in range(len(d) - 1))


# ------------------------------------------------- source variance is used


def test_variance_shifts_bits_toward_directions_that_actually_vary():
    """Sensitivity alone is not the objective; lambda * sigma^2 is."""
    P = np.diag([1.0, 1.0] + [1e-6] * (D - 2))
    acts = RNG.standard_normal((512, D))
    acts[:, 1] *= 20.0  # second direction carries far more signal
    alloc = allocate_bits(P, budget_bits=6.0, activations=acts)
    # the operator's eigenbasis is the standard one here, in descending
    # eigenvalue order with ties, so find which column each axis maps to
    axis0 = int(np.argmax(np.abs(alloc.basis[0])))
    axis1 = int(np.argmax(np.abs(alloc.basis[1])))
    assert alloc.bits[axis1] > alloc.bits[axis0]


def test_whitened_mode_is_recorded():
    alloc = allocate_bits(np.eye(D), budget_bits=D)
    assert alloc.meta["whitened"] is True


def test_activation_channel_mismatch_is_an_error():
    with pytest.raises(ValueError, match="channels"):
        allocate_bits(
            np.eye(D), budget_bits=D, activations=RNG.standard_normal((10, D + 3))
        )


# ------------------------------------------------------- the honest caution


def test_a_concentrated_spectrum_carries_the_probe_budget_caution():
    """A rank-one operator is what a sub-dimensional probe returns whatever
    the consumer reads, so the report must say so rather than allocate
    confidently."""
    v = RNG.standard_normal(D)
    rep = allocation_report(np.outer(v, v), budget_bits=4.0 * D)
    assert rep["spectrum_concentrated"] is True
    assert "cliff at k = d" in rep["caution"]


def test_a_broad_spectrum_carries_no_caution():
    rep = allocation_report(np.eye(D), budget_bits=4.0 * D)
    assert rep["spectrum_concentrated"] is False
    assert rep["caution"] == ""


# --------------------------------------------- prediction versus reality


def test_predicted_and_realised_distortion_agree_for_a_known_codec():
    """The rate-distortion model predicts; a real codec's error is measured.
    A uniform quantizer at b bits has error variance ~ step^2/12, so the two
    should track rather than merely both be small."""
    rng = np.random.default_rng(21)
    n, d = 4000, 6
    x = rng.standard_normal((n, d))
    P = np.diag(np.array([8.0, 4.0, 2.0, 1.0, 0.5, 0.25]))

    bits = 4
    step = (x.max(axis=0) - x.min(axis=0)) / (2**bits - 1)
    q = np.round(x / step) * step

    realised = realised_distortion(P, x, q)
    sigma_q = (step**2) / 12.0
    predicted = float(np.sum(np.diag(P) * sigma_q))
    assert realised == pytest.approx(predicted, rel=0.15)


def test_realised_distortion_ignores_error_the_consumer_cannot_read():
    rng = np.random.default_rng(22)
    x = rng.standard_normal((256, 4))
    P = np.diag([1.0, 1.0, 0.0, 0.0])
    y = x.copy()
    y[:, 2:] += 50.0 * rng.standard_normal((256, 2))
    assert realised_distortion(P, x, y) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------- wired to the provider registry


def test_allocates_against_a_provider_supplied_operator():
    """End to end through the extension point rather than a hand-made matrix."""
    keys = RNG.standard_normal((64, D))
    q = RNG.standard_normal((4, D))
    P = create_read_operator("attention_analytic", n_probe_keys=8).operator(
        keys, queries=q
    )
    rep = allocation_report(P, budget_bits=3.0 * D, activations=keys)
    assert rep["allocated_distortion"] <= rep["uniform_distortion"]
    # a head reads at most as many directions as it has queries
    assert rep["allocation"]["effective_rank"] <= 4.0 + 1e-9


def test_identity_provider_allocation_is_uniform():
    """The null consumer must produce the null allocation, which is what
    makes 'reconstruction error is a consumer' concrete."""
    acts = RNG.standard_normal((256, D))
    P = create_read_operator("identity").operator(acts)
    alloc = allocate_bits(P, budget_bits=4.0 * D)
    assert np.allclose(alloc.bits, uniform_bits(D, 4.0 * D), atol=1e-6)


def test_predicted_distortion_matches_the_formula():
    lam = np.array([4.0, 1.0])
    var = np.array([2.0, 3.0])
    bits = np.array([1.0, 2.0])
    want = 4.0 * 2.0 * 2 ** (-2) + 1.0 * 3.0 * 2 ** (-4)
    assert predicted_distortion(lam, var, bits) == pytest.approx(want)

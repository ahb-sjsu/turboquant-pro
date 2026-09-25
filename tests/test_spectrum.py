"""Bits that follow the spectrum: the allocation rule and the segmented ADC index.

The allocation minimises ``sum lambda_j D(b_j)`` under a bit budget; on a sorted
spectrum it is monotone, so it lands as contiguous segments. The segmented index
scores those segments with per-row energy fractions, and the numpy scan must agree
with a direct reconstruction of what it stores.
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro import ADCIndex, PCAMatryoshka
from turboquant_pro.spectrum import (
    DISTORTION,
    allocate_bits,
    allocate_for_bytes,
    expected_distortion,
    segments,
    stored_bytes,
)

# --------------------------------------------------------------------------- #
# Allocation                                                                  #
# --------------------------------------------------------------------------- #


def _spectrum(d, decay=0.97, seed=0):
    return decay ** np.arange(d) * (1 + 0.01 * np.random.default_rng(seed).random(d))


def test_distortion_table_is_decreasing_with_diminishing_returns():
    b = sorted(DISTORTION)
    d = [DISTORTION[x] for x in b]
    assert all(x > y for x, y in zip(d, d[1:]))
    gains = [x - y for x, y in zip(d, d[1:])]
    assert all(g1 > g2 for g1, g2 in zip(gains, gains[1:]))


@pytest.mark.parametrize("budget", [0, 37, 128, 300, 512])
def test_allocation_spends_the_budget_and_stays_monotone(budget):
    lam = _spectrum(128)
    bits = allocate_bits(lam, budget)
    assert bits.shape == (128,) and bits.min() >= 0 and bits.max() <= 4
    assert bits.sum() == budget  # unit steps spend every bit that fits
    assert np.all(np.diff(bits) <= 0)  # sorted spectrum -> non-increasing widths


def test_allocation_is_optimal_against_brute_force():
    """On a tiny problem, the greedy answer equals the exhaustive minimum."""
    import itertools

    lam = np.asarray([5.0, 2.0, 1.0, 0.4])
    budget = 7
    best, best_cost = None, np.inf
    for combo in itertools.product(range(5), repeat=4):
        if sum(combo) <= budget:
            cost = sum(lam_j * DISTORTION[b] for lam_j, b in zip(lam, combo))
            if cost < best_cost - 1e-12:
                best, best_cost = combo, cost
    got = allocate_bits(lam, budget)
    assert expected_distortion(lam, got) == pytest.approx(
        best_cost / lam.sum(), rel=1e-9
    )
    assert tuple(got.tolist()) == best


def test_thinned_choices_and_minimum_width():
    lam = _spectrum(64)
    bits = allocate_bits(lam, 64 * 3, choices=(2, 3, 4))
    assert set(bits.tolist()) <= {2, 3, 4} and bits.sum() <= 64 * 3
    with pytest.raises(ValueError):
        allocate_bits(lam, 10, choices=(2, 3, 4))  # cannot give every dim 2 bits


def test_zero_eigenvalues_get_nothing_and_flat_spectrum_is_uniform():
    lam = np.concatenate([np.ones(8), np.zeros(8)])
    bits = allocate_bits(lam, 8 * 4 + 5)
    assert bits[:8].tolist() == [4] * 8 and bits[8:].tolist() == [0] * 8
    flat = allocate_bits(np.ones(10), 25)
    assert flat.max() - flat.min() <= 1 and flat.sum() == 25


def test_segments_are_runs_and_drop_zero_dims():
    assert segments(np.asarray([4, 4, 3, 3, 3, 1, 0, 0])) == [(2, 4), (3, 3), (1, 1)]
    assert segments(np.asarray([0, 0])) == []
    assert stored_bytes(np.asarray([4, 4, 3, 3, 3, 1, 0, 0])) == 1 + 2 + 1 + 4 + 3


def test_allocate_for_bytes_fits_the_budget_including_segment_overhead():
    lam = _spectrum(256)
    for budget in (40, 72, 132, 200):
        bits = allocate_for_bytes(lam, budget, choices=(1, 2, 3, 4))
        assert stored_bytes(bits) <= budget
        assert bits.min() >= 1
        if bits.min() < 4:  # tight unless every dim already has the widest choice
            assert stored_bytes(bits) > budget - 8


# --------------------------------------------------------------------------- #
# The pipeline's schedule and the segmented index                             #
# --------------------------------------------------------------------------- #


def _anisotropic(n, dim, seed=0, decay=0.96):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32) * (decay ** np.arange(dim))
    x = x @ np.linalg.qr(rng.standard_normal((dim, dim)))[0].astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_auto_schedule_follows_the_spectrum():
    x = _anisotropic(3000, 96)
    pca = PCAMatryoshka(input_dim=96, output_dim=64)
    pca.fit(x)
    sched = pca._auto_bit_schedule(3.0)
    assert sum(n for n, _ in sched) == 64
    bits = [b for _, b in sched]
    assert bits == sorted(bits, reverse=True) and set(bits) <= {2, 3, 4}
    assert abs(sum(n * b for n, b in sched) - 3.0 * 64) <= 2


def test_with_spectrum_quantizer_meets_the_byte_budget():
    x = _anisotropic(3000, 96)
    pca = PCAMatryoshka(input_dim=96, output_dim=64)
    pca.fit(x)
    pipe = pca.with_spectrum_quantizer(budget_bytes=30)
    ix = ADCIndex(pipe)
    assert ix.stored_bytes_per_row <= 30
    assert ix._coder.nseg == len(pipe.bit_schedule)
    with pytest.raises(ValueError):
        pca.with_spectrum_quantizer(budget_bytes=30, avg_bits=3.0)


def test_segmented_index_numpy_scan_matches_direct_reconstruction():
    x = _anisotropic(2000, 64, seed=1)
    pca = PCAMatryoshka(input_dim=64, output_dim=48)
    pca.fit(x[:1500])
    pipe = pca.with_weighted_quantizer(bit_schedule=[(16, 4), (16, 2), (16, 1)])
    ix = ADCIndex(pipe).add(x[:1300]).add(x[1300:])
    assert ix._segw.shape == (2000, 3)
    np.testing.assert_allclose((ix._segw**2).sum(axis=1), 1.0, atol=1e-2)
    np.testing.assert_array_equal(ix._segw * 255, np.round(ix._segw * 255))
    q = x[:7]
    q_rot, qbias = ix._query_terms(q)
    idx, sc = ix._search_numpy(q_rot, qbias, 10)
    # direct: reconstruct rows in the rotated PCA space and score the cosine
    cc = ix._coder.reconstruct(np.asarray(ix._codes), ix._segw)
    inner = qbias[:, None] + ix._cnorm[None, :] * (q_rot @ cc.T)
    direct = inner * ix._vrnorm[None, :]
    for i in range(len(q)):
        np.testing.assert_allclose(sc[i], np.sort(direct[i])[::-1][:10], rtol=1e-5)
    # and the codes really carry three widths
    codes = np.asarray(ix._codes)
    assert codes[:, :16].max() == 15 and codes[:, 16:32].max() <= 3
    assert codes[:, 32:].max() <= 1


def test_segmented_index_recall_at_matched_bytes_on_anisotropic_data():
    """At the same stored bytes the spectrum allocation should stay close to the
    uniform one on a steeply decaying spectrum; a sanity floor, not the measured
    claim (that is the preregistered cell, docs/PREREG_spectrum_bits.md). An
    earlier form of this test at the 2-bit level (28 bytes) lost 0.12 recall when
    the fraction cost two bytes per segment: at small budgets the overhead is
    what decides, which is why the fraction is one byte and the claim is measured."""
    x = _anisotropic(6000, 128, seed=2, decay=0.94)
    pca = PCAMatryoshka(input_dim=128, output_dim=96)
    pca.fit(x[:4000])
    uniform = ADCIndex(pca.with_quantizer(bits=3)).add(x)
    budget = uniform.stored_bytes_per_row
    spectrum = ADCIndex(pca.with_spectrum_quantizer(budget_bytes=budget)).add(x)
    assert spectrum.stored_bytes_per_row <= budget
    assert spectrum._coder.nseg <= 4
    q = x[:300]
    gt = np.argsort(-(q @ x.T), axis=1)[:, :10]

    def recall(ix):
        got, _ = ix.search(q, k=10)
        return np.mean([len(set(a) & set(b)) / 10 for a, b in zip(got, gt)])

    assert recall(spectrum) >= recall(uniform) - 0.05

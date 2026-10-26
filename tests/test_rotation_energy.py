"""The large-dimension rotation must spread energy across coordinates.

Before the fix, TurboQuantKV with head_dim > 4096 used a sign flip plus a
permutation. That leaves every coordinate's magnitude unchanged, so an input
whose energy sits in a few coordinates (outlier channels) quantized with about
0.9 relative squared error, against about 0.05 for a spreading rotation.
Gaussian inputs hide the problem, which is why a Gaussian-only test missed it.
"""

import logging

import numpy as np
import pytest

from turboquant_pro.core import TurboQuantKV, _TwoWindowHadamard
from turboquant_pro.pgvector import TurboQuantPGVector


def _spiky(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = 0.01 * rng.standard_normal((1, 1, n, d)).astype(np.float32)
    x[..., :8] += 5.0
    return x


def _rel_mse(x: np.ndarray, xh: np.ndarray) -> float:
    return float(np.sum((xh - x) ** 2) / np.sum(x**2))


@pytest.mark.parametrize("d", [4097, 5000, 6144, 8192])
def test_two_window_hadamard_is_orthogonal(d: int) -> None:
    rot = _TwoWindowHadamard(d, np.random.default_rng(3), np)
    x = np.random.default_rng(4).standard_normal((5, d)).astype(np.float32)
    y = rot.rotate(x)
    np.testing.assert_allclose(
        np.linalg.norm(y, axis=1), np.linalg.norm(x, axis=1), rtol=1e-4
    )
    np.testing.assert_allclose(rot.unrotate(y), x, atol=1e-4)


@pytest.mark.parametrize("d", [5000, 6144, 8192])
@pytest.mark.parametrize("hot", [0, 4500, -1])
def test_rotation_spreads_a_spiky_vector(d: int, hot: int) -> None:
    # hot = 4500 or -1 starts in the tail that the first window may miss; 6144
    # is the case where a second plain Hadamard would re-concentrate energy.
    rot = _TwoWindowHadamard(d, np.random.default_rng(1), np)
    x = np.zeros(d, np.float32)
    x[hot] = 1.0
    y = rot.rotate(x)
    # A one-hot input must end up with no coordinate holding much energy. For a
    # well-spread unit vector the largest squared coordinate is about
    # 2 log(d) / d (about 0.003 at d = 5000); without spreading it stays 1.0.
    assert float(np.max(y**2)) < 0.02


@pytest.mark.parametrize("d", [5000, 8192])
def test_large_head_dim_quantizes_spiky_inputs(d: int) -> None:
    tq = TurboQuantKV(head_dim=d, n_heads=1, bits=3, use_gpu=False, seed=0)
    x = _spiky(4, d)
    xh = tq.decompress(tq.compress(x))
    assert _rel_mse(x, xh) < 0.15


def test_large_head_dim_gaussian_unchanged_quality() -> None:
    d = 8192
    tq = TurboQuantKV(head_dim=d, n_heads=1, bits=3, use_gpu=False, seed=0)
    x = np.random.default_rng(5).standard_normal((1, 1, 4, d)).astype(np.float32)
    assert _rel_mse(x, tq.decompress(tq.compress(x))) < 0.06


def test_pgvector_qr_above_4096_is_legacy_and_warns(caplog) -> None:
    # "qr" must keep its exact historical construction so stored records decode.
    d, seed = 5000, 7
    with caplog.at_level(logging.WARNING, logger="turboquant_pro.pgvector"):
        pq = TurboQuantPGVector(dim=d, bits=3, seed=seed, rotation="qr")
    assert pq._structured
    rng = np.random.default_rng(seed)
    expected_signs = rng.choice([-1.0, 1.0], size=d).astype(np.float32)
    expected_perm = rng.permutation(d)
    np.testing.assert_array_equal(pq._sign_flip, expected_signs)
    np.testing.assert_array_equal(pq._perm, expected_perm)
    assert any("does not spread energy" in r.getMessage() for r in caplog.records)

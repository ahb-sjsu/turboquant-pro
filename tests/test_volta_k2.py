# TurboQuant Pro: tests for the Volta K2 fused key-score kernel.
# Copyright (c) 2026 Andrew H. Bond. MIT License.
"""Exactness tests for :mod:`turboquant_pro.volta_kernels` (K2).

The CPU-fallback and format-integration tests run everywhere (NumPy only). The
CuPy kernel test skips when CuPy / a CUDA device is unavailable, mirroring the
torch-optional tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro.kv_fused_pck import _grid_params, pck_key_scores
from turboquant_pro.per_channel_kv import _NF4, PerChannelKV, _pack_indices
from turboquant_pro.volta_kernels import k2_key_scores, k2_key_scores_packed


def _synth(H=8, S=257, D=128, seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 16, size=(H, S, D), dtype=np.uint8)
    q = rng.standard_normal((H, D)).astype(np.float32)
    weight = (0.5 + rng.random((H, D))).astype(np.float32)
    mu = (0.1 * rng.standard_normal((H, D))).astype(np.float32)
    w = (q * weight).astype(np.float32)
    bias = (q * mu).sum(1).astype(np.float32)
    return codes, w, bias


def test_cpu_fallback_matches_reference():
    codes, w, bias = _synth()
    got = k2_key_scores(codes, w, bias, _NF4)
    ref = bias[:, None] + np.einsum("hd,hsd->hs", w, _NF4[codes])
    assert got.shape == (codes.shape[0], codes.shape[1])
    assert np.allclose(got, ref, atol=1e-5)


def test_matches_pck_reference_dense():
    """The kernel primitive reproduces the real per-channel-KV score path
    (asym-NF4, no outliers) computed by ``kv_fused_pck.pck_key_scores``."""
    H, S, D = 6, 129, 128
    rng = np.random.default_rng(1)
    x = (0.3 * rng.standard_normal((1, H, S, D))).astype(np.float32)
    x[:, :, :, ::7] += 2.0  # a DC offset the asym zero-point must absorb
    q_kv = PerChannelKV(head_dim=D, n_heads=H, bits=4, nf4_asym=True)
    c = q_kv.compress(x)  # outlier_frac=0 -> dense only

    q = rng.standard_normal((H, D)).astype(np.float32)
    mu, weight, grid = _grid_params(q_kv, c)
    w = (q * weight).astype(np.float32)
    bias = (q * mu).sum(1).astype(np.float32)
    codes = c.indices[0]  # (H,S,D) uint8 (unpacked)

    got = k2_key_scores(codes, w, bias, grid)
    ref = pck_key_scores(q, q_kv, c)
    assert np.allclose(got, ref, atol=1e-4)


def test_cupy_kernel_exact():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # pragma: no cover
        pytest.skip("no CUDA device")
    # S a multiple of 4 (tuned vec4+ns4 path) and a non-multiple (tail handling).
    for S in (1024, 1023):
        codes, w, bias = _synth(H=32, S=S, D=128, seed=2)
        ref = bias[:, None] + np.einsum("hd,hsd->hs", w, _NF4[codes])
        got = k2_key_scores(
            cp.asarray(codes), cp.asarray(w), cp.asarray(bias), cp.asarray(_NF4)
        )
        assert float(cp.abs(got - cp.asarray(ref)).max()) < 1e-4, f"S={S}"


def test_packed_cpu_matches_reference():
    """Packed decode (CPU fallback) reproduces the unpacked reference for 2/3/4-bit."""
    for bits in (2, 3, 4):
        H, S, D = 4, 65, 128
        rng = np.random.default_rng(10 + bits)
        codes = rng.integers(0, 2**bits, size=(H, S, D), dtype=np.uint8)
        q = rng.standard_normal((H, D)).astype(np.float32)
        weight = (0.5 + rng.random((H, D))).astype(np.float32)
        w = (q * weight).astype(np.float32)
        bias = np.zeros(H, dtype=np.float32)
        grid = np.linspace(-1, 1, 2**bits).astype(np.float32)
        packed = _pack_indices(codes, bits)
        got = k2_key_scores_packed(packed, w, bias, grid, H, S, D, bits)
        ref = bias[:, None] + np.einsum("hd,hsd->hs", w, grid[codes])
        assert np.allclose(got, ref, atol=1e-5), f"bits={bits}"


def test_packed_matches_pck_reference():
    """Packed path matches ``pck_key_scores`` on a real packed asym-NF4 container."""
    H, S, D = 5, 130, 128
    rng = np.random.default_rng(7)
    x = (0.3 * rng.standard_normal((1, H, S, D))).astype(np.float32)
    x[:, :, :, ::5] += 1.5
    q_kv = PerChannelKV(head_dim=D, n_heads=H, bits=4, nf4_asym=True)
    c = q_kv.compress(x, packed=True)
    q = rng.standard_normal((H, D)).astype(np.float32)
    mu, weight, grid = _grid_params(q_kv, c)
    w = (q * weight).astype(np.float32)
    bias = (q * mu).sum(1).astype(np.float32)
    got = k2_key_scores_packed(c.indices, w, bias, grid, H, S, D, c.bits)
    ref = pck_key_scores(q, q_kv, c)
    assert np.allclose(got, ref, atol=1e-4)


def test_packed_cupy_exact():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # pragma: no cover
        pytest.skip("no CUDA device")
    H, S, D, bits = 32, 1024, 128, 4
    rng = np.random.default_rng(3)
    codes = rng.integers(0, 2**bits, size=(H, S, D), dtype=np.uint8)
    q = rng.standard_normal((H, D)).astype(np.float32)
    weight = (0.5 + rng.random((H, D))).astype(np.float32)
    w = (q * weight).astype(np.float32)
    bias = (0.1 * rng.standard_normal(H)).astype(np.float32)
    packed = _pack_indices(codes, bits)
    ref = bias[:, None] + np.einsum("hd,hsd->hs", w, _NF4[codes])
    got = k2_key_scores_packed(
        cp.asarray(packed),
        cp.asarray(w),
        cp.asarray(bias),
        cp.asarray(_NF4),
        H,
        S,
        D,
        bits,
    )
    assert float(cp.abs(got - cp.asarray(ref)).max()) < 1e-4

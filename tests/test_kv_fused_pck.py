"""
Tests for the M4 fused per-channel decode reference (kv_fused_pck).

The gate is exactness: the fused score decomposition (bias + dense grid sum +
CSR outlier deltas) must reproduce decompress-then-attend to float tolerance
for every per-channel key variant, with PolarQuant values.

Usage:
    pytest tests/test_kv_fused_pck.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro.core import TurboQuantKV
from turboquant_pro.kv_fused import _rot_matrices, _softmax
from turboquant_pro.kv_fused_pck import (
    build_outlier_csr,
    fused_decode_pck,
    pck_key_scores,
)
from turboquant_pro.per_channel_kv import PerChannelKV

RNG = np.random.default_rng(42)
H, S, D = 4, 96, 64
THETA = 1e6


def _keys(offset_scale: float = 4.0) -> np.ndarray:
    """Keys with a per-channel DC offset (the regime PerChannelKV serves)."""
    off = RNG.uniform(-offset_scale, offset_scale, size=(1, H, 1, D))
    return (off + RNG.standard_normal((1, H, S, D))).astype(np.float32)


def _values_polar():
    tq = TurboQuantKV(head_dim=D, n_heads=H, bits=4, use_gpu=False, seed=0)
    v = RNG.standard_normal((H, S, D)).astype(np.float32)
    norm_v = np.linalg.norm(v, axis=-1)
    unit = v / np.maximum(norm_v[..., None], 1e-30)
    rot = np.einsum("hsd,de->hse", unit, np.asarray(tq._Pi_T, dtype=np.float32))
    vcodes = np.searchsorted(tq.boundaries, rot).astype(np.uint8)
    return tq, vcodes, norm_v.astype(np.float32)


def _truth(q, k_full, tq, vcodes, norm_v):
    """Ground truth: standard attention over reconstructed K and dequant V."""
    _, pi, cent = _rot_matrices(tq, np)
    v = norm_v[..., None] * (cent[vcodes] @ pi)
    sc = np.einsum("hd,hsd->hs", q, k_full) / np.sqrt(q.shape[-1])
    p = _softmax(sc, np)
    return np.einsum("hs,hsd->hd", p, v)


def _check(quantizer, x, packed=False, position_start=0, hot=False, atol=2e-4):
    q = RNG.standard_normal((H, D)).astype(np.float32)
    c = quantizer.compress(x, packed=packed, position_start=position_start)
    tq, vcodes, norm_v = _values_polar()
    k_full = quantizer.decompress(c)[0]  # (H, S, D)
    if hot:
        hot_k = RNG.standard_normal((H, 8, D)).astype(np.float32)
        hot_v = RNG.standard_normal((H, 8, D)).astype(np.float32)
        got = fused_decode_pck(q, hot_k, hot_v, quantizer, c, vcodes, norm_v, tq)
        _, pi, cent = _rot_matrices(tq, np)
        v = norm_v[..., None] * (cent[vcodes] @ pi)
        k_all = np.concatenate([k_full, hot_k], axis=1)
        v_all = np.concatenate([v, hot_v], axis=1)
        sc = np.einsum("hd,hsd->hs", q, k_all) / np.sqrt(D)
        p = _softmax(sc, np)
        want = np.einsum("hs,hsd->hd", p, v_all)
    else:
        got = fused_decode_pck(q, None, None, quantizer, c, vcodes, norm_v, tq)
        want = _truth(q, k_full, tq, vcodes, norm_v)
    assert np.allclose(got, want, atol=atol), float(np.abs(got - want).max())


class TestExactness:
    """Fused score path == decompress-then-attend for every key variant."""

    def test_uniform(self) -> None:
        _check(PerChannelKV(head_dim=D, n_heads=H, bits=4), _keys())

    def test_uniform_3bit(self) -> None:
        _check(PerChannelKV(head_dim=D, n_heads=H, bits=3), _keys())

    def test_nf4_symmetric(self) -> None:
        _check(PerChannelKV(head_dim=D, n_heads=H, nf4=True), _keys())

    def test_nf4_asym(self) -> None:
        _check(PerChannelKV(head_dim=D, n_heads=H, nf4_asym=True), _keys())

    def test_nf4_asym_outliers(self) -> None:
        _check(
            PerChannelKV(head_dim=D, n_heads=H, nf4_asym=True, outlier_frac=0.02),
            _keys(),
        )

    def test_heavy_outliers(self) -> None:
        _check(
            PerChannelKV(head_dim=D, n_heads=H, nf4_asym=True, outlier_frac=0.10),
            _keys(offset_scale=8.0),
        )

    def test_zero_point_sparse(self) -> None:
        _check(
            PerChannelKV(
                head_dim=D,
                n_heads=H,
                nf4_asym=True,
                zero_point="sparse",
                rope_theta=THETA,
                outlier_frac=0.02,
            ),
            _keys(),
        )

    def test_zero_point_bias(self) -> None:
        bias = RNG.uniform(-2, 2, size=(H, D)).astype(np.float32)
        _check(
            PerChannelKV(
                head_dim=D,
                n_heads=H,
                nf4_asym=True,
                zero_point="bias",
                rope_theta=THETA,
                k_bias=bias,
                outlier_frac=0.02,
            ),
            _keys(),
            position_start=100,
        )

    def test_packed_codes(self) -> None:
        _check(
            PerChannelKV(head_dim=D, n_heads=H, nf4_asym=True, outlier_frac=0.02),
            _keys(),
            packed=True,
        )

    def test_hot_cold_merge(self) -> None:
        _check(
            PerChannelKV(head_dim=D, n_heads=H, nf4_asym=True, outlier_frac=0.02),
            _keys(),
            hot=True,
        )

    def test_nuq_rejected(self) -> None:
        q = RNG.standard_normal((H, D)).astype(np.float32)
        quantizer = PerChannelKV(head_dim=D, n_heads=H, nuq=True)
        c = quantizer.compress(_keys())
        with pytest.raises(NotImplementedError):
            pck_key_scores(q, quantizer, c)


class TestCSR:
    """The token-major CSR reproduces the container's overlay exactly."""

    def test_csr_matches_scatter(self) -> None:
        quantizer = PerChannelKV(
            head_dim=D, n_heads=H, nf4_asym=True, outlier_frac=0.05
        )
        x = _keys()
        c = quantizer.compress(x)
        row_ptr, cols, deltas = build_outlier_csr(quantizer, c)
        assert row_ptr[-1] == len(cols) == len(deltas)
        # dense dequant WITHOUT the overlay:
        c_no = quantizer.compress(x)
        c_no.outlier_idx = c_no.outlier_val = None
        dense = quantizer.decompress(c_no)[0]
        # apply CSR deltas on top -> must equal the container's own decompress
        rows = np.repeat(np.arange(H * S), np.diff(row_ptr))
        h, s = rows // S, rows % S
        dense[h, s, cols.astype(np.int64)] += deltas
        want = quantizer.decompress(c)[0]
        assert np.allclose(dense, want, atol=2e-3)

    def test_no_outliers_returns_none(self) -> None:
        quantizer = PerChannelKV(head_dim=D, n_heads=H, nf4_asym=True)
        assert build_outlier_csr(quantizer, quantizer.compress(_keys())) is None

    def test_row_density(self) -> None:
        """~outlier_frac of tokens per channel -> frac*D entries per token."""
        quantizer = PerChannelKV(
            head_dim=D, n_heads=H, nf4_asym=True, outlier_frac=0.02
        )
        c = quantizer.compress(_keys())
        row_ptr, _, _ = build_outlier_csr(quantizer, c)
        per_row = np.diff(row_ptr)
        assert abs(per_row.mean() - 0.02 * D) < 0.02 * D  # right order


class TestCudaKernel:
    """M4 CUDA kernel vs the reference (runs only where CuPy + GPU exist)."""

    def test_kernel_matches_reference(self) -> None:
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("no CUDA device")
        except Exception:
            pytest.skip("CUDA unavailable")
        from turboquant_pro.kv_kernel import fused_decode_pck_cuda

        bias = RNG.uniform(-2, 2, size=(H, D)).astype(np.float32)
        quantizer = PerChannelKV(
            head_dim=D,
            n_heads=H,
            nf4_asym=True,
            zero_point="bias",
            rope_theta=THETA,
            k_bias=bias,
            outlier_frac=0.02,
        )
        x = _keys()
        c = quantizer.compress(x)
        q = RNG.standard_normal((H, D)).astype(np.float32)
        tq, vcodes, norm_v = _values_polar()
        want = fused_decode_pck(q, None, None, quantizer, c, vcodes, norm_v, tq)
        got = cp.asnumpy(fused_decode_pck_cuda(q, quantizer, c, vcodes, norm_v, tq))
        assert np.allclose(got, want, atol=1e-4), float(np.abs(got - want).max())

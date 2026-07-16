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

from turboquant_pro.core import TurboQuantKV, TurboQuantKVCache
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


def _fill(cache: TurboQuantKVCache, n: int, offset_scale: float = 4.0) -> None:
    """Append n tokens with the per-channel DC-offset key regime."""
    off = RNG.uniform(-offset_scale, offset_scale, size=(H, D)).astype(np.float32)
    for _ in range(n):
        k = (off + RNG.standard_normal((H, D))).astype(np.float32)
        v = RNG.standard_normal((H, D)).astype(np.float32)
        cache.append(k, v)


def _cache_truth(cache: TurboQuantKVCache, q: np.ndarray) -> np.ndarray:
    """Decompress-then-attend over the whole cache via the public getters."""
    k = np.asarray(cache.get_keys(0, cache.length))[0]
    v = np.asarray(cache.get_values(0, cache.length))[0]
    sc = np.einsum("hd,hsd->hs", q, k) / np.sqrt(D)
    p = _softmax(sc, np)
    return np.einsum("hs,hsd->hd", p, v)


class TestCacheDispatch:
    """TurboQuantKVCache.fused_decode routes per-channel key pages through the
    M4 fused path, with the query-independent work (codes, grid params, outlier
    CSR) prepared once per cold flush and cached."""

    def _cache(self, **kw) -> TurboQuantKVCache:
        kw.setdefault("hot_window", 16)  # small window -> several cold pages
        return TurboQuantKVCache(
            head_dim=D,
            n_heads=H,
            bits=4,
            use_gpu=False,
            seed=0,
            per_channel_keys=True,
            **kw,
        )

    def test_exact_uniform(self) -> None:
        cache = self._cache()
        _fill(cache, 80)
        assert len(cache._cold_keys) >= 2
        q = RNG.standard_normal((H, D)).astype(np.float32)
        got = cache.fused_decode(q)
        assert np.allclose(got, _cache_truth(cache, q), atol=2e-4)

    def test_exact_nf4_asym_outliers(self) -> None:
        cache = self._cache(key_nf4_asym=True, key_outlier_frac=0.02)
        _fill(cache, 80)
        q = RNG.standard_normal((H, D)).astype(np.float32)
        got = cache.fused_decode(q)
        assert np.allclose(got, _cache_truth(cache, q), atol=2e-4)

    def test_exact_bias_zero_point_multipage(self) -> None:
        """The 'bias' zero-point depends on each page's absolute position_start;
        multiple pages exercise the per-page offsets."""
        bias = RNG.uniform(-2, 2, size=(H, D)).astype(np.float32)
        cache = self._cache(
            key_nf4_asym=True,
            key_outlier_frac=0.02,
            key_zero_point="bias",
            key_rope_theta=THETA,
            key_k_bias=bias,
        )
        _fill(cache, 80)
        assert len(cache._cold_keys) >= 2
        q = RNG.standard_normal((H, D)).astype(np.float32)
        got = cache.fused_decode(q)
        assert np.allclose(got, _cache_truth(cache, q), atol=2e-4)

    def test_pages_prepared_once(self, monkeypatch) -> None:
        """Each cold page is prepared exactly once: repeat decodes rebuild
        nothing, and new flushes only append."""
        import turboquant_pro.kv_fused_pck as mod

        builds: list[int] = []
        orig = mod.build_outlier_csr
        monkeypatch.setattr(
            mod, "build_outlier_csr", lambda *a: (builds.append(1), orig(*a))[1]
        )
        cache = self._cache(key_nf4_asym=True, key_outlier_frac=0.02)
        _fill(cache, 80)
        n_pages = len(cache._cold_keys)
        q = RNG.standard_normal((H, D)).astype(np.float32)
        out1 = cache.fused_decode(q)
        assert len(cache._pck_blocks) == n_pages
        first = list(cache._pck_blocks)
        n_builds = len(builds)
        out2 = cache.fused_decode(q)
        assert len(builds) == n_builds  # nothing rebuilt on the second decode
        assert all(a is b for a, b in zip(cache._pck_blocks, first))
        assert np.allclose(out1, out2)
        _fill(cache, 40)  # forces at least one new flush
        assert len(cache._cold_keys) > n_pages
        cache.fused_decode(q)
        assert len(cache._pck_blocks) == len(cache._cold_keys)
        assert all(a is b for a, b in zip(cache._pck_blocks[:n_pages], first))

    def test_nuq_falls_back_to_reconstruction(self) -> None:
        """Data-fit quantile grids have no fused form; the dispatch must fall
        back to decompress-then-attend and stay exact."""
        cache = self._cache(key_nuq=True)
        _fill(cache, 60)
        q = RNG.standard_normal((H, D)).astype(np.float32)
        got = cache.fused_decode(q)
        assert cache._pck_fused_ok is False
        assert cache._pck_blocks == []
        assert np.allclose(got, _cache_truth(cache, q), atol=2e-4)

    def test_clear_drops_prepared_pages(self) -> None:
        cache = self._cache(key_nf4_asym=True, key_outlier_frac=0.02)
        _fill(cache, 60)
        q = RNG.standard_normal((H, D)).astype(np.float32)
        cache.fused_decode(q)
        assert cache._pck_blocks
        cache.clear()
        assert cache._pck_blocks == [] and cache._pck_fused_ok is None
        assert cache.length == 0


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

    def test_cache_dispatch_gpu(self) -> None:
        """End-to-end cache path on GPU: prepared pages hold device arrays and
        the kernel partials merge exactly with the hot window."""
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("no CUDA device")
        except Exception:
            pytest.skip("CUDA unavailable")
        cache = TurboQuantKVCache(
            head_dim=D,
            n_heads=H,
            bits=4,
            use_gpu=True,
            seed=0,
            per_channel_keys=True,
            key_nf4_asym=True,
            key_outlier_frac=0.02,
            hot_window=16,
        )
        _fill(cache, 80)
        assert len(cache._cold_keys) >= 2
        q = RNG.standard_normal((H, D)).astype(np.float32)
        got = cp.asnumpy(cache.fused_decode(q))
        assert np.allclose(got, _cache_truth(cache, q), atol=2e-4)
        assert len(cache._pck_blocks) == len(cache._cold_keys)
        assert all(blk.xp is cp for blk in cache._pck_blocks)

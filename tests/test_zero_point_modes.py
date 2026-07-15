"""
Tests for PerChannelKV zero-point modes ("calibrated" / "sparse" / "bias").

The synthetic keys are built with an independent per-position rotation (not
the module's own position-averaged helper), so the bias mode is tested
against ground truth rather than against itself.

Usage:
    pytest tests/test_zero_point_modes.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro.per_channel_kv import PerChannelKV

RNG = np.random.default_rng(42)

H, D, S = 4, 64, 256
THETA = 1e6


def _rope_rotate(vec: np.ndarray, pos: int, theta: float, d: int) -> np.ndarray:
    """Independent rotate_half RoPE of a (H, D) vector at one position."""
    half = d // 2
    inv = theta ** (-2.0 * np.arange(half) / d)
    ang = pos * inv
    c = np.concatenate([np.cos(ang), np.cos(ang)])
    s = np.concatenate([np.sin(ang), np.sin(ang)])
    rh = np.concatenate([-vec[:, half:], vec[:, :half]], axis=1)
    return vec * c[None, :] + rh * s[None, :]


def _keylike(bias: np.ndarray, n: int = S, noise: float = 0.1, start: int = 0):
    """Post-RoPE-like keys: rotated bias per position + small residual."""
    x = np.stack(
        [_rope_rotate(bias, start + t, THETA, D) for t in range(n)], axis=1
    )  # (H, S, D)
    x = x + noise * RNG.standard_normal(x.shape)
    return x[None].astype(np.float32)  # (1, H, S, D)


def _bias() -> np.ndarray:
    return np.random.default_rng(7).uniform(-8.0, 8.0, size=(H, D)).astype(np.float32)


def _relerr(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(x - y) / np.linalg.norm(x))


def _relerr_dc(x: np.ndarray, y: np.ndarray, window: int) -> float:
    """Relative error restricted to the DC channels, where the offset damage
    concentrates (fast channels dilute the global norm)."""
    dc = PerChannelKV.dc_channel_mask(THETA, D, window)
    return _relerr(x[..., dc], y[..., dc])


class TestHelpers:
    """Config-only RoPE geometry helpers."""

    def test_dc_mask_matches_wavelength_math(self) -> None:
        """Mask is exactly the wavelength > window criterion, both thetas."""
        for theta, window in ((1e6, 512), (1e4, 512)):
            mask = PerChannelKV.dc_channel_mask(theta, 128, window)
            half = 64
            inv = theta ** (-2.0 * np.arange(half) / 128)
            expect = np.tile(2 * np.pi / inv, 2) > window
            assert mask.shape == (128,)
            assert np.array_equal(mask, expect)

    def test_dc_mask_fractions(self) -> None:
        """The measured channel fractions: 67.2% at theta=1e6, 51.6% at 1e4."""
        assert abs(PerChannelKV.dc_channel_mask(1e6, 128, 512).mean() - 0.672) < 0.01
        assert abs(PerChannelKV.dc_channel_mask(1e4, 128, 512).mean() - 0.516) < 0.01

    def test_rope_averaged_bias_matches_bruteforce(self) -> None:
        """Closed-form position average equals the explicit position loop."""
        bias = _bias()
        start, n = 13, 100
        brute = np.mean(
            [_rope_rotate(bias, start + t, THETA, D) for t in range(n)], axis=0
        )
        fast = PerChannelKV.rope_averaged_bias(bias, THETA, D, start, n)
        assert np.allclose(fast, brute, atol=1e-4)


class TestBiasMode:
    """zero_point='bias': zero calibration, zero stored zero-point metadata."""

    def _pair(self):
        bias = _bias()
        calib = PerChannelKV(head_dim=D, n_heads=H, nf4_asym=True)
        biasq = PerChannelKV(
            head_dim=D,
            n_heads=H,
            nf4_asym=True,
            zero_point="bias",
            rope_theta=THETA,
            k_bias=bias,
        )
        return bias, calib, biasq

    def test_no_stored_mean_and_smaller_container(self) -> None:
        """Bias mode stores no nf4_mean; the container is measurably smaller."""
        bias, calib, biasq = self._pair()
        x = _keylike(bias)
        cb = biasq.compress(x)
        cc = calib.compress(x)
        assert cb.zp_mode == "bias"
        assert cb.nf4_mean is None
        assert cc.nf4_mean is not None
        assert cb.nbytes() == cc.nbytes() - cc.nf4_mean.nbytes

    def test_roundtrip_quality_matches_calibrated(self) -> None:
        """On RoPE-bias keys, bias-mode recon ~ calibrated recon, << symmetric."""
        bias, calib, biasq = self._pair()
        sym = PerChannelKV(head_dim=D, n_heads=H, nf4=True)
        x = _keylike(bias)
        xb = biasq.decompress(biasq.compress(x))
        xc = calib.decompress(calib.compress(x))
        xs = sym.decompress(sym.compress(x))
        assert _relerr(x, xb) < 1.5 * _relerr(x, xc)  # ~ calibrated globally
        assert _relerr(x, xb) < 0.8 * _relerr(x, xs)  # better than symmetric
        # the offset damage concentrates in the DC channels: decisive there
        assert _relerr_dc(x, xb, S) < 0.5 * _relerr_dc(x, xs, S)

    def test_position_start_respected(self) -> None:
        """A block starting at position 1000 quantizes better with the correct
        start than with the wrong one."""
        bias, _, biasq = self._pair()
        x = _keylike(bias, start=1000)
        good = biasq.decompress(biasq.compress(x, position_start=1000))
        bad = biasq.decompress(biasq.compress(x, position_start=0))
        assert _relerr(x, good) <= _relerr(x, bad)

    def test_packed_roundtrip(self) -> None:
        """Bit-packed path works with the bias zero-point."""
        bias, _, biasq = self._pair()
        x = _keylike(bias)
        unpacked = biasq.decompress(biasq.compress(x, packed=False))
        packed = biasq.decompress(biasq.compress(x, packed=True))
        assert np.allclose(unpacked, packed)

    def test_decompress_requires_bias(self) -> None:
        """A bias-mode container cannot be decoded without the bias."""
        bias, calib, biasq = self._pair()
        c = biasq.compress(_keylike(bias))
        with pytest.raises(ValueError, match="k_bias"):
            calib.decompress(c)

    def test_head_count_mismatch_raises(self) -> None:
        """Input head count must match the bias the quantizer was built with."""
        bias, _, biasq = self._pair()
        x = _keylike(bias)[:, : H - 1]
        with pytest.raises(ValueError, match="heads"):
            biasq.compress(x)


class TestSparseMode:
    """zero_point='sparse': calibrated means on config-identified DC channels."""

    def _quantizer(self):
        return PerChannelKV(
            head_dim=D, n_heads=H, nf4_asym=True, zero_point="sparse", rope_theta=THETA
        )

    def test_compact_storage(self) -> None:
        """Only the DC channels' means are stored."""
        bias = _bias()
        x = _keylike(bias)
        c = self._quantizer().compress(x)
        n_dc = int(PerChannelKV.dc_channel_mask(THETA, D, S).sum())
        assert c.zp_mode == "sparse"
        assert c.nf4_mean.shape == (1, H, n_dc)
        full = PerChannelKV(head_dim=D, n_heads=H, nf4_asym=True).compress(x)
        assert c.nbytes() < full.nbytes()

    def test_roundtrip_close_to_calibrated_on_dc_data(self) -> None:
        """When the offset lives in DC channels (as measured on real keys),
        sparse recon ~ calibrated recon, << symmetric."""
        bias = _bias()
        x = _keylike(bias)
        sparse = self._quantizer()
        calib = PerChannelKV(head_dim=D, n_heads=H, nf4_asym=True)
        sym = PerChannelKV(head_dim=D, n_heads=H, nf4=True)
        xp = sparse.decompress(sparse.compress(x))
        xc = calib.decompress(calib.compress(x))
        xs = sym.decompress(sym.compress(x))
        assert _relerr(x, xp) < 1.5 * _relerr(x, xc)  # ~ calibrated globally
        assert _relerr(x, xp) < 0.8 * _relerr(x, xs)  # better than symmetric
        assert _relerr_dc(x, xp, S) < 0.5 * _relerr_dc(x, xs, S)

    def test_all_dc_equals_calibrated(self) -> None:
        """A tiny window makes every channel DC: sparse == calibrated exactly."""
        bias = _bias()
        # the fastest rotary channel always has wavelength 2*pi ~ 6.3, so the
        # window must be <= 6 for EVERY channel to qualify as DC
        x = _keylike(bias, n=6)
        sparse = self._quantizer()
        calib = PerChannelKV(head_dim=D, n_heads=H, nf4_asym=True)
        assert np.allclose(
            sparse.decompress(sparse.compress(x)),
            calib.decompress(calib.compress(x)),
        )


class TestValidation:
    """Constructor contracts."""

    def test_mode_requires_nf4_asym(self) -> None:
        with pytest.raises(ValueError, match="nf4_asym"):
            PerChannelKV(zero_point="sparse", rope_theta=1e4)

    def test_mode_requires_theta(self) -> None:
        with pytest.raises(ValueError, match="rope_theta"):
            PerChannelKV(nf4_asym=True, zero_point="sparse")

    def test_bias_mode_requires_bias(self) -> None:
        with pytest.raises(ValueError, match="k_bias"):
            PerChannelKV(nf4_asym=True, zero_point="bias", rope_theta=1e4)

    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero_point"):
            PerChannelKV(nf4_asym=True, zero_point="frequency")

    def test_default_unchanged(self) -> None:
        """Default construction is byte-identical to the legacy behavior."""
        x = _keylike(_bias())
        q = PerChannelKV(head_dim=D, n_heads=H, nf4_asym=True)
        c = q.compress(x)
        assert c.zp_mode == "calibrated"
        assert c.rope_theta is None
        assert c.nf4_mean is not None


class TestCacheIntegration:
    """TurboQuantKVCache plumbing for the zero-point modes."""

    def test_streaming_cache_with_bias_zero_point(self) -> None:
        """A streaming cache with zero_point='bias' round-trips keys well and
        passes advancing absolute positions to each cold flush."""
        from turboquant_pro import TurboQuantKVCache

        bias = _bias()
        cache = TurboQuantKVCache(
            head_dim=D,
            n_heads=H,
            key_bits=4,
            value_bits=4,
            hot_window=32,
            use_gpu=False,
            per_channel_keys=True,
            key_nf4_asym=True,
            key_zero_point="bias",
            key_rope_theta=THETA,
            key_k_bias=bias,
        )
        n = 96
        x = _keylike(bias, n=n)  # (1, H, n, D)
        for t in range(n):
            cache.append(x[:, :, t, :].reshape(H, D), x[:, :, t, :].reshape(H, D))
        got = cache.get_keys(0, n)
        assert got.shape == (1, H, n, D)
        assert _relerr(x, got) < 0.1
        # at least one cold flush happened with a non-zero absolute start
        assert len(cache._cold_lengths) >= 2

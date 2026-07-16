"""
Tests for the torch backend milestone of P1 (docs/DESIGN_hardware_and_plugins.md
section 4.1): instruments accept torch tensors from any device via the
to_numpy boundary, and torch_decode is the decompress-then-attend reference
running natively on any torch device.

Usage:
    pytest tests/test_torch_backend.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro.a2_probe import tangential_fraction
from turboquant_pro.backend import is_torch_tensor, to_numpy
from turboquant_pro.rank_certificate import certify, measure_kappa, mu_hat

torch = pytest.importorskip("torch")

RNG = np.random.default_rng(11)
H, D = 4, 64


def _devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devs.append("mps")
    return devs


class TestToNumpy:
    def test_numpy_passthrough(self) -> None:
        x = RNG.standard_normal(5)
        assert to_numpy(x) is not None and np.shares_memory(to_numpy(x), x)

    def test_list_input(self) -> None:
        assert to_numpy([1.0, 2.0]).shape == (2,)

    @pytest.mark.parametrize("device", _devices())
    def test_torch_any_device(self, device) -> None:
        t = torch.randn(3, 4, device=device)
        out = to_numpy(t)
        assert isinstance(out, np.ndarray) and out.shape == (3, 4)
        assert np.allclose(out, t.detach().cpu().numpy())

    def test_is_torch_tensor(self) -> None:
        assert is_torch_tensor(torch.zeros(1))
        assert not is_torch_tensor(np.zeros(1))


class TestInstrumentsAcceptTorch:
    """The exit criterion's instrument half: identical numbers from torch
    inputs on every available device."""

    @pytest.mark.parametrize("device", _devices())
    def test_certificate_matches_numpy(self, device) -> None:
        exact = np.abs(RNG.standard_normal(4000)) + 0.1
        approx = exact * np.exp(RNG.normal(0, 0.05, exact.shape))
        want = certify(exact, approx)
        got = certify(
            torch.as_tensor(exact, device=device),
            torch.as_tensor(approx, device=device),
        )
        assert got.kappa == want.kappa
        assert got.tau_floor == want.tau_floor

    @pytest.mark.parametrize("device", _devices())
    def test_kappa_mu_match(self, device) -> None:
        exact = np.abs(RNG.standard_normal(2000)) + 0.1
        approx = exact * 1.05
        te = torch.as_tensor(exact, device=device)
        ta = torch.as_tensor(approx, device=device)
        assert measure_kappa(te, ta) == measure_kappa(exact, approx)
        assert mu_hat(te, 1.2) == mu_hat(exact, 1.2)

    @pytest.mark.parametrize("device", _devices())
    def test_tangential_fraction_matches(self, device) -> None:
        x = RNG.standard_normal(64)
        y = x + 0.01 * RNG.standard_normal(64)
        want = tangential_fraction(x, y)
        got = tangential_fraction(
            torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
        )
        assert got == pytest.approx(want, abs=1e-12)


class TestTorchDecode:
    """The exit criterion's decode half: decompress-then-attend natively in
    torch matches the cache's own fused decode."""

    def _cache(self):
        from turboquant_pro.core import TurboQuantKVCache

        cache = TurboQuantKVCache(
            head_dim=D,
            n_heads=H,
            bits=4,
            use_gpu=False,
            seed=0,
            per_channel_keys=True,
            key_nf4_asym=True,
            key_outlier_frac=0.02,
            hot_window=16,
        )
        off = RNG.uniform(-4, 4, size=(H, D)).astype(np.float32)
        for _ in range(60):
            cache.append(
                (off + RNG.standard_normal((H, D))).astype(np.float32),
                RNG.standard_normal((H, D)).astype(np.float32),
            )
        return cache

    @pytest.mark.parametrize("device", _devices())
    def test_matches_fused_decode(self, device) -> None:
        from turboquant_pro.backend import torch_decode

        cache = self._cache()
        q = RNG.standard_normal((H, D)).astype(np.float32)
        want = np.asarray(cache.fused_decode(q))
        got = torch_decode(cache, q, device=device)
        assert str(got.device).startswith(device)
        assert np.allclose(to_numpy(got), want, atol=5e-5), float(
            np.abs(to_numpy(got) - want).max()
        )

    def test_device_follows_query_tensor(self) -> None:
        from turboquant_pro.backend import torch_decode

        cache = self._cache()
        q = torch.randn(H, D)
        out = torch_decode(cache, q)
        assert out.device == q.device

    def test_empty_cache_raises(self) -> None:
        from turboquant_pro.backend import torch_decode
        from turboquant_pro.core import TurboQuantKVCache

        cache = TurboQuantKVCache(head_dim=D, n_heads=H, use_gpu=False)
        with pytest.raises(RuntimeError, match="empty"):
            torch_decode(cache, np.zeros((H, D), dtype=np.float32))


class TestTorchXPReferencePaths:
    """P1 slice 2: the kv_fused reference einsums run with xp=torch_xp on any
    device and match the NumPy reference."""

    def _polar(self):
        from turboquant_pro.core import TurboQuantKV

        tq = TurboQuantKV(head_dim=D, n_heads=H, bits=4, use_gpu=False, seed=0)
        S = 48
        k = RNG.standard_normal((H, S, D)).astype(np.float32)
        v = RNG.standard_normal((H, S, D)).astype(np.float32)
        codes = {}
        norms = {}
        for name, x in (("k", k), ("v", v)):
            n = np.linalg.norm(x, axis=-1)
            unit = x / np.maximum(n[..., None], 1e-30)
            rot = np.einsum("hsd,de->hse", unit, np.asarray(tq._Pi_T, dtype=np.float32))
            codes[name] = np.searchsorted(tq.boundaries, rot).astype(np.uint8)
            norms[name] = n.astype(np.float32)
        return tq, codes, norms

    @pytest.mark.parametrize("device", _devices())
    def test_fused_decode_attention_matches(self, device) -> None:
        from turboquant_pro.backend import torch_xp
        from turboquant_pro.kv_fused import fused_decode_attention

        tq, codes, norms = self._polar()
        q = RNG.standard_normal((H, D)).astype(np.float32)
        want = fused_decode_attention(
            q, codes["k"], codes["v"], norms["k"], norms["v"], tq, xp=np
        )
        dev = torch.device(device)
        got = fused_decode_attention(
            torch.as_tensor(q, device=dev),
            torch.as_tensor(codes["k"], device=dev),
            torch.as_tensor(codes["v"], device=dev),
            torch.as_tensor(norms["k"], device=dev),
            torch.as_tensor(norms["v"], device=dev),
            tq,
            xp=torch_xp,
        )
        assert np.allclose(to_numpy(got), want, atol=5e-5)

    @pytest.mark.parametrize("device", _devices())
    def test_full_fused_decode_with_hot_window(self, device) -> None:
        from turboquant_pro.backend import torch_xp
        from turboquant_pro.kv_fused import fused_decode

        tq, codes, norms = self._polar()
        q = RNG.standard_normal((H, D)).astype(np.float32)
        hot_k = RNG.standard_normal((H, 8, D)).astype(np.float32)
        hot_v = RNG.standard_normal((H, 8, D)).astype(np.float32)
        want = fused_decode(
            q, hot_k, hot_v, codes["k"], codes["v"], norms["k"], norms["v"], tq, xp=np
        )
        dev = torch.device(device)
        args = [q, hot_k, hot_v, codes["k"], codes["v"], norms["k"], norms["v"]]
        targs = [torch.as_tensor(a, device=dev) for a in args]
        got = fused_decode(*targs, tq, xp=torch_xp)
        assert np.allclose(to_numpy(got), want, atol=5e-5)

    def test_pck_reference_scores_match(self) -> None:
        from turboquant_pro.backend import torch_xp
        from turboquant_pro.kv_fused_pck import pck_key_scores
        from turboquant_pro.per_channel_kv import PerChannelKV

        quant = PerChannelKV(head_dim=D, n_heads=H, nf4_asym=True, outlier_frac=0.02)
        off = RNG.uniform(-4, 4, size=(1, H, 1, D))
        x = (off + RNG.standard_normal((1, H, 48, D))).astype(np.float32)
        c = quant.compress(x)
        q = RNG.standard_normal((H, D)).astype(np.float32)
        want = pck_key_scores(q, quant, c, xp=np)
        got = pck_key_scores(torch.as_tensor(q), quant, c, xp=torch_xp)
        assert np.allclose(to_numpy(got), want, atol=5e-5)

"""Tests for PerChannelKV (per-channel key quantizer).

Beyond round-trip fidelity, these encode *why the class exists*: it must stay robust
when one channel is a large outlier (the key regime where per-vector PolarQuant
collapses), and it must preserve the relative **dot-product** structure attention needs.
"""

import numpy as np
import pytest

from turboquant_pro import CompressedPerChannelKV, PerChannelKV


def _rel(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-9)


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("nuq", [False, True])
def test_roundtrip_shape_and_fidelity(bits, nuq):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 128, 64)).astype(np.float32)
    q = PerChannelKV(head_dim=64, n_heads=3, bits=bits, nuq=nuq)
    xh = q.decompress(q.compress(x))
    assert xh.shape == x.shape and xh.dtype == np.float32
    tol = {2: 0.55, 3: 0.25, 4: 0.13}[bits]  # 2-bit = 4 levels -> ~0.5 rel on N(0,1)
    assert _rel(xh, x) < tol, f"bits={bits} nuq={nuq} rel={_rel(xh, x):.3f}"


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("nuq", [False, True])
def test_packed_roundtrip_matches_unpacked(bits, nuq):
    rng = np.random.default_rng(7)
    x = rng.standard_normal((1, 2, 96, 64)).astype(np.float32)
    q = PerChannelKV(head_dim=64, bits=bits, nuq=nuq)
    unpacked = q.decompress(q.compress(x, packed=False))
    packed = q.decompress(q.compress(x, packed=True))
    assert np.array_equal(unpacked, packed)  # packing is lossless
    c = q.compress(x, packed=True)
    assert c.packed and c.compression_ratio(64) > 1.0


def test_outlier_channel_is_isolated():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 2, 128, 64)).astype(np.float32)
    x[..., 0] *= 50.0
    q = PerChannelKV(head_dim=64, bits=4)
    xh = q.decompress(q.compress(x))
    assert _rel(xh[..., 1:], x[..., 1:]) < 0.12  # non-outlier channels stay accurate


def test_preserves_dot_product_ordering():
    rng = np.random.default_rng(2)
    K = rng.standard_normal((1, 1, 256, 64)).astype(np.float32)
    K[..., 3] *= 30.0
    q = rng.standard_normal((64,)).astype(np.float32)
    quant = PerChannelKV(head_dim=64, bits=4)
    xh = quant.decompress(quant.compress(K))
    top_true = set(np.argsort(K[0, 0] @ q)[-10:].tolist())
    top_quant = set(np.argsort(xh[0, 0] @ q)[-10:].tolist())
    assert len(top_true & top_quant) >= 8


@pytest.mark.parametrize("packed", [False, True])
def test_nf4_levels_beat_uniform(packed):
    """NF4 (calibration-free normal-float) should reconstruct N(0,1) keys at least as
    well as uniform 4-bit -- it allocates levels by the Gaussian, not evenly."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal((2, 3, 200, 64)).astype(np.float32)
    uni = PerChannelKV(head_dim=64, bits=4)
    nf4 = PerChannelKV(head_dim=64, bits=4, nf4=True)
    assert _rel(nf4.decompress(nf4.compress(x, packed=packed)), x) <= _rel(
        uni.decompress(uni.compress(x, packed=packed)), x
    )
    with pytest.raises(ValueError):  # NF4 is a 4-bit codebook
        PerChannelKV(head_dim=64, bits=3, nf4=True)


def test_dense_sparse_outliers_recover_outlier_channel():
    """1% dense-sparse fp16 outliers should sharply reduce error on a heavy-outlier
    key channel that uniform 4-bit otherwise crushes (the LongBench/qasper regime)."""
    rng = np.random.default_rng(12)
    x = rng.standard_normal((1, 4, 300, 32)).astype(np.float32)
    x[..., 5] *= 25.0  # outlier channel
    base = PerChannelKV(head_dim=32, n_heads=4, bits=4)
    out = PerChannelKV(head_dim=32, n_heads=4, bits=4, outlier_frac=0.01)
    c = out.compress(x, packed=True)
    assert c.outlier_idx is not None
    assert len(c.outlier_idx) == max(1, round(300 * 0.01)) * 4 * 32  # 1% per channel
    err_base = _rel(base.decompress(base.compress(x)), x)
    err_out = _rel(out.decompress(c), x)
    assert err_out < err_base  # outliers help
    assert c.compression_ratio(32) > 1.0  # still compresses despite fp16 sidecar


def test_nf4_plus_outliers_roundtrips_through_cache():
    from turboquant_pro import TurboQuantKVCache

    rng = np.random.default_rng(13)
    cache = TurboQuantKVCache(
        head_dim=64,
        n_heads=2,
        bits=4,
        hot_window=8,
        use_gpu=False,
        key_nf4=True,
        key_outlier_frac=0.01,
    )
    assert cache._kq is not None and cache._kq.nf4 and cache._kq.outlier_frac == 0.01
    keys = []
    for _ in range(40):
        k = rng.standard_normal((2, 64)).astype(np.float32)
        keys.append(k)
        cache.append(k, rng.standard_normal((2, 64)).astype(np.float32))
    got = cache.get_keys(0, cache.length)
    assert got.shape == (1, 2, cache.length, 64)
    ref = np.stack(keys, axis=1)[np.newaxis]
    cold = cache.cold_length
    rel = np.linalg.norm(got[:, :, :cold] - ref[:, :, :cold]) / np.linalg.norm(
        ref[:, :, :cold]
    )
    assert rel < 0.2


def test_compression_ratio_and_type():
    x = np.random.default_rng(3).standard_normal((1, 2, 512, 128)).astype(np.float32)
    c = PerChannelKV(head_dim=128, bits=4).compress(x, packed=True)
    assert isinstance(c, CompressedPerChannelKV)
    assert c.compression_ratio(128) > 3.0


def test_kvcache_uses_per_channel_keys_by_default():
    from turboquant_pro import TurboQuantKVCache

    rng = np.random.default_rng(5)
    cache = TurboQuantKVCache(
        head_dim=64, n_heads=2, bits=4, hot_window=8, use_gpu=False
    )
    assert (
        cache.per_channel_keys and cache._kq is not None
    )  # correct architecture is the default
    keys = []
    for _ in range(40):
        k = rng.standard_normal((2, 64)).astype(np.float32)
        v = rng.standard_normal((2, 64)).astype(np.float32)
        keys.append(k)
        cache.append(k, v)
    assert cache.cold_length > 0  # some keys went through PerChannelKV
    got = cache.get_keys(0, cache.length)  # cold (per-channel) + hot path round-trips
    assert got.shape == (1, 2, cache.length, 64)
    # the per-channel key path reconstructs cold keys with bounded error
    ref = np.stack(keys, axis=1)[np.newaxis]  # (1,2,T,64)
    cold = cache.cold_length
    rel = np.linalg.norm(got[:, :, :cold] - ref[:, :, :cold]) / np.linalg.norm(
        ref[:, :, :cold]
    )
    assert rel < 0.2

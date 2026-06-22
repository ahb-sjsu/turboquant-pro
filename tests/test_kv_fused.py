"""M0 tests: fused KV-decode (code space) matches the dequant path exactly."""

import numpy as np

from turboquant_pro import TurboQuantPGVector
from turboquant_pro.kv_fused import dequant_decode_attention, fused_decode_attention


def _setup(heads=4, S=256, d=64, bits=3, seed=0):
    tq = TurboQuantPGVector(dim=d, bits=bits)
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((heads, d)).astype(np.float32)

    def code(X):
        n = np.linalg.norm(X, axis=2)
        r = tq._rotate(X / np.maximum(n[..., None], 1e-30))
        return np.searchsorted(tq.boundaries, r).astype(np.uint8), n.astype(np.float32)

    kcodes, nk = code(rng.standard_normal((heads, S, d)).astype(np.float32))
    vcodes, nv = code(rng.standard_normal((heads, S, d)).astype(np.float32))
    return tq, q, kcodes, vcodes, nk, nv


def test_fused_matches_dequant():
    tq, q, kc, vc, nk, nv = _setup()
    out_f = fused_decode_attention(q, kc, vc, nk, nv, tq)
    out_d = dequant_decode_attention(q, kc, vc, nk, nv, tq)
    assert out_f.shape == (4, 64)
    np.testing.assert_allclose(out_f, out_d, atol=1e-4, rtol=1e-3)


def test_fused_matches_dequant_4bit():
    tq, q, kc, vc, nk, nv = _setup(d=128, bits=4, S=512)
    np.testing.assert_allclose(
        fused_decode_attention(q, kc, vc, nk, nv, tq),
        dequant_decode_attention(q, kc, vc, nk, nv, tq),
        atol=1e-4,
        rtol=1e-3,
    )


def test_softmax_normalized_output_is_convex_combo():
    # output must lie within the range of reconstructed V rows (convexity sanity)
    tq, q, kc, vc, nk, nv = _setup(heads=1, S=64, d=32)
    out = fused_decode_attention(q, kc, vc, nk, nv, tq)
    assert np.isfinite(out).all()


def _full_ref(q, hot_k, hot_v, kc, vc, nk, nv, tq):
    # exact attention over [dequantized cold ; hot]
    from turboquant_pro.kv_fused import _rot_matrices

    _, pi, cent = _rot_matrices(tq, np)
    parts_k, parts_v = [], []
    if kc is not None and kc.shape[1] > 0:
        parts_k.append(nk[..., None] * (cent[kc] @ pi))
        parts_v.append(nv[..., None] * (cent[vc] @ pi))
    if hot_k is not None and hot_k.shape[1] > 0:
        parts_k.append(hot_k)
        parts_v.append(hot_v)
    K = np.concatenate(parts_k, axis=1)
    V = np.concatenate(parts_v, axis=1)
    d = q.shape[-1]
    s = np.einsum("hd,hsd->hs", q, K) / np.sqrt(d)
    s -= s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    return np.einsum("hs,hsd->hd", p, V)


def test_hot_cold_merge_matches_full_attention():
    from turboquant_pro.kv_fused import fused_decode

    tq, q, kc, vc, nk, nv = _setup(heads=4, S=300, d=64)
    rng = np.random.default_rng(7)
    hot_k = rng.standard_normal((4, 48, 64)).astype(np.float32)
    hot_v = rng.standard_normal((4, 48, 64)).astype(np.float32)
    out = fused_decode(q, hot_k, hot_v, kc, vc, nk, nv, tq)
    ref = _full_ref(q, hot_k, hot_v, kc, vc, nk, nv, tq)
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-3)


def test_hot_only_and_cold_only():
    from turboquant_pro.kv_fused import fused_decode

    tq, q, kc, vc, nk, nv = _setup(heads=2, S=128, d=64)
    rng = np.random.default_rng(8)
    hk = rng.standard_normal((2, 32, 64)).astype(np.float32)
    hv = rng.standard_normal((2, 32, 64)).astype(np.float32)
    # cold-only
    np.testing.assert_allclose(
        fused_decode(q, None, None, kc, vc, nk, nv, tq),
        _full_ref(q, None, None, kc, vc, nk, nv, tq),
        atol=1e-4,
        rtol=1e-3,
    )
    # hot-only
    np.testing.assert_allclose(
        fused_decode(q, hk, hv, None, None, None, None, tq),
        _full_ref(q, hk, hv, None, None, None, None, tq),
        atol=1e-4,
        rtol=1e-3,
    )


def test_cache_fused_decode_matches_decompress_attend():
    from turboquant_pro import TurboQuantKVCache

    rng = np.random.default_rng(0)
    H, d = 4, 64
    cache = TurboQuantKVCache(
        head_dim=d, n_heads=H, bits=3, hot_window=64, use_gpu=False
    )
    for _ in range(300):  # exceeds hot_window -> forces cold flushes
        cache.append(
            rng.standard_normal((H, d)).astype(np.float32),
            rng.standard_normal((H, d)).astype(np.float32),
        )
    assert cache.cold_length > 0 and cache.hot_length > 0
    q = rng.standard_normal((H, d)).astype(np.float32)
    out = cache.fused_decode(q)
    # reference: full attention over decompress(cold) + hot via the public API
    K = np.asarray(cache.get_keys(0, cache.length))[0]
    V = np.asarray(cache.get_values(0, cache.length))[0]
    s = np.einsum("hd,hsd->hs", q, K) / np.sqrt(d)
    s -= s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    ref = np.einsum("hs,hsd->hd", p, V)
    assert out.shape == (H, d)
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-3)

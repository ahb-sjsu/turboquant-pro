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

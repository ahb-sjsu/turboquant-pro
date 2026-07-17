# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""P5 Triton port: exactness vs the NumPy reference (the RawKernel oracle).

Two layers:

* :func:`_kernel_sim_pck` replays the Triton M4 kernel's *exact* control flow
  in NumPy -- split-K over the key axis, per-token CSR outlier loop, online
  (flash) softmax, host flash-combine. It runs everywhere and gates the port's
  algorithm/indexing before any GPU time is spent; if the decomposition is
  wrong this fails on CPU.
* the ``triton_*`` tests call the real kernels and compare to the reference;
  they skip unless :func:`turboquant_pro.kv_triton.has_triton` (Triton + CUDA).
  These are the P5 exit-criterion checks, run on NRP V100/A100.
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro.core import TurboQuantKVCache
from turboquant_pro.kv_triton import _nsplit, has_triton


def _make_block(H=4, D=64, n_tok=50, outlier=0.02, seed=0):
    """Build one cold PreparedPCKBlock (numpy) + its reference partials."""
    cache = TurboQuantKVCache(
        head_dim=D, n_heads=H, bits=4, use_gpu=False, seed=seed,
        per_channel_keys=True, key_nf4_asym=True, key_outlier_frac=outlier,
        hot_window=4,
    )
    rng = np.random.default_rng(seed)
    off = rng.uniform(-4, 4, size=(H, D)).astype(np.float32)
    for _ in range(n_tok):
        cache.append(
            (off + rng.standard_normal((H, D))).astype(np.float32),
            rng.standard_normal((H, D)).astype(np.float32),
        )
    # small hot_window flushes most tokens to cold automatically
    blocks = cache._prepared_pck_blocks()
    q = rng.standard_normal((H, D)).astype(np.float32)
    return cache, blocks, q


def _kernel_sim_pck(blk, q, tq, scale):
    """NumPy replay of the Triton M4 kernel: split-K + CSR + online softmax +
    flash-combine. Mirrors _pck_partials_kernel one-to-one (per (h, split)
    program, per-token CSR row, running (m, lsum, acc))."""
    H, S, D = blk.H, blk.S, blk.D
    q = np.asarray(q, np.float32).reshape(H, D)
    w = q * np.asarray(blk.weight, np.float32)          # (H, D)
    bias = (q * np.asarray(blk.mu, np.float32)).sum(1)  # (H,)
    grid = np.asarray(blk.grid, np.float32)
    kcodes = np.asarray(blk.kcodes)                     # (H, S, D) uint8
    vcodes = np.asarray(blk.vcodes)
    norm_v = np.asarray(blk.norm_v, np.float32)         # (H, S)
    cent = np.asarray(tq.centroids, np.float32)
    pi = np.asarray(tq._Pi, np.float32)
    row_ptr = np.asarray(blk.row_ptr, np.int64)
    cols = np.asarray(blk.cols, np.int64)
    deltas = np.asarray(blk.deltas, np.float32)
    nsplit = _nsplit(S)
    chunk = (S + nsplit - 1) // nsplit

    m_p = np.full((H, nsplit), -1e30, np.float32)
    l_p = np.zeros((H, nsplit), np.float32)
    acc_p = np.zeros((H, nsplit, D), np.float32)
    for h in range(H):
        for split in range(nsplit):
            s0 = split * chunk
            s1 = min(s0 + chunk, S)
            m, lsum = np.float32(-1e30), np.float32(0.0)
            acc = np.zeros(D, np.float32)
            for s in range(s0, s1):
                partial = float((w[h] * grid[kcodes[h, s]]).sum())
                e0, e1 = int(row_ptr[h * S + s]), int(row_ptr[h * S + s + 1])
                for e in range(e0, e1):
                    partial += float(q[h, cols[e]] * deltas[e])
                score = np.float32((partial + bias[h]) * scale)
                mn = max(m, score)
                corr = np.exp(m - mn, dtype=np.float32)
                p = np.exp(score - mn, dtype=np.float32)
                lsum = lsum * corr + p
                m = mn
                acc = acc * corr + (p * norm_v[h, s]) * cent[vcodes[h, s]]
            m_p[h, split], l_p[h, split], acc_p[h, split] = m, lsum, acc
    # flash-combine
    m = m_p.max(1, keepdims=True)
    wgt = np.exp(m_p - m)
    denom = (l_p * wgt).sum(1)
    acc = (acc_p * wgt[:, :, None]).sum(1)
    return m[:, 0], denom, acc @ pi


@pytest.mark.parametrize("D,outlier", [(64, 0.0), (64, 0.02), (96, 0.05), (128, 0.02)])
def test_pck_kernel_logic_numpy(D, outlier):
    """The port's split-K + CSR + online-softmax decomposition equals the
    all-at-once reference einsum -- validated on CPU, no GPU needed."""
    cache, blocks, q = _make_block(H=4, D=D, n_tok=50, outlier=outlier, seed=1)
    assert len(blocks) >= 1
    tq, scale = cache._tq, 1.0 / np.sqrt(D)
    for blk in blocks:
        m_ref, l_ref, acc_ref = blk.partials(q, tq, scale)   # numpy reference
        m_s, l_s, acc_s = _kernel_sim_pck(blk, q, tq, scale)
        assert np.allclose(m_ref, m_s, atol=1e-4), "m mismatch"
        assert np.allclose(l_ref, l_s, rtol=1e-4, atol=1e-4), "lsum mismatch"
        assert np.allclose(acc_ref, acc_s, rtol=1e-4, atol=1e-3), "acc mismatch"


@pytest.mark.skipif(not has_triton(), reason="needs Triton + CUDA/ROCm")
def test_triton_pck_matches_reference():
    """M4 Triton kernel == NumPy reference partials (per page)."""
    cache, blocks, q = _make_block(H=8, D=128, n_tok=200, outlier=0.02, seed=2)
    from turboquant_pro.kv_triton import pck_block_partials_triton

    tq, scale = cache._tq, 1.0 / np.sqrt(128)
    for blk in blocks:
        m_ref, l_ref, acc_ref = blk.partials(q, tq, scale)
        m_t, l_t, acc_t = pck_block_partials_triton(q, blk, tq, scale)
        to_np = lambda x: x.detach().cpu().numpy()  # noqa: E731
        assert np.allclose(m_ref, to_np(m_t), atol=1e-3)
        assert np.allclose(l_ref, to_np(l_t), rtol=1e-3, atol=1e-3)
        assert np.allclose(acc_ref, to_np(acc_t), rtol=1e-3, atol=1e-2)


@pytest.mark.skipif(not has_triton(), reason="needs Triton + CUDA/ROCm")
def test_triton_pck_batched_matches_full_decode():
    """Batched-page kernel over all cold pages == the full fused_decode output."""
    import torch

    cache, blocks, q = _make_block(H=8, D=128, n_tok=600, outlier=0.02, seed=3)
    from turboquant_pro.kv_triton import pck_batched_partials_triton

    tq, scale = cache._tq, 1.0 / np.sqrt(128)
    m_b, l_b, acc_b = pck_batched_partials_triton(q, blocks, tq, scale)
    out_b = (acc_b / torch.clamp(l_b, min=1e-30)[:, None]).detach().cpu().numpy()
    # reference: attention over reconstructed cold keys/values
    k = np.asarray(cache.get_keys(0, cache.cold_length))[0]
    v = np.asarray(cache.get_values(0, cache.cold_length))[0]
    sc = np.einsum("hd,hsd->hs", q, k) / np.sqrt(128)
    p = np.exp(sc - sc.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    want = np.einsum("hs,hsd->hd", p, v)
    assert np.allclose(out_b, want, rtol=1e-3, atol=1e-2)


@pytest.mark.skipif(not has_triton(), reason="needs Triton + CUDA/ROCm")
def test_triton_polar_matches_reference():
    """M2 PolarQuant Triton kernel == the CuPy/NumPy fused-decode reference."""
    import torch

    from turboquant_pro.core import TurboQuantKV
    from turboquant_pro.kv_fused import fused_decode_attention
    from turboquant_pro.kv_triton import polar_partials_triton

    H, S, D = 8, 300, 128
    rng = np.random.default_rng(4)
    tq = TurboQuantKV(head_dim=D, n_heads=H, bits=4, use_gpu=False, seed=0)
    v = rng.standard_normal((H, S, D)).astype(np.float32)
    nv = np.linalg.norm(v, axis=-1).astype(np.float32) + 1e-6
    unit = v / nv[..., None]
    rot = np.einsum("hsd,de->hse", unit, np.asarray(tq._Pi_T, np.float32))
    codes = np.searchsorted(tq.boundaries, rot).astype(np.uint8)
    q = rng.standard_normal((H, D)).astype(np.float32)
    want = fused_decode_attention(q, codes, codes, nv, nv, tq, xp=np)
    m, lsum, acc = polar_partials_triton(q, codes, codes, nv, nv, tq)
    got = (acc / torch.clamp(lsum, min=1e-30)[:, None]).detach().cpu().numpy()
    assert np.allclose(got, want, rtol=1e-3, atol=1e-2)

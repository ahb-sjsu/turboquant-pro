"""Fused KV-decode reference (M0): one attention step over compressed K/V codes.

Computes a decode-step attention output directly from tq-pro key/value *codes* with
no per-token reconstruction and a single rotation at each boundary:

    score_s = ||K||_s * (rotate(q) . cent[kcode_s]) / sqrt(d)   # ADC over K codes
    p       = softmax(score)
    acc     = sum_s p_s * ||V||_s * cent[vcode_s]               # V sum in code space
    out     = unrotate(acc)                                     # one inverse-rotation

This is the algorithm the fused CUDA kernel (M1+) implements; here it is expressed in
array ops that run on NumPy *or* CuPy (pass ``xp``), so it validates correctness on
GPU and serves as the reference. See ``docs/DESIGN_fused_kv_decode.md``.

Shapes (per decode step): ``q`` (heads, d); ``kcodes``/``vcodes`` (heads, S, d) uint8;
``norm_k``/``norm_v`` (heads, S). Returns ``out`` (heads, d).
"""

from __future__ import annotations

import numpy as np


def _rot_matrices(tq, xp):
    if getattr(tq, "_structured", False):
        raise NotImplementedError(
            "fused KV-decode reference currently supports full-QR rotations "
            "(head_dim <= 4096); structured rotations are future work"
        )
    pit = xp.asarray(tq._Pi_T, dtype=xp.float32)  # rotate:   x @ Pi_T
    pi = xp.asarray(tq._Pi, dtype=xp.float32)  # unrotate: y @ Pi
    cent = xp.asarray(tq.centroids, dtype=xp.float32)
    return pit, pi, cent


def _softmax(x, xp):
    x = x - xp.max(x, axis=-1, keepdims=True)
    e = xp.exp(x)
    return e / xp.sum(e, axis=-1, keepdims=True)


def fused_decode_attention(q, kcodes, vcodes, norm_k, norm_v, tq, xp=np):
    """One decode-step attention output computed in code space (the fused path)."""
    d = q.shape[-1]
    pit, pi, cent = _rot_matrices(tq, xp)
    q_rot = xp.asarray(q, dtype=xp.float32) @ pit
    scores = (
        xp.asarray(norm_k, dtype=xp.float32)
        * xp.einsum("hsd,hd->hs", cent[kcodes], q_rot)
    ) * (1.0 / np.sqrt(d))
    p = _softmax(scores, xp)
    acc = xp.einsum(
        "hs,hsd->hd", p * xp.asarray(norm_v, dtype=xp.float32), cent[vcodes]
    )
    return acc @ pi


def dequant_decode_attention(q, kcodes, vcodes, norm_k, norm_v, tq, xp=np):
    """Reference: reconstruct K/V to fp32, then standard attention (same math)."""
    d = q.shape[-1]
    _, pi, cent = _rot_matrices(tq, xp)
    k = xp.asarray(norm_k, dtype=xp.float32)[..., None] * (cent[kcodes] @ pi)
    v = xp.asarray(norm_v, dtype=xp.float32)[..., None] * (cent[vcodes] @ pi)
    scores = xp.einsum("hd,hsd->hs", xp.asarray(q, dtype=xp.float32), k) * (
        1.0 / np.sqrt(d)
    )
    p = _softmax(scores, xp)
    return xp.einsum("hs,hsd->hd", p, v)

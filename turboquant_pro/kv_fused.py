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
    x = x - xp.amax(x, axis=-1, keepdims=True)
    e = xp.exp(x)
    return e / xp.sum(e, axis=-1, keepdims=True)


def fused_decode_attention(q, kcodes, vcodes, norm_k, norm_v, tq, xp=np):
    """One decode-step attention output computed in code space (the fused path)."""
    kcodes, vcodes = xp.asarray(kcodes), xp.asarray(vcodes)
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
    kcodes, vcodes = xp.asarray(kcodes), xp.asarray(vcodes)
    d = q.shape[-1]
    _, pi, cent = _rot_matrices(tq, xp)
    k = xp.asarray(norm_k, dtype=xp.float32)[..., None] * (cent[kcodes] @ pi)
    v = xp.asarray(norm_v, dtype=xp.float32)[..., None] * (cent[vcodes] @ pi)
    scores = xp.einsum("hd,hsd->hs", xp.asarray(q, dtype=xp.float32), k) * (
        1.0 / np.sqrt(d)
    )
    p = _softmax(scores, xp)
    return xp.einsum("hs,hsd->hd", p, v)


def _cold_partials(q, kcodes, vcodes, norm_k, norm_v, tq, scale, xp):
    """Unnormalized (m, l, acc) for cold (coded) keys/values, in real space."""
    kcodes, vcodes = xp.asarray(kcodes), xp.asarray(vcodes)
    pit, pi, cent = _rot_matrices(tq, xp)
    q_rot = xp.asarray(q, dtype=xp.float32) @ pit
    sc = (
        xp.asarray(norm_k, dtype=xp.float32)
        * xp.einsum("hsd,hd->hs", cent[kcodes], q_rot)
    ) * scale
    m = xp.amax(sc, axis=1)
    e = xp.exp(sc - m[:, None])
    lsum = e.sum(axis=1)
    acc_code = xp.einsum(
        "hs,hsd->hd", e * xp.asarray(norm_v, dtype=xp.float32), cent[vcodes]
    )
    return m, lsum, acc_code @ pi  # unrotate to real space


def _hot_partials(q, hot_k, hot_v, scale, xp):
    """Unnormalized (m, l, acc) for the uncompressed hot window."""
    sc = (
        xp.einsum(
            "hd,hsd->hs",
            xp.asarray(q, dtype=xp.float32),
            xp.asarray(hot_k, dtype=xp.float32),
        )
        * scale
    )
    m = xp.amax(sc, axis=1)
    e = xp.exp(sc - m[:, None])
    return (
        m,
        e.sum(axis=1),
        xp.einsum("hs,hsd->hd", e, xp.asarray(hot_v, dtype=xp.float32)),
    )


def fused_decode(q, hot_k, hot_v, kcodes, vcodes, norm_k, norm_v, tq, xp=np):
    """Full two-tier decode step: fp16 hot window + coded cold pages, merged by
    online softmax. ``hot_k``/``hot_v`` (H, Sh, d) or None; cold codes (H, Sc, d) or
    None. Returns the attention output (H, d). The GPU kernel
    (:func:`turboquant_pro.kv_kernel.fused_decode_cuda`) accelerates the cold term;
    this reference defines the exact result.
    """
    d = q.shape[-1]
    scale = 1.0 / np.sqrt(d)
    parts = []
    if kcodes is not None and kcodes.shape[1] > 0:
        parts.append(_cold_partials(q, kcodes, vcodes, norm_k, norm_v, tq, scale, xp))
    if hot_k is not None and hot_k.shape[1] > 0:
        parts.append(_hot_partials(q, hot_k, hot_v, scale, xp))
    if not parts:
        raise ValueError("no keys: provide hot and/or cold")
    return merge_partials(parts, xp)


def merge_partials(parts, xp=np):
    """Merge unnormalized online-softmax states ``(m, l, acc)`` -- each (H,), (H,),
    (H, d) -- into the final attention output (H, d). Combines the hot window and the
    coded cold pages (the cold state may come from the GPU kernel via
    ``fused_decode_cuda(..., return_partials=True)``)."""
    big_m = parts[0][0]
    for m, _, _ in parts[1:]:
        big_m = xp.maximum(big_m, m)
    num = xp.zeros_like(parts[0][2])
    den = xp.zeros_like(parts[0][1])
    for m, lsum, acc in parts:
        w = xp.exp(m - big_m)
        num = num + acc * w[:, None]
        den = den + lsum * w
    return num / den[:, None]

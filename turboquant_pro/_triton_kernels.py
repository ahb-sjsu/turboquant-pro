# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Triton ``@triton.jit`` kernels for the P5 port (M2/M4 fused decode).

These MUST live at module scope: ``@triton.jit`` resolves names in the
kernel body and its annotations (``tl.constexpr``) against the function's
*module globals* (``fn.__globals__``), not any enclosing closure. Defining
them inside a builder function makes ``tl`` a local and the compiler raises
``NameError('tl is not defined')`` at the first annotation. So ``triton`` and
``tl`` are imported here at module top -- which means this module is imported
only from :func:`turboquant_pro.kv_triton._build_kernels`, gated behind
:func:`turboquant_pro.kv_triton.has_triton`, so CPU-only installs never load
it. See ``turboquant_pro/kv_triton.py`` for the host wrappers and contract.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def pck_partials_kernel(
    kcodes_ptr,
    vcodes_ptr,
    w_ptr,
    bias_ptr,
    grid_ptr,
    qk_ptr,
    row_ptr_ptr,
    cols_ptr,
    deltas_ptr,
    norm_v_ptr,
    cent_ptr,
    m_out_ptr,
    l_out_ptr,
    acc_out_ptr,
    H,
    S,
    D,
    scale,
    nsplit,
    MAX_NNZ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # one program per (head, key-split); mirror of fused_decode_pck
    pid = tl.program_id(0)
    h = pid // nsplit
    split = pid % nsplit
    if h >= H:
        return
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    w = tl.load(w_ptr + h * D + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + h)
    chunk = (S + nsplit - 1) // nsplit
    s0 = split * chunk
    s1 = tl.minimum(s0 + chunk, S)

    m = -1e30
    lsum = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    hS = h * S
    for s in range(s0, s1):
        base = (hS + s) * D + offs
        kc = tl.load(kcodes_ptr + base, mask=mask, other=0).to(tl.int32)
        gk = tl.load(grid_ptr + kc, mask=mask, other=0.0)
        partial = tl.sum(w * gk, axis=0)
        # sparse outlier correction: this token's CSR row (bounded MAX_NNZ)
        e0 = tl.load(row_ptr_ptr + hS + s)
        e1 = tl.load(row_ptr_ptr + hS + s + 1)
        n_e = e1 - e0
        for e in range(0, MAX_NNZ):
            valid = e < n_e
            col = tl.load(cols_ptr + e0 + e, mask=valid, other=0).to(tl.int32)
            dl = tl.load(deltas_ptr + e0 + e, mask=valid, other=0.0)
            qv = tl.load(qk_ptr + h * D + col, mask=valid, other=0.0)
            partial += tl.where(valid, qv * dl, 0.0)
        score = (partial + b) * scale
        mn = tl.maximum(m, score)
        corr = tl.exp(m - mn)
        p = tl.exp(score - mn)
        lsum = lsum * corr + p
        m = mn
        nv = tl.load(norm_v_ptr + hS + s)
        vc = tl.load(vcodes_ptr + base, mask=mask, other=0).to(tl.int32)
        cv = tl.load(cent_ptr + vc, mask=mask, other=0.0)
        acc = acc * corr + (p * nv) * cv

    po = h * nsplit + split
    tl.store(m_out_ptr + po, m)
    tl.store(l_out_ptr + po, lsum)
    tl.store(acc_out_ptr + po * D + offs, acc, mask=mask)


@triton.jit
def pck_batched_kernel(
    kcodes_ptr,
    vcodes_ptr,
    w_ptr,
    bias_ptr,
    grid_ptr,
    qk_ptr,
    row_ptr_ptr,
    cols_ptr,
    deltas_ptr,
    norm_v_ptr,
    cent_ptr,
    page_S_ptr,
    page_koff_ptr,
    page_toff_ptr,
    m_out_ptr,
    l_out_ptr,
    acc_out_ptr,
    P,
    H,
    D,
    scale,
    nsplit,
    MAX_NNZ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # one program per (page, head, key-split): every cold page in one launch
    pid = tl.program_id(0)
    hn = H * nsplit
    p = pid // hn
    rem = pid % hn
    h = rem // nsplit
    split = rem % nsplit
    if p >= P:
        return
    S = tl.load(page_S_ptr + p)
    koff = tl.load(page_koff_ptr + p)  # element offset into kcodes/vcodes cat
    toff = tl.load(page_toff_ptr + p)  # token offset (= sum_{q<p} H*S_q)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    # w/bias are per-(page,head): laid out (P*H, D) and (P*H,)
    ph = p * H + h
    w = tl.load(w_ptr + ph * D + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + ph)
    chunk = (S + nsplit - 1) // nsplit
    s0 = split * chunk
    s1 = tl.minimum(s0 + chunk, S)

    m = -1e30
    lsum = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    hS = h * S
    for s in range(s0, s1):
        base = koff + (hS + s) * D + offs
        kc = tl.load(kcodes_ptr + base, mask=mask, other=0).to(tl.int32)
        gk = tl.load(grid_ptr + kc, mask=mask, other=0.0)
        partial = tl.sum(w * gk, axis=0)
        tok = toff + hS + s  # global token index for the shared CSR
        e0 = tl.load(row_ptr_ptr + tok)
        e1 = tl.load(row_ptr_ptr + tok + 1)
        n_e = e1 - e0
        for e in range(0, MAX_NNZ):
            valid = e < n_e
            col = tl.load(cols_ptr + e0 + e, mask=valid, other=0).to(tl.int32)
            dl = tl.load(deltas_ptr + e0 + e, mask=valid, other=0.0)
            qv = tl.load(qk_ptr + ph * D + col, mask=valid, other=0.0)
            partial += tl.where(valid, qv * dl, 0.0)
        score = (partial + b) * scale
        mn = tl.maximum(m, score)
        corr = tl.exp(m - mn)
        pw = tl.exp(score - mn)
        lsum = lsum * corr + pw
        m = mn
        nv = tl.load(norm_v_ptr + toff + hS + s)
        vc = tl.load(vcodes_ptr + base, mask=mask, other=0).to(tl.int32)
        cv = tl.load(cent_ptr + vc, mask=mask, other=0.0)
        acc = acc * corr + (pw * nv) * cv

    po = (p * H + h) * nsplit + split
    tl.store(m_out_ptr + po, m)
    tl.store(l_out_ptr + po, lsum)
    tl.store(acc_out_ptr + po * D + offs, acc, mask=mask)


@triton.jit
def polar_split_kernel(
    kcodes_ptr,
    vcodes_ptr,
    norm_k_ptr,
    norm_v_ptr,
    qrot_ptr,
    cent_ptr,
    m_out_ptr,
    l_out_ptr,
    acc_out_ptr,
    H,
    S,
    D,
    ncent,
    scale,
    nsplit,
    BLOCK_D: tl.constexpr,
):
    # PolarQuant code-space split-K (port of fused_decode_split); keys and
    # values share the centroid grid, score = norm_k * <q_rot, cent[code]>.
    pid = tl.program_id(0)
    h = pid // nsplit
    split = pid % nsplit
    if h >= H:
        return
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    qr = tl.load(qrot_ptr + h * D + offs, mask=mask, other=0.0)
    chunk = (S + nsplit - 1) // nsplit
    s0 = split * chunk
    s1 = tl.minimum(s0 + chunk, S)

    m = -1e30
    lsum = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    hS = h * S
    for s in range(s0, s1):
        base = (hS + s) * D + offs
        kc = tl.load(kcodes_ptr + base, mask=mask, other=0).to(tl.int32)
        gk = tl.load(cent_ptr + kc, mask=mask, other=0.0)
        partial = tl.sum(qr * gk, axis=0)
        nk = tl.load(norm_k_ptr + hS + s)
        score = nk * partial * scale
        mn = tl.maximum(m, score)
        corr = tl.exp(m - mn)
        pw = tl.exp(score - mn)
        lsum = lsum * corr + pw
        m = mn
        nv = tl.load(norm_v_ptr + hS + s)
        vc = tl.load(vcodes_ptr + base, mask=mask, other=0).to(tl.int32)
        cv = tl.load(cent_ptr + vc, mask=mask, other=0.0)
        acc = acc * corr + (pw * nv) * cv

    po = h * nsplit + split
    tl.store(m_out_ptr + po, m)
    tl.store(l_out_ptr + po, lsum)
    tl.store(acc_out_ptr + po * D + offs, acc, mask=mask)

# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""P5: portable Triton port of the fused compute-on-codes kernels.

The M1--M4 fast paths ship as CuPy ``RawKernel`` (CUDA-only). This module
ports the two that matter -- **M2** (PolarQuant split-K value decode) and
**M4** (per-channel keys + CSR outliers, the recommended key format) -- to
Triton, keyed off the *same* prepared-page structs
(:class:`turboquant_pro.kv_fused_pck.PreparedPCKBlock`). Triton compiles one
source to NVIDIA and (where a target exists) ROCm; the CuPy RawKernel stays
the CUDA reference implementation and the exactness oracle. See
``docs/DESIGN_hardware_and_plugins.md`` sections 4.2 / P5.

Design decisions carried over verbatim from the RawKernel so the ports are
byte-comparable against the oracle:

  * split-K over the key axis for occupancy (``nsplit`` per head), each
    program emitting an **unnormalized** online-softmax partial ``(m, lsum,
    acc)`` that the host flash-combines -- identical to
    :func:`turboquant_pro.kv_kernel.pck_block_partials_cuda`;
  * the M4 dense score is branch-free (``sum_j w_j grid[code]``) and the
    sparse outlier correction is a short per-token CSR loop folded in *before*
    the softmax update;
  * values stay PolarQuant code-space; the single inverse-rotation ``@ Pi`` is
    applied on the host after the combine.

The two M4 "next" items from ``docs/DESIGN_fused_kv_decode.md`` section 8.5
land here rather than in the RawKernel (design doc 4.2 -- "so the effort
compounds"): a **batched per-page launch** (:func:`pck_batched_partials_triton`,
one kernel over every cold page) and a **grid tile over head_dim** (``BLOCK_D``)
that removes the ``d/32 <= 16`` register-tiling limit of the warp kernel.

Triton and torch are imported lazily; this module loads on CPU-only installs.
Calling a kernel requires a CUDA (or ROCm) GPU with Triton available.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "has_triton",
    "pck_block_partials_triton",
    "pck_batched_partials_triton",
    "fused_decode_pck_triton",
    "polar_partials_triton",
]

_TRITON = {"checked": False, "ok": False}


def has_triton() -> bool:
    """True when both torch (with a CUDA/ROCm device) and Triton are importable.

    Cached after the first probe. Never raises -- a missing Triton or a
    CPU-only torch just means the caller stays on the CuPy or NumPy path.
    """
    if not _TRITON["checked"]:
        _TRITON["checked"] = True
        try:
            import torch  # noqa: F401
            import triton  # noqa: F401

            _TRITON["ok"] = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001 - any import/runtime failure = no triton
            _TRITON["ok"] = False
    return _TRITON["ok"]


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _nsplit(S: int) -> int:
    """Split-K factor -- identical schedule to the RawKernel path."""
    return max(1, min(32, (S + 511) // 512))


# --------------------------------------------------------------------------
# The @triton.jit kernels live in _triton_kernels (module scope: the compiler
# resolves `tl` against the kernel's module globals, so it CANNOT be a closure
# local). That module imports triton at top, so it is imported lazily here --
# only after has_triton() -- and never on a CPU-only install. Cached in _K.
# --------------------------------------------------------------------------
_K: dict = {}


def _build_kernels():
    if _K:
        return _K
    import triton

    from . import _triton_kernels as tk

    _K["pck"] = tk.pck_partials_kernel
    _K["pck_batched"] = tk.pck_batched_kernel
    _K["polar"] = tk.polar_split_kernel
    _K["triton"] = triton
    return _K


# --------------------------------------------------------------------------
# Host wrappers. Each mirrors its CuPy counterpart's flash-combine and returns
# real-space (unrotated) partials for turboquant_pro.kv_fused.merge_partials.
# --------------------------------------------------------------------------
def _cuda(t, dtype=None):
    """Move an array/tensor to a contiguous CUDA torch tensor."""
    import torch

    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(np.asarray(t))
    if dtype is not None:
        t = t.to(dtype)
    if not t.is_cuda:
        t = t.cuda()
    return t.contiguous()


def _max_row_nnz(row_ptr) -> int:
    import torch

    rp = (
        row_ptr
        if isinstance(row_ptr, torch.Tensor)
        else torch.as_tensor(np.asarray(row_ptr))
    )
    if rp.numel() <= 1:
        return 0
    return int((rp[1:] - rp[:-1]).max().item())


def _flash_combine(m_p, l_p, acc_p, pi):
    """Combine split-K partials (H, nsplit),(H, nsplit),(H, nsplit, D) -> real
    space (m, denom, acc @ pi); identical math to the CuPy path."""
    import torch

    m = m_p.amax(dim=1, keepdim=True)
    w = torch.exp(m_p - m)
    denom = (l_p * w).sum(dim=1)
    acc = (acc_p * w[:, :, None]).sum(dim=1)
    return m[:, 0], denom, acc @ pi


def pck_block_partials_triton(q, blk, tq, scale=None):
    """Cold partials (m, lsum, acc) for one prepared per-channel page via the M4
    Triton kernel. ``blk`` a :class:`~turboquant_pro.kv_fused_pck.PreparedPCKBlock`
    (any backend -- arrays are moved to CUDA here). Real-space (unrotated)
    result for :func:`turboquant_pro.kv_fused.merge_partials`; exact vs the
    NumPy reference and the CuPy RawKernel oracle."""
    import torch

    if getattr(tq, "_structured", False):
        raise NotImplementedError("structured rotation not supported by the kernel")
    K = _build_kernels()
    H, d = int(blk.H), int(blk.D)
    S = int(blk.S)
    qk = _cuda(q, torch.float32).reshape(H, d)
    weight = _cuda(blk.weight, torch.float32)
    mu = _cuda(blk.mu, torch.float32)
    w = (qk * weight).contiguous()
    bias = (qk * mu).sum(dim=1).contiguous()
    grid = _cuda(blk.grid, torch.float32)
    cent = _cuda(tq.centroids, torch.float32)
    pi = _cuda(tq._Pi, torch.float32)
    kcodes = _cuda(blk.kcodes, torch.uint8)
    vcodes = _cuda(blk.vcodes, torch.uint8)
    norm_v = _cuda(blk.norm_v, torch.float32)
    row_ptr = _cuda(blk.row_ptr, torch.int32)
    cols = _cuda(blk.cols, torch.int32)  # uint16 -> int32 for gather
    deltas = _cuda(blk.deltas, torch.float32)
    if scale is None:
        scale = 1.0 / np.sqrt(d)

    nsplit = _nsplit(S)
    max_nnz = _max_row_nnz(row_ptr)
    m_p = torch.empty((H, nsplit), dtype=torch.float32, device="cuda")
    l_p = torch.empty((H, nsplit), dtype=torch.float32, device="cuda")
    acc_p = torch.empty((H, nsplit, d), dtype=torch.float32, device="cuda")

    K["pck"][(H * nsplit,)](
        kcodes,
        vcodes,
        w,
        bias,
        grid,
        qk,
        row_ptr,
        cols,
        deltas,
        norm_v,
        cent,
        m_p,
        l_p,
        acc_p,
        H,
        S,
        d,
        float(scale),
        nsplit,
        MAX_NNZ=max(max_nnz, 1),
        BLOCK_D=_next_pow2(d),
    )
    return _flash_combine(m_p, l_p, acc_p, pi)


def pck_batched_partials_triton(q, blocks, tq, scale=None):
    """All cold pages in a **single** launch (M4 §8.5 batched-page item).

    Concatenates the per-page code / CSR / norm arrays and launches one kernel
    over ``P*H*nsplit`` programs, then flash-combines every (page, split) into
    one partial per head. Grid parameters (mu, weight) are per page -- pages are
    independently quantized, and ``zero_point="bias"`` makes ``mu`` position- and
    length-dependent -- so ``w``/``bias`` are packed ``(P*H, D)``/``(P*H,)`` and
    indexed per page inside the kernel. Returns one real-space (m, lsum, acc)
    partial for the whole cold cache; feeds :func:`merge_partials` alongside the
    hot window exactly like the per-page path, but with a single kernel launch
    instead of ``P``."""
    import torch

    if getattr(tq, "_structured", False):
        raise NotImplementedError("structured rotation not supported by the kernel")
    if not blocks:
        raise ValueError("no cold pages")
    K = _build_kernels()
    H, d = int(blocks[0].H), int(blocks[0].D)
    P = len(blocks)
    qk = _cuda(q, torch.float32).reshape(H, d)
    cent = _cuda(tq.centroids, torch.float32)
    pi = _cuda(tq._Pi, torch.float32)
    grid = _cuda(blocks[0].grid, torch.float32)  # shared: NF4 / arange grid
    if scale is None:
        scale = 1.0 / np.sqrt(d)

    # per-page projections + concatenation
    page_S = [int(b.S) for b in blocks]
    w_cat = torch.empty((P * H, d), dtype=torch.float32, device="cuda")
    bias_cat = torch.empty((P * H,), dtype=torch.float32, device="cuda")
    kc_parts, vc_parts, nv_parts = [], [], []
    rp_parts, col_parts, dl_parts = [], [], []
    koff = [0] * P
    toff = [0] * P
    k_acc = 0
    t_acc = 0
    nnz_acc = 0
    max_nnz = 0
    for p, b in enumerate(blocks):
        weight = _cuda(b.weight, torch.float32)
        mu = _cuda(b.mu, torch.float32)
        w_cat[p * H : (p + 1) * H] = qk * weight
        bias_cat[p * H : (p + 1) * H] = (qk * mu).sum(dim=1)
        kc_parts.append(_cuda(b.kcodes, torch.uint8).reshape(-1))
        vc_parts.append(_cuda(b.vcodes, torch.uint8).reshape(-1))
        nv_parts.append(_cuda(b.norm_v, torch.float32).reshape(-1))
        rp = _cuda(b.row_ptr, torch.int32)  # (H*S+1,)
        max_nnz = max(max_nnz, _max_row_nnz(rp))
        rp_parts.append(rp[:-1] + nnz_acc)  # globalize into shared cols/deltas
        col_parts.append(_cuda(b.cols, torch.int32))
        dl_parts.append(_cuda(b.deltas, torch.float32))
        koff[p] = k_acc
        toff[p] = t_acc
        k_acc += H * page_S[p] * d
        t_acc += H * page_S[p]
        nnz_acc += (
            int(b.deltas.shape[0]) if hasattr(b.deltas, "shape") else len(b.deltas)
        )

    kcodes = torch.cat(kc_parts).contiguous()
    vcodes = torch.cat(vc_parts).contiguous()
    norm_v = torch.cat(nv_parts).contiguous()
    row_ptr = torch.cat(
        rp_parts + [torch.tensor([nnz_acc], dtype=torch.int32, device="cuda")]
    ).contiguous()
    cols = (
        torch.cat(col_parts).contiguous()
        if any(c.numel() for c in col_parts)
        else torch.zeros(0, dtype=torch.int32, device="cuda")
    )
    deltas = (
        torch.cat(dl_parts).contiguous()
        if any(dd.numel() for dd in dl_parts)
        else torch.zeros(0, dtype=torch.float32, device="cuda")
    )
    page_S_t = torch.tensor(page_S, dtype=torch.int32, device="cuda")
    page_koff_t = torch.tensor(koff, dtype=torch.int64, device="cuda")
    page_toff_t = torch.tensor(toff, dtype=torch.int32, device="cuda")
    # raw query tiled per page so qk_ptr[ph*D + col] gives q[h, col] for every page
    qk_cat = qk.repeat(P, 1).contiguous()  # (P*H, D)

    nsplit = _nsplit(max(page_S))
    m_p = torch.empty((P * H, nsplit), dtype=torch.float32, device="cuda")
    l_p = torch.empty((P * H, nsplit), dtype=torch.float32, device="cuda")
    acc_p = torch.empty((P * H, nsplit, d), dtype=torch.float32, device="cuda")

    K["pck_batched"][(P * H * nsplit,)](
        kcodes,
        vcodes,
        w_cat,
        bias_cat,
        grid,
        qk_cat,
        row_ptr,
        cols,
        deltas,
        norm_v,
        cent,
        page_S_t,
        page_koff_t,
        page_toff_t,
        m_p,
        l_p,
        acc_p,
        P,
        H,
        d,
        float(scale),
        nsplit,
        MAX_NNZ=max(max_nnz, 1),
        BLOCK_D=_next_pow2(d),
    )
    # combine across pages AND splits per head: reshape (P, H, nsplit)->(H, P*nsplit)
    m_p = m_p.reshape(P, H, nsplit).permute(1, 0, 2).reshape(H, P * nsplit)
    l_p = l_p.reshape(P, H, nsplit).permute(1, 0, 2).reshape(H, P * nsplit)
    acc_p = acc_p.reshape(P, H, nsplit, d).permute(1, 0, 2, 3).reshape(H, P * nsplit, d)
    return _flash_combine(m_p, l_p, acc_p, pi)


def fused_decode_pck_triton(
    q, key_quantizer, key_container, vcodes, norm_v, tq, return_partials=False
):
    """M4 fused decode via Triton (self-contained one-shot, mirrors
    :func:`turboquant_pro.kv_kernel.fused_decode_pck_cuda`).

    Builds a :class:`~turboquant_pro.kv_fused_pck.PreparedPCKBlock` and runs the
    Triton kernel. Exact vs the NumPy reference
    :func:`turboquant_pro.kv_fused_pck.fused_decode_pck` and the CuPy oracle."""
    import torch

    from .kv_fused_pck import PreparedPCKBlock

    blk = PreparedPCKBlock(key_quantizer, key_container, vcodes, norm_v, xp=np)
    m, lsum, acc = pck_block_partials_triton(q, blk, tq)
    if return_partials:
        return m, lsum, acc
    return acc / torch.clamp(lsum, min=1e-30)[:, None]


def polar_partials_triton(q, kcodes, vcodes, norm_k, norm_v, tq, scale=None):
    """Cold partials for PolarQuant code-space keys+values via the M2 Triton
    kernel (port of :func:`turboquant_pro.kv_kernel.fused_decode_cuda`
    ``method="warp"``). ``q`` (H, d); codes (H, S, d) uint8; norms (H, S).
    Returns real-space (m, lsum, acc)."""
    import torch

    if getattr(tq, "_structured", False):
        raise NotImplementedError("structured rotation not supported by the kernel")
    K = _build_kernels()
    q = _cuda(q, torch.float32)
    H, d = q.shape
    S = (
        int(np.asarray(kcodes).shape[1])
        if not isinstance(kcodes, torch.Tensor)
        else int(kcodes.shape[1])
    )
    cent = _cuda(tq.centroids, torch.float32)
    pit = _cuda(tq._Pi_T, torch.float32)
    pi = _cuda(tq._Pi, torch.float32)
    q_rot = (q @ pit).contiguous()
    kc = _cuda(kcodes, torch.uint8)
    vc = _cuda(vcodes, torch.uint8)
    nk = _cuda(norm_k, torch.float32)
    nv = _cuda(norm_v, torch.float32)
    ncent = int(cent.shape[0])
    if scale is None:
        scale = 1.0 / np.sqrt(d)

    nsplit = _nsplit(S)
    m_p = torch.empty((H, nsplit), dtype=torch.float32, device="cuda")
    l_p = torch.empty((H, nsplit), dtype=torch.float32, device="cuda")
    acc_p = torch.empty((H, nsplit, d), dtype=torch.float32, device="cuda")
    K["polar"][(H * nsplit,)](
        kc,
        vc,
        nk,
        nv,
        q_rot,
        cent,
        m_p,
        l_p,
        acc_p,
        H,
        S,
        d,
        ncent,
        float(scale),
        nsplit,
        BLOCK_D=_next_pow2(d),
    )
    return _flash_combine(m_p, l_p, acc_p, pi)

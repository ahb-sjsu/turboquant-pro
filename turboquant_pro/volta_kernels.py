# TurboQuant Pro: Volta-native fused KV-decode kernels.
# Copyright (c) 2026 Andrew H. Bond. MIT License.
"""Volta (``sm_70``) fused kernels for KV attention-on-codes decode (**K2**).

The recommended *keys* format is
:class:`~turboquant_pro.per_channel_kv.PerChannelKV` (per-channel asym-NF4). Keys
enter attention only through ``q . k``, so the whole score can be computed from
the codes without ever materializing an fp16/fp32 key tensor:

    score[h,s] = bias[h] + sum_d w[h,d] * grid[code[h,s,d]]
        w    = q * per-channel weight   (H, D)
        bias = sum_d q * per-channel mu  (H,)   -- zero-point term, folded
        grid = the NF4 / uniform codebook (L,)

:mod:`turboquant_pro.kv_fused_pck` expresses this with ``einsum(w, grid[codes])``,
which is correct but materializes ``grid[codes]`` -- an ``(H,S,D)`` fp32 tensor,
the same bandwidth as decompressing. This module instead reads the 1-byte codes
directly and does the LUT-dequant + dot in registers, so the only large tensor
touched is the codes themselves (2x smaller than fp16 K here; 4x once packed).

Why Volta specifically: the score is a per-channel-coded GEMV -- memory-bound at
decode -- so the win is *bandwidth*, not tensor cores. The GV100 has no
``cp.async``/``ldmatrix`` and none are needed. The kernel is one warp per
``(h, s)``: 32 lanes stride the head dimension (coalesced byte reads), the NF4
grid lives in shared memory, and a warp shuffle reduces the partial dot. Measured
on a Quadro GV100 vs decompress-then-attend: ~12--20x at ``S in {4k, 8k}``,
exact to fp32 rounding.

Compiled lazily via CuPy's NVRTC (arch auto = ``compute_70`` on a GV100), so the
import is free on CPU-only hosts; a NumPy reference is used when CuPy is absent.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp  # type: ignore[import-untyped]

    _HAS_CUPY = True
except ImportError:  # pragma: no cover - exercised on CPU-only hosts
    cp = None  # type: ignore[assignment]
    _HAS_CUPY = False

__all__ = ["k2_key_scores", "K2_MAX_LEVELS"]

# The shared-memory grid buffer is sized for this many codebook levels (covers
# 2/3/4-bit uniform and NF4). Raising it means widening ``sgrid`` in the source.
K2_MAX_LEVELS = 64

_KERNEL_SRC = r"""
extern "C" __global__
void k2_key_scores_u8(
    const unsigned char* __restrict__ codes,  // (H,S,D) uint8 indices in [0,L)
    const float* __restrict__ w,              // (H,D) = q * weight
    const float* __restrict__ bias,           // (H,)
    const float* __restrict__ grid,           // (L,) codebook
    float* __restrict__ scores,               // (H,S)
    int H, int S, int D, int L)
{
    __shared__ float sgrid[64];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    for (int i = tid; i < L; i += blockDim.x) sgrid[i] = grid[i];
    __syncthreads();

    long gwarp = (long)blockIdx.x * warps_per_block + (tid >> 5);
    long total = (long)H * S;
    if (gwarp >= total) return;
    int h = (int)(gwarp / S);
    int s = (int)(gwarp % S);
    const unsigned char* crow = codes + ((long)h * S + s) * D;
    const float* wrow = w + (long)h * D;
    float acc = 0.f;
    for (int d = lane; d < D; d += 32) acc += wrow[d] * sgrid[crow[d]];
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) scores[(long)h * S + s] = acc + bias[h];
}
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        if not _HAS_CUPY:  # pragma: no cover
            raise RuntimeError("k2_key_scores requires CuPy on a CUDA device")
        _kernel = cp.RawKernel(_KERNEL_SRC, "k2_key_scores_u8")
    return _kernel


def _reference(codes, w, bias, grid):
    """NumPy reference: bias + sum_d w * grid[codes]. Also the CPU fallback."""
    return bias[:, None] + np.einsum("hd,hsd->hs", w, grid[codes])


def k2_key_scores(codes, w, bias, grid, out=None):
    """Fused per-channel key scores ``(H, S)`` from unpacked uint8 codes.

    Args:
        codes: ``(H, S, D)`` uint8 index array, values in ``[0, len(grid))``.
        w:     ``(H, D)`` float32, the per-channel table weights ``q * weight``.
        bias:  ``(H,)`` float32, the folded zero-point term ``sum_d q * mu``.
        grid:  ``(L,)`` float32 codebook, ``L <= K2_MAX_LEVELS``.
        out:   optional ``(H, S)`` float32 output buffer (CuPy path only).

    Returns a ``(H, S)`` float32 array on the same backend as the inputs
    (CuPy when the inputs are CuPy arrays, else NumPy via the reference). The
    dense path only; outlier CSR deltas (:func:`kv_fused_pck.build_outlier_csr`)
    are added by the caller.
    """
    H, S, D = codes.shape
    L = int(grid.shape[0])
    if L > K2_MAX_LEVELS:
        raise ValueError(f"grid has {L} levels; kernel caps at {K2_MAX_LEVELS}")
    if w.shape != (H, D):
        raise ValueError(f"w must be (H,D)=({H},{D}), got {tuple(w.shape)}")
    if bias.shape != (H,):
        raise ValueError(f"bias must be (H,)=({H},), got {tuple(bias.shape)}")

    if not (_HAS_CUPY and isinstance(codes, cp.ndarray)):
        return _reference(
            np.asarray(codes), np.asarray(w), np.asarray(bias), np.asarray(grid)
        )

    codes = cp.ascontiguousarray(codes, dtype=cp.uint8)
    w = cp.ascontiguousarray(w, dtype=cp.float32)
    bias = cp.ascontiguousarray(bias, dtype=cp.float32)
    grid = cp.ascontiguousarray(grid, dtype=cp.float32)
    if out is None:
        out = cp.empty((H, S), dtype=cp.float32)
    tpb = 256
    warps_per_block = tpb // 32
    grid_x = (H * S + warps_per_block - 1) // warps_per_block
    _get_kernel()(
        (grid_x,),
        (tpb,),
        (codes, w, bias, grid, out, np.int32(H), np.int32(S), np.int32(D), np.int32(L)),
    )
    return out

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

__all__ = ["k2_key_scores", "k2_key_scores_packed", "K2_MAX_LEVELS"]

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

# Packed variant: codes are bit-packed exactly as ``per_channel_kv._pack_indices``
# (value i occupies stream bits [i*bits, i*bits+bits); each stream bit lives
# MSB-first within its byte -- the inverse of ``_unpack_indices``). Reads
# ``bits/8`` byte per value instead of 1, halving DRAM traffic at 4-bit.
_KERNEL_SRC_PACKED = r"""
extern "C" __global__
void k2_key_scores_packed(
    const unsigned char* __restrict__ packed,  // bit-packed codes (_pack_indices)
    const float* __restrict__ w,               // (H,D) = q * weight
    const float* __restrict__ bias,            // (H,)
    const float* __restrict__ grid,            // (L,) codebook
    float* __restrict__ scores,                // (H,S)
    int H, int S, int D, int L, int bits)
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
    long rowbase = ((long)h * S + s) * D;  // value index of channel d=0
    const float* wrow = w + (long)h * D;
    float acc = 0.f;
    for (int d = lane; d < D; d += 32) {
        long g0 = (rowbase + d) * bits;    // first stream bit of this value
        int code = 0;
        for (int k = 0; k < bits; k++) {
            long g = g0 + k;
            int byte = packed[g >> 3];
            int bit = (byte >> (7 - (int)(g & 7))) & 1;
            code |= bit << k;
        }
        acc += wrow[d] * sgrid[code];
    }
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) scores[(long)h * S + s] = acc + bias[h];
}
"""

# Fast bits=4 path: a 256-entry format-decode LUT (byte -> its two 4-bit codes)
# plus contiguous channel-pair assignment, so each byte is read once (coalesced)
# and decoded with a table lookup instead of per-bit reconstruction. Requires
# ``D % 64 == 0`` (two codes per byte, D/32 channels per lane). At parity with the
# unpacked kernel while touching half the bytes -- K2 is latency-bound, so packing
# is a storage win (half the KV cache) at ~equal decode speed, not a speedup.
_KERNEL_SRC_PACKED4 = r"""
extern "C" __global__
void k2_key_scores_packed4(
    const unsigned char* __restrict__ packed,  // 4-bit packed codes
    const short* __restrict__ codeLUT,         // byte -> code_even | code_odd<<8
    const float* __restrict__ w,               // (H,D)
    const float* __restrict__ bias,            // (H,)
    const float* __restrict__ grid,            // (L,)
    float* __restrict__ scores,                // (H,S)
    int H, int S, int D, int L)
{
    __shared__ float sgrid[64];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    for (int i = tid; i < L; i += blockDim.x) sgrid[i] = grid[i];
    __syncthreads();

    long gwarp = (long)blockIdx.x * warps_per_block + (tid >> 5);
    if (gwarp >= (long)H * S) return;
    int h = (int)(gwarp / S);
    int s = (int)(gwarp % S);
    long rb = ((long)h * S + s) * D;
    const float* wrow = w + (long)h * D;
    int cpl = D >> 5;             // channels per lane
    int d0 = lane * cpl;
    long b0 = (rb + d0) >> 1;     // byte holding the lane's first channel pair
    float acc = 0.f;
    for (int j = 0; j < cpl; j += 2) {
        int cc = codeLUT[packed[b0 + (j >> 1)]];
        acc += wrow[d0 + j] * sgrid[cc & 0xff]
             + wrow[d0 + j + 1] * sgrid[(cc >> 8) & 0xff];
    }
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) scores[(long)h * S + s] = acc + bias[h];
}
"""

_kernels: dict = {}
_code_lut = None  # lazily built cupy int16 array, format-decode only (grid-free)


def _get_code_lut():
    """byte -> (code_even | code_odd<<8) for the 4-bit ``_pack_indices`` layout."""
    global _code_lut
    if _code_lut is None:
        lut = np.zeros(256, dtype=np.int16)
        for b in range(256):
            ce = sum(((b >> (7 - k)) & 1) << k for k in range(4))
            co = sum(((b >> (3 - k)) & 1) << k for k in range(4))
            lut[b] = ce | (co << 8)
        _code_lut = cp.asarray(lut)
    return _code_lut


def _get_kernel(name, src):
    if name not in _kernels:
        if not _HAS_CUPY:  # pragma: no cover
            raise RuntimeError("k2 kernels require CuPy on a CUDA device")
        _kernels[name] = cp.RawKernel(src, name)
    return _kernels[name]


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
    _get_kernel("k2_key_scores_u8", _KERNEL_SRC)(
        (grid_x,),
        (tpb,),
        (codes, w, bias, grid, out, np.int32(H), np.int32(S), np.int32(D), np.int32(L)),
    )
    return out


def _unpack_ref(packed, H, S, D, bits):
    """NumPy unpack matching ``per_channel_kv._unpack_indices`` (CPU fallback)."""
    n = H * S * D
    bitstream = np.unpackbits(np.asarray(packed, dtype=np.uint8))[: n * bits]
    bitstream = bitstream.reshape(n, bits)
    weights = (1 << np.arange(bits, dtype=np.uint16)).astype(np.uint16)
    idx = (bitstream.astype(np.uint16) * weights).sum(axis=1).astype(np.uint8)
    return idx.reshape(H, S, D)


def k2_key_scores_packed(packed, w, bias, grid, H, S, D, bits, out=None):
    """Fused per-channel key scores ``(H, S)`` from **bit-packed** codes.

    ``packed`` is the flat byte array produced by
    ``per_channel_kv._pack_indices`` (i.e. ``CompressedPerChannelKV.indices``
    when ``packed=True``) -- half the KV-cache bytes of the unpacked path at
    4-bit. Decode is at parity with :func:`k2_key_scores` (K2 is latency-bound,
    so fewer bytes does not mean faster), so packing buys storage, not speed.
    ``bits=4`` uses the fast LUT kernel (requires ``D % 64 == 0``); ``bits`` in
    ``{2, 3}`` (or odd ``D``) use the general per-bit kernel. Arguments and the
    return value match :func:`k2_key_scores`.
    """
    L = int(grid.shape[0])
    if L > K2_MAX_LEVELS:
        raise ValueError(f"grid has {L} levels; kernel caps at {K2_MAX_LEVELS}")
    if bits not in (2, 3, 4):
        raise ValueError(f"bits must be 2, 3, or 4, got {bits}")
    if w.shape != (H, D):
        raise ValueError(f"w must be (H,D)=({H},{D}), got {tuple(w.shape)}")
    if bias.shape != (H,):
        raise ValueError(f"bias must be (H,)=({H},), got {tuple(bias.shape)}")

    if not (_HAS_CUPY and isinstance(packed, cp.ndarray)):
        codes = _unpack_ref(packed, H, S, D, bits)
        return _reference(codes, np.asarray(w), np.asarray(bias), np.asarray(grid))

    packed = cp.ascontiguousarray(packed, dtype=cp.uint8)
    w = cp.ascontiguousarray(w, dtype=cp.float32)
    bias = cp.ascontiguousarray(bias, dtype=cp.float32)
    grid = cp.ascontiguousarray(grid, dtype=cp.float32)
    if out is None:
        out = cp.empty((H, S), dtype=cp.float32)
    tpb = 256
    warps_per_block = tpb // 32
    grid_x = (H * S + warps_per_block - 1) // warps_per_block
    if bits == 4 and D % 64 == 0:
        _get_kernel("k2_key_scores_packed4", _KERNEL_SRC_PACKED4)(
            (grid_x,),
            (tpb,),
            (
                packed,
                _get_code_lut(),
                w,
                bias,
                grid,
                out,
                np.int32(H),
                np.int32(S),
                np.int32(D),
                np.int32(L),
            ),
        )
    else:
        _get_kernel("k2_key_scores_packed", _KERNEL_SRC_PACKED)(
            (grid_x,),
            (tpb,),
            (
                packed,
                w,
                bias,
                grid,
                out,
                np.int32(H),
                np.int32(S),
                np.int32(D),
                np.int32(L),
                np.int32(bits),
            ),
        )
    return out

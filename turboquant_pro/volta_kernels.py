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
``cp.async``/``ldmatrix`` and none are needed. The tuned kernel
(``_KERNEL_SRC_VEC4NS``, default when ``D % 4 == 0``) is one warp per group of
four ``s`` rows: each lane reads a ``uint32`` (4 codes) for one coalesced 128B
transaction and keeps four independent rows in flight for latency-hiding
memory-level parallelism, the NF4 grid lives in shared memory, and a warp shuffle
reduces each row's dot. That reaches ~55--63% of HBM2 peak (1.6--1.9x over the
one-warp-per-``(h,s)`` scalar kernel, kept as the odd-``D`` fallback). Measured on
a Quadro GV100 vs decompress-then-attend: ~15--30x at ``S in {4k..16k}``, exact
to fp32 rounding.

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

__all__ = [
    "k2_key_scores",
    "k2_key_scores_packed",
    "value_accum",
    "apply_outlier_csr",
    "K2_MAX_LEVELS",
]

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

# Tuned unpacked kernel (default when D % 4 == 0). Two changes over the scalar
# kernel take it from ~34% to ~60% of HBM2 peak on a GV100 (1.6--1.9x): (1) each
# lane reads a ``uint32`` (4 codes) -> one coalesced 128B transaction per warp
# instead of four 32B ones; (2) each warp handles NS=4 independent ``s`` rows, so
# 4 loads are in flight before any reduction -- the memory-level parallelism that
# hides latency (the actual bottleneck; vectorization alone barely helped). NS=8
# gave nothing over 4. Bounds-checked, so S need not be a multiple of NS.
_KERNEL_SRC_VEC4NS = r"""
extern "C" __global__
void k2_key_scores_vec4ns(
    const unsigned int* __restrict__ codes32,  // (H,S,D/4) uint8 codes as uint32
    const float* __restrict__ w,               // (H,D) = q * weight
    const float* __restrict__ bias,            // (H,)
    const float* __restrict__ grid,            // (L,) codebook
    float* __restrict__ scores,                // (H,S)
    int H, int S, int D, int L)
{
    const int NS = 4;
    __shared__ float sgrid[64];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    for (int i = tid; i < L; i += blockDim.x) sgrid[i] = grid[i];
    __syncthreads();

    int SN = (S + NS - 1) / NS;                 // s-groups per head
    long g = (long)blockIdx.x * warps_per_block + (tid >> 5);
    if (g >= (long)H * SN) return;
    int h = (int)(g / SN);
    int s0 = (int)(g % SN) * NS;
    int nv = S - s0;
    if (nv > NS) nv = NS;                        // valid rows in this group
    int D4 = D >> 2;
    const float* wrow = w + (long)h * D;
    float acc[NS];
    #pragma unroll
    for (int i = 0; i < NS; i++) acc[i] = 0.f;
    for (int t = lane; t < D4; t += 32) {
        int d = t << 2;
        float w0 = wrow[d], w1 = wrow[d + 1], w2 = wrow[d + 2], w3 = wrow[d + 3];
        #pragma unroll
        for (int i = 0; i < NS; i++) {
            if (i < nv) {
                unsigned int wd = codes32[((long)h * S + s0 + i) * D4 + t];
                acc[i] += w0 * sgrid[wd & 0xff] + w1 * sgrid[(wd >> 8) & 0xff]
                        + w2 * sgrid[(wd >> 16) & 0xff] + w3 * sgrid[(wd >> 24) & 0xff];
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < NS; i++) {
        if (i < nv) {
            float a = acc[i];
            for (int off = 16; off > 0; off >>= 1)
                a += __shfl_down_sync(0xffffffffu, a, off);
            if (lane == 0) scores[(long)h * S + s0 + i] = a + bias[h];
        }
    }
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

# Fast bits=4 path (default when D % 4 == 0). The same vec+ns4 recipe as the
# unpacked kernel, adapted to packed bytes: each lane reads a ``uint16`` (2 bytes
# = 4 codes) per row for one coalesced load, a 256-entry format-decode LUT turns
# each byte into its two codes, and NS=4 ``s`` rows stay in flight (MLP). This is
# ~1.5x faster than the naive one-warp-per-(h,s) LUT kernel and reaches near
# parity (~0.85-0.98x) with the unpacked kernel. Note packing is still a *storage*
# win (half the KV cache) at ~parity decode, not a speedup: a D=128 packed row is
# only 64B, half a 128B transaction, so it cannot become bandwidth-bound the way
# unpacked (uint32, full 128B line) does. Requires ``D % 4 == 0`` (uint16 view,
# 4 channels per word); other cases use the general per-bit kernel.
_KERNEL_SRC_PACKED4 = r"""
extern "C" __global__
void k2_key_scores_packed4(
    const unsigned short* __restrict__ p16,    // 4-bit packed codes as uint16
    const short* __restrict__ codeLUT,         // byte -> code_even | code_odd<<8
    const float* __restrict__ w,               // (H,D)
    const float* __restrict__ bias,            // (H,)
    const float* __restrict__ grid,            // (L,)
    float* __restrict__ scores,                // (H,S)
    int H, int S, int D, int L)
{
    const int NS = 4;
    __shared__ float sgrid[64];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warps_per_block = blockDim.x >> 5;
    for (int i = tid; i < L; i += blockDim.x) sgrid[i] = grid[i];
    __syncthreads();

    int SN = (S + NS - 1) / NS;
    long g = (long)blockIdx.x * warps_per_block + (tid >> 5);
    if (g >= (long)H * SN) return;
    int h = (int)(g / SN);
    int s0 = (int)(g % SN) * NS;
    int nv = S - s0;
    if (nv > NS) nv = NS;
    int U = D >> 2;                 // uint16 words per row (4 codes each)
    const float* wrow = w + (long)h * D;
    float acc[NS];
    #pragma unroll
    for (int i = 0; i < NS; i++) acc[i] = 0.f;
    for (int u = lane; u < U; u += 32) {
        int d0 = u << 2;
        float w0 = wrow[d0], w1 = wrow[d0 + 1], w2 = wrow[d0 + 2], w3 = wrow[d0 + 3];
        #pragma unroll
        for (int i = 0; i < NS; i++) {
            if (i < nv) {
                unsigned short v = p16[((long)h * S + s0 + i) * U + u];
                int lo = codeLUT[v & 0xff], hi = codeLUT[(v >> 8) & 0xff];
                acc[i] += w0 * sgrid[lo & 0xff] + w1 * sgrid[(lo >> 8) & 0xff]
                        + w2 * sgrid[hi & 0xff] + w3 * sgrid[(hi >> 8) & 0xff];
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < NS; i++) {
        if (i < nv) {
            float a = acc[i];
            for (int off = 16; off > 0; off >>= 1)
                a += __shfl_down_sync(0xffffffffu, a, off);
            if (lane == 0) scores[(long)h * S + s0 + i] = a + bias[h];
        }
    }
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
    args = (np.int32(H), np.int32(S), np.int32(D), np.int32(L))
    if D % 4 == 0:
        # tuned path: uint32 loads + NS=4 s-rows/warp (see _KERNEL_SRC_VEC4NS)
        codes32 = codes.view(cp.uint32)  # (H, S, D/4)
        sn = (S + 3) // 4
        grid_x = (H * sn + warps_per_block - 1) // warps_per_block
        _get_kernel("k2_key_scores_vec4ns", _KERNEL_SRC_VEC4NS)(
            (grid_x,), (tpb,), (codes32, w, bias, grid, out, *args)
        )
    else:
        grid_x = (H * S + warps_per_block - 1) // warps_per_block
        _get_kernel("k2_key_scores_u8", _KERNEL_SRC)(
            (grid_x,), (tpb,), (codes, w, bias, grid, out, *args)
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
    if bits == 4 and D % 4 == 0:
        # tuned path: uint16 loads + LUT + NS=4 s-rows/warp (_KERNEL_SRC_PACKED4)
        p16 = packed.view(cp.uint16)
        sn = (S + 3) // 4
        grid_x = (H * sn + warps_per_block - 1) // warps_per_block
        _get_kernel("k2_key_scores_packed4", _KERNEL_SRC_PACKED4)(
            (grid_x,),
            (tpb,),
            (
                p16,
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
        grid_x = (H * S + warps_per_block - 1) // warps_per_block
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


# ---------------------------------------------------------------------------- #
# Values leg (P4): the transpose of the key score. PolarQuant values enter the  #
# decode as  acc_code[h,d] = sum_s wv[h,s] * cent[vcode[h,s,d]]  (wv = e*norm_v, #
# cent a scalar codebook). The cupy path materializes cent[vcodes] (H,S,d) fp32; #
# this kernel reads vcodes (uint8) directly. Block per (h, s-chunk); thread per  #
# d so vcodes reads coalesce; S split across blocks with atomicAdd for occupancy #
# (SCH=128 measured best on a GV100). ~13-17x over the cupy einsum.              #
# ---------------------------------------------------------------------------- #
_KERNEL_SRC_VALUE = r"""
extern "C" __global__
void value_accum(
    const unsigned char* __restrict__ vcodes,  // (H,S,D)
    const float* __restrict__ wv,              // (H,S) = e * norm_v
    const float* __restrict__ cent,            // (Lv,) scalar codebook
    float* __restrict__ acc,                   // (H,D), pre-zeroed
    int H, int S, int D, int Lv, int SCH)
{
    __shared__ float scent[256];
    for (int i = threadIdx.x; i < Lv; i += blockDim.x) scent[i] = cent[i];
    __syncthreads();
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (d >= D) return;
    int s0 = blockIdx.y * SCH;
    int s1 = s0 + SCH;
    if (s1 > S) s1 = S;
    const unsigned char* base = vcodes + (long)h * S * D + d;
    const float* wrow = wv + (long)h * S;
    float a = 0.f;
    for (int s = s0; s < s1; s++) a += wrow[s] * scent[base[(long)s * D]];
    atomicAdd(&acc[(long)h * D + d], a);
}
"""

# Key outlier deltas (P4): dense-and-sparse keeps the top magnitude entries in
# fp16; build_outlier_csr turns them into token-major score deltas. This applies
# them on-GPU: one thread per (h,s) row adds sum_k q[h, cols[k]] * deltas[k].
_KERNEL_SRC_OUTLIER = r"""
extern "C" __global__
void apply_outlier_csr(
    const int* __restrict__ row_ptr,           // (H*S + 1,)
    const unsigned short* __restrict__ cols,    // (nnz,) channel index
    const float* __restrict__ deltas,           // (nnz,)
    const float* __restrict__ q,                // (H,D)
    float* __restrict__ scores,                 // (H,S) in/out
    int H, int S, int D)
{
    long row = (long)blockIdx.x * blockDim.x + threadIdx.x;  // = h*S + s
    if (row >= (long)H * S) return;
    int a = row_ptr[row];
    int b = row_ptr[row + 1];
    if (a == b) return;
    const float* qh = q + (row / S) * D;
    float corr = 0.f;
    for (int k = a; k < b; k++) corr += qh[cols[k]] * deltas[k];
    scores[row] += corr;
}
"""


def value_accum(vcodes, wv, cent, out=None, s_chunk=128):
    """Fused PolarQuant value accumulation ``acc_code[h,d] = sum_s wv * cent[vcode]``.

    Args:
        vcodes: ``(H, S, D)`` uint8 value codes.
        wv:     ``(H, S)`` float32 per-token weights (``e * norm_v``).
        cent:   ``(Lv,)`` float32 scalar codebook (``Lv <= 256``).
        out:    optional ``(H, D)`` float32 accumulator (CuPy path only).
        s_chunk: rows of ``S`` per block (occupancy knob; 128 best on GV100).

    Returns ``(H, D)`` float32 (rotate to real space with ``@ pi`` afterward).
    On CuPy inputs runs the kernel; otherwise the NumPy reference.
    """
    H, S, D = vcodes.shape
    Lv = int(cent.shape[0])
    if Lv > 256:
        raise ValueError(f"cent has {Lv} levels; value kernel caps at 256")
    if wv.shape != (H, S):
        raise ValueError(f"wv must be (H,S)=({H},{S}), got {tuple(wv.shape)}")

    if not (_HAS_CUPY and isinstance(vcodes, cp.ndarray)):
        return np.einsum(
            "hs,hsd->hd", np.asarray(wv), np.asarray(cent)[np.asarray(vcodes)]
        ).astype(np.float32)

    vcodes = cp.ascontiguousarray(vcodes, dtype=cp.uint8)
    wv = cp.ascontiguousarray(wv, dtype=cp.float32)
    cent = cp.ascontiguousarray(cent, dtype=cp.float32)
    if out is None:
        out = cp.zeros((H, D), dtype=cp.float32)
    else:
        out.fill(0)
    n_chunks = (S + s_chunk - 1) // s_chunk
    _get_kernel("value_accum", _KERNEL_SRC_VALUE)(
        (H, n_chunks),
        (D,),
        (
            vcodes,
            wv,
            cent,
            out,
            np.int32(H),
            np.int32(S),
            np.int32(D),
            np.int32(Lv),
            np.int32(s_chunk),
        ),
    )
    return out


def apply_outlier_csr(scores, row_ptr, cols, deltas, q):
    """Add key dense-and-sparse outlier score deltas in place: ``scores[h,s] +=
    sum_k q[h, cols[k]] * deltas[k]`` over CSR row ``h*S + s``.

    ``row_ptr`` ``(H*S+1,)`` int32, ``cols`` ``(nnz,)`` uint16, ``deltas``
    ``(nnz,)`` float32 -- the output of ``kv_fused_pck.build_outlier_csr``.
    ``scores`` ``(H, S)`` and ``q`` ``(H, D)``. Returns ``scores``.
    """
    H, S = scores.shape
    D = q.shape[1]
    if not (_HAS_CUPY and isinstance(scores, cp.ndarray)):
        rp = np.asarray(row_ptr)
        rows = np.repeat(np.arange(H * S), np.diff(rp))
        heads = rows // S
        contrib = np.asarray(q, dtype=np.float64)[
            heads, np.asarray(cols, dtype=np.int64)
        ] * np.asarray(deltas, dtype=np.float64)
        corr = np.zeros(H * S, dtype=np.float64)
        np.add.at(corr, rows, contrib)
        return scores + corr.reshape(H, S).astype(scores.dtype)

    scores = cp.ascontiguousarray(scores, dtype=cp.float32)
    row_ptr = cp.ascontiguousarray(row_ptr, dtype=cp.int32)
    cols = cp.ascontiguousarray(cols, dtype=cp.uint16)
    deltas = cp.ascontiguousarray(deltas, dtype=cp.float32)
    q = cp.ascontiguousarray(q, dtype=cp.float32)
    tpb = 256
    grid_x = (H * S + tpb - 1) // tpb
    _get_kernel("apply_outlier_csr", _KERNEL_SRC_OUTLIER)(
        (grid_x,),
        (tpb,),
        (row_ptr, cols, deltas, q, scores, np.int32(H), np.int32(S), np.int32(D)),
    )
    return scores

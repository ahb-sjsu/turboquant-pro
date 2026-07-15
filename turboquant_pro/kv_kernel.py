"""M1: single-head fused KV-decode CUDA kernel (CuPy RawKernel).

One thread block per head computes the code-space decode output without
reconstructing K or V: build a per-query LUT, stream keys with **online (flash)
softmax** over ADC scores, and accumulate the V weighted-sum in code space; the
caller applies the single inverse-rotation. Validated against the NumPy/CuPy
reference in :mod:`turboquant_pro.kv_fused`. See ``docs/DESIGN_fused_kv_decode.md``.

CuPy is imported lazily, so this module loads on CPU-only installs; calling the
kernel requires a CUDA GPU. Codes are unpacked (one ``uint8`` per dimension) for M1;
bit-packing is an M2/M3 optimization.
"""

from __future__ import annotations

import numpy as np

# One block per head; blockDim.x == d (head_dim, a power of two). Online softmax +
# code-space V accumulation, fp32 accumulators. Uses precise expf for correctness.
_SRC = r"""
extern "C" __global__ void fused_decode(
    const unsigned char* __restrict__ kcodes,   // (H, S, d)
    const unsigned char* __restrict__ vcodes,   // (H, S, d)
    const float* __restrict__ norm_k,           // (H, S)
    const float* __restrict__ norm_v,           // (H, S)
    const float* __restrict__ q_rot,            // (H, d)  rotated query
    const float* __restrict__ cent,             // (ncent,)
    float* __restrict__ out,                     // (H, d) code-space acc, pre-unrotate
    const int S, const int d, const int ncent, const float scale)
{
    const int h = blockIdx.x;
    const int tid = threadIdx.x;                 // 0..d-1
    extern __shared__ float sh[];
    float* lut = sh;                             // d*ncent
    float* red = lut + d * ncent;                // d (score reduction)
    __shared__ float m_s, l_s, p_s, corr_s;

    const float qj = q_rot[h * d + tid];
    for (int s = 0; s < ncent; ++s) lut[tid * ncent + s] = qj * cent[s];
    if (tid == 0) { m_s = -1e30f; l_s = 0.f; }
    float acc = 0.f;
    __syncthreads();

    const unsigned char* kc = kcodes + (size_t)h * S * d;
    const unsigned char* vc = vcodes + (size_t)h * S * d;
    const float* nk = norm_k + (size_t)h * S;
    const float* nv = norm_v + (size_t)h * S;

    for (int s = 0; s < S; ++s) {
        red[tid] = lut[tid * ncent + kc[(size_t)s * d + tid]];
        __syncthreads();
        for (int off = d >> 1; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        if (tid == 0) {
            float score = nk[s] * red[0] * scale;
            float mn = fmaxf(m_s, score);
            corr_s = expf(m_s - mn);
            p_s = expf(score - mn);
            l_s = l_s * corr_s + p_s;
            m_s = mn;
        }
        __syncthreads();
        acc = acc * corr_s + p_s * nv[s] * cent[vc[(size_t)s * d + tid]];
        __syncthreads();
    }
    out[h * d + tid] = acc / l_s;
}
"""

# M2 fast path: split-K flash-decode. One warp per (head, key-split); each lane owns
# d/32 dims (stride 32) and holds its q_rot in registers; the per-key score is an
# all-reduce via __shfl_xor (warp-synchronous -- no __syncthreads, no shared LUT).
# Each warp emits an UNnormalized partial (running max m, denom l, code-space acc) for
# its key range; a cheap host-side flash-combine merges splits per head. Splitting the
# keys gives H*nsplit blocks (occupancy) and parallelizes the otherwise-serial S loop.
# Requires d % 32 == 0 and d/32 <= 16.
_SRC_SPLIT = r"""
extern "C" __global__ void fused_decode_split(
    const unsigned char* __restrict__ kcodes,
    const unsigned char* __restrict__ vcodes,
    const float* __restrict__ norm_k,
    const float* __restrict__ norm_v,
    const float* __restrict__ q_rot,
    const float* __restrict__ cent,
    float* __restrict__ m_out,      // (H, nsplit)
    float* __restrict__ l_out,      // (H, nsplit)
    float* __restrict__ acc_out,    // (H, nsplit, d)  unnormalized
    const int H, const int S, const int d, const int ncent,
    const float scale, const int nsplit)
{
    const int warps = blockDim.x >> 5;
    const int gw = blockIdx.x * warps + (threadIdx.x >> 5);  // global warp id
    const int h = gw / nsplit, split = gw % nsplit;
    const int lane = threadIdx.x & 31;
    if (h >= H) return;
    const int chunk = (S + nsplit - 1) / nsplit;
    const int s0 = split * chunk;
    const int s1 = min(s0 + chunk, S);
    const int ndl = d >> 5;
    float qr[16], acc[16];
    #pragma unroll
    for (int i = 0; i < ndl; ++i) {
        qr[i] = q_rot[h * d + lane + (i << 5)]; acc[i] = 0.f;
    }
    const unsigned char* kc = kcodes + (size_t)h * S * d;
    const unsigned char* vc = vcodes + (size_t)h * S * d;
    const float* nk = norm_k + (size_t)h * S;
    const float* nv = norm_v + (size_t)h * S;
    float m = -1e30f, l = 0.f;
    for (int s = s0; s < s1; ++s) {
        const size_t base = (size_t)s * d + lane;
        float partial = 0.f;
        #pragma unroll
        for (int i = 0; i < ndl; ++i) partial += qr[i] * cent[kc[base + (i << 5)]];
        for (int o = 16; o > 0; o >>= 1)
            partial += __shfl_xor_sync(0xffffffffu, partial, o);
        const float score = nk[s] * partial * scale;
        const float mn = fmaxf(m, score);
        const float corr = expf(m - mn), p = expf(score - mn);
        l = l * corr + p; m = mn;
        const float pv = p * nv[s];
        #pragma unroll
        for (int i = 0; i < ndl; ++i)
            acc[i] = acc[i] * corr + pv * cent[vc[base + (i << 5)]];
    }
    const int po = h * nsplit + split;
    if (lane == 0) { m_out[po] = m; l_out[po] = l; }
    #pragma unroll
    for (int i = 0; i < ndl; ++i) acc_out[(size_t)po * d + lane + (i << 5)] = acc[i];
}
"""

_KERNEL = {}


def _kernel(cp, name, src):
    k = _KERNEL.get(name)
    if k is None:
        k = _KERNEL[name] = cp.RawKernel(src, name)
    return k


def fused_decode_cuda(
    q, kcodes, vcodes, norm_k, norm_v, tq, method="warp", return_partials=False
):
    """Fused KV-decode on GPU. ``q`` (H, d); codes (H, S, d) uint8; norms (H, S).

    Returns ``out`` (H, d). ``method="warp"`` (default) is the M2 fast path (one warp
    per head, ``__shfl`` score reduction, ``d % 32 == 0``); ``method="block"`` is the
    M1 reference (one block/head, block reduction, ``d`` a power of two). The rotation
    must be full-QR (``tq._Pi`` present); structured rotations are future work.

    With ``return_partials=True`` (warp only) returns unnormalized online-softmax state
    ``(m, l, acc)`` in real space -- (H,), (H,), (H, d) -- for merging with a hot window
    via :func:`turboquant_pro.kv_fused.merge_partials`.
    """
    import cupy as cp

    if getattr(tq, "_structured", False):
        raise NotImplementedError("structured rotation not supported by the kernel")
    H, d = q.shape
    S = int(kcodes.shape[1])
    cent = cp.ascontiguousarray(cp.asarray(tq.centroids, dtype=cp.float32))
    pit = cp.asarray(tq._Pi_T, dtype=cp.float32)
    pi = cp.asarray(tq._Pi, dtype=cp.float32)
    q_rot = cp.ascontiguousarray(cp.asarray(q, dtype=cp.float32) @ pit)
    kc = cp.ascontiguousarray(cp.asarray(kcodes, dtype=cp.uint8))
    vc = cp.ascontiguousarray(cp.asarray(vcodes, dtype=cp.uint8))
    nk = cp.ascontiguousarray(cp.asarray(norm_k, dtype=cp.float32))
    nv = cp.ascontiguousarray(cp.asarray(norm_v, dtype=cp.float32))
    ncent = int(cent.shape[0])
    scale = np.float32(1.0 / np.sqrt(d))

    if method == "warp":
        if d % 32 or d // 32 > 16:
            raise ValueError(f"warp kernel needs d % 32 == 0 and d/32 <= 16, got {d}")
        nsplit = max(1, min(32, (S + 511) // 512))  # split-K for occupancy
        m_p = cp.empty((H, nsplit), dtype=cp.float32)
        l_p = cp.empty((H, nsplit), dtype=cp.float32)
        acc_p = cp.empty((H, nsplit, d), dtype=cp.float32)
        warps = 4
        total = H * nsplit
        grid = ((total + warps - 1) // warps,)
        _kernel(cp, "fused_decode_split", _SRC_SPLIT)(
            grid,
            (32 * warps,),
            (
                kc,
                vc,
                nk,
                nv,
                q_rot,
                cent,
                m_p,
                l_p,
                acc_p,
                np.int32(H),
                np.int32(S),
                np.int32(d),
                np.int32(ncent),
                scale,
                np.int32(nsplit),
            ),
        )
        # flash-combine the unnormalized partials across splits (cheap, on GPU)
        m = m_p.max(axis=1, keepdims=True)
        w = cp.exp(m_p - m)
        denom = (l_p * w).sum(axis=1, keepdims=True)
        acc = (acc_p * w[:, :, None]).sum(axis=1)
        if return_partials:
            return m[:, 0], denom[:, 0], acc @ pi  # unnormalized real-space state
        return (acc / cp.maximum(denom, 1e-30)) @ pi
    elif method == "block":
        if return_partials:
            raise ValueError("return_partials is only supported for method='warp'")
        if d & (d - 1):
            raise ValueError(f"block kernel needs head_dim a power of two, got {d}")
        out = cp.empty((H, d), dtype=cp.float32)
        _kernel(cp, "fused_decode", _SRC)(
            (H,),
            (d,),
            (
                kc,
                vc,
                nk,
                nv,
                q_rot,
                cent,
                out,
                np.int32(S),
                np.int32(d),
                np.int32(ncent),
                scale,
            ),
            shared_mem=(d * ncent + d) * 4,
        )
    else:
        raise ValueError(f"unknown method {method!r}")
    return out @ pi  # single inverse-rotation


# M4: per-channel keys (uniform / NF4 family, any zero-point mode) with
# dense-sparse fp16 outliers, fused. Same split-K skeleton as _SRC_SPLIT; only the
# score computation differs:
#   score_s = ( bias[h]                      -- q.mu, all zero-point modes fold here
#             + sum_j w[h,j]*grid[code]      -- dense, branch-free, no shared LUT
#             + sum_{e in csr row (h,s)} qk[h,col_e]*delta_e ) * scale
# The sparse pass is warp-cooperative: lanes stride the row's ~3 entries and one
# __shfl reduction folds the correction in BEFORE the online-softmax update, so the
# dense loop never branches and divergence is bounded to the ragged tail of a short
# list (docs/DESIGN_fused_kv_decode.md section 8.2). Keys are in the original
# post-RoPE basis (no rotation); values stay PolarQuant code-space (unrotate on host).
_SRC_PCK = r"""
extern "C" __global__ void fused_decode_pck(
    const unsigned char*  __restrict__ kcodes,   // (H, S, d)
    const unsigned char*  __restrict__ vcodes,   // (H, S, d)
    const float*          __restrict__ w,        // (H, d)  q * per-channel weight
    const float*          __restrict__ bias,     // (H,)    q . mu
    const float*          __restrict__ grid,     // (nlev,) key grid (NF4 or arange)
    const float*          __restrict__ qk,       // (H, d)  raw query (outlier corr)
    const int*            __restrict__ row_ptr,  // (H*S+1,) outlier CSR
    const unsigned short* __restrict__ cols,     // (nnz,)
    const float*          __restrict__ deltas,   // (nnz,)
    const float*          __restrict__ norm_v,   // (H, S)
    const float*          __restrict__ cent,     // (ncent,) value centroids
    float* __restrict__ m_out, float* __restrict__ l_out,
    float* __restrict__ acc_out,                  // (H, nsplit, d) unnormalized
    const int H, const int S, const int d, const float scale, const int nsplit)
{
    const int warps = blockDim.x >> 5;
    const int gw = blockIdx.x * warps + (threadIdx.x >> 5);
    const int h = gw / nsplit, split = gw % nsplit;
    const int lane = threadIdx.x & 31;
    if (h >= H) return;
    const int chunk = (S + nsplit - 1) / nsplit;
    const int s0 = split * chunk;
    const int s1 = min(s0 + chunk, S);
    const int ndl = d >> 5;
    float wr[16], acc[16];
    #pragma unroll
    for (int i = 0; i < ndl; ++i) {
        wr[i] = w[h * d + lane + (i << 5)]; acc[i] = 0.f;
    }
    const float b = bias[h];
    const unsigned char* kc = kcodes + (size_t)h * S * d;
    const unsigned char* vc = vcodes + (size_t)h * S * d;
    const float* nv = norm_v + (size_t)h * S;
    float m = -1e30f, l = 0.f;
    for (int s = s0; s < s1; ++s) {
        const size_t base = (size_t)s * d + lane;
        float partial = 0.f;
        #pragma unroll
        for (int i = 0; i < ndl; ++i) partial += wr[i] * grid[kc[base + (i << 5)]];
        // sparse outlier correction: lanes stride this token's CSR row
        const int e0 = row_ptr[h * S + s], e1 = row_ptr[h * S + s + 1];
        for (int e = e0 + lane; e < e1; e += 32)
            partial += qk[h * d + (int)cols[e]] * deltas[e];
        for (int o = 16; o > 0; o >>= 1)
            partial += __shfl_xor_sync(0xffffffffu, partial, o);
        const float score = (partial + b) * scale;
        const float mn = fmaxf(m, score);
        const float corr = expf(m - mn), p = expf(score - mn);
        l = l * corr + p; m = mn;
        const float pv = p * nv[s];
        #pragma unroll
        for (int i = 0; i < ndl; ++i)
            acc[i] = acc[i] * corr + pv * cent[vc[base + (i << 5)]];
    }
    const int po = h * nsplit + split;
    if (lane == 0) { m_out[po] = m; l_out[po] = l; }
    #pragma unroll
    for (int i = 0; i < ndl; ++i) acc_out[(size_t)po * d + lane + (i << 5)] = acc[i];
}
"""


def fused_decode_pck_cuda(
    q, key_quantizer, key_container, vcodes, norm_v, tq, return_partials=False
):
    """M4 fused decode on GPU: per-channel keys (+outliers), PolarQuant values.

    ``q`` (H, d) raw post-RoPE query (per-channel keys are unrotated);
    ``key_container`` a ``CompressedPerChannelKV`` (B=1); values as in
    :func:`fused_decode_cuda`. Exact vs the reference
    :func:`turboquant_pro.kv_fused_pck.fused_decode_pck`.
    """
    import cupy as cp

    from .kv_fused_pck import _codes, _grid_params, build_outlier_csr

    if getattr(tq, "_structured", False):
        raise NotImplementedError("structured rotation not supported by the kernel")
    H, d = q.shape
    if d % 32 or d // 32 > 16:
        raise ValueError(f"kernel needs d % 32 == 0 and d/32 <= 16, got {d}")
    S = int(key_container.shape[2])
    mu, weight, grid_tab = _grid_params(key_quantizer, key_container)
    qf = np.asarray(q, dtype=np.float32)
    csr = build_outlier_csr(key_quantizer, key_container)
    if csr is None:
        row_ptr = np.zeros(H * S + 1, dtype=np.int32)
        cols = np.zeros(0, dtype=np.uint16)
        deltas = np.zeros(0, dtype=np.float32)
    else:
        row_ptr, cols, deltas = csr

    kc = cp.ascontiguousarray(cp.asarray(_codes(key_container)[0], dtype=cp.uint8))
    vc = cp.ascontiguousarray(cp.asarray(vcodes, dtype=cp.uint8))
    w_d = cp.ascontiguousarray(cp.asarray(qf * weight, dtype=cp.float32))
    bias_d = cp.ascontiguousarray(cp.asarray((qf * mu).sum(axis=1), dtype=cp.float32))
    grid_d = cp.ascontiguousarray(cp.asarray(grid_tab, dtype=cp.float32))
    qk_d = cp.ascontiguousarray(cp.asarray(qf, dtype=cp.float32))
    rp_d = cp.ascontiguousarray(cp.asarray(row_ptr, dtype=cp.int32))
    cols_d = cp.ascontiguousarray(cp.asarray(cols, dtype=cp.uint16))
    del_d = cp.ascontiguousarray(cp.asarray(deltas, dtype=cp.float32))
    nv = cp.ascontiguousarray(cp.asarray(norm_v, dtype=cp.float32))
    cent = cp.ascontiguousarray(cp.asarray(tq.centroids, dtype=cp.float32))
    pi = cp.asarray(tq._Pi, dtype=cp.float32)
    scale = np.float32(1.0 / np.sqrt(d))

    nsplit = max(1, min(32, (S + 511) // 512))
    m_p = cp.empty((H, nsplit), dtype=cp.float32)
    l_p = cp.empty((H, nsplit), dtype=cp.float32)
    acc_p = cp.empty((H, nsplit, d), dtype=cp.float32)
    warps = 4
    total = H * nsplit
    _kernel(cp, "fused_decode_pck", _SRC_PCK)(
        ((total + warps - 1) // warps,),
        (32 * warps,),
        (
            kc,
            vc,
            w_d,
            bias_d,
            grid_d,
            qk_d,
            rp_d,
            cols_d,
            del_d,
            nv,
            cent,
            m_p,
            l_p,
            acc_p,
            np.int32(H),
            np.int32(S),
            np.int32(d),
            scale,
            np.int32(nsplit),
        ),
    )
    m = m_p.max(axis=1, keepdims=True)
    wgt = cp.exp(m_p - m)
    denom = (l_p * wgt).sum(axis=1, keepdims=True)
    acc = (acc_p * wgt[:, :, None]).sum(axis=1)
    if return_partials:
        return m[:, 0], denom[:, 0], acc @ pi
    return (acc / cp.maximum(denom, 1e-30)) @ pi

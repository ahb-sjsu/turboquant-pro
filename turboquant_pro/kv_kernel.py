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

_KERNEL = None


def _kernel(cp):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = cp.RawKernel(_SRC, "fused_decode")
    return _KERNEL


def fused_decode_cuda(q, kcodes, vcodes, norm_k, norm_v, tq):
    """Fused KV-decode on GPU. ``q`` (H, d); codes (H, S, d) uint8; norms (H, S).

    Returns ``out`` (H, d). ``head_dim`` must be a power of two ``<= 1024`` and the
    rotation must be full-QR (``tq._Pi`` present); structured rotations are future work.
    """
    import cupy as cp

    if getattr(tq, "_structured", False):
        raise NotImplementedError("structured rotation not supported in the M1 kernel")
    H, d = q.shape
    S = int(kcodes.shape[1])
    if d & (d - 1):
        raise ValueError(f"head_dim must be a power of two for the M1 kernel, got {d}")
    cent = cp.ascontiguousarray(cp.asarray(tq.centroids, dtype=cp.float32))
    pit = cp.asarray(tq._Pi_T, dtype=cp.float32)
    pi = cp.asarray(tq._Pi, dtype=cp.float32)
    q_rot = cp.ascontiguousarray(cp.asarray(q, dtype=cp.float32) @ pit)
    kc = cp.ascontiguousarray(cp.asarray(kcodes, dtype=cp.uint8))
    vc = cp.ascontiguousarray(cp.asarray(vcodes, dtype=cp.uint8))
    nk = cp.ascontiguousarray(cp.asarray(norm_k, dtype=cp.float32))
    nv = cp.ascontiguousarray(cp.asarray(norm_v, dtype=cp.float32))
    out = cp.empty((H, d), dtype=cp.float32)
    ncent = int(cent.shape[0])
    shmem = (d * ncent + d) * 4
    _kernel(cp)(
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
            np.float32(1.0 / np.sqrt(d)),
        ),
        shared_mem=shmem,
    )
    return out @ pi  # single inverse-rotation

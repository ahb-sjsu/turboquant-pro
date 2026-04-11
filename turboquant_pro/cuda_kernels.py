# TurboQuant Pro: Open-source PolarQuant+QJL for LLM KV cache compression
# Copyright (c) 2025 Andrew H. Bond
# MIT License
#
# Algorithm: Zandieh et al. "Sub-linear Memory Inference via PolarQuant
# and QJL" (ICLR 2026). Implementation adapted from Theory Radar.

"""
CuPy CUDA RawKernels for GPU-accelerated bit-packing.

All kernels are Volta-compatible (compute capability 7.0+).
Kernels are lazily compiled on first use to avoid startup overhead
when running CPU-only.
"""

from __future__ import annotations

try:
    import cupy as cp  # type: ignore[import-untyped]

    _HAS_CUPY = True
except ImportError:
    cp = None  # type: ignore[assignment]
    _HAS_CUPY = False


# ------------------------------------------------------------------ #
# CUDA kernel sources                                                 #
# ------------------------------------------------------------------ #

PACK_KERNEL_3BIT_SRC = r"""
extern "C" __global__
void pack_3bit(const unsigned char* indices, unsigned char* packed,
               int n) {
    // Each thread packs 8 x 3-bit values into 3 bytes (24 bits).
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx * 8 >= n) return;

    int src = idx * 8;
    int dst = idx * 3;

    unsigned int v0 = (src + 0 < n) ? indices[src + 0] : 0;
    unsigned int v1 = (src + 1 < n) ? indices[src + 1] : 0;
    unsigned int v2 = (src + 2 < n) ? indices[src + 2] : 0;
    unsigned int v3 = (src + 3 < n) ? indices[src + 3] : 0;
    unsigned int v4 = (src + 4 < n) ? indices[src + 4] : 0;
    unsigned int v5 = (src + 5 < n) ? indices[src + 5] : 0;
    unsigned int v6 = (src + 6 < n) ? indices[src + 6] : 0;
    unsigned int v7 = (src + 7 < n) ? indices[src + 7] : 0;

    unsigned int bits24 = (v0)
                        | (v1 << 3)
                        | (v2 << 6)
                        | (v3 << 9)
                        | (v4 << 12)
                        | (v5 << 15)
                        | (v6 << 18)
                        | (v7 << 21);

    packed[dst + 0] = (unsigned char)(bits24 & 0xFF);
    packed[dst + 1] = (unsigned char)((bits24 >> 8) & 0xFF);
    packed[dst + 2] = (unsigned char)((bits24 >> 16) & 0xFF);
}
"""

UNPACK_KERNEL_3BIT_SRC = r"""
extern "C" __global__
void unpack_3bit(const unsigned char* packed, unsigned char* indices,
                 int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx * 8 >= n) return;

    int src = idx * 3;
    int dst = idx * 8;

    unsigned int b0 = packed[src + 0];
    unsigned int b1 = packed[src + 1];
    unsigned int b2 = packed[src + 2];
    unsigned int bits24 = b0 | (b1 << 8) | (b2 << 16);

    if (dst + 0 < n) indices[dst + 0] = (unsigned char)( bits24        & 0x7);
    if (dst + 1 < n) indices[dst + 1] = (unsigned char)((bits24 >>  3) & 0x7);
    if (dst + 2 < n) indices[dst + 2] = (unsigned char)((bits24 >>  6) & 0x7);
    if (dst + 3 < n) indices[dst + 3] = (unsigned char)((bits24 >>  9) & 0x7);
    if (dst + 4 < n) indices[dst + 4] = (unsigned char)((bits24 >> 12) & 0x7);
    if (dst + 5 < n) indices[dst + 5] = (unsigned char)((bits24 >> 15) & 0x7);
    if (dst + 6 < n) indices[dst + 6] = (unsigned char)((bits24 >> 18) & 0x7);
    if (dst + 7 < n) indices[dst + 7] = (unsigned char)((bits24 >> 21) & 0x7);
}
"""

PACK_KERNEL_2BIT_SRC = r"""
extern "C" __global__
void pack_2bit(const unsigned char* indices, unsigned char* packed,
               int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx * 4 >= n) return;

    int src = idx * 4;
    unsigned int v0 = (src + 0 < n) ? indices[src + 0] : 0;
    unsigned int v1 = (src + 1 < n) ? indices[src + 1] : 0;
    unsigned int v2 = (src + 2 < n) ? indices[src + 2] : 0;
    unsigned int v3 = (src + 3 < n) ? indices[src + 3] : 0;

    packed[idx] = (unsigned char)(v0 | (v1 << 2) | (v2 << 4) | (v3 << 6));
}
"""

UNPACK_KERNEL_2BIT_SRC = r"""
extern "C" __global__
void unpack_2bit(const unsigned char* packed, unsigned char* indices,
                 int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx * 4 >= n) return;

    int dst = idx * 4;
    unsigned int byte_val = packed[idx];

    if (dst + 0 < n) indices[dst + 0] = (unsigned char)( byte_val       & 0x3);
    if (dst + 1 < n) indices[dst + 1] = (unsigned char)((byte_val >> 2) & 0x3);
    if (dst + 2 < n) indices[dst + 2] = (unsigned char)((byte_val >> 4) & 0x3);
    if (dst + 3 < n) indices[dst + 3] = (unsigned char)((byte_val >> 6) & 0x3);
}
"""

PACK_KERNEL_4BIT_SRC = r"""
extern "C" __global__
void pack_4bit(const unsigned char* indices, unsigned char* packed,
               int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx * 2 >= n) return;

    int src = idx * 2;
    unsigned int v0 = (src + 0 < n) ? indices[src + 0] : 0;
    unsigned int v1 = (src + 1 < n) ? indices[src + 1] : 0;

    packed[idx] = (unsigned char)(v0 | (v1 << 4));
}
"""

UNPACK_KERNEL_4BIT_SRC = r"""
extern "C" __global__
void unpack_4bit(const unsigned char* packed, unsigned char* indices,
                 int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx * 2 >= n) return;

    int dst = idx * 2;
    unsigned int byte_val = packed[idx];

    if (dst + 0 < n) indices[dst + 0] = (unsigned char)( byte_val       & 0xF);
    if (dst + 1 < n) indices[dst + 1] = (unsigned char)((byte_val >> 4) & 0xF);
}
"""


# ------------------------------------------------------------------ #
# Batch quantization kernels                                          #
# Unrolled binary search on precomputed boundaries — avoids generic   #
# searchsorted overhead.  One kernel per bit-width so the boundary    #
# count is compile-time constant.                                     #
# ------------------------------------------------------------------ #

QUANTIZE_2BIT_SRC = r"""
extern "C" __global__
void quantize_2bit(const float* __restrict__ x,
                   unsigned char* __restrict__ idx,
                   const float* __restrict__ bounds,
                   int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;
    float val = x[i];
    /* 3 boundaries -> 4 bins, 2 comparisons (balanced binary tree) */
    unsigned char r;
    if (val >= bounds[1]) {
        r = (val >= bounds[2]) ? 3 : 2;
    } else {
        r = (val >= bounds[0]) ? 1 : 0;
    }
    idx[i] = r;
}
"""

QUANTIZE_3BIT_SRC = r"""
extern "C" __global__
void quantize_3bit(const float* __restrict__ x,
                   unsigned char* __restrict__ idx,
                   const float* __restrict__ bounds,
                   int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;
    float val = x[i];
    /* 7 boundaries -> 8 bins, 3 comparisons */
    unsigned char r;
    if (val >= bounds[3]) {
        if (val >= bounds[5]) {
            r = (val >= bounds[6]) ? 7 : 6;
        } else {
            r = (val >= bounds[4]) ? 5 : 4;
        }
    } else {
        if (val >= bounds[1]) {
            r = (val >= bounds[2]) ? 3 : 2;
        } else {
            r = (val >= bounds[0]) ? 1 : 0;
        }
    }
    idx[i] = r;
}
"""

QUANTIZE_4BIT_SRC = r"""
extern "C" __global__
void quantize_4bit_q(const float* __restrict__ x,
                     unsigned char* __restrict__ idx,
                     const float* __restrict__ bounds,
                     int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;
    float val = x[i];
    /* 15 boundaries -> 16 bins, 4 comparisons */
    unsigned char r;
    if (val >= bounds[7]) {
        if (val >= bounds[11]) {
            if (val >= bounds[13]) {
                r = (val >= bounds[14]) ? 15 : 14;
            } else {
                r = (val >= bounds[12]) ? 13 : 12;
            }
        } else {
            if (val >= bounds[9]) {
                r = (val >= bounds[10]) ? 11 : 10;
            } else {
                r = (val >= bounds[8]) ? 9 : 8;
            }
        }
    } else {
        if (val >= bounds[3]) {
            if (val >= bounds[5]) {
                r = (val >= bounds[6]) ? 7 : 6;
            } else {
                r = (val >= bounds[4]) ? 5 : 4;
            }
        } else {
            if (val >= bounds[1]) {
                r = (val >= bounds[2]) ? 3 : 2;
            } else {
                r = (val >= bounds[0]) ? 1 : 0;
            }
        }
    }
    idx[i] = r;
}
"""


# ------------------------------------------------------------------ #
# Fused rotation + quantization kernel                                #
# Tiled GEMM: (N, dim) x (dim, dim) -> quantize inline -> uint8      #
# Saves one full (N, dim) float32 global memory round-trip.           #
# Only for QR rotation (dim <= 4096).                                 #
# ------------------------------------------------------------------ #

FUSED_ROTATE_QUANTIZE_3BIT_SRC = r"""
extern "C" __global__
void fused_rotate_quantize_3bit(
    const float* __restrict__ x,       /* (N, dim) normalised input   */
    const float* __restrict__ Pi_T,    /* (dim, dim) rotation matrix  */
    const float* __restrict__ bounds,  /* (7,) quantization bounds    */
    unsigned char* __restrict__ idx,   /* (N, dim) output indices     */
    int N, int dim
) {
    /* Grid: (ceil(dim/TILE), N) — one block computes TILE output
       elements for one row of x. */
    const int TILE = 32;

    int col_base = blockIdx.x * TILE;  /* starting output column */
    int row      = blockIdx.y;         /* which input row        */
    int tx       = threadIdx.x;        /* column within tile     */

    if (row >= N) return;
    int out_col = col_base + tx;

    /* Accumulate dot product: x[row, :] . Pi_T[:, out_col] */
    float acc = 0.0f;

    /* Shared memory tile for a column-slice of Pi_T */
    __shared__ float s_Pi[32];

    for (int t = 0; t < dim; t += TILE) {
        /* Load one tile of Pi_T column into shared memory */
        int pi_row = t + tx;
        s_Pi[tx] = (pi_row < dim && out_col < dim) ? Pi_T[pi_row * dim + out_col] : 0.0f;
        __syncthreads();

        /* Dot product over this tile */
        for (int k = 0; k < TILE && (t + k) < dim; ++k) {
            acc += x[row * dim + t + k] * s_Pi[k];
        }
        __syncthreads();
    }

    if (out_col >= dim) return;

    /* Inline 3-bit quantization (7 boundaries, 3 comparisons) */
    unsigned char r;
    if (acc >= bounds[3]) {
        if (acc >= bounds[5]) {
            r = (acc >= bounds[6]) ? 7 : 6;
        } else {
            r = (acc >= bounds[4]) ? 5 : 4;
        }
    } else {
        if (acc >= bounds[1]) {
            r = (acc >= bounds[2]) ? 3 : 2;
        } else {
            r = (acc >= bounds[0]) ? 1 : 0;
        }
    }
    idx[row * dim + out_col] = r;
}
"""

FUSED_ROTATE_QUANTIZE_2BIT_SRC = r"""
extern "C" __global__
void fused_rotate_quantize_2bit(
    const float* __restrict__ x,
    const float* __restrict__ Pi_T,
    const float* __restrict__ bounds,
    unsigned char* __restrict__ idx,
    int N, int dim
) {
    const int TILE = 32;
    int col_base = blockIdx.x * TILE;
    int row      = blockIdx.y;
    int tx       = threadIdx.x;
    if (row >= N) return;
    int out_col = col_base + tx;

    float acc = 0.0f;
    __shared__ float s_Pi[32];

    for (int t = 0; t < dim; t += TILE) {
        int pi_row = t + tx;
        s_Pi[tx] = (pi_row < dim && out_col < dim) ? Pi_T[pi_row * dim + out_col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE && (t + k) < dim; ++k) {
            acc += x[row * dim + t + k] * s_Pi[k];
        }
        __syncthreads();
    }

    if (out_col >= dim) return;
    unsigned char r;
    if (acc >= bounds[1]) {
        r = (acc >= bounds[2]) ? 3 : 2;
    } else {
        r = (acc >= bounds[0]) ? 1 : 0;
    }
    idx[row * dim + out_col] = r;
}
"""

FUSED_ROTATE_QUANTIZE_4BIT_SRC = r"""
extern "C" __global__
void fused_rotate_quantize_4bit(
    const float* __restrict__ x,
    const float* __restrict__ Pi_T,
    const float* __restrict__ bounds,
    unsigned char* __restrict__ idx,
    int N, int dim
) {
    const int TILE = 32;
    int col_base = blockIdx.x * TILE;
    int row      = blockIdx.y;
    int tx       = threadIdx.x;
    if (row >= N) return;
    int out_col = col_base + tx;

    float acc = 0.0f;
    __shared__ float s_Pi[32];

    for (int t = 0; t < dim; t += TILE) {
        int pi_row = t + tx;
        s_Pi[tx] = (pi_row < dim && out_col < dim) ? Pi_T[pi_row * dim + out_col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE && (t + k) < dim; ++k) {
            acc += x[row * dim + t + k] * s_Pi[k];
        }
        __syncthreads();
    }

    if (out_col >= dim) return;
    unsigned char r;
    if (acc >= bounds[7]) {
        if (acc >= bounds[11]) {
            if (acc >= bounds[13]) {
                r = (acc >= bounds[14]) ? 15 : 14;
            } else {
                r = (acc >= bounds[12]) ? 13 : 12;
            }
        } else {
            if (acc >= bounds[9]) {
                r = (acc >= bounds[10]) ? 11 : 10;
            } else {
                r = (acc >= bounds[8]) ? 9 : 8;
            }
        }
    } else {
        if (acc >= bounds[3]) {
            if (acc >= bounds[5]) {
                r = (acc >= bounds[6]) ? 7 : 6;
            } else {
                r = (acc >= bounds[4]) ? 5 : 4;
            }
        } else {
            if (acc >= bounds[1]) {
                r = (acc >= bounds[2]) ? 3 : 2;
            } else {
                r = (acc >= bounds[0]) ? 1 : 0;
            }
        }
    }
    idx[row * dim + out_col] = r;
}
"""


# ------------------------------------------------------------------ #
# Lazy kernel compilation                                             #
# ------------------------------------------------------------------ #

_gpu_kernels: dict[str, object] = {}

_KERNEL_SOURCES = {
    "pack_3bit": (PACK_KERNEL_3BIT_SRC, "pack_3bit"),
    "unpack_3bit": (UNPACK_KERNEL_3BIT_SRC, "unpack_3bit"),
    "pack_2bit": (PACK_KERNEL_2BIT_SRC, "pack_2bit"),
    "unpack_2bit": (UNPACK_KERNEL_2BIT_SRC, "unpack_2bit"),
    "pack_4bit": (PACK_KERNEL_4BIT_SRC, "pack_4bit"),
    "unpack_4bit": (UNPACK_KERNEL_4BIT_SRC, "unpack_4bit"),
    "quantize_2bit": (QUANTIZE_2BIT_SRC, "quantize_2bit"),
    "quantize_3bit": (QUANTIZE_3BIT_SRC, "quantize_3bit"),
    "quantize_4bit": (QUANTIZE_4BIT_SRC, "quantize_4bit_q"),
    "fused_rotate_quantize_2bit": (
        FUSED_ROTATE_QUANTIZE_2BIT_SRC,
        "fused_rotate_quantize_2bit",
    ),
    "fused_rotate_quantize_3bit": (
        FUSED_ROTATE_QUANTIZE_3BIT_SRC,
        "fused_rotate_quantize_3bit",
    ),
    "fused_rotate_quantize_4bit": (
        FUSED_ROTATE_QUANTIZE_4BIT_SRC,
        "fused_rotate_quantize_4bit",
    ),
}


def get_gpu_kernel(name: str) -> object:
    """Lazily compile and cache a CuPy RawKernel.

    Args:
        name: Kernel name, one of ``pack_2bit``, ``unpack_2bit``,
            ``pack_3bit``, ``unpack_3bit``, ``pack_4bit``, ``unpack_4bit``.

    Returns:
        Compiled CuPy RawKernel.

    Raises:
        RuntimeError: If CuPy is not available.
        KeyError: If the kernel name is unknown.
    """
    if name in _gpu_kernels:
        return _gpu_kernels[name]
    if not _HAS_CUPY:
        raise RuntimeError("CuPy not available for GPU kernels")
    src, func_name = _KERNEL_SOURCES[name]
    kernel = cp.RawKernel(src, func_name)
    _gpu_kernels[name] = kernel
    return kernel


# ------------------------------------------------------------------ #
# High-level wrappers                                                 #
# ------------------------------------------------------------------ #

_QUANTIZE_KERNEL_NAMES = {2: "quantize_2bit", 3: "quantize_3bit", 4: "quantize_4bit"}
_FUSED_KERNEL_NAMES = {
    2: "fused_rotate_quantize_2bit",
    3: "fused_rotate_quantize_3bit",
    4: "fused_rotate_quantize_4bit",
}


def gpu_batch_quantize(
    x_rotated: cp.ndarray,
    boundaries: cp.ndarray,
    bits: int,
) -> cp.ndarray:
    """Quantize rotated float32 values to uint8 indices on GPU.

    Uses an unrolled binary-search kernel (no generic searchsorted).

    Args:
        x_rotated: Arbitrary-shape float32 array of rotated values.
        boundaries: 1-D float32 boundary array (``2**bits - 1`` elements).
        bits: Quantisation width (2, 3, or 4).

    Returns:
        uint8 array with the same shape as *x_rotated*.
    """
    if not _HAS_CUPY:
        raise RuntimeError("CuPy not available for GPU kernels")
    kernel = get_gpu_kernel(_QUANTIZE_KERNEL_NAMES[bits])
    flat = x_rotated.ravel().astype(cp.float32, copy=False)
    n = flat.size
    out = cp.empty(n, dtype=cp.uint8)
    threads = 256
    blocks = (n + threads - 1) // threads
    kernel((blocks,), (threads,), (flat, out, boundaries, n))
    return out.reshape(x_rotated.shape)


def gpu_batch_rotate_quantize(
    x_unit: cp.ndarray,
    Pi_T: cp.ndarray,
    boundaries: cp.ndarray,
    bits: int,
) -> cp.ndarray:
    """Fused rotation + quantization on GPU.

    Performs ``quantize(x_unit @ Pi_T)`` in a single kernel pass,
    avoiding the float32 intermediate write to global memory.

    Only supports QR rotation (dim <= 4096).

    Args:
        x_unit: 2-D float32 array of shape ``(N, dim)``, unit-normalised.
        Pi_T: 2-D float32 rotation matrix transpose, shape ``(dim, dim)``.
        boundaries: 1-D float32 boundary array.
        bits: Quantisation width (2, 3, or 4).

    Returns:
        uint8 array of shape ``(N, dim)``.
    """
    if not _HAS_CUPY:
        raise RuntimeError("CuPy not available for GPU kernels")
    kernel = get_gpu_kernel(_FUSED_KERNEL_NAMES[bits])
    N, dim = x_unit.shape
    out = cp.empty((N, dim), dtype=cp.uint8)
    tile = 32
    grid = ((dim + tile - 1) // tile, N)
    block = (tile,)
    kernel(grid, block, (x_unit, Pi_T, boundaries, out, N, dim))
    return out

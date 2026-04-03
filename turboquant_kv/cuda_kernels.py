# TurboQuant-KV: Open-source PolarQuant+QJL for LLM KV cache compression
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

from typing import Dict

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
# Lazy kernel compilation                                             #
# ------------------------------------------------------------------ #

_gpu_kernels: Dict[str, object] = {}

_KERNEL_SOURCES = {
    "pack_3bit": (PACK_KERNEL_3BIT_SRC, "pack_3bit"),
    "unpack_3bit": (UNPACK_KERNEL_3BIT_SRC, "unpack_3bit"),
    "pack_2bit": (PACK_KERNEL_2BIT_SRC, "pack_2bit"),
    "unpack_2bit": (UNPACK_KERNEL_2BIT_SRC, "unpack_2bit"),
    "pack_4bit": (PACK_KERNEL_4BIT_SRC, "pack_4bit"),
    "unpack_4bit": (UNPACK_KERNEL_4BIT_SRC, "unpack_4bit"),
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

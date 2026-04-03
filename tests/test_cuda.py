"""
GPU kernel tests for TurboQuant-KV.

These tests verify that CuPy CUDA kernels produce identical results
to the NumPy CPU fallback. All tests are skipped when CuPy is not
available.

Usage:
    pytest tests/test_cuda.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_kv import TurboQuantKV

try:
    import cupy as _cp

    _HAS_CUPY = True
except ImportError:
    _cp = None
    _HAS_CUPY = False


def _random_kv(
    batch: int = 1,
    n_heads: int = 4,
    seq_len: int = 32,
    head_dim: int = 64,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, n_heads, seq_len, head_dim)).astype("float32")


@pytest.mark.skipif(not _HAS_CUPY, reason="CuPy not available")
class TestGPUKernels:
    """Verify GPU kernels match CPU results."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_gpu_pack_matches_cpu(self, bits: int) -> None:
        """GPU packing produces identical bytes as CPU."""
        tq_cpu = TurboQuantKV(head_dim=64, bits=bits, use_gpu=False, seed=0)
        tq_gpu = TurboQuantKV(head_dim=64, bits=bits, use_gpu=True, seed=0)
        rng = np.random.default_rng(42)
        n = 2048
        indices_np = rng.integers(0, 2**bits, size=n, dtype=np.uint8)
        indices_cp = _cp.asarray(indices_np)

        packed_cpu = tq_cpu._pack_bits(indices_np)
        packed_gpu = _cp.asnumpy(tq_gpu._pack_bits(indices_cp))
        np.testing.assert_array_equal(packed_cpu, packed_gpu)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_gpu_roundtrip(self, bits: int) -> None:
        """GPU pack then unpack recovers original indices."""
        tq = TurboQuantKV(head_dim=64, bits=bits, use_gpu=True, seed=0)
        rng = np.random.default_rng(42)
        n = 2048
        indices_np = rng.integers(0, 2**bits, size=n, dtype=np.uint8)
        indices_cp = _cp.asarray(indices_np)

        packed = tq._pack_bits(indices_cp)
        unpacked = _cp.asnumpy(tq._unpack_bits(packed, n))
        np.testing.assert_array_equal(indices_np, unpacked)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_gpu_compress_decompress(self, bits: int) -> None:
        """Full GPU compress/decompress with packing produces
        the same reconstruction as CPU."""
        tensor = _random_kv(head_dim=64, seed=42)
        tq_cpu = TurboQuantKV(head_dim=64, n_heads=4, bits=bits, use_gpu=False, seed=0)
        tq_gpu = TurboQuantKV(head_dim=64, n_heads=4, bits=bits, use_gpu=True, seed=0)
        r_cpu = tq_cpu.decompress(tq_cpu.compress(tensor, packed=True))
        r_gpu = tq_gpu.decompress(tq_gpu.compress(tensor, packed=True))
        np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-5)

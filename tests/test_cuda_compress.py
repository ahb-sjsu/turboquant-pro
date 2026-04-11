"""
GPU compression kernel tests for TurboQuant Pro.

Verifies that the batch quantization, fused rotation+quantization, and
integrated compress pipelines produce identical results on GPU vs CPU.

All tests are skipped when CuPy is not available.

Usage:
    pytest tests/test_cuda_compress.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro import TurboQuantKV
from turboquant_pro.pgvector import TurboQuantPGVector

try:
    import cupy as _cp

    _HAS_CUPY = True
except ImportError:
    _cp = None
    _HAS_CUPY = False


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _random_kv(
    batch: int = 1,
    n_heads: int = 4,
    seq_len: int = 32,
    head_dim: int = 64,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, n_heads, seq_len, head_dim)).astype("float32")


def _random_embeddings(
    n: int = 100,
    dim: int = 128,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype("float32")


# ------------------------------------------------------------------ #
# Standalone batch quantize kernel                                     #
# ------------------------------------------------------------------ #


@pytest.mark.skipif(not _HAS_CUPY, reason="CuPy not available")
class TestGPUBatchQuantize:
    """Verify GPU batch quantize kernels match np.searchsorted."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_gpu_batch_quantize_matches_cpu(self, bits: int) -> None:
        from turboquant_pro.cuda_kernels import gpu_batch_quantize

        # Build boundaries matching TurboQuantKV conventions
        import math

        CODEBOOKS = {
            2: np.array([-1.510, -0.453, 0.453, 1.510]),
            3: np.array(
                [-1.748, -1.050, -0.500, -0.069, 0.069, 0.500, 1.050, 1.748]
            ),
            4: np.array(
                [
                    -2.401, -1.844, -1.437, -1.099, -0.800, -0.524, -0.262,
                    -0.066, 0.066, 0.262, 0.524, 0.800, 1.099, 1.437, 1.844,
                    2.401,
                ]
            ),
        }
        dim = 128
        scale = 1.0 / math.sqrt(dim)
        centroids = (CODEBOOKS[bits] * scale).astype(np.float32)
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        rng = np.random.default_rng(99)
        x = rng.standard_normal(10_000).astype(np.float32) * scale

        cpu_idx = np.searchsorted(boundaries, x).astype(np.uint8)

        bounds_d = _cp.asarray(boundaries)
        x_d = _cp.asarray(x)
        gpu_idx = _cp.asnumpy(gpu_batch_quantize(x_d, bounds_d, bits))

        np.testing.assert_array_equal(cpu_idx, gpu_idx)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_2d_shape_preserved(self, bits: int) -> None:
        from turboquant_pro.cuda_kernels import gpu_batch_quantize
        import math

        CODEBOOKS = {
            2: np.array([-1.510, -0.453, 0.453, 1.510]),
            3: np.array(
                [-1.748, -1.050, -0.500, -0.069, 0.069, 0.500, 1.050, 1.748]
            ),
            4: np.array(
                [
                    -2.401, -1.844, -1.437, -1.099, -0.800, -0.524, -0.262,
                    -0.066, 0.066, 0.262, 0.524, 0.800, 1.099, 1.437, 1.844,
                    2.401,
                ]
            ),
        }
        dim = 64
        scale = 1.0 / math.sqrt(dim)
        centroids = (CODEBOOKS[bits] * scale).astype(np.float32)
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        rng = np.random.default_rng(7)
        x = rng.standard_normal((50, dim)).astype(np.float32) * scale
        result = gpu_batch_quantize(
            _cp.asarray(x), _cp.asarray(boundaries), bits
        )
        assert result.shape == (50, dim)


# ------------------------------------------------------------------ #
# Fused rotation + quantization kernel                                 #
# ------------------------------------------------------------------ #


@pytest.mark.skipif(not _HAS_CUPY, reason="CuPy not available")
class TestGPUFusedRotateQuantize:
    """Verify fused GPU rotation+quantization matches sequential CPU."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_fused_matches_sequential(self, bits: int, dim: int) -> None:
        from turboquant_pro.cuda_kernels import gpu_batch_rotate_quantize
        import math

        CODEBOOKS = {
            2: np.array([-1.510, -0.453, 0.453, 1.510]),
            3: np.array(
                [-1.748, -1.050, -0.500, -0.069, 0.069, 0.500, 1.050, 1.748]
            ),
            4: np.array(
                [
                    -2.401, -1.844, -1.437, -1.099, -0.800, -0.524, -0.262,
                    -0.066, 0.066, 0.262, 0.524, 0.800, 1.099, 1.437, 1.844,
                    2.401,
                ]
            ),
        }
        scale = 1.0 / math.sqrt(dim)
        centroids = (CODEBOOKS[bits] * scale).astype(np.float32)
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        rng = np.random.default_rng(42)
        G = rng.standard_normal((dim, dim)).astype(np.float32)
        Q, _ = np.linalg.qr(G)
        Pi_T = Q.T.copy()

        N = 200
        x_unit = rng.standard_normal((N, dim)).astype(np.float32)
        norms = np.linalg.norm(x_unit, axis=1, keepdims=True)
        x_unit = x_unit / np.maximum(norms, 1e-30)

        # CPU reference: sequential rotate then searchsorted
        x_rot = x_unit @ Pi_T
        cpu_idx = np.searchsorted(boundaries, x_rot.ravel()).astype(np.uint8)
        cpu_idx = cpu_idx.reshape(N, dim)

        # GPU fused kernel
        gpu_idx = _cp.asnumpy(
            gpu_batch_rotate_quantize(
                _cp.asarray(x_unit),
                _cp.asarray(Pi_T),
                _cp.asarray(boundaries),
                bits,
            )
        )

        np.testing.assert_array_equal(cpu_idx, gpu_idx)


# ------------------------------------------------------------------ #
# Full pipeline: TurboQuantKV GPU vs CPU                               #
# ------------------------------------------------------------------ #


@pytest.mark.skipif(not _HAS_CUPY, reason="CuPy not available")
class TestGPUFullPipeline:
    """Verify full compress/decompress pipeline matches on GPU vs CPU."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_gpu_compress_decompress_packed(self, bits: int) -> None:
        tensor = _random_kv(head_dim=64, seed=42)
        tq_cpu = TurboQuantKV(
            head_dim=64, n_heads=4, bits=bits, use_gpu=False, seed=0
        )
        tq_gpu = TurboQuantKV(
            head_dim=64, n_heads=4, bits=bits, use_gpu=True, seed=0
        )
        r_cpu = tq_cpu.decompress(tq_cpu.compress(tensor, packed=True))
        r_gpu = tq_gpu.decompress(tq_gpu.compress(tensor, packed=True))
        np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-5)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_gpu_compress_decompress_unpacked(self, bits: int) -> None:
        tensor = _random_kv(head_dim=128, seq_len=16, seed=7)
        tq_cpu = TurboQuantKV(
            head_dim=128, n_heads=4, bits=bits, use_gpu=False, seed=0
        )
        tq_gpu = TurboQuantKV(
            head_dim=128, n_heads=4, bits=bits, use_gpu=True, seed=0
        )
        r_cpu = tq_cpu.decompress(tq_cpu.compress(tensor, packed=False))
        r_gpu = tq_gpu.decompress(tq_gpu.compress(tensor, packed=False))
        np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-5)


# ------------------------------------------------------------------ #
# TurboQuantPGVector GPU batch path                                    #
# ------------------------------------------------------------------ #


@pytest.mark.skipif(not _HAS_CUPY, reason="CuPy not available")
class TestGPUPGVectorBatch:
    """Verify TurboQuantPGVector compress_batch(use_gpu=True) matches CPU."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_gpu_batch_matches_cpu(self, bits: int) -> None:
        dim = 128
        embs = _random_embeddings(n=50, dim=dim, seed=42)
        tq = TurboQuantPGVector(dim=dim, bits=bits, seed=0)

        cpu_results = tq.compress_batch(embs, use_gpu=False)
        gpu_results = tq.compress_batch(embs, use_gpu=True)

        assert len(cpu_results) == len(gpu_results)
        for c, g in zip(cpu_results, gpu_results):
            assert c.packed_bytes == g.packed_bytes
            np.testing.assert_allclose(c.norm, g.norm, atol=1e-5)

    def test_gpu_large_batch(self) -> None:
        """Compress 10K embeddings on GPU without error."""
        dim = 256
        embs = _random_embeddings(n=10_000, dim=dim, seed=99)
        tq = TurboQuantPGVector(dim=dim, bits=3, seed=0)
        results = tq.compress_batch(embs, use_gpu=True)
        assert len(results) == 10_000
        assert all(r.dim == dim for r in results)

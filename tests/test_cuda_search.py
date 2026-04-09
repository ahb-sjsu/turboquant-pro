"""Tests for CUDA search kernels (ADC and Hamming)."""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro.cuda_search import pack_binary

# GPU tests are skipped if CuPy unavailable
try:
    import importlib.util

    HAS_CUPY = importlib.util.find_spec("cupy") is not None
except Exception:
    HAS_CUPY = False


class TestPackBinary:
    def test_basic_packing(self) -> None:
        # 2 vectors, 8 dims
        vecs = np.array(
            [[1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8
        )
        packed = pack_binary(vecs)
        assert packed.shape == (2, 1)  # ceil(8/64) = 1 word
        assert packed.dtype == np.uint64
        # First vec: bits 0,2,4,6 set = 0b01010101 = 85
        assert packed[0, 0] == 85
        # Second vec: bits 1,3,5,7 set = 0b10101010 = 170
        assert packed[1, 0] == 170

    def test_384_dims(self) -> None:
        rng = np.random.default_rng(42)
        vecs = rng.integers(0, 2, size=(100, 384), dtype=np.uint8)
        packed = pack_binary(vecs)
        assert packed.shape == (100, 6)  # ceil(384/64) = 6 words
        assert packed.dtype == np.uint64

    def test_round_trip(self) -> None:
        rng = np.random.default_rng(42)
        original = rng.integers(0, 2, size=(10, 64), dtype=np.uint8)
        packed = pack_binary(original)
        # Unpack to verify
        for i in range(10):
            word = packed[i, 0]
            for b in range(64):
                bit = int((word >> b) & 1)
                assert bit == original[i, b], f"Mismatch at vec={i}, bit={b}"


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
class TestGPUHammingSearch:
    def test_self_is_nearest(self) -> None:
        from turboquant_pro.cuda_search import gpu_hamming_search

        rng = np.random.default_rng(42)
        vecs = rng.integers(0, 2, size=(1000, 384), dtype=np.uint8)
        packed = pack_binary(vecs)

        # Query = first vector, should be distance 0 to itself
        top_idx, top_dist = gpu_hamming_search(packed[0], packed, top_k=1)
        assert top_idx[0] == 0
        assert top_dist[0] == 0

    def test_top_k_ordering(self) -> None:
        from turboquant_pro.cuda_search import gpu_hamming_search

        rng = np.random.default_rng(42)
        vecs = rng.integers(0, 2, size=(500, 128), dtype=np.uint8)
        packed = pack_binary(vecs)

        top_idx, top_dist = gpu_hamming_search(packed[0], packed, top_k=10)
        # Distances should be non-decreasing
        for i in range(len(top_dist) - 1):
            assert top_dist[i] <= top_dist[i + 1]


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
class TestGPUADCSearch:
    def test_basic_search(self) -> None:
        from turboquant_pro import TurboQuantPGVector
        from turboquant_pro.cuda_search import gpu_adc_search

        rng = np.random.default_rng(42)
        dim = 384
        n = 500
        corpus = rng.standard_normal((n, dim)).astype(np.float32)
        # Normalize
        corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)

        tq = TurboQuantPGVector(dim=dim, bits=3, seed=42)
        compressed = tq.compress_batch(corpus)

        query = corpus[0]
        top_idx, top_scores = gpu_adc_search(query, compressed, tq, top_k=10)

        assert len(top_idx) == 10
        # Self should be in top results (not necessarily #1 due to quantization)
        assert 0 in top_idx[:5], f"Self not in top-5: {top_idx}"

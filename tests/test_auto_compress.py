"""
Tests for general-purpose auto_compress API (Issue #12).

Usage:
    pytest tests/test_auto_compress.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro.auto_compress import (
    AutoCompressResult,
    auto_compress,
)


def _random_embeddings(n: int = 200, dim: int = 128, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


class TestTargetParsing:
    """Test target constraint string parsing."""

    def test_cosine_gt(self) -> None:
        from turboquant_pro.auto_compress import _parse_target

        metric, op, val = _parse_target("cosine > 0.97")
        assert metric == "cosine"
        assert op == ">"
        assert val == 0.97

    def test_cosine_gte(self) -> None:
        from turboquant_pro.auto_compress import _parse_target

        metric, op, val = _parse_target("cosine >= 0.95")
        assert metric == "cosine"
        assert op == ">="
        assert val == 0.95

    def test_ratio_gt(self) -> None:
        from turboquant_pro.auto_compress import _parse_target

        metric, op, val = _parse_target("ratio > 20")
        assert metric == "ratio"
        assert val == 20.0

    def test_invalid_raises(self) -> None:
        from turboquant_pro.auto_compress import _parse_target

        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_target("foo bar baz")


class TestAutoCompress:
    """Test the auto_compress sweep."""

    def test_basic_sweep(self) -> None:
        embs = _random_embeddings(100, 128)
        result = auto_compress(embs, target="cosine > 0.90", verbose=False)
        assert isinstance(result, AutoCompressResult)
        assert result.mean_cosine > 0.90
        assert result.ratio > 1.0
        assert len(result.compressed) > 0

    def test_high_quality_target(self) -> None:
        embs = _random_embeddings(100, 128)
        result = auto_compress(embs, target="cosine > 0.95", verbose=False)
        assert result.mean_cosine > 0.95

    def test_compression_target(self) -> None:
        embs = _random_embeddings(100, 128)
        result = auto_compress(
            embs, target="ratio > 5", verbose=False, bit_widths=[2, 3]
        )
        assert result.ratio >= 5.0

    def test_returns_pareto_candidates(self) -> None:
        embs = _random_embeddings(100, 128)
        result = auto_compress(embs, target="cosine > 0.90", verbose=False)
        assert len(result.candidates) >= 1
        # Pareto candidates should have increasing quality
        for c in result.candidates:
            assert "mean_cosine" in c
            assert "ratio" in c

    def test_config_has_label(self) -> None:
        embs = _random_embeddings(100, 128)
        result = auto_compress(embs, target="cosine > 0.90", verbose=False)
        assert "label" in result.config
        assert len(result.config["label"]) > 0

    def test_custom_pca_dims(self) -> None:
        embs = _random_embeddings(100, 128)
        result = auto_compress(
            embs,
            target="cosine > 0.85",
            pca_dims=[128, 64],
            verbose=False,
        )
        assert result.mean_cosine > 0.85

    def test_small_batch(self) -> None:
        embs = _random_embeddings(10, 64)
        result = auto_compress(
            embs, target="cosine > 0.80", pca_dims=[64], verbose=False
        )
        assert len(result.compressed) == 10

    def test_unmet_target_returns_best(self) -> None:
        """If no config meets the target, returns highest quality."""
        embs = _random_embeddings(50, 64)
        result = auto_compress(embs, target="cosine > 0.9999", verbose=False)
        # Should still return something
        assert isinstance(result, AutoCompressResult)
        assert result.mean_cosine > 0.5

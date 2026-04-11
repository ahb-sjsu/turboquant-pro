"""
Tests for the unified auto-configuration API.

Usage:
    pytest tests/test_autoconfig.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro import AutoConfig, TurboQuantKV, TurboQuantKVCache
from turboquant_pro.autoconfig import list_models, list_targets

# ------------------------------------------------------------------ #
# Factory methods                                                      #
# ------------------------------------------------------------------ #


class TestFromDict:
    """Test AutoConfig.from_dict()."""

    def test_basic_dict(self) -> None:
        cfg = AutoConfig.from_dict(
            {"head_dim": 128, "num_key_value_heads": 8, "num_hidden_layers": 32}
        )
        assert cfg.head_dim == 128
        assert cfg.n_kv_heads == 8
        assert cfg.n_layers == 32

    def test_infer_head_dim_from_hidden_size(self) -> None:
        cfg = AutoConfig.from_dict({"hidden_size": 4096, "num_attention_heads": 32})
        assert cfg.head_dim == 128

    def test_target_quality(self) -> None:
        cfg = AutoConfig.from_dict({"head_dim": 128}, target="quality")
        assert cfg.key_bits == 4
        assert cfg.value_bits == 4
        assert cfg.rope_aware is True

    def test_target_balanced(self) -> None:
        cfg = AutoConfig.from_dict({"head_dim": 128}, target="balanced")
        assert cfg.key_bits == 4
        assert cfg.value_bits == 3

    def test_target_compression(self) -> None:
        cfg = AutoConfig.from_dict({"head_dim": 128}, target="compression")
        assert cfg.key_bits == 3
        assert cfg.value_bits == 2

    def test_target_extreme(self) -> None:
        cfg = AutoConfig.from_dict({"head_dim": 128}, target="extreme")
        assert cfg.key_bits == 2
        assert cfg.value_bits == 2
        assert cfg.rope_aware is False

    def test_invalid_target_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown target"):
            AutoConfig.from_dict({"head_dim": 128}, target="turbo")

    def test_overrides(self) -> None:
        cfg = AutoConfig.from_dict(
            {"head_dim": 128},
            target="balanced",
            key_bits=3,
            rope_aware=False,
        )
        assert cfg.key_bits == 3  # overridden
        assert cfg.value_bits == 3  # from preset
        assert cfg.rope_aware is False  # overridden

    def test_rope_theta(self) -> None:
        cfg = AutoConfig.from_dict({"head_dim": 128, "rope_theta": 500000.0})
        assert cfg.rope_theta == 500000.0

    def test_max_seq_len(self) -> None:
        cfg = AutoConfig.from_dict({"head_dim": 128, "max_position_embeddings": 131072})
        assert cfg.max_seq_len == 131072


class TestFromPretrained:
    """Test AutoConfig.from_pretrained() with built-in registry."""

    def test_known_model(self) -> None:
        cfg = AutoConfig.from_pretrained("llama-3-8b")
        assert cfg.head_dim == 128
        assert cfg.n_kv_heads == 8
        assert cfg.n_layers == 32
        assert cfg.model_name == "llama-3-8b"

    def test_known_model_case_insensitive(self) -> None:
        cfg = AutoConfig.from_pretrained("Llama-3-8B")
        assert cfg.head_dim == 128

    def test_alias(self) -> None:
        cfg = AutoConfig.from_pretrained("meta-llama/Llama-3-8B")
        assert cfg.model_name == "llama-3-8b"

    def test_gemma(self) -> None:
        cfg = AutoConfig.from_pretrained("gemma-2-27b")
        assert cfg.head_dim == 128
        assert cfg.n_kv_heads == 16
        assert cfg.n_layers == 46

    def test_qwen(self) -> None:
        cfg = AutoConfig.from_pretrained("qwen2.5-7b")
        assert cfg.rope_theta == 1000000.0
        assert cfg.max_seq_len == 131072

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not resolve"):
            AutoConfig.from_pretrained("totally-unknown-model-xyz")

    def test_all_registered_models_valid(self) -> None:
        for model in list_models():
            cfg = AutoConfig.from_pretrained(model)
            assert cfg.head_dim > 0
            assert cfg.n_kv_heads > 0
            assert cfg.n_layers > 0


# ------------------------------------------------------------------ #
# Builder methods                                                      #
# ------------------------------------------------------------------ #


class TestBuilders:
    """Test build_quantizer, build_cache, etc."""

    def test_build_quantizer(self) -> None:
        cfg = AutoConfig.from_pretrained("llama-3-8b")
        tq = cfg.build_quantizer()
        assert isinstance(tq, TurboQuantKV)
        assert tq.key_bits == 4  # balanced default
        assert tq.value_bits == 3
        assert tq.head_dim == 128

    def test_build_cache(self) -> None:
        cfg = AutoConfig.from_pretrained("llama-3-8b")
        cache = cfg.build_cache(hot_window=256)
        assert isinstance(cache, TurboQuantKVCache)
        assert cache.key_bits == 4
        assert cache.value_bits == 3
        assert cache.hot_window == 256

    def test_build_rope_quantizer_enabled(self) -> None:
        cfg = AutoConfig.from_pretrained("llama-3-8b", target="balanced")
        rq = cfg.build_rope_quantizer()
        from turboquant_pro.rope import RoPEAwareQuantizer

        assert isinstance(rq, RoPEAwareQuantizer)

    def test_build_rope_quantizer_disabled(self) -> None:
        cfg = AutoConfig.from_pretrained("llama-3-8b", target="extreme")
        tq = cfg.build_rope_quantizer()
        assert isinstance(tq, TurboQuantKV)

    def test_build_manager(self) -> None:
        cfg = AutoConfig.from_pretrained("llama-3-8b")
        mgr = cfg.build_manager(hot_window=128)
        assert mgr.n_layers == 32


# ------------------------------------------------------------------ #
# TurboQuantKV.from_model() convenience                                #
# ------------------------------------------------------------------ #


class TestFromModel:
    """Test the TurboQuantKV.from_model() class method."""

    def test_from_model_string(self) -> None:
        tq = TurboQuantKV.from_model("llama-3-8b")
        assert isinstance(tq, TurboQuantKV)
        assert tq.head_dim == 128
        assert tq.key_bits == 4
        assert tq.value_bits == 3

    def test_from_model_dict(self) -> None:
        tq = TurboQuantKV.from_model({"head_dim": 256, "num_key_value_heads": 16})
        assert tq.head_dim == 256

    def test_from_model_target(self) -> None:
        tq = TurboQuantKV.from_model("gemma-2-27b", target="compression")
        assert tq.key_bits == 3
        assert tq.value_bits == 2

    def test_from_model_compress_works(self) -> None:
        tq = TurboQuantKV.from_model("llama-3-8b")
        rng = np.random.default_rng(42)
        tensor = rng.standard_normal((1, 8, 32, 128)).astype(np.float32)
        ck = tq.compress(tensor, packed=True, kind="key")
        cv = tq.compress(tensor, packed=True, kind="value")
        assert ck.bits == 4
        assert cv.bits == 3


# ------------------------------------------------------------------ #
# Memory estimation and summary                                        #
# ------------------------------------------------------------------ #


class TestEstimateAndSummary:
    """Test estimate_memory and summary."""

    def test_estimate_memory(self) -> None:
        cfg = AutoConfig.from_pretrained("llama-3-8b")
        est = cfg.estimate_memory(seq_len=8192)
        assert est["original_gb"] > 0
        assert est["compressed_gb"] > 0
        assert est["ratio"] > 1.0
        assert est["saved_gb"] > 0

    def test_summary(self) -> None:
        cfg = AutoConfig.from_pretrained("llama-3-8b")
        s = cfg.summary()
        assert s["model"] == "llama-3-8b"
        assert s["target"] == "balanced"
        assert s["key_bits"] == 4
        assert s["value_bits"] == 3
        assert "estimated_kv_cache_gb" in s
        assert "compression_ratio" in s

    def test_repr(self) -> None:
        cfg = AutoConfig.from_pretrained("llama-3-8b")
        r = repr(cfg)
        assert "llama-3-8b" in r
        assert "K4/V3" in r


# ------------------------------------------------------------------ #
# Utility functions                                                    #
# ------------------------------------------------------------------ #


class TestUtilities:
    """Test list_models and list_targets."""

    def test_list_models(self) -> None:
        models = list_models()
        assert len(models) >= 7
        assert "llama-3-8b" in models
        assert "gemma-2-27b" in models

    def test_list_targets(self) -> None:
        targets = list_targets()
        assert "quality" in targets
        assert "balanced" in targets
        assert "compression" in targets
        assert "extreme" in targets
        assert all(isinstance(v, str) for v in targets.values())

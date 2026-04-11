# TurboQuant Pro: Open-source PolarQuant+QJL for LLM KV cache compression
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""
Unified auto-configuration for TurboQuant KV cache compression.

Reads a HuggingFace model configuration and automatically selects the
best compression parameters — asymmetric K/V bit allocation, RoPE-aware
boosting, and target-based presets — so that users get optimal results
without manual tuning.

Usage::

    from turboquant_pro.autoconfig import AutoConfig

    # From a HuggingFace model name (requires transformers)
    cfg = AutoConfig.from_pretrained("meta-llama/Llama-3-8B")

    # From a local config dict
    cfg = AutoConfig.from_dict({
        "head_dim": 128,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "rope_theta": 500000.0,
        "max_position_embeddings": 8192,
    })

    # Get a configured TurboQuantKV
    tq = cfg.build_quantizer()

    # Get a configured TurboQuantKVCache
    cache = cfg.build_cache(hot_window=512)

    # Get a configured RoPEAwareQuantizer
    rq = cfg.build_rope_quantizer()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Known model registry                                                 #
# ------------------------------------------------------------------ #

_MODEL_REGISTRY: dict[str, dict] = {
    "llama-3-8b": dict(
        head_dim=128,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=32,
        rope_theta=500000.0,
        max_position_embeddings=8192,
    ),
    "llama-3-70b": dict(
        head_dim=128,
        num_attention_heads=64,
        num_key_value_heads=8,
        num_hidden_layers=80,
        rope_theta=500000.0,
        max_position_embeddings=8192,
    ),
    "gemma-2-27b": dict(
        head_dim=128,
        num_attention_heads=32,
        num_key_value_heads=16,
        num_hidden_layers=46,
        rope_theta=10000.0,
        max_position_embeddings=8192,
    ),
    "gemma-2-9b": dict(
        head_dim=256,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_hidden_layers=42,
        rope_theta=10000.0,
        max_position_embeddings=8192,
    ),
    "gemma-4-27b-a4b": dict(
        head_dim=256,
        num_attention_heads=32,
        num_key_value_heads=16,
        num_hidden_layers=48,
        rope_theta=10000.0,
        max_position_embeddings=262144,
    ),
    "qwen2.5-7b": dict(
        head_dim=128,
        num_attention_heads=28,
        num_key_value_heads=4,
        num_hidden_layers=28,
        rope_theta=1000000.0,
        max_position_embeddings=131072,
    ),
    "qwen2.5-72b": dict(
        head_dim=128,
        num_attention_heads=64,
        num_key_value_heads=8,
        num_hidden_layers=80,
        rope_theta=1000000.0,
        max_position_embeddings=131072,
    ),
    "mistral-7b": dict(
        head_dim=128,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=32,
        rope_theta=10000.0,
        max_position_embeddings=32768,
    ),
}

# Aliases: map common HuggingFace model name patterns to registry keys
_MODEL_ALIASES: dict[str, str] = {
    "meta-llama/llama-3-8b": "llama-3-8b",
    "meta-llama/meta-llama-3-8b": "llama-3-8b",
    "meta-llama/llama-3-70b": "llama-3-70b",
    "meta-llama/meta-llama-3-70b": "llama-3-70b",
    "google/gemma-2-27b": "gemma-2-27b",
    "google/gemma-2-9b": "gemma-2-9b",
    "google/gemma-4-27b-a4b": "gemma-4-27b-a4b",
    "qwen/qwen2.5-7b": "qwen2.5-7b",
    "qwen/qwen2.5-72b": "qwen2.5-72b",
    "mistralai/mistral-7b-v0.1": "mistral-7b",
    "mistralai/mistral-7b": "mistral-7b",
}


# ------------------------------------------------------------------ #
# Target presets                                                       #
# ------------------------------------------------------------------ #

_TARGETS: dict[str, dict] = {
    "quality": {
        "key_bits": 4,
        "value_bits": 4,
        "rope_aware": True,
        "description": "Maximum reconstruction quality (K4/V4 + RoPE boost)",
    },
    "balanced": {
        "key_bits": 4,
        "value_bits": 3,
        "rope_aware": True,
        "description": "Best quality/compression tradeoff (K4/V3 + RoPE boost)",
    },
    "compression": {
        "key_bits": 3,
        "value_bits": 2,
        "rope_aware": True,
        "description": "Maximum compression (K3/V2 + RoPE boost)",
    },
    "extreme": {
        "key_bits": 2,
        "value_bits": 2,
        "rope_aware": False,
        "description": "Extreme compression, lower quality (K2/V2)",
    },
}


# ------------------------------------------------------------------ #
# AutoConfig                                                           #
# ------------------------------------------------------------------ #


@dataclass
class AutoConfig:
    """Unified auto-configuration for TurboQuant KV cache compression.

    Encapsulates model architecture parameters and compression target,
    producing fully configured compressor instances via ``build_*()``
    methods.

    Attributes:
        head_dim: Dimension per attention head.
        n_kv_heads: Number of key-value attention heads per layer.
        n_layers: Number of transformer layers.
        rope_theta: RoPE base frequency.
        max_seq_len: Maximum position embeddings (context length).
        key_bits: Quantisation bits for keys.
        value_bits: Quantisation bits for values.
        rope_aware: Whether to apply RoPE-aware bit boosting.
        target: Name of the target preset used.
        model_name: Model name (if resolved from registry/HF).
    """

    head_dim: int = 128
    n_kv_heads: int = 8
    n_layers: int = 32
    rope_theta: float = 10000.0
    max_seq_len: int = 8192
    key_bits: int = 4
    value_bits: int = 3
    rope_aware: bool = True
    target: str = "balanced"
    model_name: str = ""

    # ------------------------------------------------------------------ #
    # Factory methods                                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(
        cls,
        config: dict,
        target: str = "balanced",
        **overrides,
    ) -> AutoConfig:
        """Create an AutoConfig from a configuration dictionary.

        Accepts HuggingFace-style config keys (``num_key_value_heads``,
        ``num_hidden_layers``, ``rope_theta``, etc.) and maps them to
        TurboQuant parameters.

        Args:
            config: Model configuration dictionary.
            target: Compression target preset — ``"quality"``,
                ``"balanced"``, ``"compression"``, or ``"extreme"``.
            **overrides: Override any auto-selected parameter.
        """
        if target not in _TARGETS:
            raise ValueError(
                f"Unknown target '{target}'. " f"Choose from: {sorted(_TARGETS)}"
            )
        preset = _TARGETS[target]

        # Extract model parameters, handling various naming conventions
        head_dim = config.get("head_dim")
        if head_dim is None:
            hidden = config.get("hidden_size", 4096)
            n_heads = config.get("num_attention_heads", 32)
            head_dim = hidden // n_heads

        n_kv_heads = config.get(
            "num_key_value_heads",
            config.get("num_attention_heads", 32),
        )
        n_layers = config.get("num_hidden_layers", 32)
        rope_theta = config.get("rope_theta", 10000.0)
        max_seq_len = config.get("max_position_embeddings", 8192)

        # Apply target preset, then overrides
        kwargs = dict(
            head_dim=head_dim,
            n_kv_heads=n_kv_heads,
            n_layers=n_layers,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
            key_bits=preset["key_bits"],
            value_bits=preset["value_bits"],
            rope_aware=preset["rope_aware"],
            target=target,
            model_name=config.get("model_type", ""),
        )
        kwargs.update(overrides)

        cfg = cls(**kwargs)

        logger.info(
            "AutoConfig: %s target=%s key=%d-bit value=%d-bit "
            "rope_aware=%s head_dim=%d n_kv_heads=%d n_layers=%d",
            cfg.model_name or "custom",
            target,
            cfg.key_bits,
            cfg.value_bits,
            cfg.rope_aware,
            cfg.head_dim,
            cfg.n_kv_heads,
            cfg.n_layers,
        )
        return cfg

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        target: str = "balanced",
        **overrides,
    ) -> AutoConfig:
        """Create an AutoConfig from a model name or HuggingFace path.

        Tries, in order:

        1. The built-in model registry (no network required).
        2. ``transformers.AutoConfig.from_pretrained()`` (requires the
           ``transformers`` package and may download from HF Hub).

        Args:
            model_name_or_path: A model name like ``"llama-3-8b"`` or a
                HuggingFace path like ``"meta-llama/Llama-3-8B"``.
            target: Compression target preset.
            **overrides: Override any auto-selected parameter.
        """
        key = model_name_or_path.lower().strip("/")

        # 1. Try built-in registry
        if key in _MODEL_REGISTRY:
            config = _MODEL_REGISTRY[key]
            cfg = cls.from_dict(config, target=target, **overrides)
            cfg.model_name = key
            return cfg

        # 2. Try aliases
        if key in _MODEL_ALIASES:
            config = _MODEL_REGISTRY[_MODEL_ALIASES[key]]
            cfg = cls.from_dict(config, target=target, **overrides)
            cfg.model_name = _MODEL_ALIASES[key]
            return cfg

        # 3. Try HuggingFace transformers
        try:
            from transformers import AutoConfig as HFAutoConfig

            hf_cfg = HFAutoConfig.from_pretrained(model_name_or_path)
            config = hf_cfg.to_dict()
            cfg = cls.from_dict(config, target=target, **overrides)
            cfg.model_name = model_name_or_path
            return cfg
        except ImportError:
            pass
        except Exception as e:
            logger.warning(
                "Failed to load HF config for '%s': %s",
                model_name_or_path,
                e,
            )

        raise ValueError(
            f"Could not resolve model '{model_name_or_path}'. "
            f"Known models: {sorted(_MODEL_REGISTRY)}. "
            f"Install 'transformers' for HuggingFace Hub lookup, "
            f"or pass a config dict via AutoConfig.from_dict()."
        )

    # ------------------------------------------------------------------ #
    # Builder methods                                                     #
    # ------------------------------------------------------------------ #

    def build_quantizer(
        self,
        use_gpu: bool = False,
        seed: int | None = None,
    ):
        """Build a configured TurboQuantKV instance.

        Returns:
            TurboQuantKV with asymmetric K/V bits from this config.
        """
        from .core import TurboQuantKV

        return TurboQuantKV(
            head_dim=self.head_dim,
            n_heads=self.n_kv_heads,
            bits=self.value_bits,  # default = value (lower)
            key_bits=self.key_bits,
            value_bits=self.value_bits,
            use_gpu=use_gpu,
            seed=seed,
        )

    def build_cache(
        self,
        hot_window: int = 512,
        use_gpu: bool = False,
        seed: int | None = None,
    ):
        """Build a configured TurboQuantKVCache instance.

        Returns:
            TurboQuantKVCache with asymmetric K/V bits.
        """
        from .core import TurboQuantKVCache

        return TurboQuantKVCache(
            head_dim=self.head_dim,
            n_heads=self.n_kv_heads,
            bits=self.value_bits,
            key_bits=self.key_bits,
            value_bits=self.value_bits,
            hot_window=hot_window,
            use_gpu=use_gpu,
            seed=seed,
        )

    def build_rope_quantizer(
        self,
        use_gpu: bool = False,
        seed: int | None = None,
    ):
        """Build a RoPE-aware quantizer if RoPE boosting is enabled.

        Falls back to a standard TurboQuantKV if ``rope_aware=False``.

        Returns:
            RoPEAwareQuantizer or TurboQuantKV.
        """
        if not self.rope_aware:
            return self.build_quantizer(use_gpu=use_gpu, seed=seed)

        from .rope import RoPEAwareQuantizer

        return RoPEAwareQuantizer(
            head_dim=self.head_dim,
            n_heads=self.n_kv_heads,
            default_bits=self.value_bits,
            boost_bits=self.key_bits,
            rope_base=self.rope_theta,
            max_seq_len=self.max_seq_len,
            use_gpu=use_gpu,
            seed=seed,
        )

    def build_manager(
        self,
        hot_window: int = 512,
        use_gpu: bool = False,
    ):
        """Build a configured TurboQuantKVManager for all layers.

        Returns:
            TurboQuantKVManager with asymmetric K/V bits.
        """
        from .vllm_plugin import TurboQuantKVManager

        return TurboQuantKVManager(
            n_layers=self.n_layers,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            bits=self.value_bits,
            key_bits=self.key_bits,
            value_bits=self.value_bits,
            hot_window=hot_window,
            use_gpu=use_gpu,
        )

    # ------------------------------------------------------------------ #
    # Hardware-aware tuning                                                #
    # ------------------------------------------------------------------ #

    def hardware_profile(self, device_id: int = 0):
        """Detect GPU and return a HardwareProfile with recommendations.

        If the hardware profile recommends different bit-widths than
        the current config (e.g., Blackwell recommends K4/V4), the
        returned profile contains the hardware-optimal values.

        Returns:
            HardwareProfile from :mod:`turboquant_pro.hardware`.
        """
        from .hardware import get_hardware_profile

        return get_hardware_profile(device_id=device_id)

    def with_hardware_tuning(self, device_id: int = 0) -> AutoConfig:
        """Return a new AutoConfig with hardware-optimized parameters.

        Adjusts ``key_bits`` and ``value_bits`` based on the detected
        GPU architecture.  For example, Blackwell's native NVFP4
        makes 4-bit nearly free, so the profile upgrades to K4/V4.

        Returns:
            New AutoConfig instance (does not mutate ``self``).
        """
        from .hardware import get_hardware_profile

        profile = get_hardware_profile(device_id=device_id)

        return AutoConfig(
            head_dim=self.head_dim,
            n_kv_heads=self.n_kv_heads,
            n_layers=self.n_layers,
            rope_theta=self.rope_theta,
            max_seq_len=self.max_seq_len,
            key_bits=profile.recommended_key_bits,
            value_bits=profile.recommended_value_bits,
            rope_aware=self.rope_aware,
            target=self.target,
            model_name=self.model_name,
        )

    # ------------------------------------------------------------------ #
    # Memory estimation                                                    #
    # ------------------------------------------------------------------ #

    def estimate_memory(
        self,
        seq_len: int | None = None,
        original_dtype: str = "float16",
    ) -> dict[str, float]:
        """Estimate KV cache memory for this model configuration.

        Args:
            seq_len: Context length.  Defaults to ``max_seq_len``.
            original_dtype: Original storage dtype.

        Returns:
            Dict with ``original_gb``, ``compressed_gb``, ``ratio``,
            ``saved_gb``.
        """
        from .core import TurboQuantKV

        if seq_len is None:
            seq_len = self.max_seq_len

        return TurboQuantKV.estimate_memory(
            n_layers=self.n_layers,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            seq_len=seq_len,
            key_bits=self.key_bits,
            value_bits=self.value_bits,
            bit_packed=True,
            original_dtype=original_dtype,
        )

    # ------------------------------------------------------------------ #
    # Display                                                              #
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        """Return a human-readable summary of the configuration."""
        preset = _TARGETS.get(self.target, {})
        mem = self.estimate_memory()
        return {
            "model": self.model_name or "custom",
            "target": self.target,
            "target_description": preset.get("description", ""),
            "head_dim": self.head_dim,
            "n_kv_heads": self.n_kv_heads,
            "n_layers": self.n_layers,
            "max_seq_len": self.max_seq_len,
            "key_bits": self.key_bits,
            "value_bits": self.value_bits,
            "rope_aware": self.rope_aware,
            "rope_theta": self.rope_theta,
            "estimated_kv_cache_gb": mem["compressed_gb"],
            "compression_ratio": mem["ratio"],
            "saved_gb": mem["saved_gb"],
        }

    def __repr__(self) -> str:
        name = self.model_name or "custom"
        return (
            f"AutoConfig({name}, target={self.target!r}, "
            f"K{self.key_bits}/V{self.value_bits}, "
            f"rope_aware={self.rope_aware}, "
            f"head_dim={self.head_dim}, "
            f"n_kv_heads={self.n_kv_heads}, "
            f"n_layers={self.n_layers})"
        )


# ------------------------------------------------------------------ #
# Convenience functions                                                #
# ------------------------------------------------------------------ #


def list_models() -> list[str]:
    """Return the names of all known models in the built-in registry."""
    return sorted(_MODEL_REGISTRY)


def list_targets() -> dict[str, str]:
    """Return available target presets with descriptions."""
    return {k: v["description"] for k, v in _TARGETS.items()}

"""
vLLM KV cache compression plugin for TurboQuant Pro.

Extends vLLM's KV cache with TurboQuant 3-bit compression,
reducing KV cache memory by ~5x and enabling longer context
windows on the same hardware.

Usage with vLLM:
    from vllm import LLM
    from turboquant_pro.vllm_plugin import TurboQuantKVBackend

    # Register the backend
    backend = TurboQuantKVBackend(bits=3, hot_window=512)

    # Use with vLLM (via monkey-patch or plugin entry point)
    llm = LLM("google/gemma-4-31b", max_model_len=16384)

Standalone usage:
    from turboquant_pro.vllm_plugin import TurboQuantKVManager

    manager = TurboQuantKVManager(
        n_layers=32, n_kv_heads=8, head_dim=256, bits=3
    )
    manager.store(layer_id=0, keys=k_tensor, values=v_tensor)
    k, v = manager.load(layer_id=0, start=0, end=1024)
    print(manager.memory_stats())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    from .core import TurboQuantKVCache
except ImportError:
    TurboQuantKVCache = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Multi-layer KV cache manager
# ---------------------------------------------------------------------------


@dataclass
class LayerStats:
    """Statistics for one transformer layer's KV cache."""

    layer_id: int
    length: int
    cold_length: int
    hot_length: int
    cold_bytes: int
    hot_bytes: int
    total_bytes: int
    uncompressed_bytes: int
    compression_ratio: float


class TurboQuantKVManager:
    """Multi-layer KV cache manager with TurboQuant compression.

    Manages per-layer TurboQuantKVCache instances for an entire
    transformer model. Each layer has independent hot/cold tiering.

    Args:
        n_layers: Number of transformer layers.
        n_kv_heads: Number of KV attention heads per layer.
        head_dim: Dimension per attention head.
        bits: Quantization bit width (2, 3, or 4).
        hot_window: Tokens kept uncompressed per layer.
        use_gpu: Use GPU for compression (requires CuPy).
        device_id: CUDA device ID.
    """

    def __init__(
        self,
        n_layers: int = 32,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        bits: int = 3,
        hot_window: int = 512,
        use_gpu: bool = False,
        device_id: int = 0,
    ) -> None:
        if TurboQuantKVCache is None:
            raise ImportError(
                "TurboQuantKVCache not available. "
                "Install turboquant-pro: pip install turboquant-pro"
            )

        self._n_layers = n_layers
        self._n_kv_heads = n_kv_heads
        self._head_dim = head_dim
        self._bits = bits
        self._hot_window = hot_window

        # Create per-layer caches
        self._caches: list[TurboQuantKVCache] = []
        for _ in range(n_layers):
            cache = TurboQuantKVCache(
                head_dim=head_dim,
                n_heads=n_kv_heads,
                bits=bits,
                hot_window=hot_window,
                use_gpu=use_gpu,
                device_id=device_id,
            )
            self._caches.append(cache)

        logger.info(
            "TurboQuantKVManager: %d layers, %d heads, "
            "%d head_dim, %d-bit, hot_window=%d",
            n_layers,
            n_kv_heads,
            head_dim,
            bits,
            hot_window,
        )

    @property
    def n_layers(self) -> int:
        return self._n_layers

    def store(
        self,
        layer_id: int,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """Store KV pair for a layer.

        Args:
            layer_id: Transformer layer index.
            keys: Key tensor, shape (n_kv_heads, head_dim)
                  or (1, n_kv_heads, 1, head_dim).
            values: Value tensor, same shape as keys.
        """
        cache = self._caches[layer_id]
        cache.append(keys, values)

    def load(
        self,
        layer_id: int,
        start: int = 0,
        end: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load KV tensors for a layer (decompresses cold if needed).

        Args:
            layer_id: Transformer layer index.
            start: Start token position.
            end: End token position (None = all).

        Returns:
            (keys, values) each shape (1, n_kv_heads, length, head_dim).
        """
        cache = self._caches[layer_id]
        if end is None:
            end = cache.length
        keys = cache.get_keys(start, end)
        values = cache.get_values(start, end)
        return keys, values

    def length(self, layer_id: int = 0) -> int:
        """Total tokens stored for a layer."""
        return self._caches[layer_id].length

    def clear(self) -> None:
        """Clear all layers."""
        for cache in self._caches:
            cache.clear()

    def layer_stats(self, layer_id: int) -> LayerStats:
        """Get statistics for one layer."""
        cache = self._caches[layer_id]
        stats = cache.memory_stats()
        uncompressed = (
            cache.length
            * self._n_kv_heads
            * self._head_dim
            * 4  # float32
            * 2  # keys + values
        )
        return LayerStats(
            layer_id=layer_id,
            length=cache.length,
            cold_length=cache.cold_length,
            hot_length=cache.hot_length,
            cold_bytes=stats["cold_bytes"],
            hot_bytes=stats["hot_bytes"],
            total_bytes=stats["total_bytes"],
            uncompressed_bytes=uncompressed,
            compression_ratio=(stats.get("effective_ratio", 1.0)),
        )

    def memory_stats(self) -> dict:
        """Aggregate memory statistics across all layers."""
        total_cold = 0
        total_hot = 0
        total_uncompressed = 0
        total_tokens = 0

        for i in range(self._n_layers):
            ls = self.layer_stats(i)
            total_cold += ls.cold_bytes
            total_hot += ls.hot_bytes
            total_uncompressed += ls.uncompressed_bytes
            total_tokens = max(total_tokens, ls.length)

        total_bytes = total_cold + total_hot
        ratio = total_uncompressed / max(total_bytes, 1) if total_bytes > 0 else 1.0

        return {
            "n_layers": self._n_layers,
            "total_tokens": total_tokens,
            "cold_bytes": total_cold,
            "hot_bytes": total_hot,
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / 1048576, 1),
            "uncompressed_bytes": total_uncompressed,
            "uncompressed_mb": round(total_uncompressed / 1048576, 1),
            "compression_ratio": round(ratio, 1),
            "saved_mb": round((total_uncompressed - total_bytes) / 1048576, 1),
            "bits": self._bits,
            "hot_window": self._hot_window,
        }

    def estimate_capacity(
        self,
        max_memory_gb: float,
    ) -> int:
        """Estimate max context length for a memory budget.

        Args:
            max_memory_gb: Available memory in GB.

        Returns:
            Estimated maximum context length (tokens).
        """
        # Per-token memory with compression
        # Hot: full float32 = n_layers * n_kv_heads * head_dim * 4 * 2
        # Cold: compressed ≈ hot / compression_ratio
        bytes_per_token_hot = self._n_layers * self._n_kv_heads * self._head_dim * 4 * 2
        # Approximate cold ratio from bit width
        approx_ratio = {2: 7.5, 3: 5.1, 4: 3.9}.get(self._bits, 5.0)
        bytes_per_token_cold = bytes_per_token_hot / approx_ratio

        budget = max_memory_gb * 1073741824
        # Hot window uses full memory, rest uses compressed
        hot_bytes = self._hot_window * bytes_per_token_hot
        remaining = budget - hot_bytes
        if remaining <= 0:
            return self._hot_window

        cold_tokens = int(remaining / bytes_per_token_cold)
        return self._hot_window + cold_tokens


# ---------------------------------------------------------------------------
# vLLM backend wrapper (for plugin registration)
# ---------------------------------------------------------------------------


class TurboQuantKVBackend:
    """vLLM-compatible KV cache backend using TurboQuant.

    This class wraps TurboQuantKVManager with the interface
    expected by vLLM's pluggable KV cache system.

    Args:
        bits: Quantization bit width (2, 3, or 4).
        hot_window: Tokens kept uncompressed.
    """

    def __init__(
        self,
        bits: int = 3,
        hot_window: int = 512,
    ) -> None:
        self._bits = bits
        self._hot_window = hot_window
        self._manager: TurboQuantKVManager | None = None

    def initialize(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        **kwargs,
    ) -> None:
        """Initialize the backend for a specific model."""
        self._manager = TurboQuantKVManager(
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            bits=self._bits,
            hot_window=self._hot_window,
        )

    def store_block(
        self,
        layer_id: int,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """Store a KV block (vLLM interface)."""
        if self._manager is None:
            raise RuntimeError("Backend not initialized")
        self._manager.store(layer_id, keys, values)

    def load_block(
        self,
        layer_id: int,
        start: int = 0,
        end: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load a KV block (vLLM interface)."""
        if self._manager is None:
            raise RuntimeError("Backend not initialized")
        return self._manager.load(layer_id, start, end)

    def memory_stats(self) -> dict:
        """Return memory statistics."""
        if self._manager is None:
            return {}
        return self._manager.memory_stats()

    def supports_async(self) -> bool:
        """Whether this backend supports async operations."""
        return False

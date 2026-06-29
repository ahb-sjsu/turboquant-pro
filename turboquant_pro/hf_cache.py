# TurboQuant Pro: HuggingFace `transformers` drop-in KV cache.
# Copyright (c) 2026 Andrew H. Bond. MIT License.
"""One-line HuggingFace drop-in for TurboQuant asym-NF4 KV-cache compression.

Use TurboQuant's calibration-free, architecture-robust KV compression inside any
``model.generate(...)`` call with a single line::

    from turboquant_pro import TurboQuantCache
    out = model.generate(**inputs, past_key_values=TurboQuantCache(hot_window=512))

or, even shorter, patch the model so *every* ``generate`` uses it::

    from turboquant_pro import enable_turboquant_cache
    enable_turboquant_cache(model, hot_window=512)
    out = model.generate(**inputs)   # now compressed transparently

Design (tiered hot/cold, calibration-free)
-------------------------------------------
Each transformer layer keeps the most-recent ``hot_window`` tokens uncompressed in
the model's native dtype (a rolling fp16/fp32 buffer) so recent-token attention is
lossless. Older "cold" tokens are spilled into a compressed store using the shipped
quantizers (we do **not** reinvent them):

* **keys**  -> :class:`turboquant_pro.per_channel_kv.PerChannelKV` with asymmetric
  (zero-point) NF4 + ``outlier_frac`` dense-sparse fp16 outliers. This is the same
  recipe as :meth:`TurboQuantKVCache.robust` and is the recommended robust default
  for keys (ties NF4 on Llama/Mistral, recovers the Qwen2.5 collapse).
* **values** -> :class:`turboquant_pro.core.TurboQuantKV` (PolarQuant: random
  rotation + per-vector norm + Lloyd-Max), bit-packed, per-token.

On every ``update`` the layer returns the **full** per-layer K/V (cold dequantized,
concatenated with the hot fp16 window along the sequence axis) so the attention op
sees an ordinary dense tensor and nothing downstream needs to change.

This targets the v5 ``transformers`` Cache contract: a :class:`Cache` is a thin
container delegating to one ``CacheLayerMixin`` per layer; the layer implements
``lazy_initialization``, ``update``, ``get_seq_length``, ``get_mask_sizes`` and
``get_max_cache_shape`` (plus ``crop`` / ``reset`` / ``reorder_cache`` and the batch
helpers used by generation).
"""

from __future__ import annotations

from functools import partial

import numpy as np

# Optional transformers import -- keep the package importable without transformers,
# matching the repo's optional-import pattern (see vllm_plugin.py / faiss_index.py).
try:
    import torch
    from transformers.cache_utils import Cache, CacheLayerMixin

    _HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover - exercised only when deps are absent
    torch = None  # type: ignore[assignment]
    _HAS_TRANSFORMERS = False

    # Minimal stand-ins so the rest of the module can still be imported / introspected
    # without transformers installed. The real classes are required to actually use it.
    class Cache:  # type: ignore[no-redef]
        pass

    class CacheLayerMixin:  # type: ignore[no-redef]
        pass


from .core import TurboQuantKV
from .per_channel_kv import PerChannelKV

__all__ = ["TurboQuantCache", "TurboQuantLayer", "enable_turboquant_cache"]


class TurboQuantLayer(CacheLayerMixin):
    """One layer's tiered hot/cold TurboQuant cache.

    Keeps the most recent ``hot_window`` tokens uncompressed (native dtype) and
    compresses spilled older tokens with asym-NF4 per-channel keys + PolarQuant
    per-token values. Implements the v5 ``CacheLayerMixin`` contract.
    """

    is_compileable = False
    is_sliding = False

    def __init__(
        self,
        hot_window: int = 512,
        key_bits: int = 4,
        value_bits: int = 4,
        outlier_frac: float = 0.02,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if hot_window < 1:
            raise ValueError("hot_window must be >= 1")
        self.hot_window = int(hot_window)
        self.key_bits = int(key_bits)
        self.value_bits = int(value_bits)
        self.outlier_frac = float(outlier_frac)
        self.seed = int(seed)

        # Set in lazy_initialization (once shapes/device/dtype are known).
        self.head_dim: int | None = None
        self.n_kv_heads: int | None = None
        self._kq: PerChannelKV | None = None  # key quantizer (per-channel asym-NF4)
        self._vq: TurboQuantKV | None = None  # value quantizer (PolarQuant)

        # Hot window: dense tensors [B, H, hot_len, D] in native dtype.
        self._hot_keys = None
        self._hot_values = None
        # Cold store: parallel lists of compressed chunks + their token lengths.
        self._cold_keys: list = []
        self._cold_values: list = []
        self._cold_lengths: list[int] = []
        self._seq_len: int = 0

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #
    def lazy_initialization(self, key_states, value_states) -> None:
        """Record device/dtype/shape and build the quantizers (calibration-free)."""
        self.dtype, self.device = key_states.dtype, key_states.device
        b, h, _s, d = key_states.shape
        self.n_kv_heads, self.head_dim = int(h), int(d)

        # Keys: per-channel asymmetric NF4 + dense-sparse outliers (the robust recipe).
        self._kq = PerChannelKV(
            head_dim=self.head_dim,
            n_heads=self.n_kv_heads,
            bits=self.key_bits,
            nf4_asym=True,
            outlier_frac=self.outlier_frac,
        )
        # Values: PolarQuant (random rotation + per-vector norm + Lloyd-Max), packed.
        self._vq = TurboQuantKV(
            head_dim=self.head_dim,
            n_heads=self.n_kv_heads,
            bits=self.value_bits,
            value_bits=self.value_bits,
            use_gpu=False,
            seed=self.seed,
        )

        empty = key_states.new_zeros((b, h, 0, d))
        self._hot_keys = empty
        self._hot_values = empty.clone()
        self._cold_keys = []
        self._cold_values = []
        self._cold_lengths = []
        self._seq_len = 0
        self.is_initialized = True

    # ------------------------------------------------------------------ #
    # numpy <-> torch bridges                                             #
    # ------------------------------------------------------------------ #
    def _to_numpy(self, t):
        return t.detach().to("cpu", torch.float32).numpy()

    def _to_torch(self, arr: np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(arr)).to(
            device=self.device, dtype=self.dtype
        )

    # ------------------------------------------------------------------ #
    # Core update                                                         #
    # ------------------------------------------------------------------ #
    def update(self, key_states, value_states, *args, **kwargs):
        """Append new K/V, spill over-capacity tokens, return the full K/V."""
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self._ingest(key_states, value_states)
        return self._full()

    def _ingest(self, key_states, value_states) -> None:
        """Append to the hot window, spilling the oldest tokens once over capacity."""
        self._hot_keys = torch.cat([self._hot_keys, key_states], dim=-2)
        self._hot_values = torch.cat([self._hot_values, value_states], dim=-2)
        self._seq_len += int(key_states.shape[-2])

        overflow = self._hot_keys.shape[-2] - self.hot_window
        if overflow > 0:
            spill_k = self._hot_keys[:, :, :overflow, :]
            spill_v = self._hot_values[:, :, :overflow, :]
            # Keys -> per-channel asym-NF4 (+outliers); values -> PolarQuant. Packed.
            self._cold_keys.append(
                self._kq.compress(self._to_numpy(spill_k), packed=True)
            )
            self._cold_values.append(
                self._vq.compress(self._to_numpy(spill_v), packed=True, kind="value")
            )
            self._cold_lengths.append(int(overflow))
            self._hot_keys = self._hot_keys[:, :, overflow:, :].contiguous()
            self._hot_values = self._hot_values[:, :, overflow:, :].contiguous()

    def _full(self):
        """Return (keys, values) over the whole cache: cold dequantized + hot."""
        if not self._cold_keys:
            return self._hot_keys, self._hot_values

        key_parts = [self._to_torch(self._kq.decompress(c)) for c in self._cold_keys]
        val_parts = [self._to_torch(self._vq.decompress(c)) for c in self._cold_values]
        key_parts.append(self._hot_keys)
        val_parts.append(self._hot_values)
        return torch.cat(key_parts, dim=-2), torch.cat(val_parts, dim=-2)

    # ------------------------------------------------------------------ #
    # Contract: sizes / masks                                            #
    # ------------------------------------------------------------------ #
    def get_seq_length(self) -> int:
        if not self.is_initialized:
            return 0
        return self._seq_len

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """(kv_length, kv_offset) for mask construction (called pre-update)."""
        return self.get_seq_length() + query_length, 0

    def get_max_cache_shape(self) -> int:
        """Dynamic cache: no fixed maximum length."""
        return -1

    # ------------------------------------------------------------------ #
    # Contract: mutation (generation utilities)                          #
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Reset to empty while preserving the layer object and quantizers."""
        if not self.is_initialized:
            return
        self._hot_keys = self._hot_keys[:, :, :0, :].contiguous()
        self._hot_values = self._hot_values[:, :, :0, :].contiguous()
        self._cold_keys = []
        self._cold_values = []
        self._cold_lengths = []
        self._seq_len = 0

    def _reload(self, keys, values) -> None:
        """Drop all state and re-ingest a full (already-dense) K/V pair."""
        b, h, _s, d = keys.shape
        self._hot_keys = keys.new_zeros((b, h, 0, d))
        self._hot_values = values.new_zeros((b, h, 0, d))
        self._cold_keys = []
        self._cold_values = []
        self._cold_lengths = []
        self._seq_len = 0
        if keys.shape[-2] > 0:
            self._ingest(keys, values)

    def crop(self, max_length: int) -> None:
        """Crop to ``max_length`` tokens (negative => drop that many from the end)."""
        if not self.is_initialized:
            return
        seq_len = self.get_seq_length()
        if max_length < 0:
            max_length = seq_len - abs(max_length)
        if seq_len <= max_length:
            return
        keys, values = self._full()
        self._reload(keys[:, :, :max_length, :], values[:, :, :max_length, :])

    def reorder_cache(self, beam_idx) -> None:
        """Reorder the batch dim for beam search."""
        if not self.is_initialized or self.get_seq_length() == 0:
            return
        keys, values = self._full()
        idx = beam_idx.to(keys.device)
        self._reload(keys.index_select(0, idx), values.index_select(0, idx))

    def batch_repeat_interleave(self, repeats: int) -> None:
        if not self.is_initialized or self.get_seq_length() == 0:
            return
        keys, values = self._full()
        self._reload(
            keys.repeat_interleave(repeats, dim=0),
            values.repeat_interleave(repeats, dim=0),
        )

    def batch_select_indices(self, indices) -> None:
        if not self.is_initialized or self.get_seq_length() == 0:
            return
        keys, values = self._full()
        self._reload(keys[indices, ...], values[indices, ...])

    # Offloading is a no-op here (compressed cold store already lives on CPU/numpy).
    def offload(self) -> None:  # pragma: no cover - GPU/offload path
        pass

    def prefetch(self) -> None:  # pragma: no cover - GPU/offload path
        pass


class TurboQuantCache(Cache):
    """Drop-in ``transformers`` Cache with TurboQuant asym-NF4 KV compression.

    Pass it straight to ``generate``::

        model.generate(**inputs, past_key_values=TurboQuantCache(hot_window=512))

    Args:
        hot_window: recent tokens kept uncompressed per layer (default 512).
        key_bits: key quantization width (asym-NF4 requires 4; default 4).
        value_bits: value (PolarQuant) quantization width (default 4).
        outlier_frac: per-channel fraction of key entries kept in fp16 (default 0.02).
        seed: rotation seed for the value quantizer (determinism).
    """

    def __init__(
        self,
        hot_window: int = 512,
        key_bits: int = 4,
        value_bits: int = 4,
        outlier_frac: float = 0.02,
        seed: int = 0,
    ) -> None:
        if not _HAS_TRANSFORMERS:  # pragma: no cover - dependency guard
            raise ImportError(
                "TurboQuantCache requires `transformers` (v5+) and `torch`. "
                "Install them to use the HuggingFace drop-in cache."
            )
        self.hot_window = int(hot_window)
        self.key_bits = int(key_bits)
        self.value_bits = int(value_bits)
        self.outlier_frac = float(outlier_frac)
        self.seed = int(seed)
        # Lazily replicate one TurboQuantLayer per model layer as `update` is called.
        layer_factory = partial(
            TurboQuantLayer,
            hot_window=self.hot_window,
            key_bits=self.key_bits,
            value_bits=self.value_bits,
            outlier_frac=self.outlier_frac,
            seed=self.seed,
        )
        super().__init__(layer_class_to_replicate=layer_factory)


def enable_turboquant_cache(model, **kwargs):
    """Make ``model.generate`` use a :class:`TurboQuantCache` by default.

    Wraps ``model.generate`` so that calls without an explicit ``past_key_values``
    transparently get a fresh ``TurboQuantCache(**kwargs)``. Returns the model.

    Example::

        enable_turboquant_cache(model, hot_window=512)
        out = model.generate(**inputs)            # compressed
        model.generate(**inputs, past_key_values=other)  # respects explicit cache
    """
    if not _HAS_TRANSFORMERS:  # pragma: no cover - dependency guard
        raise ImportError(
            "enable_turboquant_cache requires `transformers` and `torch`."
        )

    if getattr(model, "_turboquant_generate_patched", False):
        model._turboquant_cache_kwargs = dict(kwargs)
        return model

    original_generate = model.generate
    model._turboquant_cache_kwargs = dict(kwargs)

    def generate(*args, **gen_kwargs):
        use_cache = gen_kwargs.get("use_cache", True)
        if use_cache and gen_kwargs.get("past_key_values") is None:
            gen_kwargs["past_key_values"] = TurboQuantCache(
                **model._turboquant_cache_kwargs
            )
        return original_generate(*args, **gen_kwargs)

    model.generate = generate
    model._turboquant_generate_patched = True
    return model

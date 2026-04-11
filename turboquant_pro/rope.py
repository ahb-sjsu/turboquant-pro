# TurboQuant Pro: Open-source PolarQuant+QJL for LLM KV cache compression
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""
RoPE-aware KV cache quantization.

Rotary Position Embeddings (RoPE) apply sinusoidal rotations at different
frequencies to pairs of head dimensions.  Low-frequency pairs (small
index) carry long-range positional information — a single rotation cycle
spans more positions than the context window — and are therefore more
sensitive to quantization noise.

This module provides:

- :class:`RoPEFrequencyAnalyzer`: inspects the RoPE frequency spectrum
  and computes a per-dimension bit allocation that gives extra precision
  to the low-frequency (long-wavelength) pairs.
- :class:`RoPEAwareQuantizer`: a KV cache compressor that splits dimensions
  into *boosted* (low-frequency) and *default* (high-frequency) groups,
  compresses each with its own :class:`TurboQuantKV` instance, and
  reassembles the result on decompression.

Typical usage::

    raq = RoPEAwareQuantizer(head_dim=128, max_seq_len=32768)
    compressed = raq.compress(kv_tensor)
    reconstructed = raq.decompress(compressed)

For long-context models (32K+) this yields measurably better attention
accuracy in the low-frequency subspace at a modest increase in average
bits per dimension (e.g. 3.25 instead of 3.0).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .core import CompressedKV, TurboQuantKV

# ------------------------------------------------------------------ #
# RoPE frequency analysis                                             #
# ------------------------------------------------------------------ #


class RoPEFrequencyAnalyzer:
    """Analyze RoPE frequency spectrum and compute per-dimension bit allocation.

    RoPE applies sinusoidal rotations at different frequencies to pairs of
    head dimensions.  Low-frequency pairs (small index) carry long-range
    positional information and are more sensitive to quantization noise.

    Args:
        head_dim: Dimension per attention head (must be even).
        base: RoPE base frequency (default 10000.0 for LLaMA/Gemma).
        max_seq_len: Maximum context length to consider.
    """

    def __init__(
        self,
        head_dim: int = 128,
        base: float = 10000.0,
        max_seq_len: int = 8192,
    ) -> None:
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")

        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len

        n_pairs = head_dim // 2
        i = np.arange(n_pairs, dtype=np.float64)

        # RoPE frequency for each pair: 1 / (base ** (2i / head_dim))
        self.frequencies: np.ndarray = (1.0 / (base ** (2.0 * i / head_dim))).astype(
            np.float64
        )

        # Wavelength: how many positions before the rotation repeats
        self.wavelengths: np.ndarray = (2.0 * math.pi / self.frequencies).astype(
            np.float64
        )

    # ------------------------------------------------------------------ #
    # Bit allocation                                                      #
    # ------------------------------------------------------------------ #

    def bit_allocation(
        self,
        default_bits: int = 3,
        boost_bits: int = 4,
    ) -> np.ndarray:
        """Compute per-dimension bit widths based on RoPE wavelength.

        Dimensions whose wavelength exceeds *max_seq_len* (i.e. the
        rotation does not complete a full cycle within the context window)
        are allocated *boost_bits*.  All other dimensions receive
        *default_bits*.  Each pair ``(2i, 2i+1)`` receives the same
        bit width because RoPE rotates them together.

        Args:
            default_bits: Bit width for high-frequency dimensions.
            boost_bits: Bit width for low-frequency dimensions.

        Returns:
            ``(head_dim,)`` uint8 array of per-dimension bit widths.
        """
        alloc = np.full(self.head_dim, default_bits, dtype=np.uint8)
        for pair_idx in range(self.head_dim // 2):
            if self.wavelengths[pair_idx] > self.max_seq_len:
                alloc[2 * pair_idx] = boost_bits
                alloc[2 * pair_idx + 1] = boost_bits
        return alloc

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #

    def summary(self) -> dict[str, Any]:
        """Return a summary of the frequency analysis.

        Returns:
            Dict with keys: ``n_boosted_dims``, ``n_default_dims``,
            ``avg_bits``, ``frequencies_range``, ``wavelengths_range``.
        """
        alloc = self.bit_allocation()
        n_boosted = int(np.sum(alloc > alloc.min()))
        n_default = self.head_dim - n_boosted
        avg_bits = float(np.mean(alloc))
        return {
            "n_boosted_dims": n_boosted,
            "n_default_dims": n_default,
            "avg_bits": avg_bits,
            "frequencies_range": (
                float(self.frequencies.min()),
                float(self.frequencies.max()),
            ),
            "wavelengths_range": (
                float(self.wavelengths.min()),
                float(self.wavelengths.max()),
            ),
        }


# ------------------------------------------------------------------ #
# RoPE-aware quantizer                                                #
# ------------------------------------------------------------------ #


class RoPEAwareQuantizer:
    """KV cache compressor with RoPE-aware per-dimension bit allocation.

    Automatically allocates higher precision to low-frequency RoPE
    dimensions that carry long-range positional information.

    Internally this creates two :class:`TurboQuantKV` instances — one for
    the *boosted* (low-frequency) dimensions and one for the *default*
    (high-frequency) dimensions — and splits the head dimension
    accordingly during compression.

    Args:
        head_dim: Dimension per attention head.
        n_heads: Number of KV heads.
        default_bits: Bit width for high-frequency dimensions (default 3).
        boost_bits: Bit width for low-frequency dimensions (default 4).
        rope_base: RoPE base frequency (default 10000.0).
        max_seq_len: Maximum context length.
        use_gpu: Use GPU for compression.
        seed: Random seed.
    """

    def __init__(
        self,
        head_dim: int = 128,
        n_heads: int = 32,
        default_bits: int = 3,
        boost_bits: int = 4,
        rope_base: float = 10000.0,
        max_seq_len: int = 8192,
        use_gpu: bool = False,
        seed: int | None = None,
    ) -> None:
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.default_bits = default_bits
        self.boost_bits = boost_bits

        # Frequency analysis and bit allocation ----------------------------
        self._analyzer = RoPEFrequencyAnalyzer(
            head_dim=head_dim,
            base=rope_base,
            max_seq_len=max_seq_len,
        )
        alloc = self._analyzer.bit_allocation(
            default_bits=default_bits,
            boost_bits=boost_bits,
        )

        # Boolean mask: True for dimensions that get boost_bits
        self._boost_mask: np.ndarray = alloc == boost_bits
        self._n_boost: int = int(self._boost_mask.sum())
        self._n_default: int = head_dim - self._n_boost

        # Create separate compressors for each segment --------------------
        # Guard against edge cases where all dims fall into one bucket.
        self._tq_boost: TurboQuantKV | None = None
        self._tq_default: TurboQuantKV | None = None

        if self._n_boost > 0:
            self._tq_boost = TurboQuantKV(
                head_dim=self._n_boost,
                n_heads=n_heads,
                bits=boost_bits,
                use_gpu=use_gpu,
                seed=seed,
            )
        if self._n_default > 0:
            # Use a different seed for the default compressor so the
            # rotation matrices are independent.
            default_seed = (seed + 1) if seed is not None else None
            self._tq_default = TurboQuantKV(
                head_dim=self._n_default,
                n_heads=n_heads,
                bits=default_bits,
                use_gpu=use_gpu,
                seed=default_seed,
            )

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def analyzer(self) -> RoPEFrequencyAnalyzer:
        """The underlying :class:`RoPEFrequencyAnalyzer`."""
        return self._analyzer

    @property
    def avg_bits(self) -> float:
        """Weighted average bits per dimension."""
        if self.head_dim == 0:
            return 0.0
        return (
            self._n_boost * self.boost_bits + self._n_default * self.default_bits
        ) / self.head_dim

    # ------------------------------------------------------------------ #
    # Compress / decompress                                               #
    # ------------------------------------------------------------------ #

    def compress(
        self,
        tensor: np.ndarray,
        packed: bool = True,
    ) -> dict[str, Any]:
        """Compress a KV cache tensor with RoPE-aware bit allocation.

        The tensor is split along the head dimension into *boosted* and
        *default* segments.  Each segment is compressed independently
        with its own :class:`TurboQuantKV` instance.

        Args:
            tensor: KV cache tensor of shape ``(B, H, S, D)`` in any
                float dtype.
            packed: If True, bit-pack indices for maximum compression.

        Returns:
            Dict with keys ``boost`` (:class:`CompressedKV` or None),
            ``default`` (:class:`CompressedKV` or None), ``boost_mask``,
            ``shape``, and ``head_dim``.
        """
        if tensor.shape[-1] != self.head_dim:
            raise ValueError(
                f"Expected head_dim={self.head_dim} in last axis, "
                f"got {tensor.shape[-1]}"
            )

        # Split along head_dim ------------------------------------------
        boost_tensor = tensor[..., self._boost_mask] if self._n_boost > 0 else None
        default_tensor = tensor[..., ~self._boost_mask] if self._n_default > 0 else None

        boost_compressed: CompressedKV | None = None
        default_compressed: CompressedKV | None = None

        if boost_tensor is not None and self._tq_boost is not None:
            boost_compressed = self._tq_boost.compress(boost_tensor, packed=packed)
        if default_tensor is not None and self._tq_default is not None:
            default_compressed = self._tq_default.compress(
                default_tensor, packed=packed
            )

        return {
            "boost": boost_compressed,
            "default": default_compressed,
            "boost_mask": self._boost_mask,
            "shape": tensor.shape,
            "head_dim": self.head_dim,
        }

    def decompress(self, compressed: dict[str, Any]) -> np.ndarray:
        """Reconstruct a KV cache tensor from a RoPE-aware compressed form.

        Decompresses the *boosted* and *default* segments and interleaves
        them back into the original dimension order.

        Args:
            compressed: Dict returned by :meth:`compress`.

        Returns:
            Reconstructed tensor of shape ``(B, H, S, D)`` in float32.
        """
        shape = compressed["shape"]
        boost_mask = compressed["boost_mask"]

        # Allocate output tensor
        out = np.empty(shape, dtype=np.float32)

        if compressed["boost"] is not None and self._tq_boost is not None:
            boost_recon = self._tq_boost.decompress(compressed["boost"])
            out[..., boost_mask] = boost_recon

        if compressed["default"] is not None and self._tq_default is not None:
            default_recon = self._tq_default.decompress(compressed["default"])
            out[..., ~boost_mask] = default_recon

        return out

    # ------------------------------------------------------------------ #
    # Stats                                                               #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        """Return allocation statistics.

        Returns:
            Dict with keys: ``n_boost_dims``, ``n_default_dims``,
            ``boost_bits``, ``default_bits``, ``avg_bits``,
            ``analyzer_summary``.
        """
        return {
            "n_boost_dims": self._n_boost,
            "n_default_dims": self._n_default,
            "boost_bits": self.boost_bits,
            "default_bits": self.default_bits,
            "avg_bits": self.avg_bits,
            "analyzer_summary": self._analyzer.summary(),
        }

# TurboQuant Pro: Open-source PolarQuant+QJL for LLM KV cache compression
# Copyright (c) 2025 Andrew H. Bond
# MIT License
#
# Algorithm: Zandieh et al. "Sub-linear Memory Inference via PolarQuant
# and QJL" (ICLR 2026). Implementation adapted from Theory Radar.

"""
TurboQuant KV cache compression for LLM inference.

Reduces KV cache memory by ~5.1x (3-bit) to ~7.9x (2-bit) with
bit-packing, enabling longer context windows on VRAM-constrained GPUs
such as the Quadro GV100 (32 GB, Volta / compute 7.0).

Algorithm overview:

  1. Random rotation Pi (QR of Gaussian or structured Hadamard + sign
     flip for large dimensions) maps each head-dim vector onto the unit
     hypersphere where coordinates are approximately i.i.d. Gaussian.
  2. Optimal Lloyd-Max scalar quantizer maps each rotated coordinate
     to a *b*-bit index using precomputed centroids for N(0, 1/sqrt(d)).
  3. Bit-pack indices (8 x 3-bit = 3 bytes) + per-vector L2 norm.

The key difference from beam-search quantization is that KV cache
compression must preserve attention-pattern accuracy (cosine similarity
of key/query dot products), not just candidate rankings.  Empirically,
3-bit quantization yields cosine similarity > 0.99 for head_dim >= 64.

Target model dimensions (Gemma 4 27B):
  - n_heads=16, n_kv_heads=16, head_dim=256
  - 8K context at fp16: ~2.0 GB KV cache -> ~340 MB at 3-bit (packed)

Provides three main components:
  - ``TurboQuantKV``: stateless compress/decompress with optional bit-packing
  - ``TurboQuantKVCache``: streaming KV cache with hot/cold tiered storage
  - CuPy CUDA RawKernels for GPU-accelerated bit-packing (optional)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from .cuda_kernels import (
    get_gpu_kernel,
    gpu_batch_quantize,
    gpu_batch_rotate_quantize,
)

logger = logging.getLogger(__name__)

try:
    import cupy as cp  # type: ignore[import-untyped]

    _HAS_CUPY = True
except ImportError:
    cp = None  # type: ignore[assignment]
    _HAS_CUPY = False


# ------------------------------------------------------------------ #
# Lloyd-Max codebook centroids for standard normal distribution       #
# (scaled by 1/sqrt(d) at runtime where d = head_dim)                #
# ------------------------------------------------------------------ #

_CODEBOOKS: dict[int, np.ndarray] = {
    2: np.array([-1.510, -0.453, 0.453, 1.510]),
    3: np.array([-1.748, -1.050, -0.500, -0.069, 0.069, 0.500, 1.050, 1.748]),
    4: np.array(
        [
            -2.401,
            -1.844,
            -1.437,
            -1.099,
            -0.800,
            -0.524,
            -0.262,
            -0.066,
            0.066,
            0.262,
            0.524,
            0.800,
            1.099,
            1.437,
            1.844,
            2.401,
        ]
    ),
}


@dataclass
class CompressedKV:
    """Container for a quantized KV tensor.

    Attributes:
        indices: uint8 array of quantisation bin indices.
            When ``packed=False``: shape matches the original tensor
            shape ``(batch, n_heads, seq_len, head_dim)`` with dtype
            uint8 (one index per byte).
            When ``packed=True``: flat uint8 byte array with indices
            bit-packed (e.g. 8 x 3-bit = 3 bytes).
        norms: L2 norms of the original vectors along ``head_dim``.
            Shape is ``(batch, n_heads, seq_len)``.
        bits: Number of quantisation bits (2, 3, or 4).
        original_dtype: The dtype of the tensor before compression.
        packed: Whether indices are bit-packed.
        n_values: Total number of index values (needed for unpacking
            when ``packed=True``).
        shape: Original index array shape before packing.
    """

    indices: np.ndarray  # uint8 (packed or unpacked)
    norms: np.ndarray  # float32 (B, H, S)
    bits: int
    original_dtype: np.dtype = field(default_factory=lambda: np.dtype("float32"))
    packed: bool = False
    n_values: int = 0
    shape: tuple[int, ...] = ()

    def nbytes(self) -> int:
        """Total memory consumed by the compressed representation."""
        return self.indices.nbytes + self.norms.nbytes

    def original_nbytes(self, head_dim: int) -> int:
        """Memory that the uncompressed tensor would occupy."""
        n_vectors = int(np.prod(self.norms.shape))
        return n_vectors * head_dim * self.original_dtype.itemsize

    def compression_ratio(self, head_dim: int) -> float:
        """Ratio of original size to compressed size."""
        return self.original_nbytes(head_dim) / max(self.nbytes(), 1)


class TurboQuantKV:
    """Compress KV cache tensors using TurboQuant (b-bit quantisation).

    Reduces KV cache memory by ~6x (3-bit) to enable longer context
    windows.  Based on Zandieh et al. (ICLR 2026) PolarQuant + QJL,
    adapted from Theory Radar's TurboBeam implementation.

    Supports **asymmetric K/V bit allocation** via *key_bits* and
    *value_bits*.  Keys are more sensitive to quantisation noise than
    values because attention scores are computed as
    ``softmax(Q @ K^T / sqrt(d))`` — small errors in K are amplified
    by the softmax.  Using ``key_bits=4, value_bits=3`` gives
    near-4-bit attention quality at near-3-bit storage cost.

    Usage::

        tq = TurboQuantKV(head_dim=256, n_heads=16, bits=3)
        compressed_k = tq.compress(key_tensor)    # (B, H, S, D)
        compressed_v = tq.compress(value_tensor)
        key_approx = tq.decompress(compressed_k)
        value_approx = tq.decompress(compressed_v)

        # Asymmetric K/V bits
        tq = TurboQuantKV(head_dim=256, key_bits=4, value_bits=3)
        compressed_k = tq.compress(key_tensor, kind="key")
        compressed_v = tq.compress(value_tensor, kind="value")

    Args:
        head_dim: Dimension of each attention head (e.g. 128, 256).
        n_heads: Number of KV heads.  Only used for logging; the actual
            head count is inferred from the input tensor shape.
        bits: Default quantisation width -- 2, 3, or 4.  Used for both
            keys and values unless *key_bits* or *value_bits* is set.
        key_bits: Quantisation width for key tensors.  Overrides *bits*
            for keys when set.
        value_bits: Quantisation width for value tensors.  Overrides
            *bits* for values when set.
        use_gpu: If True *and* CuPy is available, perform rotation and
            quantisation on the GPU.  Falls back to NumPy otherwise.
        device_id: CUDA device ordinal when ``use_gpu=True``.
        seed: Random seed for the rotation matrix (for reproducibility).
    """

    def __init__(
        self,
        head_dim: int = 128,
        n_heads: int = 32,
        bits: int = 3,
        key_bits: int | None = None,
        value_bits: int | None = None,
        use_gpu: bool = True,
        device_id: int = 0,
        seed: int | None = None,
    ) -> None:
        # Resolve per-tensor bit widths ----------------------------------
        self.key_bits = key_bits if key_bits is not None else bits
        self.value_bits = value_bits if value_bits is not None else bits

        for b_name, b_val in [
            ("bits", bits),
            ("key_bits", self.key_bits),
            ("value_bits", self.value_bits),
        ]:
            if b_val not in _CODEBOOKS:
                raise ValueError(
                    f"Unsupported {b_name}={b_val}; "
                    f"choose from {sorted(_CODEBOOKS)}"
                )

        self.head_dim = head_dim
        self.n_heads = n_heads
        self.bits = bits  # kept for backward compat / default
        self.n_centroids = 2**bits

        # Decide backend -------------------------------------------------
        self._gpu = use_gpu and _HAS_CUPY
        self._device_id = device_id
        self._xp: object = cp if self._gpu else np  # type: ignore[assignment]

        logger.info(
            "TurboQuantKV: head_dim=%d, n_heads=%d, bits=%d "
            "(key=%d, value=%d), backend=%s",
            head_dim,
            n_heads,
            bits,
            self.key_bits,
            self.value_bits,
            "cupy" if self._gpu else "numpy",
        )

        # Codebooks for every bit-width in use ----------------------------
        scale = 1.0 / math.sqrt(head_dim)
        self._codebooks: dict[int, tuple] = {}
        for b in sorted({self.bits, self.key_bits, self.value_bits}):
            raw = _CODEBOOKS[b]
            if self._gpu:
                with cp.cuda.Device(device_id):
                    c = cp.asarray(raw * scale, dtype=cp.float32)
                    bnd = (c[:-1] + c[1:]) / 2.0
            else:
                c = (raw * scale).astype(np.float32)
                bnd = (c[:-1] + c[1:]) / 2.0
            self._codebooks[b] = (c, bnd)

        # Default centroids / boundaries (for decompress backward compat)
        self.centroids, self.boundaries = self._codebooks[self.bits]

        # Random rotation matrix ------------------------------------------
        rng = np.random.default_rng(seed)

        if head_dim <= 4096:
            # Full QR factorisation for moderate dimensions
            G = rng.standard_normal((head_dim, head_dim)).astype(np.float32)
            Q, _ = np.linalg.qr(G)
            if self._gpu:
                with cp.cuda.Device(device_id):
                    self._Pi = cp.asarray(Q)
                    self._Pi_T = self._Pi.T.copy()
            else:
                self._Pi = Q
                self._Pi_T = Q.T.copy()
            self._structured = False
        else:
            # Structured rotation (sign flip + permutation) for very
            # large dims -- O(d) memory and O(d) application cost.
            signs = rng.choice([-1.0, 1.0], size=head_dim).astype(np.float32)
            perm = rng.permutation(head_dim)
            if self._gpu:
                with cp.cuda.Device(device_id):
                    self._sign_flip = cp.asarray(signs)
                    self._perm = cp.asarray(perm)
                    self._inv_perm = cp.argsort(self._perm)
            else:
                self._sign_flip = signs
                self._perm = perm
                self._inv_perm = np.argsort(perm)
            self._structured = True

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _to_device(self, x: np.ndarray) -> np.ndarray:
        """Move a NumPy array to the compute backend."""
        if self._gpu:
            with cp.cuda.Device(self._device_id):
                return cp.asarray(x)
        return x

    def _to_numpy(self, x: object) -> np.ndarray:
        """Move an array back to host NumPy."""
        if self._gpu and isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
        return x  # type: ignore[return-value]

    def _rotate(self, x: np.ndarray) -> np.ndarray:
        """Apply random rotation along the last axis (head_dim)."""
        if self._structured:
            return (x * self._sign_flip)[..., self._perm]
        xp = cp if self._gpu else np
        return xp.einsum("...d,de->...e", x, self._Pi_T)

    def _unrotate(self, y: np.ndarray) -> np.ndarray:
        """Inverse rotation along the last axis."""
        if self._structured:
            return y[..., self._inv_perm] / self._sign_flip
        xp = cp if self._gpu else np
        return xp.einsum("...d,de->...e", y, self._Pi)

    # ------------------------------------------------------------------ #
    # Bit-packing                                                         #
    # ------------------------------------------------------------------ #

    def _pack_bits(self, indices: np.ndarray, *, bits: int | None = None) -> np.ndarray:
        """Pack *b*-bit indices into bytes.

        For 3-bit: 8 values into 3 bytes (24 bits = 8 x 3).
        For 2-bit: 4 values into 1 byte (8 bits  = 4 x 2).
        For 4-bit: 2 values into 1 byte (8 bits  = 2 x 4).

        Args:
            indices: Flat uint8 array of quantisation bin indices.
            bits: Override bit-width (for asymmetric K/V).

        Returns:
            Packed uint8 byte array.
        """
        if self._gpu:
            return self._pack_bits_gpu(indices, bits=bits)
        return self._pack_bits_cpu(indices, bits=bits)

    def _unpack_bits(
        self, packed: np.ndarray, n_values: int, *, bits: int | None = None
    ) -> np.ndarray:
        """Unpack bytes back to *b*-bit indices.

        Args:
            packed: Packed uint8 byte array from :meth:`_pack_bits`.
            n_values: Number of original index values (needed because
                the last group may have been zero-padded).
            bits: Override bit-width (for asymmetric K/V).

        Returns:
            Flat uint8 array of ``n_values`` indices.
        """
        if self._gpu:
            return self._unpack_bits_gpu(packed, n_values, bits=bits)
        return self._unpack_bits_cpu(packed, n_values, bits=bits)

    # -- CPU (NumPy) implementations ---------------------------------- #

    def _pack_bits_cpu(
        self, indices: np.ndarray, *, bits: int | None = None
    ) -> np.ndarray:
        """CPU bit-packing using NumPy vectorised operations."""
        bits = bits if bits is not None else self.bits
        flat = indices.ravel().astype(np.uint32)
        n = len(flat)

        if bits == 2:
            pad = (4 - n % 4) % 4
            if pad:
                flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint32)])
            flat = flat.reshape(-1, 4)
            packed = (
                flat[:, 0] | (flat[:, 1] << 2) | (flat[:, 2] << 4) | (flat[:, 3] << 6)
            )
            return packed.astype(np.uint8)

        elif bits == 3:
            pad = (8 - n % 8) % 8
            if pad:
                flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint32)])
            flat = flat.reshape(-1, 8)
            bits24 = (
                flat[:, 0]
                | (flat[:, 1] << 3)
                | (flat[:, 2] << 6)
                | (flat[:, 3] << 9)
                | (flat[:, 4] << 12)
                | (flat[:, 5] << 15)
                | (flat[:, 6] << 18)
                | (flat[:, 7] << 21)
            )
            b0 = (bits24 & 0xFF).astype(np.uint8)
            b1 = ((bits24 >> 8) & 0xFF).astype(np.uint8)
            b2 = ((bits24 >> 16) & 0xFF).astype(np.uint8)
            packed = np.column_stack([b0, b1, b2]).ravel()
            return packed

        elif bits == 4:
            pad = (2 - n % 2) % 2
            if pad:
                flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint32)])
            flat = flat.reshape(-1, 2)
            packed = flat[:, 0] | (flat[:, 1] << 4)
            return packed.astype(np.uint8)

        else:
            raise ValueError(f"Unsupported bits={bits} for packing")

    def _unpack_bits_cpu(
        self, packed: np.ndarray, n_values: int, *, bits: int | None = None
    ) -> np.ndarray:
        """CPU bit-unpacking using NumPy vectorised operations."""
        bits = bits if bits is not None else self.bits
        packed = packed.ravel()

        if bits == 2:
            b = packed.astype(np.uint32)
            v0 = b & 0x3
            v1 = (b >> 2) & 0x3
            v2 = (b >> 4) & 0x3
            v3 = (b >> 6) & 0x3
            out = np.column_stack([v0, v1, v2, v3]).ravel()
            return out[:n_values].astype(np.uint8)

        elif bits == 3:
            packed = packed.reshape(-1, 3)
            b0 = packed[:, 0].astype(np.uint32)
            b1 = packed[:, 1].astype(np.uint32)
            b2 = packed[:, 2].astype(np.uint32)
            bits24 = b0 | (b1 << 8) | (b2 << 16)
            v0 = bits24 & 0x7
            v1 = (bits24 >> 3) & 0x7
            v2 = (bits24 >> 6) & 0x7
            v3 = (bits24 >> 9) & 0x7
            v4 = (bits24 >> 12) & 0x7
            v5 = (bits24 >> 15) & 0x7
            v6 = (bits24 >> 18) & 0x7
            v7 = (bits24 >> 21) & 0x7
            out = np.column_stack([v0, v1, v2, v3, v4, v5, v6, v7]).ravel()
            return out[:n_values].astype(np.uint8)

        elif bits == 4:
            b = packed.astype(np.uint32)
            v0 = b & 0xF
            v1 = (b >> 4) & 0xF
            out = np.column_stack([v0, v1]).ravel()
            return out[:n_values].astype(np.uint8)

        else:
            raise ValueError(f"Unsupported bits={bits} for unpacking")

    # -- GPU (CuPy) implementations ----------------------------------- #

    def _pack_bits_gpu(
        self, indices: np.ndarray, *, bits: int | None = None
    ) -> np.ndarray:
        """GPU bit-packing using CuPy RawKernels."""
        bits = bits if bits is not None else self.bits
        xp = cp
        flat = indices.ravel()
        n = len(flat)

        if bits == 2:
            groups = (n + 3) // 4
            pad = groups * 4 - n
            if pad:
                flat = xp.concatenate([flat, xp.zeros(pad, dtype=xp.uint8)])
            packed = xp.empty(groups, dtype=xp.uint8)
            kernel = get_gpu_kernel("pack_2bit")
            threads = 256
            blocks = (groups + threads - 1) // threads
            kernel((blocks,), (threads,), (flat, packed, n))
            return packed

        elif bits == 3:
            groups = (n + 7) // 8
            pad = groups * 8 - n
            if pad:
                flat = xp.concatenate([flat, xp.zeros(pad, dtype=xp.uint8)])
            packed = xp.empty(groups * 3, dtype=xp.uint8)
            kernel = get_gpu_kernel("pack_3bit")
            threads = 256
            blocks = (groups + threads - 1) // threads
            kernel((blocks,), (threads,), (flat, packed, n))
            return packed

        elif bits == 4:
            groups = (n + 1) // 2
            pad = groups * 2 - n
            if pad:
                flat = xp.concatenate([flat, xp.zeros(pad, dtype=xp.uint8)])
            packed = xp.empty(groups, dtype=xp.uint8)
            kernel = get_gpu_kernel("pack_4bit")
            threads = 256
            blocks = (groups + threads - 1) // threads
            kernel((blocks,), (threads,), (flat, packed, n))
            return packed

        else:
            raise ValueError(f"Unsupported bits={bits} for GPU packing")

    def _unpack_bits_gpu(
        self, packed: np.ndarray, n_values: int, *, bits: int | None = None
    ) -> np.ndarray:
        """GPU bit-unpacking using CuPy RawKernels."""
        bits = bits if bits is not None else self.bits
        xp = cp

        if bits == 2:
            groups = len(packed)
            indices = xp.empty(groups * 4, dtype=xp.uint8)
            kernel = get_gpu_kernel("unpack_2bit")
            threads = 256
            blocks = (groups + threads - 1) // threads
            kernel((blocks,), (threads,), (packed, indices, n_values))
            return indices[:n_values]

        elif bits == 3:
            groups = len(packed) // 3
            indices = xp.empty(groups * 8, dtype=xp.uint8)
            kernel = get_gpu_kernel("unpack_3bit")
            threads = 256
            blocks = (groups + threads - 1) // threads
            kernel((blocks,), (threads,), (packed, indices, n_values))
            return indices[:n_values]

        elif bits == 4:
            groups = len(packed)
            indices = xp.empty(groups * 2, dtype=xp.uint8)
            kernel = get_gpu_kernel("unpack_4bit")
            threads = 256
            blocks = (groups + threads - 1) // threads
            kernel((blocks,), (threads,), (packed, indices, n_values))
            return indices[:n_values]

        else:
            raise ValueError(f"Unsupported bits={self.bits} for GPU unpacking")

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def _resolve_bits(self, kind: str | None) -> int:
        """Return the bit-width for a given tensor kind."""
        if kind == "key":
            return self.key_bits
        if kind == "value":
            return self.value_bits
        return self.bits

    def compress(
        self,
        tensor: np.ndarray,
        packed: bool = False,
        kind: str | None = None,
    ) -> CompressedKV:
        """Quantise a KV tensor to *b*-bit indices + per-vector norms.

        Args:
            tensor: KV cache tensor of shape (batch, n_heads, seq_len,
                head_dim) in any float dtype.
            packed: If True, bit-pack indices for maximum compression
                (e.g. ~5.1x for 3-bit instead of ~2x with uint8).
            kind: ``"key"`` or ``"value"`` to use asymmetric bit
                allocation.  ``None`` uses the default *bits*.

        Returns:
            A :class:`CompressedKV` container holding uint8 indices and
            float32 norms.
        """
        bits = self._resolve_bits(kind)
        centroids, boundaries = self._codebooks[bits]

        original_dtype = tensor.dtype
        xp = cp if self._gpu else np

        # Move to compute device
        x = self._to_device(tensor.astype(np.float32))

        # Per-vector L2 norms: shape (B, H, S)
        norms = xp.linalg.norm(x, axis=-1)

        # Normalise to unit vectors (avoid division by zero)
        safe_norms = xp.maximum(norms, 1e-30)[..., xp.newaxis]
        x_unit = x / safe_norms

        # Rotate + quantise -------------------------------------------------
        if self._gpu and not self._structured:
            # Fused GPU rotation + quantization (one kernel pass)
            orig_shape = x_unit.shape
            flat_unit = x_unit.reshape(-1, self.head_dim)
            flat_indices = gpu_batch_rotate_quantize(
                flat_unit, self._Pi_T, boundaries, bits
            )
            indices = flat_indices.reshape(orig_shape)
        elif self._gpu:
            # Structured rotation (element-wise, already fast) + GPU quantize
            x_rot = self._rotate(x_unit)
            indices = gpu_batch_quantize(x_rot, boundaries, bits)
        else:
            x_rot = self._rotate(x_unit)
            indices = xp.searchsorted(boundaries, x_rot).astype(xp.uint8)

        if packed:
            idx_shape = tuple(int(s) for s in indices.shape)
            n_values = int(np.prod(idx_shape))
            packed_indices = self._pack_bits(indices, bits=bits)
            return CompressedKV(
                indices=self._to_numpy(packed_indices),
                norms=self._to_numpy(norms.astype(xp.float32)),
                bits=bits,
                original_dtype=np.dtype(original_dtype),
                packed=True,
                n_values=n_values,
                shape=idx_shape,
            )

        return CompressedKV(
            indices=self._to_numpy(indices),
            norms=self._to_numpy(norms.astype(xp.float32)),
            bits=bits,
            original_dtype=np.dtype(original_dtype),
            packed=False,
            n_values=int(np.prod(indices.shape)),
            shape=tuple(int(s) for s in indices.shape),
        )

    def decompress(self, compressed: CompressedKV) -> np.ndarray:
        """Reconstruct an approximate KV tensor from compressed form.

        Automatically uses the correct codebook based on the bit-width
        stored in the :class:`CompressedKV`.

        Args:
            compressed: A :class:`CompressedKV` returned by
                :meth:`compress`.

        Returns:
            Reconstructed tensor of shape (B, H, S, D) in float32.
        """
        xp = cp if self._gpu else np
        bits = compressed.bits
        centroids, _boundaries = self._codebooks[bits]

        if compressed.packed:
            packed_dev = self._to_device(compressed.indices)
            flat_indices = self._unpack_bits(packed_dev, compressed.n_values, bits=bits)
            indices = flat_indices.reshape(compressed.shape)
        else:
            indices = self._to_device(compressed.indices)

        norms = self._to_device(compressed.norms)

        # Look up centroid values
        y_hat = centroids[indices]

        # Inverse rotation
        x_hat = self._unrotate(y_hat)

        # Scale by original norms
        x_hat = x_hat * norms[..., xp.newaxis]

        return self._to_numpy(x_hat)

    # ------------------------------------------------------------------ #
    # Auto-configuration                                                  #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_model(
        cls,
        model_name_or_config: str | dict,
        target: str = "balanced",
        use_gpu: bool = False,
        seed: int | None = None,
        **overrides,
    ) -> TurboQuantKV:
        """Create a TurboQuantKV with auto-tuned parameters for a model.

        Reads the model architecture (head_dim, n_kv_heads, RoPE config)
        and selects optimal compression settings based on the target
        preset.

        Args:
            model_name_or_config: A model name (e.g. ``"llama-3-8b"``),
                a HuggingFace path (e.g. ``"meta-llama/Llama-3-8B"``),
                or a config dict with HF-style keys.
            target: Compression target — ``"quality"``, ``"balanced"``,
                ``"compression"``, or ``"extreme"``.
            use_gpu: Use GPU acceleration.
            seed: Random seed.
            **overrides: Override any auto-selected parameter (e.g.
                ``key_bits=3``).

        Returns:
            Configured TurboQuantKV instance.

        Examples::

            # One-liner: auto-detect everything
            tq = TurboQuantKV.from_model("llama-3-8b")

            # Target maximum compression
            tq = TurboQuantKV.from_model("gemma-2-27b", target="compression")

            # Override specific settings
            tq = TurboQuantKV.from_model("mistral-7b", key_bits=3)
        """
        from .autoconfig import AutoConfig

        if isinstance(model_name_or_config, dict):
            cfg = AutoConfig.from_dict(model_name_or_config, target=target, **overrides)
        else:
            cfg = AutoConfig.from_pretrained(
                model_name_or_config, target=target, **overrides
            )
        return cfg.build_quantizer(use_gpu=use_gpu, seed=seed)

    # ------------------------------------------------------------------ #
    # Convenience methods                                                 #
    # ------------------------------------------------------------------ #

    def compression_ratio(self, packed: bool = False) -> float:
        """Compression ratio for fp16 originals.

        When ``packed=False`` (legacy): stores one uint8 per coordinate
        plus a float32 norm per vector.  For fp16 originals this gives
        ~2x.

        When ``packed=True``: stores *b* bits per coordinate (bit-packed)
        plus a float32 norm per vector.  For 3-bit at head_dim=256 this
        gives ~5.1x.

        Args:
            packed: If True, calculate ratio with bit-packing.

        Returns:
            Compression ratio (original / compressed).
        """
        bytes_per_element_original = 2.0  # fp16
        if packed:
            bytes_per_element_compressed = self.bits / 8.0 + 4.0 / self.head_dim
        else:
            bytes_per_element_compressed = 1.0 + 4.0 / self.head_dim
        return bytes_per_element_original / bytes_per_element_compressed

    def theoretical_compression_ratio(self) -> float:
        """Theoretical compression ratio with bit-packing.

        If indices were bit-packed (b bits per coordinate instead of
        8 bits), the ratio for fp16 originals would be much higher.

        Returns:
            Theoretical compression ratio assuming ideal bit-packing.
        """
        bits_per_element_compressed = self.bits + 32.0 / self.head_dim
        bits_per_element_original = 16.0  # fp16
        return bits_per_element_original / bits_per_element_compressed

    @staticmethod
    def estimate_memory(
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        seq_len: int,
        bits: int = 3,
        key_bits: int | None = None,
        value_bits: int | None = None,
        original_dtype: str = "float16",
        bit_packed: bool = False,
    ) -> dict[str, float]:
        """Estimate KV cache memory for a given model configuration.

        Supports asymmetric K/V bit allocation via *key_bits* and
        *value_bits*.

        Args:
            n_layers: Number of transformer layers.
            n_kv_heads: Number of key/value heads per layer.
            head_dim: Dimension per head.
            seq_len: Sequence length (context window).
            bits: Default quantisation bits (2, 3, or 4).
            key_bits: Quantisation bits for keys (overrides *bits*).
            value_bits: Quantisation bits for values (overrides *bits*).
            original_dtype: Original storage dtype ("float16" or "float32").
            bit_packed: If True, estimate with ideal bit-packing
                (b bits per coordinate).  If False (default), estimate
                with uint8 storage (1 byte per coordinate).

        Returns:
            Dict with keys ``original_gb``, ``compressed_gb``,
            ``ratio``, ``saved_gb``.
        """
        kb = key_bits if key_bits is not None else bits
        vb = value_bits if value_bits is not None else bits
        dtype_bytes = 2 if original_dtype == "float16" else 4

        n_elements_per_tensor = n_layers * n_kv_heads * seq_len * head_dim
        n_vectors_per_tensor = n_layers * n_kv_heads * seq_len

        original_bytes = 2 * n_elements_per_tensor * dtype_bytes

        if bit_packed:
            key_bytes = (n_elements_per_tensor * kb + 7) // 8 + n_vectors_per_tensor * 4
            val_bytes = (n_elements_per_tensor * vb + 7) // 8 + n_vectors_per_tensor * 4
        else:
            key_bytes = n_elements_per_tensor + n_vectors_per_tensor * 4
            val_bytes = n_elements_per_tensor + n_vectors_per_tensor * 4

        compressed_bytes = key_bytes + val_bytes
        original_gb = original_bytes / (1024**3)
        compressed_gb = compressed_bytes / (1024**3)

        return {
            "original_gb": round(original_gb, 3),
            "compressed_gb": round(compressed_gb, 3),
            "ratio": round(original_gb / max(compressed_gb, 1e-9), 2),
            "saved_gb": round(original_gb - compressed_gb, 3),
        }


# ------------------------------------------------------------------ #
# Streaming KV cache with tiered hot/cold storage                     #
# ------------------------------------------------------------------ #


class TurboQuantKVCache:
    """Manages a growing KV cache during autoregressive inference.

    Implements a two-tier memory hierarchy:

    - **L1 (hot window)**: The most recent ``hot_window`` tokens are
      stored uncompressed in VRAM for zero-latency attention.
    - **L2 (cold storage)**: Older tokens are bit-packed at *b*-bit
      precision, achieving ~5x compression.  Decompressed on demand
      when the attention window extends into cold storage.

    This is designed to sit between the model's attention mechanism
    and VRAM, transparently compressing the KV cache as context grows.

    Usage::

        cache = TurboQuantKVCache(head_dim=256, n_heads=16, bits=3)
        for token in tokens:
            k, v = model.forward_one(token)
            cache.append(k, v)        # auto-compresses old entries
            keys = cache.get_keys(0, cache.length)
            values = cache.get_values(0, cache.length)
            # ... use keys/values for attention ...

    Args:
        head_dim: Dimension of each attention head.
        n_heads: Number of key/value heads.
        bits: Quantisation width (2, 3, or 4).
        hot_window: Number of recent tokens kept uncompressed.
        use_gpu: Whether to use CuPy for compression.
        device_id: CUDA device ordinal.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        head_dim: int = 128,
        n_heads: int = 32,
        bits: int = 3,
        key_bits: int | None = None,
        value_bits: int | None = None,
        hot_window: int = 512,
        use_gpu: bool = True,
        device_id: int = 0,
        seed: int | None = None,
    ) -> None:
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.bits = bits
        self.key_bits = key_bits if key_bits is not None else bits
        self.value_bits = value_bits if value_bits is not None else bits
        self.hot_window = hot_window

        # The underlying stateless compressor
        self._tq = TurboQuantKV(
            head_dim=head_dim,
            n_heads=n_heads,
            bits=bits,
            key_bits=key_bits,
            value_bits=value_bits,
            use_gpu=use_gpu,
            device_id=device_id,
            seed=seed,
        )

        # L1 hot buffers: list of (key, value) tensors each of shape
        # (1, n_heads, 1, head_dim) -- one per token
        self._hot_keys: list[np.ndarray] = []
        self._hot_values: list[np.ndarray] = []

        # L2 cold storage: list of CompressedKV chunks
        self._cold_keys: list[CompressedKV] = []
        self._cold_values: list[CompressedKV] = []
        self._cold_lengths: list[int] = []  # seq_len per cold chunk

        # Total tokens stored
        self._total_len: int = 0

    @property
    def length(self) -> int:
        """Total number of tokens in the cache (cold + hot)."""
        return self._total_len

    @property
    def cold_length(self) -> int:
        """Number of tokens in cold (compressed) storage."""
        return sum(self._cold_lengths)

    @property
    def hot_length(self) -> int:
        """Number of tokens in the hot (uncompressed) window."""
        return len(self._hot_keys)

    def append(self, key: np.ndarray, value: np.ndarray) -> None:
        """Add a new KV pair for one token.

        If the hot window exceeds ``hot_window`` tokens, the oldest
        entries are compressed and moved to cold storage.

        Args:
            key: Key tensor of shape ``(1, n_heads, 1, head_dim)`` or
                ``(n_heads, head_dim)``.
            value: Value tensor with the same shape convention.
        """
        # Normalise to (1, n_heads, 1, head_dim)
        if key.ndim == 2:
            key = key[np.newaxis, :, np.newaxis, :]
        if value.ndim == 2:
            value = value[np.newaxis, :, np.newaxis, :]

        self._hot_keys.append(key)
        self._hot_values.append(value)
        self._total_len += 1

        # Check if we need to flush hot window to cold storage
        if len(self._hot_keys) > self.hot_window:
            self._flush_to_cold()

    def _flush_to_cold(self) -> None:
        """Compress half of the hot window and move to cold storage."""
        n_flush = len(self._hot_keys) // 2
        if n_flush == 0:
            return

        # Concatenate along seq_len axis -> (1, n_heads, n_flush, head_dim)
        keys_to_flush = np.concatenate(self._hot_keys[:n_flush], axis=2)
        values_to_flush = np.concatenate(self._hot_values[:n_flush], axis=2)

        # Compress with bit-packing (asymmetric K/V bits)
        compressed_k = self._tq.compress(keys_to_flush, packed=True, kind="key")
        compressed_v = self._tq.compress(values_to_flush, packed=True, kind="value")

        self._cold_keys.append(compressed_k)
        self._cold_values.append(compressed_v)
        self._cold_lengths.append(n_flush)

        # Remove flushed entries from hot buffers
        self._hot_keys = self._hot_keys[n_flush:]
        self._hot_values = self._hot_values[n_flush:]

    def get_keys(self, start: int, end: int) -> np.ndarray:
        """Get key tensors for token positions ``[start, end)``.

        Decompresses cold-storage entries on demand.

        Args:
            start: Start position (inclusive).
            end: End position (exclusive).

        Returns:
            Key tensor of shape ``(1, n_heads, end-start, head_dim)``.
        """
        return self._get_range(self._cold_keys, self._hot_keys, start, end)

    def get_values(self, start: int, end: int) -> np.ndarray:
        """Get value tensors for token positions ``[start, end)``.

        Decompresses cold-storage entries on demand.

        Args:
            start: Start position (inclusive).
            end: End position (exclusive).

        Returns:
            Value tensor of shape ``(1, n_heads, end-start, head_dim)``.
        """
        return self._get_range(self._cold_values, self._hot_values, start, end)

    def _get_range(
        self,
        cold_chunks: list[CompressedKV],
        hot_entries: list[np.ndarray],
        start: int,
        end: int,
    ) -> np.ndarray:
        """Retrieve a range of KV tensors spanning cold and hot storage."""
        parts: list[np.ndarray] = []
        cold_total = self.cold_length
        pos = 0

        # Walk through cold chunks
        for i, chunk_len in enumerate(self._cold_lengths):
            chunk_start = pos
            chunk_end = pos + chunk_len

            if end <= chunk_start:
                break
            if start < chunk_end:
                local_start = max(0, start - chunk_start)
                local_end = min(chunk_len, end - chunk_start)
                decompressed = self._tq.decompress(cold_chunks[i])
                parts.append(decompressed[:, :, local_start:local_end, :])

            pos = chunk_end

        # Walk through hot entries
        hot_global_start = cold_total
        for j, entry in enumerate(hot_entries):
            entry_pos = hot_global_start + j
            if entry_pos >= end:
                break
            if entry_pos >= start:
                parts.append(entry.astype(np.float32))

        if not parts:
            return np.zeros((1, self.n_heads, 0, self.head_dim), dtype=np.float32)

        return np.concatenate(parts, axis=2)

    def memory_stats(self) -> dict[str, float]:
        """Return memory usage statistics in bytes.

        Returns:
            Dict with ``cold_bytes``, ``hot_bytes``, ``total_bytes``,
            ``uncompressed_equivalent_bytes``, ``effective_ratio``.
        """
        cold_bytes = sum(c.nbytes() for c in self._cold_keys) + sum(
            c.nbytes() for c in self._cold_values
        )

        hot_bytes = sum(
            k.nbytes + v.nbytes
            for k, v in zip(self._hot_keys, self._hot_values, strict=False)
        )

        total_bytes = cold_bytes + hot_bytes

        # What it would cost uncompressed (float32)
        uncompressed = (
            self._total_len * self.n_heads * self.head_dim * 4 * 2  # float32  # K + V
        )

        return {
            "cold_bytes": float(cold_bytes),
            "hot_bytes": float(hot_bytes),
            "total_bytes": float(total_bytes),
            "uncompressed_equivalent_bytes": float(uncompressed),
            "effective_ratio": float(uncompressed) / max(total_bytes, 1),
        }

    def clear(self) -> None:
        """Reset the cache to empty state."""
        self._hot_keys.clear()
        self._hot_values.clear()
        self._cold_keys.clear()
        self._cold_values.clear()
        self._cold_lengths.clear()
        self._total_len = 0

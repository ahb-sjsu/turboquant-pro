# TurboQuant Pro: per-channel KV quantizer for *keys*.
# Copyright (c) 2026 Andrew H. Bond. MIT License.
"""Per-channel quantizer for KV-cache **keys**.

TurboQuant's :class:`~turboquant_pro.core.TurboQuantKV` (PolarQuant: random rotation +
per-vector L2 normalization + Lloyd-Max codebook) is excellent for **values** but
**catastrophic for keys**. It preserves each key vector's *norm* and quantizes its
*direction*, discarding the per-channel scale structure that attention's
``softmax(Q @ K^T)`` depends on. Measured on Qwen2.5 (post-RoPE keys, perplexity):

    PolarQuant keys (K4):   ppl ~ 10^4   (recon 0.095)   <- generation destroyed
    per-channel keys (K4):  ppl ~ 15     (recon 0.062)   <- near-fp16

Note reconstruction error is *anti-correlated* with perplexity, so cosine-similarity
benchmarks cannot detect the failure -- only generation (perplexity) can.

``PerChannelKV`` quantizes each head-dim **channel** with its own asymmetric scale
computed over the token axis (optionally non-uniform / NUQ). This is the KIVI/KVQuant
insight, packaged for TurboQuant's key path. Use it for keys; keep ``TurboQuantKV`` for
values (see :class:`~turboquant_pro.core.TurboQuantKVCache`, which wires both).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_SUPPORTED_BITS = (2, 3, 4)


def _pack_indices(idx: np.ndarray, bits: int) -> np.ndarray:
    """Bit-pack a uint8 index array (values < 2**bits) into a flat byte array."""
    flat = np.ascontiguousarray(idx).reshape(-1).astype(np.uint8)
    bitmat = np.unpackbits(flat[:, None], axis=1, bitorder="little")[
        :, :bits
    ]  # LSB-first low bits
    return np.packbits(bitmat.reshape(-1))


def _unpack_indices(packed: np.ndarray, n_values: int, bits: int) -> np.ndarray:
    """Inverse of :func:`_pack_indices`."""
    bitstream = np.unpackbits(packed)[: n_values * bits].reshape(n_values, bits)
    weights = 1 << np.arange(bits, dtype=np.uint16)
    return (bitstream.astype(np.uint16) * weights).sum(axis=1).astype(np.uint8)


@dataclass
class CompressedPerChannelKV:
    """Container for a per-channel-quantized key tensor."""

    indices: np.ndarray  # uint8 (packed bytes if `packed`, else (B,H,S,D))
    scale: np.ndarray  # float32 (B,H,1,D) -- per channel (None when nuq)
    zero: np.ndarray  # float32 (B,H,1,D) -- per channel min (None when nuq)
    bits: int
    shape: tuple[int, ...]  # original (B,H,S,D)
    packed: bool = False
    levels: np.ndarray | None = None  # float32 (B,H,D,2**bits) when non-uniform
    original_dtype: np.dtype = field(default_factory=lambda: np.dtype("float32"))

    def nbytes(self) -> int:
        n = self.indices.nbytes + (self.scale.nbytes if self.scale is not None else 0)
        n += self.zero.nbytes if self.zero is not None else 0
        n += self.levels.nbytes if self.levels is not None else 0
        return n

    def compression_ratio(self, head_dim: int) -> float:
        orig = int(np.prod(self.shape)) * self.original_dtype.itemsize
        return orig / max(self.nbytes(), 1)


class PerChannelKV:
    """Per-channel asymmetric (optionally non-uniform) quantizer for KV keys.

    Args:
        head_dim: per-head dimension D.
        n_heads:  number of KV heads (informational; inferred from the input shape).
        bits:     2, 3, or 4.
        nuq:      if True, use per-channel non-uniform (quantile) levels instead
                  of uniform -- buys ~1 bit of quality (KVQuant-style).
    """

    def __init__(
        self, head_dim: int = 128, n_heads: int = 32, bits: int = 4, nuq: bool = False
    ):
        if bits not in _SUPPORTED_BITS:
            raise ValueError(f"bits must be one of {_SUPPORTED_BITS}, got {bits}")
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.bits = bits
        self.nuq = nuq
        self.qmax = 2**bits - 1

    def compress(self, x: np.ndarray, packed: bool = False) -> CompressedPerChannelKV:
        """Compress a ``(B, H, S, D)`` key tensor (per-channel over tokens)."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError(f"expected (B,H,S,D), got shape {x.shape}")
        shape = x.shape
        if not self.nuq:
            mn = x.min(axis=2, keepdims=True)
            mx = x.max(axis=2, keepdims=True)
            scale = np.maximum(mx - mn, 1e-8) / self.qmax
            idx = np.clip(np.round((x - mn) / scale), 0, self.qmax).astype(np.uint8)
            packed_idx = _pack_indices(idx, self.bits) if packed else idx
            return CompressedPerChannelKV(
                packed_idx,
                scale.astype(np.float32),
                mn.astype(np.float32),
                self.bits,
                shape,
                packed,
            )
        B, H, S, D = shape
        qs = np.linspace(0.0, 1.0, 2**self.bits, dtype=np.float32)
        cent = np.moveaxis(np.quantile(x, qs, axis=2), 0, -1).astype(
            np.float32
        )  # (B,H,D,levels)
        xe = np.moveaxis(x, 2, -1)  # (B,H,D,S)
        idx_bhds = (
            np.abs(xe[..., None, :] - cent[..., :, None])
            .argmin(axis=-2)
            .astype(np.uint8)
        )
        idx = np.moveaxis(idx_bhds, -1, 2)  # (B,H,S,D)
        packed_idx = _pack_indices(idx, self.bits) if packed else idx
        return CompressedPerChannelKV(
            packed_idx, None, None, self.bits, shape, packed, levels=cent
        )

    def decompress(self, c: CompressedPerChannelKV) -> np.ndarray:
        B, H, S, D = c.shape
        if c.packed:
            idx = _unpack_indices(c.indices, B * H * S * D, c.bits).reshape(B, H, S, D)
        else:
            idx = c.indices
        if c.levels is None:
            return idx.astype(np.float32) * c.scale + c.zero
        idx_bhds = np.moveaxis(idx, 2, -1)  # (B,H,D,S)
        out = np.take_along_axis(c.levels, idx_bhds, axis=-1)  # (B,H,D,S)
        return np.moveaxis(out, -1, 2).astype(np.float32)

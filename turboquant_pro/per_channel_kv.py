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
computed over the token axis. Three calibration-free quality boosters (the KVQuant
recipe, *without* its Fisher/K-means calibration) are available:

* ``nuq`` / ``nf4`` -- **non-uniform** levels (per-channel quantiles, or NormalFloat-4).
* ``outlier_frac`` -- **dense-and-sparse**: the top ``outlier_frac`` magnitude entries
  per channel are kept in fp16 (the outlier *key channels* dominate attention; uniform
  4-bit crushes them). **2% is the sweet spot** (3% regresses slightly + costs storage).

Measured on LongBench (Llama-2-7B-chat, full 200-sample splits, qasper is the
outlier-sensitive task):

    fp16                                         qasper 22.06
    KVQuant nuq4-1% (Fisher + K-means)           qasper 21.06
    per-channel uniform 4-bit                    qasper 14.38   <- outlier channels lost
    per-channel NF4 + 2% outliers + sink         qasper 20.82   <- calibration-free

Use this for keys; keep ``TurboQuantKV`` for values (see
:class:`~turboquant_pro.core.TurboQuantKVCache`, which wires both).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_SUPPORTED_BITS = (2, 3, 4)

# 4-bit NormalFloat (NF4) levels (QLoRA). Calibration-free non-uniform codebook,
# scaled per channel by abs-max. Only used when ``nf4=True`` (implies 4-bit).
_NF4 = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=np.float32,
)


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
    scale: np.ndarray  # float32 (B,H,1,D) -- per channel (None when nuq/nf4)
    zero: np.ndarray  # float32 (B,H,1,D) -- per channel min (None when nuq/nf4)
    bits: int
    shape: tuple[int, ...]  # original (B,H,S,D)
    packed: bool = False
    levels: np.ndarray | None = None  # float32 (B,H,D,2**bits) for data-fit (nuq) levels
    # compact NF4 / asym-NF4: store per-channel scalars, rebuild the fixed grid on decode.
    nf4_scale: np.ndarray | None = None  # float32 (B,H,D) abs-max of (centered) channel
    nf4_mean: np.ndarray | None = None  # float32 (B,H,D) per-channel mean (asym only)
    original_dtype: np.dtype = field(default_factory=lambda: np.dtype("float32"))
    # dense-and-sparse outliers (kept fp16); None when outlier_frac == 0
    outlier_idx: np.ndarray | None = None  # int32 flat indices into (B,H,S,D)
    outlier_val: np.ndarray | None = None  # float16 original values at those indices

    def nbytes(self) -> int:
        n = self.indices.nbytes + (self.scale.nbytes if self.scale is not None else 0)
        n += self.zero.nbytes if self.zero is not None else 0
        n += self.levels.nbytes if self.levels is not None else 0
        n += self.nf4_scale.nbytes if self.nf4_scale is not None else 0
        n += self.nf4_mean.nbytes if self.nf4_mean is not None else 0
        n += self.outlier_idx.nbytes if self.outlier_idx is not None else 0
        n += self.outlier_val.nbytes if self.outlier_val is not None else 0
        return n

    def compression_ratio(self, head_dim: int) -> float:
        orig = int(np.prod(self.shape)) * self.original_dtype.itemsize
        return orig / max(self.nbytes(), 1)


class PerChannelKV:
    """Per-channel asymmetric (optionally non-uniform / dense-sparse) key quantizer.

    Args:
        head_dim: per-head dimension D.
        n_heads:  number of KV heads (informational; inferred from the input shape).
        bits:     2, 3, or 4.
        nuq:      per-channel non-uniform (quantile) levels instead of uniform.
        nf4:      use NormalFloat-4 levels (calibration-free; implies 4-bit). Overrides
                  ``nuq``. Most of the non-uniform benefit with a fixed codebook.
        nf4_asym: **asymmetric / zero-point NF4** (implies ``nf4``). Subtract the
                  per-channel mean before applying the NF4 grid, so the codebook is
                  centered on the data instead of on zero. Symmetric NF4 (``nf4=True``)
                  scales by abs-max about 0 and **silently collapses on models whose KV
                  has a large DC offset amplified by high-ratio GQA** (e.g. Qwen2.5-7B:
                  qasper 43.8 fp16 -> 4.7 with symmetric NF4 -> 41.9 with nf4_asym).
                  ``nf4_asym`` keeps NF4's nonlinear levels (so it ties NF4 on Llama /
                  Mistral) AND absorbs the offset (so it does not collapse on Qwen). It
                  is the **recommended robust default** for keys; see the cross-model
                  matrix in ``benchmarks/kvquant_matrix/``.
        outlier_frac: fraction (e.g. ``0.01``) of highest-magnitude entries **per
                  channel** to keep in fp16 (dense-and-sparse). Fixes the outlier
                  channel loss that uniform low-bit quantization inflicts on keys.
    """

    def __init__(
        self,
        head_dim: int = 128,
        n_heads: int = 32,
        bits: int = 4,
        nuq: bool = False,
        nf4: bool = False,
        nf4_asym: bool = False,
        outlier_frac: float = 0.0,
    ):
        if bits not in _SUPPORTED_BITS:
            raise ValueError(f"bits must be one of {_SUPPORTED_BITS}, got {bits}")
        if nf4_asym:
            nf4 = True  # asymmetric NF4 implies the NF4 codebook
        if nf4 and bits != 4:
            raise ValueError("nf4=True requires bits=4")
        if not (0.0 <= outlier_frac < 1.0):
            raise ValueError("outlier_frac must be in [0, 1)")
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.bits = bits
        self.nuq = nuq or nf4
        self.nf4 = nf4
        self.nf4_asym = nf4_asym
        self.outlier_frac = outlier_frac
        self.qmax = 2**bits - 1

    def _outliers(self, x: np.ndarray):
        """Per-channel top-``outlier_frac`` magnitude indices/values (dense-sparse)."""
        if self.outlier_frac <= 0.0:
            return None, None
        B, H, S, D = x.shape
        k = max(1, int(round(S * self.outlier_frac)))
        if k >= S:
            mask = np.ones(x.shape, dtype=bool)
        else:
            absx = np.abs(x)
            kth = np.partition(absx, S - k, axis=2)[:, :, S - k : S - k + 1, :]
            mask = absx >= kth
        idx = np.flatnonzero(mask).astype(np.int32)
        val = x.reshape(-1)[idx].astype(np.float16)
        return idx, val

    def compress(self, x: np.ndarray, packed: bool = False) -> CompressedPerChannelKV:
        """Compress a ``(B, H, S, D)`` key tensor (per-channel over tokens)."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError(f"expected (B,H,S,D), got shape {x.shape}")
        shape = x.shape
        B, H, S, D = shape
        outlier_idx, outlier_val = self._outliers(x)

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
                outlier_idx=outlier_idx,
                outlier_val=outlier_val,
            )

        # non-uniform: per-channel level table (B,H,D,levels). For NF4/asym-NF4 the grid
        # is fixed, so we keep only the per-channel scalars (amax, +mean) and rebuild it.
        nf4_scale = nf4_mean = None
        if self.nf4 and self.nf4_asym:
            # asymmetric / zero-point NF4: center per channel, NF4 the residual, so the
            # codebook tracks the DC offset instead of wasting half its codes about zero.
            mu = x.mean(axis=2)  # (B,H,D) per-channel DC offset
            xc = x - mu[:, :, None, :]
            amax = np.maximum(np.abs(xc).max(axis=2), 1e-8)  # (B,H,D)
            cent = (mu[..., None] + amax[..., None] * _NF4[None, None, None, :]).astype(
                np.float32
            )
            nf4_scale, nf4_mean = amax.astype(np.float32), mu.astype(np.float32)
        elif self.nf4:
            amax = np.maximum(np.abs(x).max(axis=2), 1e-8)  # (B,H,D)
            cent = (amax[..., None] * _NF4[None, None, None, :]).astype(np.float32)
            nf4_scale = amax.astype(np.float32)
        else:
            qs = np.linspace(0.0, 1.0, 2**self.bits, dtype=np.float32)
            cent = np.moveaxis(np.quantile(x, qs, axis=2), 0, -1).astype(np.float32)
        xe = np.moveaxis(x, 2, -1)  # (B,H,D,S)
        idx_bhds = (
            np.abs(xe[..., None, :] - cent[..., :, None])
            .argmin(axis=-2)
            .astype(np.uint8)
        )
        idx = np.moveaxis(idx_bhds, -1, 2)  # (B,H,S,D)
        packed_idx = _pack_indices(idx, self.bits) if packed else idx
        return CompressedPerChannelKV(
            packed_idx,
            None,
            None,
            self.bits,
            shape,
            packed,
            # NF4 stores compact scalars; quantile (nuq) keeps the full data-fit table.
            levels=None if self.nf4 else cent,
            nf4_scale=nf4_scale,
            nf4_mean=nf4_mean,
            outlier_idx=outlier_idx,
            outlier_val=outlier_val,
        )

    def decompress(self, c: CompressedPerChannelKV) -> np.ndarray:
        B, H, S, D = c.shape
        if c.packed:
            idx = _unpack_indices(c.indices, B * H * S * D, c.bits).reshape(B, H, S, D)
        else:
            idx = c.indices
        if c.nf4_scale is not None:
            # rebuild the fixed NF4 grid from per-channel scalars: (B,H,D,levels)
            cent = c.nf4_scale[..., None] * _NF4[None, None, None, :]
            if c.nf4_mean is not None:  # asymmetric / zero-point
                cent = cent + c.nf4_mean[..., None]
            idx_bhds = np.moveaxis(idx, 2, -1)  # (B,H,D,S)
            taken = np.take_along_axis(cent.astype(np.float32), idx_bhds, axis=-1)
            out = np.moveaxis(taken, -1, 2).astype(np.float32)
        elif c.levels is None:
            out = idx.astype(np.float32) * c.scale + c.zero
        else:
            idx_bhds = np.moveaxis(idx, 2, -1)  # (B,H,D,S)
            taken = np.take_along_axis(c.levels, idx_bhds, axis=-1)  # (B,H,D,S)
            out = np.moveaxis(taken, -1, 2).astype(np.float32)
        if c.outlier_idx is not None:
            flat = out.reshape(-1)
            flat[c.outlier_idx] = c.outlier_val.astype(np.float32)
            out = flat.reshape(B, H, S, D)
        return out

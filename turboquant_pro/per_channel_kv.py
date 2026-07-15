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
* ``zero_point`` -- how the asym-NF4 zero-point is obtained. The key DC offset is
  RoPE-frequency-structured (96--99% of its mass in channels whose rotary wavelength
  exceeds the window; ``benchmarks/RESULTS_rope_offsets.md``), which validates two
  calibration-lean modes measured to *beat* dense calibration on LongBench-qasper
  (Qwen2.5-7B, 200 samples: calibrated 42.35, ``"sparse"`` 42.66, ``"bias"`` 43.36;
  fp16 43.77; symmetric NF4 4.69):

  - ``"calibrated"`` (default) -- per-channel mean of the block (stored, fp32/channel).
  - ``"sparse"``     -- calibrated mean only on the config-identified DC channels
                        (wavelength > block length); stores ~1/3 less zero-point
                        metadata, needs ``rope_theta``.
  - ``"bias"``       -- the model's ``k_proj`` bias pushed through the position-averaged
                        rotation: **zero calibration data and zero stored zero-point
                        metadata** (recomputed at decode from the instance's ``k_bias``
                        and ``rope_theta``). Requires a model with key-projection biases
                        (the Qwen family); bias-free models keep the other modes.

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
    levels: np.ndarray | None = None  # float32 (B,H,D,2**bits) data-fit (nuq) levels
    # compact NF4 / asym-NF4: store per-channel scalars, rebuild fixed grid on decode
    nf4_scale: np.ndarray | None = None  # float32 (B,H,D) abs-max of (centered) channel
    nf4_mean: np.ndarray | None = None  # float32 (B,H,D) per-channel mean (asym only)
    original_dtype: np.dtype = field(default_factory=lambda: np.dtype("float32"))
    # dense-and-sparse outliers (kept fp16); None when outlier_frac == 0
    outlier_idx: np.ndarray | None = None  # int32 flat indices into (B,H,S,D)
    outlier_val: np.ndarray | None = None  # float16 original values at those indices
    # asym-NF4 zero-point mode metadata (see PerChannelKV ``zero_point``):
    #   "calibrated" -- nf4_mean is the full (B,H,D) array (legacy layout);
    #   "sparse"     -- nf4_mean holds only the DC channels (B,H,n_dc); the mask is
    #                   rebuilt at decode from rope_theta and the block length;
    #   "bias"       -- nf4_mean is None; the zero-point is recomputed at decode from
    #                   the quantizer's k_bias/rope_theta and position_start.
    zp_mode: str = "calibrated"
    rope_theta: float | None = None
    position_start: int = 0

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
        zero_point: asym-NF4 zero-point mode: ``"calibrated"`` (default),
                  ``"sparse"`` (calibrated mean on config-identified DC channels
                  only), or ``"bias"`` (RoPE-averaged ``k_bias``; zero calibration,
                  zero stored zero-point metadata). Non-default modes require
                  ``nf4_asym=True``; see the module docstring for measured quality.
        rope_theta: the model's RoPE base (e.g. ``1e6`` for Qwen2.5, ``1e4`` for
                  Llama/Mistral). Required by ``"sparse"`` and ``"bias"``.
        k_bias:   the model's key-projection bias, shape ``(n_heads * head_dim,)``
                  or ``(n_heads, head_dim)``. Required by ``"bias"``.
    """

    _ZP_MODES = ("calibrated", "sparse", "bias")

    def __init__(
        self,
        head_dim: int = 128,
        n_heads: int = 32,
        bits: int = 4,
        nuq: bool = False,
        nf4: bool = False,
        nf4_asym: bool = False,
        outlier_frac: float = 0.0,
        zero_point: str = "calibrated",
        rope_theta: float | None = None,
        k_bias: np.ndarray | None = None,
    ):
        if bits not in _SUPPORTED_BITS:
            raise ValueError(f"bits must be one of {_SUPPORTED_BITS}, got {bits}")
        if nf4_asym:
            nf4 = True  # asymmetric NF4 implies the NF4 codebook
        if nf4 and bits != 4:
            raise ValueError("nf4=True requires bits=4")
        if not (0.0 <= outlier_frac < 1.0):
            raise ValueError("outlier_frac must be in [0, 1)")
        if zero_point not in self._ZP_MODES:
            raise ValueError(f"zero_point must be one of {self._ZP_MODES}")
        if zero_point != "calibrated":
            if not nf4_asym:
                raise ValueError(f"zero_point={zero_point!r} requires nf4_asym=True")
            if rope_theta is None:
                raise ValueError(f"zero_point={zero_point!r} requires rope_theta")
        if zero_point == "bias":
            if k_bias is None:
                raise ValueError('zero_point="bias" requires k_bias')
            k_bias = np.asarray(k_bias, dtype=np.float32).reshape(n_heads, head_dim)
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.bits = bits
        self.nuq = nuq or nf4
        self.nf4 = nf4
        self.nf4_asym = nf4_asym
        self.outlier_frac = outlier_frac
        self.zero_point = zero_point
        self.rope_theta = rope_theta
        self.k_bias = k_bias
        self.qmax = 2**bits - 1

    # ------------------------------------------------------------------ #
    # RoPE geometry helpers (config-only; rotate_half convention)          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def dc_channel_mask(rope_theta: float, head_dim: int, window: int) -> np.ndarray:
        """Boolean ``(head_dim,)`` mask of channels whose rotary wavelength exceeds
        ``window`` -- the channels that carry the key DC offset (96--99% of its
        mass, measured). Derived from the config alone."""
        half = head_dim // 2
        inv = rope_theta ** (-2.0 * np.arange(half) / head_dim)
        wl = 2.0 * np.pi / inv
        return np.tile(wl, 2)[:head_dim] > float(window)

    @staticmethod
    def rope_averaged_bias(
        k_bias: np.ndarray,
        rope_theta: float,
        head_dim: int,
        start: int,
        n: int,
    ) -> np.ndarray:
        """Position-averaged rotation of ``k_bias`` over ``[start, start + n)``.

        Returns ``(n_heads, head_dim)`` float32: the deterministic asym-NF4
        zero-point (validated: matches calibrated means on WikiText-2 ppl and
        beats them on LongBench-qasper). ``k_bias`` is ``(n_heads, head_dim)``.
        """
        half = head_dim // 2
        inv = rope_theta ** (-2.0 * np.arange(half, dtype=np.float64) / head_dim)
        pos = np.arange(start, start + n, dtype=np.float64)
        ang = pos[:, None] * inv[None, :]
        mc = np.cos(ang).mean(axis=0)
        ms = np.sin(ang).mean(axis=0)
        c_full = np.concatenate([mc, mc]).astype(np.float32)
        s_full = np.concatenate([ms, ms]).astype(np.float32)
        b = np.asarray(k_bias, dtype=np.float32)
        rh = np.concatenate([-b[:, half:], b[:, :half]], axis=1)  # rotate_half(b)
        return b * c_full[None, :] + rh * s_full[None, :]

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

    def compress(
        self,
        x: np.ndarray,
        packed: bool = False,
        position_start: int = 0,
    ) -> CompressedPerChannelKV:
        """Compress a ``(B, H, S, D)`` key tensor (per-channel over tokens).

        ``position_start`` is the absolute position of the block's first token
        (0 for a prefill block); used only by ``zero_point="bias"``.
        """
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

        # non-uniform: per-channel level table (B,H,D,levels). For NF4/asym-NF4 the
        # grid is fixed, so we keep only per-channel scalars (amax, +mean) and rebuild.
        nf4_scale = nf4_mean = None
        zp_mode, zp_theta = "calibrated", None
        if self.nf4 and self.nf4_asym:
            # asymmetric / zero-point NF4: center per channel, NF4 the residual, so the
            # codebook tracks the DC offset instead of wasting half its codes near zero.
            if self.zero_point == "bias":
                if H != self.n_heads:
                    raise ValueError(
                        f"input has {H} heads but k_bias was given for "
                        f"{self.n_heads}; zero_point='bias' needs them equal"
                    )
                mu_hd = self.rope_averaged_bias(
                    self.k_bias, self.rope_theta, D, position_start, S
                )  # (H, D), identical for every batch item
                mu = np.broadcast_to(mu_hd[None, :, :], (B, H, D))
                nf4_mean = None  # recomputed at decode: zero stored metadata
                zp_mode, zp_theta = "bias", self.rope_theta
            elif self.zero_point == "sparse":
                dc = self.dc_channel_mask(self.rope_theta, D, S)  # (D,)
                mu = x.mean(axis=2) * dc[None, None, :]  # (B,H,D), zero off-DC
                # store only the DC channels: the mask is config-derived at decode
                nf4_mean = mu[:, :, dc].astype(np.float32)  # (B,H,n_dc)
                zp_mode, zp_theta = "sparse", self.rope_theta
            else:
                mu = x.mean(axis=2)  # (B,H,D) per-channel DC offset
                nf4_mean = mu.astype(np.float32)
            xc = x - mu[:, :, None, :]
            amax = np.maximum(np.abs(xc).max(axis=2), 1e-8)  # (B,H,D)
            cent = (mu[..., None] + amax[..., None] * _NF4[None, None, None, :]).astype(
                np.float32
            )
            nf4_scale = amax.astype(np.float32)
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
            zp_mode=zp_mode,
            rope_theta=zp_theta,
            position_start=position_start,
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
            if c.zp_mode == "bias":
                # zero-point recomputed from the instance's k_bias + theta:
                # nothing was stored.
                if self.k_bias is None:
                    raise ValueError(
                        'container was compressed with zero_point="bias"; '
                        "decompress needs a PerChannelKV built with the same k_bias"
                    )
                mu_hd = self.rope_averaged_bias(
                    self.k_bias, c.rope_theta, D, c.position_start, S
                )
                cent = cent + mu_hd[None, :, :, None]
            elif c.zp_mode == "sparse":
                # expand the compact DC-channel means through the config mask
                dc = self.dc_channel_mask(c.rope_theta, D, S)
                mu = np.zeros((B, H, D), dtype=np.float32)
                mu[:, :, dc] = c.nf4_mean
                cent = cent + mu[..., None]
            elif c.nf4_mean is not None:  # calibrated asymmetric / zero-point
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

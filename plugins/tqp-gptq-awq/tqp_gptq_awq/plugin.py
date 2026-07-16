# tqp-gptq-awq: GPTQ/AWQ weight formats as turboquant-pro plugins (P4)
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""GPTQ- and AWQ-style per-group int4 weight formats through the P0 contract.

Decode-side semantics only, honestly scoped: GPTQ's Hessian-aware rounding
and AWQ's activation-aware scale *search* are compressor-side cleverness --
their containers dequantize identically to per-group asymmetric uniform
int4 (`dequant = (code - zero) * scale` per group of ``group_size`` input
channels). That decode form is plain affine, so both formats inherit the
instruments and the fused path with no kernel work.

Shape convention: a weight matrix ``(O, I)`` enters as the 4-D block
``(1, 1, O, I)``; per-group scales along I expand to the block-granular
weight ``(1, O, I)`` (the P2 milestone-2 contract extension, reused
unchanged). AWQ additionally folds a per-input-channel activation scale
into the stored weight before grouping (``awq_scale``); the adapter folds
it back on decompress.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from turboquant_pro.plugins import TARGET_WEIGHT, PluginSpec

_GRID16 = np.arange(16, dtype=np.float32)


@dataclass
class GroupInt4Container:
    codes: np.ndarray  # (1, 1, O, I) uint8 in [0, 15]
    scale: np.ndarray  # (O, I // g) float32 per-group
    zero: np.ndarray  # (O, I // g) float32 per-group zero-point (in code units)
    shape: tuple
    group_size: int
    awq_scale: np.ndarray | None = None  # (I,) folded activation scale


class GPTQQuantizer:
    """Per-group asymmetric uniform int4 (GPTQ container semantics)."""

    def __init__(self, group_size: int = 128):
        self.group_size = group_size

    def _check(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = x[None, None]
        if x.ndim != 4 or x.shape[0] != 1 or x.shape[3] % self.group_size:
            raise ValueError(
                "expected (O, I) or (1, 1, O, I) with I divisible by group_size"
            )
        return x

    def compress(self, x) -> GroupInt4Container:
        x = self._check(x)
        n_out, n_in = x.shape[2], x.shape[3]
        g = self.group_size
        blocks = x[0, 0].reshape(n_out, n_in // g, g)
        mn = blocks.min(axis=-1)
        mx = blocks.max(axis=-1)
        scale = np.maximum((mx - mn) / 15.0, 1e-12).astype(np.float32)
        zero = np.round(-mn / scale).clip(0, 15).astype(np.float32)
        codes = (
            np.round(blocks / scale[..., None] + zero[..., None])
            .clip(0, 15)
            .astype(np.uint8)
            .reshape(1, 1, n_out, n_in)
        )
        return GroupInt4Container(codes, scale, zero, x.shape, g)

    def decompress(self, c: GroupInt4Container) -> np.ndarray:
        n_out, n_in = c.shape[2], c.shape[3]
        g = c.group_size
        blocks = c.codes[0, 0].astype(np.float32).reshape(n_out, n_in // g, g)
        out = (blocks - c.zero[..., None]) * c.scale[..., None]
        out = out.reshape(1, 1, n_out, n_in)
        if c.awq_scale is not None:
            out = out / c.awq_scale[None, None, None, :]
        return out.astype(np.float32)

    def grid_params(self, c: GroupInt4Container):
        if c.awq_scale is not None:
            return None  # act-scale unfold is per-input-channel on the OUTPUT
        n_out, n_in = c.shape[2], c.shape[3]
        g = c.group_size
        w = np.repeat(c.scale, g, axis=1).reshape(1, n_out, n_in).astype(np.float32)
        mu = (-np.repeat(c.zero * c.scale, g, axis=1)).reshape(1, n_out, n_in)
        return mu.astype(np.float32), w, _GRID16

    def codes(self, c: GroupInt4Container) -> np.ndarray:
        return c.codes

    def native_dtype(self):
        return None


class AWQQuantizer(GPTQQuantizer):
    """AWQ container semantics: activation scale folded in, then group int4."""

    def __init__(self, group_size: int = 128, act_scale=None):
        super().__init__(group_size)
        self.act_scale = (
            None if act_scale is None else np.asarray(act_scale, dtype=np.float32)
        )

    def compress(self, x) -> GroupInt4Container:
        x = self._check(x)
        s = (
            self.act_scale
            if self.act_scale is not None
            else np.ones(x.shape[3], dtype=np.float32)
        )
        c = super().compress(x * s[None, None, None, :])
        c.awq_scale = s
        return c


SPEC_GPTQ = PluginSpec(
    name="gptq",
    factory=GPTQQuantizer,
    targets=frozenset({TARGET_WEIGHT}),
    tier="experimental",
    description=(
        "GPTQ container semantics: per-group asymmetric uniform int4; "
        "block-granular affine (Hessian rounding is compressor-side)"
    ),
)

SPEC_AWQ = PluginSpec(
    name="awq",
    factory=AWQQuantizer,
    targets=frozenset({TARGET_WEIGHT}),
    tier="experimental",
    description=(
        "AWQ container semantics: activation scale folded + per-group int4; "
        "affine when unscaled, decompress-then-use otherwise"
    ),
)

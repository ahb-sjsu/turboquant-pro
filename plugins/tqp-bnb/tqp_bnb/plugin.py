# tqp-bnb: bitsandbytes-format plugins for turboquant-pro
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""bitsandbytes NF4 (QLoRA) as a turboquant-pro quantizer plugin.

The first external consumer of the P0 plugin contract, living out of tree
(installed separately; discovered via the ``turboquant_pro.plugins`` entry
point). Implements bnb's blockwise NF4: flatten, split into ``blocksize``
blocks, per-block absmax scale, nearest-entry lookup in the fixed 16-value
NF4 table. A pure-NumPy implementation is the reference (the same table
turboquant-pro ships); when bitsandbytes + torch are installed the tests
cross-check dequantization against ``bnb.functional`` itself.

Affine contract status: blockwise scales vary along the *token* axis, so the
(H, D) per-channel form of ``grid_params`` cannot express them —
``grid_params`` returns ``None`` and the format takes the documented
decompress-then-attend degrade. The block-granular contract extension
(weight ``(H, ceil(S/blocks), D)``, design doc §6) is milestone 2; when it
lands, this plugin inherits the fused decode with no kernel work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from turboquant_pro.per_channel_kv import _NF4
from turboquant_pro.plugins import TARGET_KV_KEY, TARGET_WEIGHT, PluginSpec


@dataclass
class BnbNF4Container:
    codes: np.ndarray  # (n_blocks, blocksize) uint8, NF4 table indices
    absmax: np.ndarray  # (n_blocks,) float32 per-block scale
    shape: tuple
    blocksize: int
    n: int  # valid elements (tail block is zero-padded)


class BnbNF4Quantizer:
    """Blockwise NF4 (bitsandbytes/QLoRA semantics), NumPy reference."""

    def __init__(self, blocksize: int = 64):
        if blocksize <= 0:
            raise ValueError("blocksize must be positive")
        self.blocksize = blocksize
        self._table = np.asarray(_NF4, dtype=np.float32)

    def compress(self, x: np.ndarray) -> BnbNF4Container:
        x = np.asarray(x, dtype=np.float32)
        flat = x.ravel()
        n = flat.size
        pad = (-n) % self.blocksize
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
        blocks = flat.reshape(-1, self.blocksize)
        absmax = np.abs(blocks).max(axis=1).astype(np.float32)
        scale = np.maximum(absmax, 1e-12)[:, None]
        codes = np.abs(blocks[..., None] / scale[..., None] - self._table).argmin(
            axis=-1
        )
        return BnbNF4Container(
            codes.astype(np.uint8), absmax, x.shape, self.blocksize, n
        )

    def decompress(self, c: BnbNF4Container) -> np.ndarray:
        vals = self._table[c.codes] * np.maximum(c.absmax, 1e-12)[:, None]
        return vals.ravel()[: c.n].reshape(c.shape).astype(np.float32)

    def grid_params(self, c: BnbNF4Container):
        """Block-granular affine form (contract extension, design doc §6):
        ``mu`` is zeros ``(H, D)`` (NF4 is symmetric); ``weight`` is the
        per-element block absmax expanded to ``(H, S, D)`` -- reference and
        conformance broadcast it to ``(B, H, S, D)``. Requires the KV block
        convention ``(1, H, S, D)``; other shapes degrade to ``None``."""
        if len(c.shape) != 4 or c.shape[0] != 1:
            return None
        _, H, S, D = c.shape
        w = np.repeat(np.maximum(c.absmax, 1e-12), self.blocksize)[: c.n].reshape(
            H, S, D
        )
        return np.zeros((H, D), dtype=np.float32), w.astype(np.float32), self._table

    def codes(self, c: BnbNF4Container) -> np.ndarray:
        if len(c.shape) != 4:
            raise NotImplementedError("codes() requires the (B,H,S,D) block form")
        return c.codes.ravel()[: c.n].reshape(c.shape)

    def native_dtype(self):
        return None


SPEC_NF4 = PluginSpec(
    name="bnb_nf4",
    factory=BnbNF4Quantizer,
    targets=frozenset({TARGET_WEIGHT, TARGET_KV_KEY}),
    tier="experimental",
    description=(
        "bitsandbytes blockwise NF4 (QLoRA): per-block absmax over the fixed "
        "NF4 table; NumPy reference, bnb cross-checked when installed; "
        "decompress-then-attend until the block-granular affine extension"
    ),
)


# ------------------------------------------------------------------ #
# LLM.int8: vector-wise int8 + fp16 outlier channels                  #
# ------------------------------------------------------------------ #


@dataclass
class LLMInt8Container:
    codes: np.ndarray  # (1, H, S, D) uint8, value = int8_code + 127
    scale: np.ndarray  # (H, D) float32 per-channel absmax / 127
    outlier_mask: np.ndarray  # (H, D) bool -- channels kept fp16
    outlier_vals: np.ndarray  # (n_outliers, S) float16, channel-major
    shape: tuple


_INT8_GRID = np.arange(-127, 128, dtype=np.float32)  # 255 levels


class LLMInt8Quantizer:
    """LLM.int8-style mixed decomposition on the KV block convention
    ``(1, H, S, D)``: per-channel int8 absmax for the dense part, whole
    channels whose peak magnitude exceeds ``outlier_threshold`` kept fp16
    (Dettmers et al.'s emergent-feature columns). The outliers surface
    through the contract's ``outlier_csr`` as dense per-token columns --
    LLM.int8's decomposition IS the dense-sparse overlay at column
    granularity, so the format is fully fused-decode-eligible."""

    def __init__(self, outlier_threshold: float = 6.0):
        self.outlier_threshold = float(outlier_threshold)

    def compress(self, x: np.ndarray) -> LLMInt8Container:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError("LLMInt8Quantizer expects the (1, H, S, D) block form")
        _, H, S, D = x.shape
        peak = np.abs(x[0]).max(axis=1)  # (H, D)
        mask = peak > self.outlier_threshold
        scale = np.maximum(peak, 1e-12).astype(np.float32) / 127.0
        codes = (np.clip(np.round(x[0] / scale[:, None, :]), -127, 127) + 127).astype(
            np.uint8
        )[None]
        vals = x[0].transpose(0, 2, 1)[mask].astype(np.float16)  # (n_out, S)
        return LLMInt8Container(codes, scale, mask, vals, x.shape)

    def decompress(self, c: LLMInt8Container) -> np.ndarray:
        out = c.scale[:, None, :] * _INT8_GRID[c.codes[0]]
        out.transpose(0, 2, 1)[c.outlier_mask] = c.outlier_vals.astype(np.float32)
        return out[None].astype(np.float32)

    def grid_params(self, c: LLMInt8Container):
        H, D = c.scale.shape
        return np.zeros((H, D), dtype=np.float32), c.scale, _INT8_GRID

    def codes(self, c: LLMInt8Container) -> np.ndarray:
        return c.codes

    def outlier_csr(self, c: LLMInt8Container):
        if not c.outlier_mask.any():
            return None
        _, H, S, D = c.shape
        dense = c.scale[:, None, :] * _INT8_GRID[c.codes[0]]  # (H, S, D)
        h_idx, d_idx = np.nonzero(c.outlier_mask)
        # token-major: for each (h, s), the outlier columns of head h
        per_head = np.bincount(h_idx, minlength=H)  # outliers per head
        counts = np.repeat(per_head, S)  # (H*S,)
        row_ptr = np.zeros(H * S + 1, dtype=np.int32)
        np.cumsum(counts, out=row_ptr[1:])
        order = np.argsort(h_idx, kind="stable")
        cols_by_head = [d_idx[order[h_idx[order] == h]] for h in range(H)]
        cols = np.concatenate(
            [np.tile(cbh, S) for cbh in cols_by_head if cbh.size]
            or [np.zeros(0, np.int64)]
        )
        # deltas: fp16 value minus dense dequant, token-major within head
        deltas = []
        for h in range(H):
            cbh = cols_by_head[h]
            if not cbh.size:
                continue
            v = c.outlier_vals[
                np.searchsorted(
                    np.flatnonzero(c.outlier_mask.ravel()), h * c.scale.shape[1] + cbh
                )
            ].astype(
                np.float32
            )  # (n_h, S)
            deltas.append((v.T - dense[h][:, cbh]).ravel())  # (S*n_h,)
        deltas = np.concatenate(deltas) if deltas else np.zeros(0, np.float32)
        return row_ptr, cols.astype(np.uint16), deltas.astype(np.float32)

    def native_dtype(self):
        return None


SPEC_INT8 = PluginSpec(
    name="bnb_llm_int8",
    factory=LLMInt8Quantizer,
    targets=frozenset({TARGET_KV_KEY, TARGET_WEIGHT}),
    tier="experimental",
    description=(
        "LLM.int8 mixed decomposition: per-channel int8 absmax + fp16 "
        "outlier channels surfaced as dense CSR columns -- fully "
        "fused-decode-eligible through the affine contract"
    ),
)

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
        codes = np.abs(
            blocks[..., None] / scale[..., None] - self._table
        ).argmin(axis=-1)
        return BnbNF4Container(
            codes.astype(np.uint8), absmax, x.shape, self.blocksize, n
        )

    def decompress(self, c: BnbNF4Container) -> np.ndarray:
        vals = self._table[c.codes] * np.maximum(c.absmax, 1e-12)[:, None]
        return vals.ravel()[: c.n].reshape(c.shape).astype(np.float32)

    def grid_params(self, c: BnbNF4Container):
        # blockwise scales vary along the token axis: not expressible as the
        # (H, D) per-channel affine form -- decompress-then-attend degrade
        # until the block-granular contract extension (milestone 2).
        return None

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

# tqp-trtllm: TensorRT-LLM-style KV formats as turboquant-pro plugins
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""FP8 (e4m3) and NVFP4 (block-16 e2m1) KV-cache formats through the P0
contract, in **code space**: the dtype is a lookup table, so correctness is
testable on any CPU (design doc section 4.3 -- correctness never waits on
exotic hardware). Native-dtype passthrough on Ada/Hopper/Blackwell is the
hardware milestone; these plugins are its semantics oracle.

FP8 KV: per-head absmax scale, values snapped to the nearest finite
e4m3fn value -- affine contract with a 253-entry grid (mu = 0, weight =
per-head scale broadcast to (H, D)).

NVFP4 KV: block-16 scales along the channel axis over the 15-value e2m1
grid -- weight is block-granular (H, S, D), the contract extension shipped
for bnb. This pair is the substrate for the pre-registered keys comparison
(per-tensor/per-head-scaled fp8 predicted fragile on DC-offset key
families vs block-scaled nvfp4 robust).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from turboquant_pro.plugins import TARGET_KV_KEY, TARGET_KV_VALUE, PluginSpec


def _e4m3_table() -> np.ndarray:
    vals = [0.0]
    for exp in range(16):
        for man in range(8):
            if exp == 15 and man == 7:
                continue  # nan encoding in e4m3fn
            m = 1 + man / 8 if exp > 0 else man / 8
            e = exp - 7 if exp > 0 else -6
            v = m * 2.0**e
            if v:
                vals.append(v)
    pos = np.asarray(sorted(set(vals)), dtype=np.float32)
    return np.concatenate([-pos[::-1][:-1], pos]).astype(np.float32)


_E4M3 = _e4m3_table()  # 253 finite values, symmetric, ascending
_E2M1 = np.asarray(
    [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32
)


def _nearest(x: np.ndarray, table: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(table, x).clip(1, len(table) - 1)
    left = table[idx - 1]
    right = table[idx]
    return np.where(np.abs(x - left) <= np.abs(x - right), idx - 1, idx).astype(
        np.uint8
    )


@dataclass
class FP8KVContainer:
    codes: np.ndarray  # (1, H, S, D) uint8 indices into the e4m3 table
    scale: np.ndarray  # (H,) per-head absmax / e4m3_max
    shape: tuple


class FP8KVQuantizer:
    """Per-head-scaled e4m3 in code space (TRT-LLM fp8 KV cache semantics)."""

    def compress(self, x: np.ndarray) -> FP8KVContainer:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError("expected the (1, H, S, D) block form")
        H = x.shape[1]
        scale = (
            np.abs(x[0]).reshape(H, -1).max(axis=1).astype(np.float32)
            / float(_E4M3[-1])
        ).clip(1e-12)
        codes = _nearest(x[0] / scale[:, None, None], _E4M3)[None]
        return FP8KVContainer(codes, scale, x.shape)

    def decompress(self, c: FP8KVContainer) -> np.ndarray:
        return (_E4M3[c.codes[0]] * c.scale[:, None, None])[None].astype(np.float32)

    def grid_params(self, c: FP8KVContainer):
        _, H, _, D = c.shape
        w = np.repeat(c.scale[:, None], D, axis=1).astype(np.float32)  # (H, D)
        return np.zeros((H, D), dtype=np.float32), w, _E4M3

    def codes(self, c: FP8KVContainer) -> np.ndarray:
        return c.codes

    def native_dtype(self):
        return "float8_e4m3fn"


@dataclass
class NVFP4Container:
    codes: np.ndarray  # (1, H, S, D) uint8 indices into the e2m1 grid
    scale: np.ndarray  # (H, S, D // 16) float32 per-block
    shape: tuple


class NVFP4KVQuantizer:
    """Block-16 e2m1 (NVFP4 semantics) in code space; block-granular affine."""

    BLOCK = 16

    def compress(self, x: np.ndarray) -> NVFP4Container:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4 or x.shape[0] != 1 or x.shape[3] % self.BLOCK:
            raise ValueError("expected (1, H, S, D) with D divisible by 16")
        _, H, S, D = x.shape
        blocks = x[0].reshape(H, S, D // self.BLOCK, self.BLOCK)
        scale = (np.abs(blocks).max(axis=-1) / 6.0).clip(1e-12).astype(np.float32)
        codes = _nearest(blocks / scale[..., None], _E2M1).reshape(H, S, D)[None]
        return NVFP4Container(codes, scale, x.shape)

    def decompress(self, c: NVFP4Container) -> np.ndarray:
        w = np.repeat(c.scale, self.BLOCK, axis=-1)  # (H, S, D)
        return (_E2M1[c.codes[0]] * w)[None].astype(np.float32)

    def grid_params(self, c: NVFP4Container):
        _, H, _, D = c.shape
        w = np.repeat(c.scale, self.BLOCK, axis=-1).astype(np.float32)  # (H, S, D)
        return np.zeros((H, D), dtype=np.float32), w, _E2M1

    def codes(self, c: NVFP4Container) -> np.ndarray:
        return c.codes

    def native_dtype(self):
        return None  # Blackwell passthrough is the hardware milestone


SPEC_FP8 = PluginSpec(
    name="fp8_kv",
    factory=FP8KVQuantizer,
    targets=frozenset({TARGET_KV_KEY, TARGET_KV_VALUE}),
    tier="experimental",
    description=(
        "TRT-LLM-style fp8 (e4m3) KV in code space; per-head scale; native "
        "passthrough on Ada/Hopper is the hardware milestone"
    ),
)

SPEC_NVFP4 = PluginSpec(
    name="nvfp4_kv",
    factory=NVFP4KVQuantizer,
    targets=frozenset({TARGET_KV_KEY, TARGET_KV_VALUE}),
    tier="experimental",
    description=(
        "NVFP4-style block-16 e2m1 KV in code space; block-granular affine "
        "weight (H, S, D)"
    ),
)

# tqp-trtllm: TensorRT-LLM-style KV formats as turboquant-pro plugins
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Native fp8 (e4m3fn) KV passthrough — the hardware milestone of the
``fp8_kv`` plugin's declared ``native_dtype``.

Storage passthrough, honestly scoped: keys/values live as real
``torch.float8_e4m3fn`` tensors (half the bytes of fp16) with a per-head
fp32 scale, and are upcast on read for attention compute. Fp8 *compute*
(scaled-mm attention) is Hopper/FlashAttention-3 territory; on Ada this
storage form is the win available today. The code-space plugin
(:class:`tqp_trtllm.plugin.FP8KVQuantizer`) is the semantics oracle: the
torch cast rounds to nearest-even over exactly the 253-value finite e4m3fn
grid, so native and code-space decompress must agree (ties excepted, which
are measure-zero for real activations).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .plugin import _E4M3


@dataclass
class FP8NativeContainer:
    data: object  # torch.Tensor, dtype float8_e4m3fn, shape (1, H, S, D)
    scale: object  # torch.Tensor float32 (H,)
    shape: tuple

    def nbytes(self) -> int:
        return self.data.element_size() * self.data.numel() + 4 * self.scale.numel()


class FP8NativeKV:
    """Per-head-scaled e4m3fn KV as real fp8 tensors (torch, any device)."""

    def compress(self, x) -> FP8NativeContainer:
        import torch

        t = (
            torch.as_tensor(np.asarray(x, dtype=np.float32))
            if not hasattr(x, "dtype") or isinstance(x, np.ndarray)
            else x
        )
        t = t.float()
        if t.ndim != 4 or t.shape[0] != 1:
            raise ValueError("expected the (1, H, S, D) block form")
        H = t.shape[1]
        scale = (t[0].abs().reshape(H, -1).amax(dim=1) / float(_E4M3[-1])).clamp_min(
            1e-12
        )
        f8 = (t[0] / scale[:, None, None]).to(torch.float8_e4m3fn)[None]
        return FP8NativeContainer(f8, scale, tuple(t.shape))

    def decompress(self, c: FP8NativeContainer):
        return (c.data[0].float() * c.scale[:, None, None])[None]

    def native_dtype(self):
        return "float8_e4m3fn"

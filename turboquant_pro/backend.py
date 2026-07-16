# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Array-backend helpers (P1 of ``docs/DESIGN_hardware_and_plugins.md``).

Torch is the portability plane: one optional dependency covers CUDA, ROCm,
Apple MPS, and Intel XPU. Two roles here:

1. **Boundary conversion** (:func:`to_numpy`) — every instrument (rank
   certificate, (A2) probe) accepts torch or CuPy arrays from *any* device.
   Instruments are calibration-time tools; their math stays NumPy on CPU,
   but a user holding device tensors should never have to know that.
2. **Native-torch reference decode** (:func:`torch_decode`) — the
   decompress-then-attend path executed on any torch device. This is the
   no-kernel-work portability story: the KV cache's decode runs on ROCm/MPS
   exactly as on CUDA, numerically matching the NumPy reference.

The fused compute-on-codes kernels remain CuPy/CUDA (and, per the design
doc, a future Triton port); this module is deliberately dependency-light —
torch is imported lazily and only when torch objects actually appear.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "is_torch_tensor",
    "is_cupy_array",
    "to_numpy",
    "torch_decode",
    "torch_xp",
]


def is_torch_tensor(x) -> bool:
    """True for ``torch.Tensor`` without importing torch."""
    mod = type(x).__module__
    return mod == "torch" or mod.startswith("torch.")


def is_cupy_array(x) -> bool:
    """True for ``cupy.ndarray`` without importing cupy."""
    return type(x).__module__.startswith("cupy")


def to_numpy(x) -> np.ndarray:
    """Convert torch (any device), CuPy, or array-like input to NumPy.

    The boundary adapter for calibration-time instruments: torch tensors are
    detached and moved off-device; CuPy arrays are downloaded; everything
    else goes through ``np.asarray`` unchanged.
    """
    if is_torch_tensor(x):
        return x.detach().cpu().numpy()
    if is_cupy_array(x):
        import cupy as cp

        return cp.asnumpy(x)
    return np.asarray(x)


def torch_decode(cache, query, *, device=None, dtype=None):
    """One decode-step attention output over a ``TurboQuantKVCache`` — a **torch
    attention reference after host-side reconstruction**, not a device-native
    decode.

    Keys/values are reconstructed on the host through the cache's public getters
    (so every key format, including nuq, is supported), moved to ``device`` as
    torch tensors, and the **attention math** (`QK^T` / softmax / `PV`) runs
    on-device in torch (CUDA, ROCm, MPS, XPU, or CPU). This is a correctness /
    portability reference: the reconstruction step is host NumPy, so it is *not*
    zero-copy or a fused on-device dequant. Numerically matches the NumPy
    reference (and hence ``cache.fused_decode``) to float tolerance.

    Args:
        cache: A ``TurboQuantKVCache`` with at least one token stored.
        query: ``(n_heads, head_dim)`` array or tensor.
        device: torch device (default: ``query``'s device if it is a tensor,
            else torch's default).
        dtype: computation dtype (default ``torch.float32``).

    Returns:
        ``torch.Tensor`` of shape ``(n_heads, head_dim)`` on ``device``.
    """
    import torch

    if cache.length == 0:
        raise RuntimeError("cache is empty")
    if device is None and is_torch_tensor(query):
        device = query.device
    dtype = dtype or torch.float32

    q = torch.as_tensor(to_numpy(query), dtype=dtype, device=device).reshape(
        cache.n_heads, cache.head_dim
    )
    k = torch.as_tensor(
        to_numpy(cache.get_keys(0, cache.length))[0], dtype=dtype, device=device
    )
    v = torch.as_tensor(
        to_numpy(cache.get_values(0, cache.length))[0], dtype=dtype, device=device
    )
    scores = torch.einsum("hd,hsd->hs", q, k) / float(np.sqrt(cache.head_dim))
    p = torch.softmax(scores, dim=-1)
    return torch.einsum("hs,hsd->hd", p, v)


class _TorchXP:
    """NumPy-signature shim over torch for the ``xp=`` reference paths
    (``kv_fused``, ``kv_fused_pck``): the three real incompatibilities are
    ``max``-returns-namedtuple (use :meth:`amax`), ``keepdims`` naming, and
    uint8 fancy-indexing (torch treats it as a mask — integer arrays without
    an explicit dtype are promoted to int64 code indices). Everything else
    delegates to torch directly. Use as ``xp=torch_xp``; pair with
    ``device=`` on the input tensors — ops inherit their device.
    """

    def __getattr__(self, name):
        import torch

        return getattr(torch, name)

    def asarray(self, x, dtype=None):
        import torch

        t = torch.as_tensor(x) if not is_torch_tensor(x) else x
        if dtype is not None:
            return t.to(dtype)
        if not t.is_floating_point() and t.dtype != torch.int64:
            return t.long()  # code indices: uint8 indexing would be a mask
        return t

    def ascontiguousarray(self, x):
        return self.asarray(x).contiguous()

    def amax(self, x, axis=None, keepdims=False):
        import torch

        return torch.amax(x, dim=axis, keepdim=keepdims)

    def concatenate(self, xs, axis=0):
        import torch

        return torch.cat(list(xs), dim=axis)


torch_xp = _TorchXP()

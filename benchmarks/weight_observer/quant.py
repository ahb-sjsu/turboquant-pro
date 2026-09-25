"""The weight codec of Part III: round-to-nearest, asymmetric, per group along the input.

One codec for every arm, so the arms differ only in which bits land where. Deterministic
and data-free (no calibration, no Hessian rounding): the predictors are scored on how well
they rank the damage this codec does, not on a codec tuned to any of them.
"""

from __future__ import annotations

import torch

GROUP = 128
LEVELS = (2, 3, 4, 5, 6, 8)


def rtn(w: torch.Tensor, bits: int, group: int = GROUP) -> torch.Tensor:
    """Quantize-dequantize ``w`` (out, in) at ``bits`` with a min/max grid per group of
    ``group`` input columns. Computed in float32; returns float32."""
    out, inp = w.shape
    if inp % group:
        raise ValueError(f"in-dimension {inp} is not a multiple of the group {group}")
    x = w.float().reshape(out, inp // group, group)
    lo = x.amin(-1, keepdim=True)
    hi = x.amax(-1, keepdim=True)
    q = 2**bits - 1
    scale = (hi - lo).clamp_min(1e-12) / q
    return (((x - lo) / scale).round().clamp(0, q) * scale + lo).reshape(out, inp)


def stored_bits(n_params: int, bits: int, group: int = GROUP) -> float:
    """Codes plus two fp16 per group (min and scale): the matched quantity of a rate."""
    return n_params * bits + (n_params // group) * 32

# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Opt-in online calibration for KV-cache keys — **experimental**.

The **zero-calibration** path (calibration-free asymmetric NF4 + 2% outliers)
stays the default and the **recommended** choice — nothing here changes it.

:func:`calibrate_key_quantizer` fits a per-(head, channel) **Lloyd-Max**
(MSE-optimal) key codebook **once** from a representative set of real key
activations and reuses it, instead of the fixed NF4 grid — the data-fit idea
behind KVQuant's offline codebook, made lightweight.

**Honest caveat** (``benchmarks/RESULTS_calibration.md``): on a controlled
attention proxy this **lowers key reconstruction error but does *not* beat the
calibration-free default on softmax-KL** — the metric attention actually
consumes. Optimizing the marginal per-channel codebook discards the DC-offset
modeling that makes asym-NF4 well-matched to post-RoPE keys. It is provided so
users can try a data-fit codebook and measure it on *their* task; whether a
stronger, Fisher-weighted objective closes the real qasper gap is left open.
Composes with `outlier_frac`, self-describing (levels travel in the container),
and — like all `nuq` keys — decodes via decompress-then-attend, not the fused
kernel.

Example::

    from turboquant_pro import calibrate_key_quantizer

    # `cal` = a modest sample of real post-RoPE keys, shape (B, H, S, D)
    kq = calibrate_key_quantizer(cal, bits=4, outlier_frac=0.02)
    c  = kq.compress(new_keys)          # data-fit codebook, reused
    k  = kq.decompress(c)
"""

from __future__ import annotations

import numpy as np

from .per_channel_kv import PerChannelKV

__all__ = ["calibrate_key_quantizer"]


def calibrate_key_quantizer(
    samples: np.ndarray,
    *,
    bits: int = 4,
    outlier_frac: float = 0.02,
    weights: np.ndarray | None = None,
    iters: int = 20,
    **kwargs,
) -> PerChannelKV:
    """Build a :class:`~turboquant_pro.per_channel_kv.PerChannelKV` with a
    per-channel codebook calibrated from ``samples``.

    Thin wrapper over ``PerChannelKV(...).calibrate(samples)``.

    Args:
        samples: real key activations — ``(N, D)``, ``(N, H, D)``, or
            ``(B, H, S, D)``. Pooled over every axis except channel ``D`` (head
            ``H`` kept when present).
        bits: code width (default 4).
        outlier_frac: dense-sparse fp16 tail kept per channel (default 0.02 — the
            measured sweet spot; set 0.0 to disable).
        weights: optional per-token importance (a light Fisher/sensitivity
            surrogate); uniform when omitted.
        **kwargs: forwarded to ``PerChannelKV`` (e.g. ``head_dim``, ``n_heads``).

    Returns:
        A calibrated ``PerChannelKV`` ready to ``compress`` / ``decompress``.
    """
    q = PerChannelKV(bits=bits, outlier_frac=outlier_frac, **kwargs)
    return q.calibrate(samples, weights=weights, iters=iters)

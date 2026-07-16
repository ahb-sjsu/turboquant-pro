# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""
Operator-dependent sensitivity for gates and state-space recurrences.

``operator_trace`` classifies a tensor into a regime; ``a2_probe`` handles the
attention score path. This module supplies the (A2) analysis for the two
regimes that hybrid / state-space architectures add -- ``GATE_SELECTION`` (MoE
routers, top-k selection) and ``STATE_DECAY`` (SSM / linear-recurrence decays)
-- because their sensitive coordinate is neither "per-channel DC" (keys) nor
"symmetric residual" (V/O). Each boundary here is *measured*, and the tooling
acts on the measurement; the formal derivations belong to the companion theory
paper (the-angular-observer).

**Gates -- selection is carried by the margin, not the magnitude.** A top-k
router reads only the *order* of the logits ``l = W_gate x``. It is invariant to
a common-mode shift of all logits (add c to every expert -> same argmax), so a
common-mode quantization error is free; only the *differential* error across
experts can flip the selection -- and only where it exceeds the **routing
margin** (the gap between the k-th and (k+1)-th logit). This is the tangential
/ (A2) split restated for selection: ``differential_fraction`` is the analog of
``tangential_fraction``, and the fragile tokens are the low-margin ones (in
measurement, low-margin tokens flip ~orders of magnitude more often than
high-margin ones as bits increase).

**Recurrences -- the slow (long-memory) channels are fragile, and error
compounds over the sequence.** For a per-channel linear recurrence
``h_t = a . h_{t-1} + b_t`` the steady-state gain is ``1/(1-a)`` and the memory
length is ``~1/(1-a)``, so a fixed error in the decay ``a`` is amplified far
more in slow channels (``a -> 1``) and accumulates over the sequence -- the
recurrent analog of the RoPE-slow-channel key finding. The discipline that
follows: quantize the decay in a **log-time-constant basis** (``tau = -log a``,
then quantize ``log tau``), which puts fine resolution where ``a -> 1`` -- the
SSM analog of NF4's non-uniform key levels, and measured to cut state drift
several-fold over linear quantization at matched bits.

All numpy, scipy-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "routing_margins",
    "differential_fraction",
    "predict_routing_flips",
    "routing_sensitivity",
    "RoutingSensitivity",
    "decay_gain",
    "decay_time_constant",
    "decay_sensitivity",
    "quantize_decay",
    "state_decay_sensitivity",
    "StateDecaySensitivity",
]


# ===========================================================================
# Gates / routing (GATE_SELECTION)
# ===========================================================================


def routing_margins(logits: np.ndarray, k: int = 1) -> np.ndarray:
    """Per-token top-k routing margin: gap between the k-th and (k+1)-th logit.

    Args:
        logits: (n_tokens, n_experts) gate logits.
        k: the top-k boundary to measure (default 1 = top-1 argmax margin).

    Returns:
        (n_tokens,) non-negative margins. A small margin means the selection is
        fragile: a quantization perturbation that exceeds it flips the routing.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError("logits must be (n_tokens, n_experts)")
    if not 1 <= k < logits.shape[1]:
        raise ValueError("k must satisfy 1 <= k < n_experts")
    part = np.sort(logits, axis=1)[:, ::-1]
    return part[:, k - 1] - part[:, k]


def differential_fraction(delta_logits: np.ndarray) -> np.ndarray:
    """Fraction of a logit perturbation that can affect selection.

    Selection is invariant to a common-mode shift, so only the component of
    ``delta_logits`` orthogonal to the all-ones direction (the per-token
    centered part) matters. Returns, per token,
    ``||delta - mean(delta)||^2 / ||delta||^2`` in [0, 1] -- the routing analog
    of :func:`turboquant_pro.a2_probe.tangential_fraction`. Near 0 = the error
    is common-mode (routing-safe); near 1 = fully differential (dangerous).
    """
    d = np.asarray(delta_logits, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError("delta_logits must be (n_tokens, n_experts)")
    total = (d**2).sum(axis=1)
    centered = d - d.mean(axis=1, keepdims=True)
    diff = (centered**2).sum(axis=1)
    out = np.divide(diff, total, out=np.full_like(total, np.nan), where=total > 0)
    return np.clip(out, 0.0, 1.0)


def predict_routing_flips(
    logits: np.ndarray, logit_noise_std: float, k: int = 1
) -> float:
    """First-order predicted top-k flip fraction under differential logit noise.

    The boundary pair sees differential noise of std ``~sqrt(2) * logit_noise_std``;
    a token flips when that exceeds its margin. Returns the fraction of tokens
    whose margin is below that scale -- an *estimate* to size the bit budget,
    not an exact rate.
    """
    margins = routing_margins(logits, k=k)
    scale = np.sqrt(2.0) * float(logit_noise_std)
    if scale <= 0:
        return 0.0
    return float(np.mean(margins < scale))


@dataclass
class RoutingSensitivity:
    """Routing fragility summary for a batch of gate logits."""

    k: int
    n_tokens: int
    margin_p10: float
    margin_p50: float
    margin_mean: float
    predicted_flip_fraction: float | None

    def as_dict(self) -> dict:
        return {
            "k": self.k,
            "n_tokens": self.n_tokens,
            "margin_p10": self.margin_p10,
            "margin_p50": self.margin_p50,
            "margin_mean": self.margin_mean,
            "predicted_flip_fraction": self.predicted_flip_fraction,
        }


def routing_sensitivity(
    logits: np.ndarray, logit_noise_std: float | None = None, k: int = 1
) -> RoutingSensitivity:
    """Summarize routing fragility: the margin distribution + a flip estimate.

    Low margin percentiles mean the router needs the gate quantized to finer
    logit precision (or the differential component protected). If
    ``logit_noise_std`` (the per-logit error a candidate quantizer induces) is
    given, also returns the first-order predicted flip fraction.
    """
    margins = routing_margins(logits, k=k)
    return RoutingSensitivity(
        k=k,
        n_tokens=int(margins.shape[0]),
        margin_p10=float(np.percentile(margins, 10)),
        margin_p50=float(np.percentile(margins, 50)),
        margin_mean=float(margins.mean()),
        predicted_flip_fraction=(
            None
            if logit_noise_std is None
            else predict_routing_flips(logits, logit_noise_std, k=k)
        ),
    )


# ===========================================================================
# State-space recurrences (STATE_DECAY)
# ===========================================================================


def decay_gain(decay: np.ndarray) -> np.ndarray:
    """Per-channel steady-state gain ``1/(1-a)`` of ``h_t = a h_{t-1} + b_t``.

    Blows up as ``a -> 1`` (long memory) -- the channels where a fixed decay
    error is most amplified.
    """
    a = np.asarray(decay, dtype=np.float64)
    return 1.0 / np.maximum(1.0 - a, 1e-12)


def decay_time_constant(decay: np.ndarray) -> np.ndarray:
    """Per-channel memory length ``tau = -1/log(a)`` (timesteps)."""
    a = np.clip(np.asarray(decay, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return -1.0 / np.log(a)


def decay_sensitivity(decay: np.ndarray, seq_len: int | None = None) -> np.ndarray:
    """Per-channel sensitivity of the accumulated state to a decay error.

    For ``S(a) = sum_{t=1}^{T} a^t`` the sensitivity is
    ``dS/da = sum_{t=1}^{T} t a^{t-1}``; at steady state (``T -> inf``) this is
    ``1/(1-a)^2``. Returns the (truncated) coefficient per channel -- it grows
    sharply toward ``a -> 1`` and with ``seq_len``, quantifying why slow
    channels over long sequences are the fragile ones.
    """
    a = np.clip(np.asarray(decay, dtype=np.float64), 0.0, 1.0 - 1e-9)
    if seq_len is None:
        return 1.0 / (1.0 - a) ** 2
    T = int(seq_len)
    # d/da of sum_{t=1}^{T} a^t = (1 - a^T(1 + T(1-a))) / (1-a)^2
    one_minus = 1.0 - a
    return (1.0 - a**T * (1.0 + T * one_minus)) / one_minus**2


def quantize_decay(decay: np.ndarray, bits: int, basis: str = "log_tau") -> np.ndarray:
    """Fake-quantize per-channel decays; ``basis`` sets the resolution warp.

    - ``"linear"``: uniform grid on ``a`` (wastes resolution away from ``a=1``).
    - ``"log_tau"`` (recommended): uniform grid on ``log(-log a)`` -- the
      log-time-constant basis, which concentrates levels where ``a -> 1`` (the
      fragile slow channels). Measured to cut state drift several-fold over
      ``"linear"`` at matched bits, the SSM analog of NF4-for-keys.

    Returns the dequantized decays.
    """
    a = np.asarray(decay, dtype=np.float64)
    if basis == "linear":
        v = a
    elif basis == "log_tau":
        v = np.log(-np.log(np.clip(a, 1e-9, 1.0 - 1e-9)))
    else:
        raise ValueError("basis must be 'linear' or 'log_tau'")
    lo, hi = float(v.min()), float(v.max())
    step = (hi - lo) / max(2**bits - 1, 1)
    vq = np.round((v - lo) / max(step, 1e-30)) * step + lo
    if basis == "linear":
        return np.clip(vq, 0.0, 1.0 - 1e-9)
    return np.clip(np.exp(-np.exp(vq)), 0.0, 1.0 - 1e-9)


@dataclass
class StateDecaySensitivity:
    """Per-channel state-decay fragility summary."""

    n_channels: int
    seq_len: int | None
    mean_gain: float
    max_gain: float
    slow_channel_fraction: float  # fraction with a > 0.9
    recommended_basis: str

    def as_dict(self) -> dict:
        return {
            "n_channels": self.n_channels,
            "seq_len": self.seq_len,
            "mean_gain": self.mean_gain,
            "max_gain": self.max_gain,
            "slow_channel_fraction": self.slow_channel_fraction,
            "recommended_basis": self.recommended_basis,
        }


def state_decay_sensitivity(
    decay: np.ndarray, seq_len: int | None = None, slow_threshold: float = 0.9
) -> StateDecaySensitivity:
    """Summarize which decay channels are fragile and the basis to use."""
    a = np.asarray(decay, dtype=np.float64)
    gain = decay_gain(a)
    return StateDecaySensitivity(
        n_channels=int(a.size),
        seq_len=seq_len,
        mean_gain=float(gain.mean()),
        max_gain=float(gain.max()),
        slow_channel_fraction=float(np.mean(a > slow_threshold)),
        recommended_basis="log_tau",
    )

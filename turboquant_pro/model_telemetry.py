# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Consumer-relative telemetry for a live model (``docs/DESIGN_inference_telemetry.md``,
layer 1). It only reads: attaching a :class:`ModelMonitor` changes no output.

**Where it looks.** Every Hugging Face attention module hands its queries, keys, values,
mask and softmax scale to a function looked up in ``AttentionInterface`` by name. The
monitor registers a wrapper there that calls the model's own implementation unchanged
and, on sampled calls, also reads the tensors it was given. These are the coordinates
the attention actually consumes and the cache actually stores: after any query/key norm,
after the rotary embedding, with the model's own scale (Gemma's
``query_pre_attn_scalar``, for one) and its own mask (causal, sliding window, padding).
Nothing is rebuilt from ``q_proj``, which is what fails on architectures with QK-norm or
per-layer RoPE.

**What it measures, per layer and key head.** For one query row with logits
``z_t = s q . k_t`` (passed through ``c tanh(z / c)`` when the model soft-caps them) and
attention ``p = softmax(z)``, an error ``dk_t`` on every key moves the row's attention
by ``dp = sum_t J_t dk_t``, and the Jacobian Gram summed over keys is

    sum_t J_t^T J_t = a s^2 q q^T,
    a = sum_t g_t^2 p_t^2 (1 - 2 p_t + S2),   S2 = sum_t p_t^2,

with ``g_t`` the derivative of the logit cap (1 without one). The inner sum over the
attended row collapses because
``sum_u p_u^2 (delta_ut - p_t)^2 = p_t^2 (1 - 2 p_t + S2)``, so the exact softmax read
operator costs ``O(S)`` per query row, not a probe per key. It is the same operator as
``read_operators.AttentionAnalyticReadOperator`` with every key probed. A key head's
operator sums over the query heads that share it and averages over query rows:

- ``softmax``: ``mean_rows sum_heads a s^2 q q^T``, what the head's attention feels;
- ``logit``: ``mean_rows sum_heads n s^2 q q^T`` with ``n`` the keys the row attends,
  what the logits feel (Part II's read weight, ``E[q^2]``, is its diagonal up to scale).

Also accumulated: the attention entropy of each sampled row, the second moments of the
new keys (the source a plan codes), and, when a
:class:`turboquant_pro.hf_cache.TurboQuantCache` is watched, the error covariance of its
key codec on every spilled block, so the consumer's realised distortion
``tr(P Sigma_delta)`` is measured rather than predicted.

**Windows, plans and status.** Readings close over windows of tokens. A
:class:`ReadPlan` freezes the operators and key covariances of calibration traffic.
Against a plan, each live head reports the fraction of its sensitivity inside the plan's
read subspace (the certificate-expiry overlap, ``validity.operator_overlap``), the
condition number of the change restricted to that subspace (Part IV's ``kappa_P``), the
change of scale, and the regret of the plan's bit allocation against one re-fitted to
the live window (``read_allocation``). Status follows the certificate-expiry rules:
VALID, STALE, or INCONCLUSIVE when the window's own split-half noise straddles the bar.

Sampled readings are measured on the sampled calls only (``sample_every``) and, during a
prefill, on up to ``max_rows`` evenly spaced query rows per call. Accumulators are
float64 on the CPU and float32 on a GPU (sums of at most a window's rows; relative error
well under the third decimal the readings are reported to).
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .read_allocation import allocate_bits
from .telemetry.metrics import reading
from .validity import (
    DEFAULT_MIN_OVERLAP,
    INCONCLUSIVE,
    NOISE_FRACTION,
    STALE,
    UNCHECKED,
    VALID,
    operator_overlap,
)

__all__ = [
    "CONSUMERS",
    "HeadComparison",
    "ModelMonitor",
    "ReadPlan",
    "Window",
    "compare_head",
    "calibration_floor",
    "plan_regret",
    "read_subspace",
    "softmax_gram_weights",
    "status_summary",
    "subspace_kappa",
]

CONSUMERS = ("softmax", "logit")
DEFAULT_MIN_ROWS = 16  # sampled rows per half-window below which status abstains
_ATTN_PREFIX = "tqp_observe"


# --------------------------------------------------------------------------- #
# Pure numerics (numpy; no torch)                                             #
# --------------------------------------------------------------------------- #
def softmax_gram_weights(p, g=None):
    """``a = sum_t g_t^2 p_t^2 (1 - 2 p_t + S2)`` along the last axis.

    ``p`` holds attention rows (masked entries exactly 0); ``g`` the logit cap's
    derivative, same shape, or None for uncapped logits. Works on numpy arrays and torch
    tensors."""
    s2 = (p * p).sum(-1, keepdims=True)
    term = p * p * (1.0 - 2.0 * p + s2)
    if g is not None:
        term = term * g * g
    return term.sum(-1)


def _sym(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    return 0.5 * (A + A.T)


def read_subspace(P: np.ndarray, share: float = 0.9) -> dict:
    """The smallest eigenspace of ``P`` carrying ``share`` of its trace: the plan's read
    subspace, as the sketch ``validity.operator_overlap`` reads (``basis`` rows are the
    eigenvectors). Sized by trace rather than by a fixed rank, because a head's read
    operator in 128 or 256 channels can spread wider than a few directions; an operator
    needing more than half its channels to reach ``share`` is too flat to test against.
    """
    vals, vecs = np.linalg.eigh(_sym(P))
    order = np.argsort(vals)[::-1]
    vals, vecs = np.clip(vals[order], 0.0, None), vecs[:, order]
    tr = float(vals.sum())
    if tr <= 0:
        r = vals.size
        carried = 0.0
    else:
        r = int(np.searchsorted(np.cumsum(vals) / tr, share - 1e-12) + 1)
        r = min(r, vals.size)
        carried = float(vals[:r].sum() / tr)
    return {
        "dim": int(vals.size),
        "rank": r,
        "share": float(share),
        "trace_fraction": carried,
        "basis": vecs[:, :r].T.copy(),
    }


def subspace_kappa(P_plan: np.ndarray, P_live: np.ndarray, basis: np.ndarray) -> float:
    """``alpha_max / alpha_min`` of ``A^{-1/2} B A^{-1/2}`` with ``A``, ``B`` the plan's
    and the live operator restricted to ``basis`` (``D x r``, orthonormal columns): Part
    IV's ``kappa_P`` on the subspace the plan reads. 1 for a pure change of scale;
    ``inf`` when the live operator is singular there."""
    U = np.asarray(basis, dtype=np.float64)
    A = U.T @ _sym(P_plan) @ U
    B = U.T @ _sym(P_live) @ U
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    if w.min() <= 0:
        return float("inf")
    Ais = V @ np.diag(w**-0.5) @ V.T
    alpha = np.linalg.eigvalsh(_sym(Ais @ B @ Ais))
    if alpha.min() <= 0:
        return float("inf")
    return float(alpha.max() / alpha.min())


def plan_regret(
    P_plan: np.ndarray,
    C_plan: np.ndarray,
    P_live: np.ndarray,
    C_live: np.ndarray,
    budget_bits: float,
) -> tuple[float, float, float]:
    """``(regret, D_plan, D_refit)`` under the live consumer.

    The plan water-fills ``budget_bits`` against its own operator and key covariance
    (``read_allocation.allocate_bits``). Coded in the plan's basis ``U`` with
    independent errors, its error covariance on live keys is
    ``U diag(v_i 2^{-2 b_i}) U^T`` with ``v_i = u_i^T C_live u_i``, so the live consumer
    feels ``D_plan = sum_i (u_i^T P_live u_i) v_i 2^{-2 b_i}``. ``D_refit`` is the same
    model's distortion for an allocation fitted to the live window. A model quantity:
    when the source's and the consumer's bases disagree the refit is not a strict
    optimum, and a ratio slightly below 1 is possible."""
    plan = allocate_bits(P_plan, budget_bits=budget_bits, covariance=C_plan)
    U = plan.basis
    lam = np.einsum("di,de,ei->i", U, _sym(P_live), U)
    var = np.maximum(np.einsum("di,de,ei->i", U, _sym(C_live), U), 0.0)
    d_plan = float(np.sum(lam * var * np.power(2.0, -2.0 * plan.bits)))
    refit = allocate_bits(P_live, budget_bits=budget_bits, covariance=C_live)
    d_refit = float(refit.predicted_distortion)
    regret = d_plan / d_refit if d_refit > 0 else (1.0 if d_plan == 0 else float("inf"))
    return regret, d_plan, d_refit


def calibration_floor(
    Pa: np.ndarray,
    Pb: np.ndarray,
    share: float = 0.9,
    *,
    Ca: np.ndarray | None = None,
    Cb: np.ndarray | None = None,
    budget_bits: float | None = None,
) -> dict:
    """What unchanged traffic scores: the two halves of the calibration window against
    each other. ``overlap`` is the mean of each half's operator scored against the
    other's read subspace (relative, as in :func:`compare_head`); ``kappa`` the
    geometric mean of the two directions' ``kappa_P``; ``regret`` (given each half's key
    covariance, from at least as many rows as channels, and the budget) the geometric
    mean of each half's allocation scored on the other. The regret floor matters most:
    a refit is fitted to the very window it is scored on, so even unchanged traffic
    scores above 1. Each half holds half the
    calibration's rows, so this is a pessimistic estimate of the plan's own noise: a
    live number is read against it, never against 1."""
    a, b = read_subspace(Pa, share), read_subspace(Pb, share)
    if a["trace_fraction"] <= 0 or b["trace_fraction"] <= 0:
        return {"overlap": None, "kappa": None, "regret": None}
    ov = 0.5 * (
        operator_overlap(a, Pb) / a["trace_fraction"]
        + operator_overlap(b, Pa) / b["trace_fraction"]
    )
    ka = subspace_kappa(Pa, Pb, np.asarray(a["basis"]).T)
    kb = subspace_kappa(Pb, Pa, np.asarray(b["basis"]).T)
    kap = math.sqrt(ka * kb) if math.isfinite(ka) and math.isfinite(kb) else None
    reg = None
    if Ca is not None and Cb is not None and budget_bits is not None:
        ra = plan_regret(Pa, Ca, Pb, Cb, budget_bits)[0]
        rb = plan_regret(Pb, Cb, Pa, Ca, budget_bits)[0]
        if math.isfinite(ra) and math.isfinite(rb):
            reg = math.sqrt(ra * rb)
    return {"overlap": float(ov), "kappa": kap, "regret": reg}


@dataclass
class HeadComparison:
    """One key head's live window against its plan."""

    layer: int
    kv_head: int
    status: str
    reason: str
    overlap: float | None = None
    overlap_noise: float | None = None
    kappa: float | None = None
    scale_log2: float | None = None
    regret: float | None = None
    trace_fraction: float | None = None
    overlap_floor: float | None = None  # the plan's calibration halves, each other
    kappa_floor: float | None = None
    regret_floor: float | None = None
    unavailable: dict = field(default_factory=dict)  # metric -> why it is not given

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def compare_head(
    layer: int,
    kv_head: int,
    plan: dict,
    P_live: np.ndarray,
    halves: tuple[np.ndarray, np.ndarray],
    C_live: np.ndarray | None,
    *,
    rows: tuple[float, float] | None = None,
    key_rows: int | None = None,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    min_rows: int = DEFAULT_MIN_ROWS,
    budget_bits: float | None = None,
) -> HeadComparison:
    """Score a live head (its window operator, and the same window as two independent
    halves, each averaged over its own rows) against its plan entry (``P``, ``C``,
    ``sketch`` and ``floor`` from :class:`ReadPlan`), by the certificate-expiry rules:
    the overlap with the plan's read subspace against ``min_overlap``, abstaining
    (INCONCLUSIVE) when the plan's own calibration halves already fall below the bar
    (:func:`calibration_floor`: unchanged traffic would fail it), when a half has fewer
    than ``min_rows`` sampled rows, when the split-half noise reaches ``NOISE_FRACTION``
    of the bar, or when the bar lies within it. The noise is half the halves'
    difference, about one standard error of the window's overlap.

    A number the window cannot support is left out with its reason in ``unavailable``:
    ``kappa`` needs at least as many sampled rows as the plan's subspace has directions
    (fewer leave the live operator singular there), and ``regret`` needs at least as
    many key rows as channels (fewer leave the key covariance rank-deficient, and the
    refit then spends nothing where the window happened to see no variance)."""
    # The overlap is relative to the plan's own: the share of the live trace inside the
    # plan's sketched subspace over the share of the plan's trace there, so 1 when the
    # live operator is the plan's. (validity.check_validity compares the raw share,
    # which a sketch carrying 0.80-0.85 of its trace fails against its own operator.)
    sk = plan["sketch"]
    out = HeadComparison(layer, kv_head, UNCHECKED, "")
    out.trace_fraction = sk["trace_fraction"]
    if sk["trace_fraction"] <= 0 or 2 * sk["rank"] > sk["dim"]:
        out.reason = (
            f"the plan's read subspace needs {sk['rank']} of {sk['dim']} directions to "
            f"carry {sk['share']:.2f} of its trace; the operator is too flat to test "
            "against"
        )
        return out
    if np.trace(halves[0]) <= 0 or np.trace(halves[1]) <= 0:
        out.reason = "a half-window saw no sampled rows for this head"
        return out
    tf = sk["trace_fraction"]
    ov = operator_overlap(sk, P_live) / tf
    oa, ob = operator_overlap(sk, halves[0]) / tf, operator_overlap(sk, halves[1]) / tf
    noise = 0.5 * abs(oa - ob)
    out.overlap, out.overlap_noise = ov, noise
    tp, tl = float(np.trace(plan["P"])), float(np.trace(P_live))
    out.scale_log2 = math.log2(tl / tp) if tp > 0 and tl > 0 else None
    n_rows = None if rows is None else float(rows[0] + rows[1])
    if n_rows is not None and n_rows < sk["rank"]:
        out.unavailable["model.plan.kappa"] = (
            f"{n_rows:g} sampled rows < the plan subspace's {sk['rank']} directions"
        )
    else:
        U = np.asarray(sk["basis"], dtype=np.float64).T
        out.kappa = subspace_kappa(plan["P"], P_live, U)
    if budget_bits is not None and C_live is not None and plan.get("C") is not None:
        d = C_live.shape[0]
        if key_rows is not None and key_rows < d:
            out.unavailable["model.plan.regret"] = (
                f"{key_rows} key rows < {d} channels: the window's key covariance is "
                "rank-deficient (use a longer --window)"
            )
        else:
            out.regret = plan_regret(plan["P"], plan["C"], P_live, C_live, budget_bits)[
                0
            ]
    out.overlap_floor = plan.get("floor", {}).get("overlap")
    out.kappa_floor = plan.get("floor", {}).get("kappa")
    out.regret_floor = plan.get("floor", {}).get("regret")
    if out.overlap_floor is not None and out.overlap_floor < min_overlap:
        out.status = INCONCLUSIVE
        out.reason = (
            f"the plan's own calibration halves overlap only {out.overlap_floor:.3f} "
            f"(< {min_overlap}): unchanged traffic would fail the bar; calibrate on "
            "more traffic"
        )
        return out
    if rows is not None and min(rows) < min_rows:
        out.status = INCONCLUSIVE
        out.reason = (
            f"a half-window has {min(rows):g} sampled rows (< {min_rows}); too few "
            "to check"
        )
        return out
    if noise >= NOISE_FRACTION * min_overlap:
        out.status = INCONCLUSIVE
        out.reason = (
            f"split-half noise {noise:.3f} reaches {NOISE_FRACTION} of the bar "
            f"{min_overlap}; the window is too short to check"
        )
        return out
    if noise >= abs(ov - min_overlap):
        out.status = INCONCLUSIVE
        out.reason = (
            f"overlap {ov:.3f} is within its split-half noise {noise:.3f} "
            f"of the bar {min_overlap}"
        )
        return out
    if ov < min_overlap:
        out.status = STALE
        out.reason = (
            f"{ov:.3f} of the live operator's trace lies in the plan's read subspace "
            f"(< {min_overlap})"
        )
    else:
        out.status = VALID
        out.reason = f"overlap {ov:.3f} >= {min_overlap} (noise {noise:.3f})"
    return out


# --------------------------------------------------------------------------- #
# Windows and plans                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class Window:
    """A closed window of observations: per layer, numpy arrays on the CPU.

    ``P[layer][consumer]`` is ``(2, H, D, D)``, the two split halves summed over
    sampled rows; ``rows[layer]`` is ``(2, H)``, their row counts. Keys, split the
    same way: ``ksum (2, H, D)``, ``kk (2, H, D, D)``, ``krows (2,)``. Entropy:
    ``ent (H,)`` summed bits over ``ent_n (H,)`` rows (summed over the query heads
    that share a key head). Spills: ``err (H, D, D)`` summed error outer products and
    ``err_n`` (rows)."""

    index: int
    tokens: int
    layers: dict = field(default_factory=dict)

    def operator(self, layer: int, consumer: str = "softmax") -> np.ndarray:
        """``(H, D, D)``: the read operator per key head, averaged over sampled rows."""
        L = self.layers[layer]
        rows = L["rows"].sum(0)
        P = L["P"][consumer].sum(0)
        return P / np.maximum(rows, 1)[:, None, None]

    def halves(self, layer: int, consumer: str = "softmax") -> np.ndarray:
        """``(2, H, D, D)``: each half's operator averaged over its own rows."""
        L = self.layers[layer]
        return L["P"][consumer] / np.maximum(L["rows"], 1)[:, :, None, None]

    def key_rows(self, layer: int, half: int | None = None) -> int:
        k = self.layers[layer]["krows"]
        return int(k.sum() if half is None else k[half])

    def key_covariance(self, layer: int, half: int | None = None) -> np.ndarray | None:
        """``(H, D, D)``: the new keys' covariance over the window, or one half's."""
        L = self.layers[layer]
        n = self.key_rows(layer, half)
        if n < 2:
            return None
        ksum = L["ksum"].sum(0) if half is None else L["ksum"][half]
        kk = L["kk"].sum(0) if half is None else L["kk"][half]
        mu = ksum / n
        return kk / n - np.einsum("hd,he->hde", mu, mu)

    def merged(self, other: Window) -> Window:
        """Sum two windows (calibration over several)."""
        out = Window(self.index, self.tokens + other.tokens, {})
        for li in set(self.layers) | set(other.layers):
            a, b = self.layers.get(li), other.layers.get(li)
            if a is None or b is None:
                out.layers[li] = a if b is None else b
                continue
            out.layers[li] = {
                "P": {c: a["P"][c] + b["P"][c] for c in a["P"]},
                "rows": a["rows"] + b["rows"],
                "ksum": a["ksum"] + b["ksum"],
                "kk": a["kk"] + b["kk"],
                "krows": a["krows"] + b["krows"],
                "ent": a["ent"] + b["ent"],
                "ent_n": a["ent_n"] + b["ent_n"],
                "err": a["err"] + b["err"],
                "err_n": a["err_n"] + b["err_n"],
            }
        return out


class ReadPlan:
    """Frozen read operators and key covariances of calibration traffic, per key head.

    Saved as one ``.npz`` (arrays) with a JSON header: the model, the consumer, the
    windows and tokens it was fitted on, and the bits per key channel its regret is
    computed at."""

    def __init__(self, entries: dict, meta: dict):
        self.entries = entries  # (layer, kv_head) -> {"P", "C", "sketch"}
        self.meta = meta

    @classmethod
    def from_window(
        cls,
        window: Window,
        *,
        consumer: str = "softmax",
        bits_per_channel: float = 4.0,
        share: float = 0.9,
        model: str | None = None,
    ) -> ReadPlan:
        entries = {}
        for li in sorted(window.layers):
            P = window.operator(li, consumer)
            halves = window.halves(li, consumer)
            C = window.key_covariance(li)
            d = P.shape[-1]
            # a half's key covariance is used only when it has as many rows as channels
            Ch = [
                window.key_covariance(li, k) if window.key_rows(li, k) >= d else None
                for k in (0, 1)
            ]
            for h in range(P.shape[0]):
                floor = calibration_floor(
                    halves[0, h],
                    halves[1, h],
                    share,
                    Ca=None if Ch[0] is None else Ch[0][h],
                    Cb=None if Ch[1] is None else Ch[1][h],
                    budget_bits=float(bits_per_channel) * d,
                )
                entries[(li, h)] = {
                    "P": P[h],
                    "C": None if C is None else C[h],
                    "sketch": read_subspace(P[h], share),
                    "floor": floor,
                }
        meta = {
            "model": model,
            "consumer": consumer,
            "bits_per_channel": float(bits_per_channel),
            "tokens": int(window.tokens),
            "subspace_share": float(share),
        }
        return cls(entries, meta)

    def save(self, path: str) -> None:
        arrays = {}
        for (li, h), e in self.entries.items():
            arrays[f"P_{li}_{h}"] = e["P"]
            if e["C"] is not None:
                arrays[f"C_{li}_{h}"] = e["C"]
        header = {
            "meta": self.meta,
            "heads": [[li, h] for (li, h) in self.entries],
            "floors": [e.get("floor") for e in self.entries.values()],
        }
        arrays["header"] = np.frombuffer(json.dumps(header).encode(), dtype=np.uint8)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str) -> ReadPlan:
        with np.load(path) as z:
            header = json.loads(bytes(z["header"]).decode())
            share = float(header["meta"].get("subspace_share", 0.9))
            entries = {}
            floors = header.get("floors") or [None] * len(header["heads"])
            for (li, h), fl in zip(header["heads"], floors):
                P = np.array(z[f"P_{li}_{h}"])
                key = f"C_{li}_{h}"
                C = np.array(z[key]) if key in z.files else None
                entries[(li, h)] = {
                    "P": P,
                    "C": C,
                    "sketch": read_subspace(P, share),
                    "floor": fl or {},
                }
        return cls(entries, header["meta"])

    def compare(
        self,
        window: Window,
        *,
        min_overlap: float = DEFAULT_MIN_OVERLAP,
        min_rows: int = DEFAULT_MIN_ROWS,
    ):
        """:class:`HeadComparison` for every head the plan and the window share."""
        consumer = self.meta["consumer"]
        out = []
        for li in sorted(window.layers):
            halves = window.halves(li, consumer)
            P = window.operator(li, consumer)
            C = window.key_covariance(li)
            L = window.layers[li]
            for h in range(halves.shape[1]):
                e = self.entries.get((li, h))
                if e is None:
                    continue
                out.append(
                    compare_head(
                        li, h, e, P[h], (halves[0, h], halves[1, h]),
                        None if C is None else C[h],
                        rows=(float(L["rows"][0, h]), float(L["rows"][1, h])),
                        key_rows=window.key_rows(li),
                        min_overlap=min_overlap,
                        min_rows=min_rows,
                        budget_bits=self.meta["bits_per_channel"] * halves.shape[-1],
                    )
                )  # fmt: skip
        return out


# --------------------------------------------------------------------------- #
# The live monitor (torch + transformers)                                     #
# --------------------------------------------------------------------------- #
class _Accum:
    """One layer's running sums on the attention's device."""

    def __init__(self, torch, n_kv: int, d: int, device, dtype):
        z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)  # noqa: E731
        self.P = {c: z(2, n_kv, d, d) for c in CONSUMERS}
        self.rows = z(2, n_kv)
        self.ksum, self.kk = z(2, n_kv, d), z(2, n_kv, d, d)
        self.krows = [0, 0]
        self.ent, self.ent_n = z(n_kv), z(n_kv)
        self.err, self.err_n = z(n_kv, d, d), 0
        self.calls = 0

    def snapshot(self) -> dict:
        f = lambda t: t.detach().to("cpu").double().numpy()  # noqa: E731
        return {
            "P": {c: f(v) for c, v in self.P.items()},
            "rows": f(self.rows),
            "ksum": f(self.ksum),
            "kk": f(self.kk),
            "krows": np.array(self.krows, dtype=np.int64),
            "ent": f(self.ent),
            "ent_n": f(self.ent_n),
            "err": f(self.err),
            "err_n": int(self.err_n),
        }


class ModelMonitor:
    """Attach to a Hugging Face model and read its consumers while it runs.

    ::

        mon = ModelMonitor(model, window_tokens=256)
        with mon:
            model.generate(**inputs)
        for w in mon.windows: ...
        plan = mon.plan()                    # calibrate on what was seen
        mon.compare(plan, mon.windows[-1])   # or score later traffic against a plan

    ``sample_every`` observes every n-th call of each layer; ``max_rows`` caps the query
    rows read per call; ``layers`` restricts observation to some layer indices.
    """

    def __init__(
        self,
        model,
        *,
        window_tokens: int = 256,
        sample_every: int = 1,
        max_rows: int = 64,
        layers=None,
        keep_windows: int = 64,
        name: str | None = None,
    ):
        import torch
        from transformers import AttentionInterface

        self._torch = torch
        self._AI = AttentionInterface
        self.model = model
        self.name = name or getattr(
            getattr(model, "config", None), "_name_or_path", None
        )
        self.window_tokens = int(window_tokens)
        self.sample_every = max(1, int(sample_every))
        self.max_rows = max(1, int(max_rows))
        self.layers = None if layers is None else set(int(i) for i in layers)
        self.keep_windows = int(keep_windows)
        self.windows: list[Window] = []
        self._acc: dict[int, _Accum] = {}
        self._tokens = 0
        self._first_layer: int | None = None
        self._index = 0
        self._lock = threading.Lock()
        self._prior: str | None = None
        self._attached = False
        self._caches: list = []

    # -- attach / detach ---------------------------------------------------- #
    def _impl_config(self):
        cfg = self.model.config
        return getattr(cfg, "get_text_config", lambda: cfg)()

    def attach(self) -> ModelMonitor:
        if self._attached:
            return self
        from transformers.masking_utils import AttentionMaskInterface

        base = self._impl_config()._attn_implementation or "sdpa"
        if base.startswith(_ATTN_PREFIX):
            raise RuntimeError("another ModelMonitor is attached to this model")
        base_fn = self._AI._global_mapping.get(base)
        if base_fn is None and base != "eager":
            raise ValueError(f"cannot wrap attention implementation {base!r}")
        name = f"{_ATTN_PREFIX}_{id(self)}_{base}"
        monitor = self

        def observed(module, query, key, value, attention_mask, **kwargs):
            fn = base_fn
            if fn is None:  # eager: the model's own, from its modeling module
                import sys

                fn = sys.modules[type(module).__module__].eager_attention_forward
            out = fn(module, query, key, value, attention_mask, **kwargs)
            try:
                monitor._observe(module, query, key, attention_mask, kwargs)
            except Exception as exc:  # telemetry never breaks inference
                monitor.errors.append(repr(exc))
            return out

        self.errors: list[str] = []
        self._AI.register(name, observed)
        mask_fn = AttentionMaskInterface._global_mapping.get(base)
        if mask_fn is not None:
            AttentionMaskInterface.register(name, mask_fn)
        self._prior, self._name = base, name
        self.model.set_attn_implementation(name)
        self._attached = True
        return self

    def detach(self) -> None:
        if not self._attached:
            return
        self.model.set_attn_implementation(self._prior)
        self._AI._global_mapping.pop(self._name, None)
        from transformers.masking_utils import AttentionMaskInterface

        AttentionMaskInterface._global_mapping.pop(self._name, None)
        for cache in self._caches:
            hooks = getattr(cache, "spill_hooks", None)
            if hooks is not None and self._on_spill in hooks:
                hooks.remove(self._on_spill)
        self._caches = []
        self._attached = False

    def __enter__(self) -> ModelMonitor:
        return self.attach()

    def __exit__(self, *exc) -> None:
        self.close_window()
        self.detach()

    def watch_cache(self, cache) -> None:
        """Measure the key codec of a :class:`TurboQuantCache` on every spilled
        block."""
        cache.spill_hooks.append(self._on_spill)
        self._caches.append(cache)

    # -- observation -------------------------------------------------------- #
    def _rows(self, n: int):
        if n <= self.max_rows:
            return self._torch.arange(n)
        idx = np.unique(np.linspace(0, n - 1, self.max_rows).round().astype(np.int64))
        return self._torch.from_numpy(idx)

    def _accum(self, li: int, n_kv: int, d: int, device) -> _Accum:
        a = self._acc.get(li)
        if a is None:
            dt = self._torch.float64 if device.type == "cpu" else self._torch.float32
            a = self._acc[li] = _Accum(self._torch, n_kv, d, device, dt)
        return a

    def _observe(self, module, query, key, mask, kwargs) -> None:
        torch = self._torch
        li = getattr(module, "layer_idx", None)
        if li is None:  # not a decoder layer (a vision tower's attention, say)
            return
        li = int(li)
        if self.layers is not None and li not in self.layers:
            return
        B, Hq, Sq, D = query.shape
        Hkv, Sk = key.shape[1], key.shape[2]
        with self._lock:
            if self._first_layer is None or li < self._first_layer:
                self._first_layer = li
            if li == self._first_layer:
                if self._tokens >= self.window_tokens:
                    self._close_locked()
                self._tokens += B * Sq
            acc = self._accum(li, Hkv, D, query.device)
            acc.calls += 1
            if (acc.calls - 1) % self.sample_every:
                return
            half = (acc.calls - 1) // self.sample_every % 2
        with torch.no_grad():
            self._measure(acc, half, query, key, mask, kwargs, module, Hq, Hkv, Sq, Sk)

    def _measure(self, acc, half, query, key, mask, kwargs, module, Hq, Hkv, Sq, Sk):
        torch = self._torch
        dt = acc.rows.dtype
        D = query.shape[-1]
        G = Hq // Hkv
        idx = self._rows(Sq).to(query.device)
        R = idx.numel()
        q = query.index_select(2, idx).to(dt)  # (B, Hq, R, D)
        k = key.to(dt)  # (B, Hkv, Sk, D)
        B = q.shape[0]
        s = kwargs.get("scaling")
        s = float(module.head_dim**-0.5 if s is None else s)
        qg = q.view(B, Hkv, G, R, D)
        z = torch.einsum("bhgrd,bhsd->bhgrs", qg, k) * s
        cap = kwargs.get("softcap")
        g = None
        if cap:
            t = torch.tanh(z / cap)
            z, g = t * cap, 1.0 - t * t
        # the row's visible keys, exactly as the model masks them
        qpos = (Sk - Sq) + idx  # absolute position of each sampled row
        if mask is not None and mask.dim() == 4:
            m = mask.index_select(2, idx)[..., :Sk]  # (B|1, 1|Hq, R, Sk)
            if m.shape[1] == 1:
                m = m.unsqueeze(2)  # (B|1, 1, 1, R, Sk)
            else:
                m = m.reshape(m.shape[0], Hkv, G, R, Sk)
            if m.dtype == torch.bool:
                keep = m.expand_as(z)
                z = z.masked_fill(~keep, float("-inf"))
            else:
                mf = m.to(dt)
                keep = (mf > torch.finfo(mask.dtype).min / 2).expand_as(z)
                z = z + mf
                z = z.masked_fill(~keep, float("-inf"))
        elif mask is None:
            kpos = torch.arange(Sk, device=z.device)
            keep2 = kpos[None, :] <= qpos[:, None]
            win = kwargs.get("sliding_window")
            if win:
                keep2 = keep2 & (kpos[None, :] > qpos[:, None] - int(win))
            keep = keep2.view(1, 1, 1, R, Sk).expand_as(z)
            z = z.masked_fill(~keep, float("-inf"))
        else:
            raise ValueError(f"unsupported attention mask of shape {tuple(mask.shape)}")
        p = torch.softmax(z, dim=-1)
        p = torch.nan_to_num(p, nan=0.0)  # fully masked rows (padding) carry nothing
        live = keep.any(-1)  # (B, Hkv, G, R)
        a = softmax_gram_weights(p, None if g is None else g.masked_fill(~keep, 0.0))
        n = keep.sum(-1).to(dt)
        lf = live.to(dt)
        s2 = s * s
        for cname, w in (("softmax", a), ("logit", n)):
            acc.P[cname][half] += torch.einsum(
                "bhgr,bhgrd,bhgre->hde", w * lf * s2, qg, qg
            )
        acc.rows[half] += lf.sum((0, 2, 3)) / G
        # entropy of the sampled rows, in bits
        plogp = torch.where(
            p > 0, p * torch.log2(p.clamp_min(1e-300)), torch.zeros_like(p)
        )
        acc.ent += (-plogp.sum(-1) * lf).sum((0, 2, 3))
        acc.ent_n += lf.sum((0, 2, 3))
        # the new keys this call appended (the source a plan codes)
        kn = key.index_select(2, (Sk - Sq) + idx).to(dt)  # (B, Hkv, R, D)
        acc.ksum[half] += kn.sum((0, 2))
        acc.kk[half] += torch.einsum("bhrd,bhre->hde", kn, kn)
        acc.krows[half] += B * R

    def _on_spill(self, layer, original, reconstructed) -> None:
        """A TurboQuantCache layer compressed a block: accumulate its key error."""
        torch = self._torch
        li = None
        for cache in self._caches:
            for i, lay in enumerate(cache.layers):
                if lay is layer:
                    li = i
        if li is None or (self.layers is not None and li not in self.layers):
            return
        with self._lock:
            B, H, S, D = original.shape
            acc = self._accum(li, H, D, original.device)
            e = (original.to(acc.err.dtype) - reconstructed.to(acc.err.dtype)).to(
                acc.err.device
            )
            acc.err += torch.einsum("bhsd,bhse->hde", e, e)
            acc.err_n += B * S

    # -- windows ------------------------------------------------------------ #
    def _close_locked(self) -> None:
        if not self._acc or self._tokens == 0:
            return
        w = Window(
            self._index, self._tokens, {li: a.snapshot() for li, a in self._acc.items()}
        )
        self.windows.append(w)
        if len(self.windows) > self.keep_windows:
            self.windows.pop(0)
        self._index += 1
        self._tokens = 0
        self._acc = {}

    def close_window(self) -> Window | None:
        """Close the current window now (also done on exit) and return it."""
        with self._lock:
            n = len(self.windows)
            self._close_locked()
            return self.windows[-1] if len(self.windows) > n else None

    def calibration(self) -> Window:
        """Every kept window summed: the traffic a plan is fitted to."""
        if not self.windows:
            raise RuntimeError("no closed window yet")
        w = self.windows[0]
        for o in self.windows[1:]:
            w = w.merged(o)
        return w

    def plan(self, **kw) -> ReadPlan:
        return ReadPlan.from_window(self.calibration(), model=self.name, **kw)

    # -- readings ----------------------------------------------------------- #
    def readings(
        self,
        window: Window,
        plan: ReadPlan | None = None,
        consumer="softmax",
        **compare_kw,
    ):
        """Every reading of a window, one dict per metric and key head, each carrying
        its spec (unit, kind, source) and ``labels``; with a plan, its status too
        (``compare_kw`` go to :meth:`ReadPlan.compare`)."""
        out = []
        cmp = {}
        if plan is not None:
            consumer = plan.meta["consumer"]
            comparisons = plan.compare(window, **compare_kw)
            cmp = {(c.layer, c.kv_head): c for c in comparisons}

        def put(name, value, labels, reason=None, **extra):
            r = reading(name, value, reason=reason)
            r["labels"] = labels
            r.update(extra)
            out.append(r)

        for li in sorted(window.layers):
            L = window.layers[li]
            P = window.operator(li, consumer)
            H = P.shape[0]
            for h in range(H):
                lab = {"model": self.name, "window": window.index, "layer": li,
                       "kv_head": h, "consumer": consumer,
                       "tokens": window.tokens}  # fmt: skip
                rows = float(L["rows"][:, h].sum())
                if rows == 0:
                    put("model.read.trace", None, lab, "no sampled rows in this window")
                    continue
                ev = np.clip(np.linalg.eigvalsh(_sym(P[h])), 0, None)
                tr, s2 = float(ev.sum()), float((ev**2).sum())
                diag = np.clip(np.diag(P[h]), 0, None)
                put(
                    "model.read.trace",
                    tr,
                    lab,
                    read_profile=list(diag / max(tr, 1e-300)),
                )
                put("model.read.effective_rank", tr * tr / s2 if s2 > 0 else 0.0, lab)
                if L["ent_n"][h] > 0:
                    put(
                        "model.attention.entropy_bits", L["ent"][h] / L["ent_n"][h], lab
                    )
                if L["err_n"]:
                    sd = L["err"][h] / L["err_n"]
                    put("model.kv.key_distortion", float(np.trace(P[h] @ sd)), lab)
                c = cmp.get((li, h))
                if c is not None:
                    ex = {"status": c.status, "status_reason": c.reason}
                    # each plan number travels with what unchanged traffic scores (the
                    # plan's calibration halves against each other), where there is one
                    base = {
                        "model.plan.overlap": c.overlap_floor,
                        "model.plan.kappa": c.kappa_floor,
                        "model.plan.regret": c.regret_floor,
                    }
                    for name, val in (("model.plan.overlap", c.overlap),
                                      ("model.plan.kappa", c.kappa),
                                      ("model.plan.scale_log2", c.scale_log2),
                                      ("model.plan.regret", c.regret)):  # fmt: skip
                        extra = dict(ex)
                        if base.get(name) is not None:
                            extra["baseline"] = base[name]
                        if val is None or not math.isfinite(val):
                            why = (
                                c.unavailable.get(name) or c.reason or "not computable"
                            )
                            put(name, None, lab, why, **extra)
                        else:
                            put(name, val, lab, **extra)
        return out

    def compare(self, plan: ReadPlan, window: Window | None = None, **kw):
        window = window if window is not None else self.windows[-1]
        return plan.compare(window, **kw)


def status_summary(comparisons) -> dict[str, Any]:
    """Counts by status, and the heads that are not VALID, worst overlap first."""
    counts: dict[str, int] = {}
    for c in comparisons:
        counts[c.status] = counts.get(c.status, 0) + 1
    bad = sorted(
        (c for c in comparisons if c.status != VALID),
        key=lambda c: (c.overlap if c.overlap is not None else -1.0),
    )
    return {"counts": counts, "not_valid": [c.to_dict() for c in bad]}

# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Runtime safe fallback — adaptive compression that knows when to back off.

Static quantization is brittle: a recipe that is safe on average can be
catastrophic on the fragile tail (a low-margin router token, a slow SSM channel,
a norm-dominated key batch, a tied retrieval boundary). Every earlier phase built
a *measurement* of that fragility; this module is the policy layer that reads
those measurements and returns a **conservative action** where the operator is
fragile — and lets the cheap path run where margins are wide.

One object, one decision type. Each ``evaluate_*`` consumes the natural input for
its situation (raw arrays, or a summary you already computed) using the shipped
instruments, compares against a floor, and returns a :class:`RuntimeDecision`:

    policy = TQPRuntimePolicy()
    d = policy.evaluate_routing(gate_logits, k=8)
    if d.conservative:
        keep_router_in_fp16()          # d.action == "keep_router_fp16"

The floors are the knobs; the mapping is fixed and matches
``docs/turboquant_pro_next_level_roadmap.md`` Phase 8:

| situation | signal (instrument) | fragile → action |
|---|---|---|
| retrieval | top-k score gap | `rerank_more` |
| certificate | vacuous / low τ floor (`rank_certificate`) | `require_exact_rerank` |
| kv_keys | unknown regime / probe (`a2_probe`) | `per_channel_or_fp16` |
| routing | margin p10 (`operator_sensitivity`) | `keep_router_fp16` |
| decay | slow-channel fraction (`operator_sensitivity`) | `log_tau_or_fp16` |
| a2 | median tangential fraction (`a2_probe`) | `recalibrate_or_disable_polar` |
| index_drift | stale PCA basis (`index.drift`) | `refit_or_migrate` |
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "RuntimeDecision",
    "TQPRuntimePolicy",
    "PROCEED",
    "RERANK_MORE",
    "REQUIRE_EXACT_RERANK",
    "PER_CHANNEL_OR_FP16",
    "KEEP_ROUTER_FP16",
    "LOG_TAU_OR_FP16",
    "RECALIBRATE_OR_DISABLE_POLAR",
    "REFIT_OR_MIGRATE",
]

# Action vocabulary (machine-readable; a caller switches on these).
PROCEED = "proceed"
RERANK_MORE = "rerank_more"
REQUIRE_EXACT_RERANK = "require_exact_rerank"
PER_CHANNEL_OR_FP16 = "per_channel_or_fp16"
WHITENED_KEYS = "whitened_keys"
KEEP_ROUTER_FP16 = "keep_router_fp16"
LOG_TAU_OR_FP16 = "log_tau_or_fp16"
RECALIBRATE_OR_DISABLE_POLAR = "recalibrate_or_disable_polar"
REFIT_OR_MIGRATE = "refit_or_migrate"


@dataclass(frozen=True)
class RuntimeDecision:
    """One adaptive decision. ``conservative`` is True when the policy backed off
    the cheap path; ``action`` is the machine-readable move; ``params`` carries
    actionable numbers (e.g. a suggested oversample)."""

    situation: str
    action: str
    conservative: bool
    reason: str
    measured: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "situation": self.situation,
            "action": self.action,
            "conservative": self.conservative,
            "reason": self.reason,
            "measured": self.measured,
            "params": self.params,
        }


class TQPRuntimePolicy:
    """Fragility floors → conservative actions. Cheap where margins are wide."""

    def __init__(
        self,
        *,
        retrieval_gap_floor: float = 0.01,
        rerank_oversample: int = 20,
        min_tau_floor: float = 0.0,
        routing_margin_floor: float = 0.02,
        decay_slow_fraction_ceiling: float = 0.02,
        radial_drift_floor: float = 0.15,
        basis_drift_floor: float = 0.05,
        unknown_operator_action: str = PER_CHANNEL_OR_FP16,
    ):
        self.retrieval_gap_floor = float(retrieval_gap_floor)
        self.rerank_oversample = int(rerank_oversample)
        self.min_tau_floor = float(min_tau_floor)
        self.routing_margin_floor = float(routing_margin_floor)
        self.decay_slow_fraction_ceiling = float(decay_slow_fraction_ceiling)
        self.radial_drift_floor = float(radial_drift_floor)
        self.basis_drift_floor = float(basis_drift_floor)
        self.unknown_operator_action = unknown_operator_action

    # -- retrieval -------------------------------------------------------
    def evaluate_retrieval(self, top_scores: np.ndarray) -> RuntimeDecision:
        """``top_scores`` (n_queries, k) descending per query. If the boundary
        gap (k-th vs (k-1)-th) is small for many queries, the k-th slot is a
        toss-up → rerank more candidates."""
        s = np.asarray(top_scores, dtype=np.float64)
        if s.ndim != 2 or s.shape[1] < 2:
            raise ValueError("top_scores must be (n_queries, k>=2), descending")
        boundary_gap = s[:, -2] - s[:, -1]
        frac_tied = float(np.mean(boundary_gap < self.retrieval_gap_floor))
        median_gap = float(np.median(boundary_gap))
        fragile = frac_tied > 0.0 and median_gap < self.retrieval_gap_floor
        return RuntimeDecision(
            situation="retrieval",
            action=RERANK_MORE if fragile else PROCEED,
            conservative=fragile,
            reason=(
                f"median boundary gap {median_gap:.4g} "
                f"{'<' if fragile else '>='} floor {self.retrieval_gap_floor:.4g} "
                f"({frac_tied:.0%} of queries tied)"
            ),
            measured={"median_boundary_gap": median_gap, "fraction_tied": frac_tied},
            params={"oversample": self.rerank_oversample} if fragile else {},
        )

    # -- certificate -----------------------------------------------------
    def evaluate_certificate(self, certificate) -> RuntimeDecision:
        """``certificate`` is a :class:`~turboquant_pro.RankCertificate` (or any
        object with ``.vacuous`` and ``.tau_floor``). A vacuous or low-floor
        certificate cannot guarantee the compressed ranking → require exact
        rerank (or refuse certification)."""
        vacuous = bool(certificate.vacuous)
        tau = float(certificate.tau_floor)
        weak = (not np.isfinite(tau)) or tau < self.min_tau_floor
        fragile = vacuous or weak
        return RuntimeDecision(
            situation="certificate",
            action=REQUIRE_EXACT_RERANK if fragile else PROCEED,
            conservative=fragile,
            reason=(
                "certificate vacuous — no rank guarantee"
                if vacuous
                else f"tau floor {tau:.4g} "
                f"{'<' if weak else '>='} min {self.min_tau_floor:.4g}"
            ),
            measured={
                "vacuous": vacuous,
                "tau_floor": None if not np.isfinite(tau) else tau,
            },
        )

    # -- kv keys ---------------------------------------------------------
    def evaluate_kv_keys(
        self,
        keys: np.ndarray | None = None,
        queries: np.ndarray | None = None,
        *,
        regime: str | None = None,
        bits: int = 4,
        seed: int = 0,
        include_whitened: bool = False,
    ) -> RuntimeDecision:
        """Unknown key operator → conservative per-channel/fp16. Otherwise run the
        (A2) consumer probe: if it does not clear the polar quotient, use the
        per-channel family (keys feed Q·Kᵀ; per-vector polar collapses them).

        With ``include_whitened=True`` the probe also considers the ZCA-whitened
        polar family; when it wins, the action is ``WHITENED_KEYS`` (still a
        back-off from the cheap per-vector polar path, but to a better family in
        the correlated direction-concentration regime)."""
        if regime is not None and str(regime).lower() in ("unknown", "unk", ""):
            return RuntimeDecision(
                situation="kv_keys",
                action=self.unknown_operator_action,
                conservative=True,
                reason="key operator regime is unknown — take the safe family",
                measured={"regime": regime},
            )
        if keys is None:
            raise ValueError("provide keys (and optionally queries), or regime")
        from .a2_probe import recommend_key_quantizer

        probe = recommend_key_quantizer(
            np.asarray(keys, dtype=np.float32),
            queries=None if queries is None else np.asarray(queries, dtype=np.float32),
            bits=bits,
            seed=seed,
            include_whitened=include_whitened,
        )
        polar_safe = probe.recommendation == "polar"
        if probe.recommendation == "whitened":
            action = WHITENED_KEYS
        elif polar_safe:
            action = PROCEED
        else:
            action = PER_CHANNEL_OR_FP16
        reason = (
            f"(A2) probe recommends {probe.recommendation!r} "
            f"(polar rho {probe.spearman_polar:.3f} vs per-channel "
            f"{probe.spearman_per_channel:.3f}"
        )
        if probe.spearman_whitened is not None:
            reason += f" vs whitened {probe.spearman_whitened:.3f}"
        reason += f", margin {probe.margin:.3f})"
        return RuntimeDecision(
            situation="kv_keys",
            action=action,
            conservative=not polar_safe,
            reason=reason,
            measured={
                "recommendation": probe.recommendation,
                "median_tangential_fraction": probe.median_tangential_fraction,
            },
        )

    # -- routing ---------------------------------------------------------
    def evaluate_routing(self, gate_logits: np.ndarray, k: int = 1) -> RuntimeDecision:
        """Tiny routing margins → keep the router in higher precision. Uses the
        p10 margin (the fragile tail) at the top-``k`` boundary."""
        from .operator_sensitivity import routing_sensitivity

        s = routing_sensitivity(np.asarray(gate_logits), k=k)
        fragile = s.margin_p10 < self.routing_margin_floor
        return RuntimeDecision(
            situation="routing",
            action=KEEP_ROUTER_FP16 if fragile else PROCEED,
            conservative=fragile,
            reason=(
                f"p10 routing margin {s.margin_p10:.4g} "
                f"{'<' if fragile else '>='} floor {self.routing_margin_floor:.4g}"
            ),
            measured={"margin_p10": s.margin_p10, "margin_p50": s.margin_p50},
        )

    # -- decay -----------------------------------------------------------
    def evaluate_decay(
        self, decays: np.ndarray, seq_len: int | None = None
    ) -> RuntimeDecision:
        """Many slow (long-memory) channels → quantize in the log-time-constant
        basis, or keep fp16. Uses the fraction of channels with ``a > 0.9``."""
        from .operator_sensitivity import state_decay_sensitivity

        s = state_decay_sensitivity(np.asarray(decays), seq_len=seq_len)
        fragile = s.slow_channel_fraction > self.decay_slow_fraction_ceiling
        return RuntimeDecision(
            situation="decay",
            action=LOG_TAU_OR_FP16 if fragile else PROCEED,
            conservative=fragile,
            reason=(
                f"slow-channel fraction {s.slow_channel_fraction:.4g} "
                f"{'>' if fragile else '<='} ceiling "
                f"{self.decay_slow_fraction_ceiling:.4g} (max gain {s.max_gain:.1f})"
            ),
            measured={
                "slow_channel_fraction": s.slow_channel_fraction,
                "max_gain": s.max_gain,
            },
        )

    # -- a2 drift --------------------------------------------------------
    def evaluate_a2(
        self,
        batch: np.ndarray | None = None,
        *,
        median_tangential_fraction: float | None = None,
    ) -> RuntimeDecision:
        """Low median tangential fraction = norm-dominated displacements, where
        the angular / polar quotient throws away signal → recalibrate or disable
        it. Pass a ``batch`` (rows are vectors) or a precomputed median."""
        if median_tangential_fraction is None:
            if batch is None:
                raise ValueError("provide batch or median_tangential_fraction")
            from .a2_probe import displacement_decomposition

            median_tangential_fraction = float(
                displacement_decomposition(np.asarray(batch)).median_tangential_fraction
            )
        mtf = float(median_tangential_fraction)
        fragile = mtf < self.radial_drift_floor
        return RuntimeDecision(
            situation="a2",
            action=RECALIBRATE_OR_DISABLE_POLAR if fragile else PROCEED,
            conservative=fragile,
            reason=(
                f"median tangential fraction {mtf:.4g} "
                f"{'<' if fragile else '>='} floor {self.radial_drift_floor:.4g}"
            ),
            measured={"median_tangential_fraction": mtf},
        )

    # -- index drift -----------------------------------------------------
    def evaluate_index_drift(self, drift_report) -> RuntimeDecision:
        """``drift_report`` is a :class:`~turboquant_pro.DriftReport` (or any
        object with ``.stale`` / ``.retained_var_drop``). A stale PCA basis →
        refit the basis or migrate the index."""
        stale = bool(drift_report.stale)
        drop = float(getattr(drift_report, "retained_var_drop", float("nan")))
        return RuntimeDecision(
            situation="index_drift",
            action=REFIT_OR_MIGRATE if stale else PROCEED,
            conservative=stale,
            reason=(
                f"PCA basis {'stale' if stale else 'fresh'}: retained-variance "
                f"drop {drop:.4g} vs floor {self.basis_drift_floor:.4g}"
            ),
            measured={
                "stale": stale,
                "retained_var_drop": None if not np.isfinite(drop) else drop,
            },
        )

    # -- aggregate -------------------------------------------------------
    def evaluate_all(self, **inputs) -> list[RuntimeDecision]:
        """Run every evaluator for which an input was supplied. Recognized keys:
        ``top_scores``, ``certificate``, ``kv_keys`` (dict of kwargs or array),
        ``gate_logits``, ``decays``, ``a2_batch``, ``drift_report``. Returns the
        decisions in a stable order."""
        out: list[RuntimeDecision] = []
        if "top_scores" in inputs:
            out.append(self.evaluate_retrieval(inputs["top_scores"]))
        if "certificate" in inputs:
            out.append(self.evaluate_certificate(inputs["certificate"]))
        if "kv_keys" in inputs:
            kw = inputs["kv_keys"]
            out.append(
                self.evaluate_kv_keys(**kw)
                if isinstance(kw, dict)
                else self.evaluate_kv_keys(kw)
            )
        if "regime" in inputs:
            out.append(self.evaluate_kv_keys(regime=inputs["regime"]))
        if "gate_logits" in inputs:
            out.append(
                self.evaluate_routing(inputs["gate_logits"], k=inputs.get("k", 1))
            )
        if "decays" in inputs:
            out.append(
                self.evaluate_decay(inputs["decays"], seq_len=inputs.get("seq_len"))
            )
        if "a2_batch" in inputs:
            out.append(self.evaluate_a2(inputs["a2_batch"]))
        if "drift_report" in inputs:
            out.append(self.evaluate_index_drift(inputs["drift_report"]))
        return out

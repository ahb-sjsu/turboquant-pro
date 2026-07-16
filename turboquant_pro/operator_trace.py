# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""
Operator-regime tracing: infer each tensor's (A2) quantization discipline.

The (A2) condition (``a2_probe``, and the companion theory paper) says a
scale-discarding quotient is safe exactly when the *consumer's* metric is
carried by the tangential part of the displacement. Until now the consumer
had to be *declared* by hand: we manually knew KV-keys feed
``softmax(Q.K^T)`` (per-channel scale is vital) while V/O weights write the
residual stream linearly (symmetric quantization is safe).

This module *infers* the consumer by looking at the model. It maps every
parameter tensor to the **operator regime** of whatever its output flows
into, and from the regime derives the quantization discipline --- no human
declaration. Two front-ends, combined (``prefer="auto"``):

* **Structural** (always available): classify each parameter from its owning
  module's type and name (``k_proj`` -> score path, ``o_proj`` -> residual,
  a router ``gate`` -> selection, a Mamba ``A_log`` -> state decay, a
  ``LayerNorm`` -> norm). Robust and dependency-light, but blind to renamed
  or novel operators.
* **Graph (torch.fx)** (best-effort): symbolically trace the model, find the
  *sink* operators --- ``softmax``/``exp``, ``topk``/``argmax`` (routing),
  ``cumsum``/scan (recurrence) --- and backtrace to the ``Linear`` layers
  that feed them. This tags the score/gate/state tensors even when their
  names give nothing away, which is what makes the pass work on an *unseen*
  architecture. Where fx has positive evidence it overrides the structural
  guess; everything else keeps the structural baseline.

The regime -> discipline table is the (A2) content, grounded in this
project's findings (``docs/KV_KEYS_FINDING.md``,
``docs/notes/projection_sensitivity_deconfounded.md``). Crucially the
discipline depends on *what is being quantized* --- a projection's **weights**
vs. its **KV-cache activations** --- because the sensitive coordinate flips
between the two operators:

* Attention **keys as cached activations** carry a per-channel DC offset the
  softmax reads directly: fragile, need per-channel + zero-point (keys
  finding). The same projection's **weights**, however, are the *robust* side
  under weight PTQ --- softmax absorbs a shared perturbation to ``W_K``
  (matched-bit finding), so symmetric and fewer bits are fine.
* **Values as cached activations** are averaged by attention: cheap (polar).
  But V/O **weights** write the residual almost linearly and are the *most*
  behaviorally sensitive projections under weight PTQ (2.3--6x more than Q/K):
  symmetric family (no DC), but allocate *more* bits.

So ``QuantizationDiscipline`` carries two independent axes --- the family /
DC-protection (a correctness choice) and the sensitivity / bit-allocation (a
budget choice) --- resolved per ``(regime, target)``.

This module needs PyTorch; ``import`` is deferred so the rest of the library
stays numpy-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "OperatorRegime",
    "QuantTarget",
    "QuantizationDiscipline",
    "TensorRegime",
    "OperatorPlan",
    "discipline_for",
    "trace_operators",
    "recommend_quantization",
]


class OperatorRegime(Enum):
    """The operator a tensor's output flows into --- its (A2) consumer."""

    SOFTMAX_SCORE = "softmax_score"  # feeds softmax(Q.K^T): the attention score path
    LINEAR_RESIDUAL = "linear_residual"  # writes the residual ~linearly (V, O, MLP)
    GATE_SELECTION = "gate_selection"  # top-k routing on gate-logit margins/order
    STATE_DECAY = "state_decay"  # SSM per-channel recurrence / decay
    NORM = "norm"  # layernorm / rmsnorm scale
    UNKNOWN = "unknown"  # no evidence: fall back to the conservative discipline


class QuantTarget(Enum):
    """What is being quantized --- the sensitive coordinate flips between these."""

    WEIGHT = "weight"  # the projection's weight matrix (weight PTQ)
    KV_ACTIVATION = "kv_activation"  # the cached key/value activations (KV-cache quant)


@dataclass(frozen=True)
class QuantizationDiscipline:
    """The (A2) prescription for a tensor.

    Two independent axes:

    * ``family`` / ``protect_dc`` --- a *correctness* choice: which quotient is
      safe. ``"per_channel"`` keeps per-channel affine structure;
      ``"symmetric"`` is a zero-centred grid; ``"polar"`` keeps the per-vector
      norm and quantizes direction; ``"keep_fp"`` says do not quantize.
      ``protect_dc`` flags that a per-channel zero-point is required (the DC
      offset the consumer reads).
    * ``sensitivity`` --- a *budget* choice: ``"low"`` tolerates aggressive
      bit-cutting, ``"high"`` should be protected with more bits.
    """

    family: str
    protect_dc: bool
    sensitivity: str
    rationale: str


# The (A2) table. Keyed by (regime, target); grounded in the project findings.
_LOW, _MED, _HIGH = "low", "medium", "high"
_DISCIPLINE: dict[tuple[OperatorRegime, QuantTarget], QuantizationDiscipline] = {
    (OperatorRegime.SOFTMAX_SCORE, QuantTarget.WEIGHT): QuantizationDiscipline(
        "symmetric",
        False,
        _LOW,
        "Q/K weight PTQ is the robust side: softmax absorbs a shared "
        "perturbation to the projection (matched-bit finding).",
    ),
    (OperatorRegime.SOFTMAX_SCORE, QuantTarget.KV_ACTIVATION): QuantizationDiscipline(
        "per_channel",
        True,
        _HIGH,
        "Cached keys carry a per-channel DC offset that softmax(Q.K^T) reads "
        "directly; per-vector/symmetric quotients collapse generation "
        "(keys finding).",
    ),
    (OperatorRegime.LINEAR_RESIDUAL, QuantTarget.WEIGHT): QuantizationDiscipline(
        "symmetric",
        False,
        _HIGH,
        "V/O/MLP write the residual ~linearly: no DC dependence (symmetric is "
        "safe) but the most behaviorally sensitive projections under weight "
        "PTQ (2.3-6x more than Q/K) -- allocate more bits.",
    ),
    (OperatorRegime.LINEAR_RESIDUAL, QuantTarget.KV_ACTIVATION): QuantizationDiscipline(
        "polar",
        False,
        _LOW,
        "Cached values are averaged by attention: per-vector polar is "
        "near-lossless (values finding).",
    ),
    (OperatorRegime.GATE_SELECTION, QuantTarget.WEIGHT): QuantizationDiscipline(
        "per_channel",
        True,
        _HIGH,
        "Top-k routing reads the relative order and margins of the gate logits, "
        "not their absolute magnitude: a common-mode shift is free, but "
        "per-expert scale/offset error changes margins and can flip the expert "
        "choice. Protect per-channel scale.",
    ),
    (OperatorRegime.GATE_SELECTION, QuantTarget.KV_ACTIVATION): QuantizationDiscipline(
        "per_channel",
        True,
        _HIGH,
        "Routing reads relative order/margins, not absolute magnitude; "
        "per-channel scale and offset errors move margins and flip selection, "
        "so preserve per-channel scale (+ zero-point).",
    ),
    (OperatorRegime.STATE_DECAY, QuantTarget.WEIGHT): QuantizationDiscipline(
        "per_channel",
        True,
        _HIGH,
        "SSM per-channel decay/selection scale is the geometry of the "
        "recurrence; a shared quotient destroys the time constant.",
    ),
    (OperatorRegime.STATE_DECAY, QuantTarget.KV_ACTIVATION): QuantizationDiscipline(
        "per_channel",
        True,
        _HIGH,
        "Hidden-state per-channel scale must survive the scan.",
    ),
}

# Norm and unknown are target-independent.
_NORM_DISCIPLINE = QuantizationDiscipline(
    "keep_fp",
    False,
    _LOW,
    "Norm scales are tiny and high-leverage; keep full precision.",
)
_UNKNOWN_DISCIPLINE = QuantizationDiscipline(
    "per_channel",
    True,
    _MED,
    "Unknown consumer: default to the conservative per-channel + zero-point "
    "discipline (never discard scale on an unclassified tensor).",
)


def discipline_for(
    regime: OperatorRegime, target: QuantTarget = QuantTarget.WEIGHT
) -> QuantizationDiscipline:
    """The (A2) discipline for a regime under a quantization target."""
    if regime is OperatorRegime.NORM:
        return _NORM_DISCIPLINE
    if regime is OperatorRegime.UNKNOWN:
        return _UNKNOWN_DISCIPLINE
    return _DISCIPLINE[(regime, target)]


@dataclass
class TensorRegime:
    """A parameter tensor's inferred regime and how it was inferred."""

    name: str
    regime: OperatorRegime
    method: str  # "structural" | "fx"
    confidence: float

    def discipline(
        self, target: QuantTarget = QuantTarget.WEIGHT
    ) -> QuantizationDiscipline:
        return discipline_for(self.regime, target)


@dataclass
class OperatorPlan:
    """Per-tensor regime plan for a model."""

    tensors: dict[str, TensorRegime]
    traced: bool  # did the fx graph pass succeed?

    @property
    def coverage(self) -> float:
        """Fraction of tensors classified to a non-UNKNOWN regime."""
        if not self.tensors:
            return 0.0
        known = sum(
            1 for t in self.tensors.values() if t.regime is not OperatorRegime.UNKNOWN
        )
        return known / len(self.tensors)

    def by_regime(self) -> dict[OperatorRegime, list[str]]:
        out: dict[OperatorRegime, list[str]] = {r: [] for r in OperatorRegime}
        for name, t in self.tensors.items():
            out[t.regime].append(name)
        return {r: names for r, names in out.items() if names}

    def families(self, target: QuantTarget = QuantTarget.WEIGHT) -> dict[str, str]:
        """name -> recommended quantizer family under ``target``."""
        return {name: t.discipline(target).family for name, t in self.tensors.items()}

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for t in self.tensors.values():
            counts[t.regime.value] = counts.get(t.regime.value, 0) + 1
        return {
            "n_tensors": len(self.tensors),
            "traced": self.traced,
            "coverage": self.coverage,
            "regime_counts": counts,
        }


# ---------------------------------------------------------------------------
# Structural front-end (always available)
# ---------------------------------------------------------------------------

# Ordered, most-specific first. Each entry: (compiled regex over the lowercased
# module path, regime, confidence). The first match wins.
_STRUCTURAL_RULES: list[tuple[re.Pattern, OperatorRegime, float]] = [
    # State-space / recurrence signals (Mamba, RetNet, RWKV, S4/S6).
    (
        re.compile(
            r"(a_log|dt_proj|x_proj|conv1d|\bssm\b|mamba|\bs6\b|"
            r"retention|\bdecay\b|time_decay|time_mix)"
        ),
        OperatorRegime.STATE_DECAY,
        0.75,
    ),
    # MoE / routing. `gate` as a router, but NOT SwiGLU's `gate_proj`/`gate_up`.
    (
        re.compile(
            r"(\brouter\b|routing|expert_gate|gate_score|" r"\bgate\b(?!_proj|_up))"
        ),
        OperatorRegime.GATE_SELECTION,
        0.8,
    ),
    # Attention score path (queries and keys enter softmax(Q.K^T)).
    (
        re.compile(
            r"(q_proj|k_proj|\bquery\b|\bkey\b|\bwq\b|\bwk\b|q_lin|k_lin|"
            r"query_key_value|c_attn|in_proj)"
        ),
        OperatorRegime.SOFTMAX_SCORE,
        0.85,
    ),
    # Residual-writing linears (values, output, MLP).
    (
        re.compile(
            r"(v_proj|o_proj|out_proj|\bvalue\b|\bwv\b|\bwo\b|v_lin|"
            r"gate_proj|up_proj|down_proj|fc1|fc2|c_fc|c_proj|"
            r"\bmlp\b|\bdense\b|\bw1\b|\bw2\b|\bw3\b|lm_head)"
        ),
        OperatorRegime.LINEAR_RESIDUAL,
        0.7,
    ),
]

_NORM_TYPES = ("LayerNorm", "RMSNorm", "GroupNorm", "BatchNorm", "RMSNormGated")


def _structural(model) -> dict[str, TensorRegime]:
    import torch.nn as nn

    module_types: dict[str, str] = {
        name: type(m).__name__ for name, m in model.named_modules()
    }
    is_mha: dict[str, bool] = {
        name: isinstance(m, nn.MultiheadAttention) for name, m in model.named_modules()
    }

    out: dict[str, TensorRegime] = {}
    for pname, _ in model.named_parameters():
        mod_name = pname.rsplit(".", 1)[0] if "." in pname else ""
        mtype = module_types.get(mod_name, "")
        # Match against the full parameter path so leaf-level names (a bare
        # ``A_log`` parameter with no owning module) are seen too.
        search = pname.lower()

        # Norm by module type (robust to naming) or an explicit norm token.
        if any(nt in mtype for nt in _NORM_TYPES) or re.search(
            r"(layer_?norm|rms_?norm|group_?norm|batch_?norm|\bnorm\b)", search
        ):
            out[pname] = TensorRegime(pname, OperatorRegime.NORM, "structural", 0.9)
            continue

        # nn.MultiheadAttention fuses QKV in in_proj_weight -> score path.
        if is_mha.get(mod_name) and "in_proj" in pname:
            out[pname] = TensorRegime(
                pname, OperatorRegime.SOFTMAX_SCORE, "structural", 0.8
            )
            continue

        regime, conf = OperatorRegime.UNKNOWN, 0.0
        for pat, reg, c in _STRUCTURAL_RULES:
            if pat.search(search):
                regime, conf = reg, c
                break

        # A bare Linear with no naming signal writes the residual by default
        # (the sensible prior; symmetric, high sensitivity). Non-Linear
        # unclassified params stay UNKNOWN -> the conservative discipline.
        if regime is OperatorRegime.UNKNOWN and "Linear" in mtype:
            regime, conf = OperatorRegime.LINEAR_RESIDUAL, 0.4

        out[pname] = TensorRegime(pname, regime, "structural", conf)
    return out


# ---------------------------------------------------------------------------
# Graph front-end (torch.fx, best-effort)
# ---------------------------------------------------------------------------

# Sink operators, matched by function object or method name, mapped to regime.
_SINK_METHODS = {
    "softmax": OperatorRegime.SOFTMAX_SCORE,
    "topk": OperatorRegime.GATE_SELECTION,
    "argmax": OperatorRegime.GATE_SELECTION,
    "argsort": OperatorRegime.GATE_SELECTION,
    "cumsum": OperatorRegime.STATE_DECAY,
    "cumprod": OperatorRegime.STATE_DECAY,
    "logcumsumexp": OperatorRegime.STATE_DECAY,
}


def _sink_regime(node) -> OperatorRegime | None:
    """Regime if ``node`` is a recognised sink operator, else None."""
    target = node.target
    name = getattr(target, "__name__", None) or (
        target if isinstance(target, str) else ""
    )
    name = str(name).lower()
    for key, regime in _SINK_METHODS.items():
        if name == key or name.endswith("." + key):
            return regime
    # scatter/one_hot are gate materialisations.
    if "scatter" in name or "one_hot" in name:
        return OperatorRegime.GATE_SELECTION
    return None


def _linear_targets(gm) -> set[str]:
    import torch.nn as nn

    out = set()
    for name, m in gm.named_modules():
        if isinstance(m, nn.Linear):
            out.add(name)
    return out


def _fx_overrides(model, example_inputs) -> dict[str, tuple[OperatorRegime, float]]:
    """Tensors for which the graph gives positive regime evidence.

    Returns ``{param_name: (regime, confidence)}`` for Linear weights (and
    directly-referenced parameters) that feed a recognised sink operator.
    Empty dict if the model cannot be traced.
    """
    import torch.fx as fx

    try:
        gm = fx.symbolic_trace(model)
    except Exception:
        return {}

    linear_mods = _linear_targets(gm)
    param_names = {n for n, _ in model.named_parameters()}
    overrides: dict[str, tuple[OperatorRegime, float]] = {}

    for node in gm.graph.nodes:
        regime = _sink_regime(node)
        if regime is None:
            continue
        # Backtrace: collect Linear call_module / get_attr param ancestors.
        seen = set()
        stack = list(node.all_input_nodes)
        producers: set[str] = set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            if n.op == "call_module" and n.target in linear_mods:
                producers.add(f"{n.target}.weight")
                # a Linear caps the backtrace: its inputs are upstream activations
                continue
            if n.op == "get_attr" and n.target in param_names:
                producers.add(n.target)
            stack.extend(n.all_input_nodes)
        for pname in producers:
            # Highest-priority sink wins if a tensor feeds several.
            prev = overrides.get(pname)
            if prev is None or _priority(regime) > _priority(prev[0]):
                overrides[pname] = (regime, 0.9)
    return overrides


def _priority(regime: OperatorRegime) -> int:
    # Specialised sinks outrank the linear-residual baseline.
    order = {
        OperatorRegime.SOFTMAX_SCORE: 3,
        OperatorRegime.GATE_SELECTION: 3,
        OperatorRegime.STATE_DECAY: 3,
        OperatorRegime.LINEAR_RESIDUAL: 1,
        OperatorRegime.NORM: 1,
        OperatorRegime.UNKNOWN: 0,
    }
    return order.get(regime, 0)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def trace_operators(
    model,
    example_inputs=None,
    prefer: str = "auto",
) -> OperatorPlan:
    """Infer the operator regime of every parameter tensor in ``model``.

    Args:
        model: a ``torch.nn.Module``.
        example_inputs: currently unused (reserved for shape-driven tracing);
            fx symbolic tracing does not need it.
        prefer: ``"auto"`` (structural baseline, upgraded by fx where the graph
            gives evidence), ``"structural"`` (names/types only), or ``"fx"``
            (require a successful graph trace; falls back to structural if it
            cannot trace, with ``traced=False``).

    Returns:
        An :class:`OperatorPlan`.
    """
    try:
        import torch.nn as nn  # noqa: F401
    except ImportError as e:  # pragma: no cover - torch is an optional dep
        raise ImportError(
            "operator_trace requires PyTorch: pip install 'turboquant-pro[torch]'"
        ) from e

    plan = _structural(model)
    traced = False

    if prefer in ("auto", "fx"):
        overrides = _fx_overrides(model, example_inputs)
        traced = bool(overrides)
        for pname, (regime, conf) in overrides.items():
            if pname in plan:
                plan[pname] = TensorRegime(pname, regime, "fx", conf)

    return OperatorPlan(tensors=plan, traced=traced)


def recommend_quantization(
    model,
    target: QuantTarget | str = QuantTarget.WEIGHT,
    example_inputs=None,
    prefer: str = "auto",
) -> dict[str, QuantizationDiscipline]:
    """One-shot: ``name -> QuantizationDiscipline`` for every tensor in ``model``.

    The human-out-of-the-loop entry point: trace the model, then map each
    tensor's regime to its (A2) discipline under the given quantization
    ``target`` (weights vs. KV-cache activations).
    """
    if isinstance(target, str):
        target = QuantTarget(target)
    plan = trace_operators(model, example_inputs=example_inputs, prefer=prefer)
    return {name: t.discipline(target) for name, t in plan.tensors.items()}

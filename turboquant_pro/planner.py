# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""The quantization control plane: which codec, for which consumer, and why.

P0 of ``docs/DESIGN_planner.md`` and the feature requested in issue #169.
The library already holds the parts -- a codec registry
(:mod:`turboquant_pro.plugins`), read operators
(:mod:`turboquant_pro.read_operators`), consumer metrics
(:mod:`turboquant_pro.consumers`), distribution-free certificates
(:mod:`turboquant_pro.rank_certificate`), a false-clear diagnostic
(:mod:`turboquant_pro.false_clear`) and conservative runtime actions
(:mod:`turboquant_pro.runtime_policy`). What was missing is the contract that
joins them into a decision a third party can check.

The decision procedure, in one paragraph. A :class:`WorkloadSpec` declares the
artifact, the consumer, the budget and the quality floor. Preflight measures
the data preconditions that change what is legal. Candidates are enumerated
from the plugin registry, not from a hard-coded list, so a third-party codec
competes on the same terms as an in-tree one. Priors on stored bytes rule out
what cannot fit. Survivors are measured on the *consumer's* metric over a
calibration split, cheaply first and then at larger samples, and the frontier
is kept under uncertainty rather than collapsed to a single winner by a mean.
The choice is verified once on a held-out split that played no part in the
search. The output is a record, not a log line: every candidate, the stage it
left at, the rule that chose the winner, the evidence with its kind, the
runtime fallback, and enough provenance to re-run it.

Three properties are load-bearing and each is a thing the package has been
wrong about before.

**The metric is the consumer's.** Reconstruction cosine never decides. It is
measured, on the same items as the consumer, for one purpose: to report how
often it would have cleared a result the consumer rejects. A candidate whose
cosine is excellent and whose consumer metric fails is not a near miss, it is
the failure mode this package exists to name.

**Uncertainty is carried, not dropped.** Quality is a bootstrap interval over
per-item scores, and a plan is accepted on the conservative end of that
interval against the declared floor. A mean that clears a floor by less than
its own noise has not cleared it.

**Not knowing is an answer.** An unregistered consumer, a target no candidate
supports, a floor nothing reaches: each returns a plan whose selection is
``ABSTAIN``, carrying the reason and the fallback. The control plane is
allowed to say that it cannot recommend anything here.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ABSTAIN",
    "Artifact",
    "Budget",
    "CandidateResult",
    "CompressionPlan",
    "CompressionPlanner",
    "QualityEstimate",
    "QualityFloor",
    "QuantizationControlPlane",
    "WorkloadSpec",
    "container_bytes",
    "plan_schema",
    "preflight",
    "replay_plan",
]

SCHEMA = "turboquant-pro/compression-plan"
SCHEMA_VERSION = 1

ABSTAIN = "ABSTAIN"

#: Bit widths tried when a codec declares no ``capabilities()`` of its own.
DEFAULT_BIT_WIDTHS = (2, 3, 4, 8)

#: Cosine above which the cheap nominal metric would have cleared a result.
#: A convention for the false-clear diagnostic, not a measurement; the KV-keys
#: finding is the reason it sits high (cosine read 0.995 while the consumer
#: failed), and a caller with a calibrated value should pass it.
NOMINAL_COSINE_CLEAR = 0.95

#: Constructor parameters a codec may declare to receive the artifact's own
#: geometry. A factory that takes one of these names is handed the measured
#: value, so a codec whose head dimension must match the data does not have to
#: be configured by hand for every artifact. The hint is offered, never
#: imposed: a factory that rejects it is built again without it.
SHAPE_HINTS = ("head_dim", "n_heads", "dim", "input_dim")


# ------------------------------------------------------------------ #
# The workload spec: the planner's intermediate representation         #
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class Budget:
    """Hard constraints. An unmet one makes a candidate infeasible, not worse.

    ``max_bytes_per_vector`` counts every stored byte of the container, codes
    and per-vector scalars alike (``docs/DESIGN_planner.md`` R6).
    ``max_bits`` is the same constraint expressed the way a request usually
    arrives ("4 bit"), and applies to a codec's declared width.
    """

    max_bytes_per_vector: float | None = None
    max_bits: float | None = None
    max_total_bytes: float | None = None

    def as_dict(self) -> dict:
        return {
            "max_bytes_per_vector": self.max_bytes_per_vector,
            "max_bits": self.max_bits,
            "max_total_bytes": self.max_total_bytes,
        }


@dataclass(frozen=True)
class QualityFloor:
    """The minimum acceptable consumer metric, and how sure we must be of it.

    ``confidence`` is one-sided: the conservative end of the bootstrap
    interval, not the mean, has to clear ``minimum``.
    """

    minimum: float
    confidence: float = 0.95

    def as_dict(self) -> dict:
        return {"minimum": self.minimum, "confidence": self.confidence}


@dataclass
class Artifact:
    """What is being compressed, plus the context the consumer needs to read it.

    ``context`` is forwarded to the consumer metric untouched, which is how a
    head's ``queries`` or a retrieval query sample reaches it. ``identity``
    defaults to a content hash so a plan can be tied to the bytes it was made
    for.
    """

    data: np.ndarray
    context: dict = field(default_factory=dict)
    name: str = "artifact"
    identity: str | None = None

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        if self.identity is None:
            self.identity = _sha256_array(self.data)

    @property
    def n_rows(self) -> int:
        return int(np.prod(self.data.shape[:-1]))

    @property
    def dim(self) -> int:
        return int(self.data.shape[-1])


@dataclass(frozen=True)
class WorkloadSpec:
    """One document declaring the problem; every later stage reads only this.

    ``consumer`` names a metric registered in
    :mod:`turboquant_pro.consumers`. ``candidates`` restricts the codecs
    considered; ``None`` means every plugin registered for the target,
    in-tree and out-of-tree alike.
    """

    target: str
    consumer: str
    consumer_config: dict = field(default_factory=dict)
    budget: Budget = field(default_factory=Budget)
    floor: QualityFloor | None = None
    candidates: tuple | None = None
    objective: str = "max_quality"
    seed: int = 0
    holdout_fraction: float = 0.3
    n_boot: int = 512
    nominal_threshold: float = NOMINAL_COSINE_CLEAR
    halving_rounds: int = 3
    min_sample: int = 256

    def __post_init__(self) -> None:
        if self.objective not in {"max_quality", "min_cost"}:
            raise ValueError(
                f"unknown objective {self.objective!r}; "
                "expected 'max_quality' or 'min_cost'"
            )
        if not 0.0 < self.holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must be in (0, 1)")

    def as_dict(self) -> dict:
        return {
            "target": self.target,
            "consumer": self.consumer,
            "consumer_config": dict(self.consumer_config),
            "budget": self.budget.as_dict(),
            "floor": self.floor.as_dict() if self.floor else None,
            "candidates": list(self.candidates) if self.candidates else None,
            "objective": self.objective,
            "seed": self.seed,
            "holdout_fraction": self.holdout_fraction,
            "n_boot": self.n_boot,
            "nominal_threshold": self.nominal_threshold,
        }


# ------------------------------------------------------------------ #
# Measurement primitives                                              #
# ------------------------------------------------------------------ #


def _sha256_array(a: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def container_bytes(container: Any) -> dict:
    """Every stored byte of a compressed container, with a breakdown.

    Walks arrays, bytes, scipy sparse matrices, dataclasses, mappings and
    sequences, and sums what each holds. The accounting is deliberately of the
    *container as it exists in memory*: a codec that packs only on
    serialization reads high here, and a codec that keeps unpacked codes reads
    what it actually costs to hold. Both are honest and the plan says which,
    because the campaign found two harnesses that quietly omitted per-vector
    correction factors and a norm.
    """
    seen = set()
    parts: dict = {}

    def walk(obj: Any, path: str) -> int:
        if obj is None or isinstance(obj, (bool, int, float, complex, str)):
            return 0
        oid = id(obj)
        if oid in seen:
            return 0
        seen.add(oid)
        if isinstance(obj, np.ndarray):
            parts[path] = int(obj.nbytes)
            return int(obj.nbytes)
        if isinstance(obj, (bytes, bytearray, memoryview)):
            n = int(len(bytes(obj)))
            parts[path] = n
            return n
        if isinstance(obj, np.generic):
            return int(obj.itemsize)
        for attr in ("data", "indices", "indptr"):  # scipy sparse
            if hasattr(obj, attr) and hasattr(obj, "nnz"):
                return sum(
                    walk(getattr(obj, a, None), f"{path}.{a}")
                    for a in ("data", "indices", "indptr")
                )
        if isinstance(obj, dict):
            return sum(walk(v, f"{path}.{k}") for k, v in obj.items())
        if isinstance(obj, (list, tuple, set, frozenset)):
            return sum(walk(v, f"{path}[{i}]") for i, v in enumerate(obj))
        state = getattr(obj, "__dict__", None)
        if state:
            return sum(walk(v, f"{path}.{k}") for k, v in state.items())
        slots = getattr(obj, "__slots__", None)
        if slots:
            return sum(
                walk(getattr(obj, s, None), f"{path}.{s}")
                for s in slots
                if hasattr(obj, s)
            )
        return 0

    total = walk(container, "container")
    return {
        "total_bytes": int(total),
        "breakdown": {k: v for k, v in sorted(parts.items()) if v},
        "accounting": "container-in-memory, every array and buffer reachable",
    }


@dataclass(frozen=True)
class QualityEstimate:
    """A consumer metric with the uncertainty it was measured with.

    ``bound`` is the conservative end of the one-sided interval: the lower
    percentile when higher is better, the upper when lower is better. Every
    accept/reject decision reads ``bound``, never ``mean``.
    """

    metric: str
    mean: float
    bound: float
    ci_low: float
    ci_high: float
    n_items: int
    confidence: float
    higher_is_better: bool

    @property
    def score(self) -> float:
        """The conservative value, oriented so larger is always better."""
        return self.bound if self.higher_is_better else -self.bound

    def clears(self, floor: QualityFloor | None) -> bool:
        if floor is None:
            return True
        if self.higher_is_better:
            return self.bound >= floor.minimum
        return self.bound <= floor.minimum

    @classmethod
    def from_dict(cls, d: dict | None):
        """Rebuild from a record, ignoring the derived fields ``as_dict`` adds."""
        if not d:
            return None
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})

    def as_dict(self) -> dict:
        return {
            "metric": self.metric,
            "mean": self.mean,
            "bound": self.bound,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_items": self.n_items,
            "confidence": self.confidence,
            "higher_is_better": self.higher_is_better,
            "bound_is": "lower" if self.higher_is_better else "upper",
        }


def _estimate(
    per_item: np.ndarray,
    *,
    metric: str,
    higher_is_better: bool,
    confidence: float,
    n_boot: int,
    seed: int,
) -> QualityEstimate:
    x = np.asarray(per_item, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        raise ValueError("consumer metric produced no finite per-item scores")
    mean = float(x.mean())
    if n == 1 or n_boot <= 0:
        return QualityEstimate(
            metric=metric,
            mean=mean,
            bound=mean,
            ci_low=mean,
            ci_high=mean,
            n_items=n,
            confidence=confidence,
            higher_is_better=higher_is_better,
        )
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    means = x[idx].mean(axis=1)
    alpha = 1.0 - float(confidence)
    lo = float(np.percentile(means, 100.0 * alpha))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha)))
    bound = lo if higher_is_better else hi
    return QualityEstimate(
        metric=metric,
        mean=mean,
        bound=bound,
        ci_low=lo,
        ci_high=hi,
        n_items=n,
        confidence=confidence,
        higher_is_better=higher_is_better,
    )


# ------------------------------------------------------------------ #
# Preflight                                                           #
# ------------------------------------------------------------------ #


def preflight(x: np.ndarray, *, seed: int = 0, sample: int = 4096) -> dict:
    """Data preconditions that change what is legal or likely.

    Each check here corresponds to something that produced a wrong answer
    once: all-zero rows dropped an exact L2 scan to recall 0.43, a narrow
    norm spread decides whether normalisation changes rankings at all, and
    spectral concentration says whether a dimension reduction is even on the
    table. The report is attached to the plan and the actions it forces are
    listed there, so nothing is silently rewritten.
    """
    a = np.asarray(x, dtype=np.float64)
    a = a.reshape(-1, a.shape[-1])
    n, d = a.shape
    rng = np.random.default_rng(seed)
    if n > sample:
        a = a[rng.choice(n, size=sample, replace=False)]
    norms = np.linalg.norm(a, axis=1)
    zero = int((norms == 0).sum())
    nonfinite = int((~np.isfinite(a)).any(axis=1).sum())
    nz = norms[norms > 0]
    spread = float(nz.max() / nz.min()) if nz.size and nz.min() > 0 else float("inf")
    flags = []
    if zero:
        flags.append(
            f"{zero} all-zero rows in the sample: an L2 or cosine ranking over "
            "them is undefined, and a metric that ignores this reads a recall "
            "that is not the consumer's"
        )
    if nonfinite:
        flags.append(f"{nonfinite} rows contain non-finite values")
    spectrum = {}
    try:
        finite = a[np.isfinite(a).all(axis=1)]
        if finite.shape[0] < 2:
            raise ValueError("too few finite rows for a spectrum")
        centred = finite - finite.mean(axis=0, keepdims=True)
        sv = np.linalg.svd(centred, compute_uv=False)
        var = sv**2
        total = float(var.sum())
        if total > 0:
            for frac in (8, 4, 2):
                keep = max(1, d // frac)
                spectrum[f"variance_at_d_over_{frac}"] = float(var[:keep].sum() / total)
    except Exception:  # noqa: BLE001 - preflight reports, never raises
        logger.debug("spectrum unavailable", exc_info=True)
    return {
        "n_rows": int(n),
        "dim": int(d),
        "sampled": int(a.shape[0]),
        "zero_rows": zero,
        "nonfinite_rows": nonfinite,
        "norm_spread": spread,
        "spectrum": spectrum,
        "flags": flags,
    }


# ------------------------------------------------------------------ #
# Candidates                                                          #
# ------------------------------------------------------------------ #


@dataclass
class CandidateResult:
    """One codec configuration and everything measured about it.

    ``verdict`` is one of ``selected``, ``candidate``, ``reject``,
    ``infeasible`` or ``unsupported``, and ``left_at`` names the stage it
    stopped in, so a reader can tell a codec that lost on quality from one
    that never ran.
    """

    codec: str
    config: dict
    verdict: str = "candidate"
    left_at: str = "enumerated"
    reason: str = ""
    source: str = "in-tree"
    tier: str = "experimental"
    declared_bits: float | None = None
    bytes_per_vector: float | None = None
    byte_accounting: dict = field(default_factory=dict)
    quality: QualityEstimate | None = None
    holdout_quality: QualityEstimate | None = None
    nominal: dict = field(default_factory=dict)
    false_clear: dict = field(default_factory=dict)
    certificate: dict = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "codec": self.codec,
            "config": dict(self.config),
            "verdict": self.verdict,
            "left_at": self.left_at,
            "reason": self.reason,
            "source": self.source,
            "tier": self.tier,
            "declared_bits": self.declared_bits,
            "bytes_per_vector": self.bytes_per_vector,
            "byte_accounting": self.byte_accounting,
            "quality": self.quality.as_dict() if self.quality else None,
            "holdout_quality": (
                self.holdout_quality.as_dict() if self.holdout_quality else None
            ),
            "nominal": self.nominal,
            "false_clear": self.false_clear,
            "certificate": self.certificate,
            "error": self.error,
        }


def _shape_hints(art: Artifact) -> dict:
    """The artifact's geometry, in the names a codec factory might use."""
    values = {"head_dim": art.dim, "dim": art.dim, "input_dim": art.dim, "n_heads": 1}
    return {k: values[k] for k in SHAPE_HINTS}


def _create_codec(name: str, config: dict, hints: dict | None = None):
    """Build a codec, offering shape hints first and retrying without them."""
    from . import plugins

    if hints:
        merged = dict(hints)
        merged.update(config)
        try:
            return plugins.create(name, **merged)
        except Exception:  # noqa: BLE001 - the hint was an offer, not a demand
            logger.debug("shape hints rejected by %s", name, exc_info=True)
    return plugins.create(name, **config)


def _enumerate_candidates(spec: WorkloadSpec, hints: dict | None = None) -> list:
    """Codec configurations from the registry, never from a hard-coded list.

    A plugin that declares ``capabilities()`` is asked what bit widths it
    supports; one that declares nothing is tried at
    :data:`DEFAULT_BIT_WIDTHS`, and one whose factory does not take ``bits``
    contributes a single default-configuration candidate. A codec that cannot
    be instantiated at all is recorded as ``unsupported`` with the error
    rather than dropped, because a silent disappearance is how a planner ends
    up recommending from a list of one.
    """
    from . import plugins

    registry = plugins.available_plugins(target=spec.target)
    names = list(spec.candidates) if spec.candidates else list(registry)
    out = []
    for name in names:
        try:
            plugin_spec = plugins.get_plugin(name)
        except KeyError as e:
            out.append(
                CandidateResult(
                    codec=name,
                    config={},
                    verdict="unsupported",
                    left_at="enumeration",
                    reason=str(e),
                    error=str(e),
                )
            )
            continue
        source = "in-tree" if name in _IN_TREE else "plugin"
        if spec.target not in plugin_spec.targets:
            out.append(
                CandidateResult(
                    codec=name,
                    config={},
                    verdict="unsupported",
                    left_at="enumeration",
                    reason=(
                        f"codec declares targets {sorted(plugin_spec.targets)}, "
                        f"which do not include {spec.target!r}"
                    ),
                    source=source,
                    tier=plugin_spec.tier,
                )
            )
            continue
        try:
            probe = _create_codec(name, {}, hints)
            caps = plugins.capabilities(probe)
        except Exception as e:  # noqa: BLE001
            caps = {}
            logger.debug("capability probe failed for %s", name, exc_info=True)
            probe_error = f"{type(e).__name__}: {e}"
        else:
            probe_error = None
        widths = caps.get("bit_widths") or DEFAULT_BIT_WIDTHS
        made_one = False
        for bits in widths:
            try:
                _create_codec(name, {"bits": int(bits)}, hints)
            except TypeError:
                break  # factory takes no bits: one default candidate instead
            except Exception as e:  # noqa: BLE001
                out.append(
                    CandidateResult(
                        codec=name,
                        config={"bits": int(bits)},
                        verdict="unsupported",
                        left_at="enumeration",
                        reason=f"{type(e).__name__}: {e}",
                        source=source,
                        tier=plugin_spec.tier,
                        declared_bits=float(bits),
                        error=f"{type(e).__name__}: {e}",
                    )
                )
                continue
            made_one = True
            out.append(
                CandidateResult(
                    codec=name,
                    config={"bits": int(bits)},
                    source=source,
                    tier=plugin_spec.tier,
                    declared_bits=float(bits),
                )
            )
        if not made_one:
            out.append(
                CandidateResult(
                    codec=name,
                    config={},
                    verdict="candidate" if probe_error is None else "unsupported",
                    left_at="enumeration",
                    reason=probe_error or "codec takes no bit-width argument",
                    source=source,
                    tier=plugin_spec.tier,
                    declared_bits=caps.get("default_bits"),
                    error=probe_error,
                )
            )
    return out


_IN_TREE = frozenset({"per_channel", "polar"})


# ------------------------------------------------------------------ #
# The plan record                                                     #
# ------------------------------------------------------------------ #


@dataclass
class CompressionPlan:
    """The decision, the evidence for it, and what to do when it stops holding.

    Field names follow the request in issue #169 so a reader of that issue
    finds what it asked for. ``selected_codec`` is :data:`ABSTAIN` when
    nothing could be recommended, and the reason is then in
    ``selection["reason"]`` rather than absent.
    """

    target: str
    consumer: dict
    operator_regime: dict
    candidate_results: list
    selected_codec: str
    selected_parameters: dict
    expected_cost: dict
    expected_quality: QualityEstimate | None
    certificate_requirement: dict
    fallback_policy: dict
    evidence: list
    selection: dict
    workload: dict
    preflight: dict
    environment: dict
    artifact: dict
    created_utc: str = field(default_factory=_now_utc)

    @property
    def abstained(self) -> bool:
        return self.selected_codec == ABSTAIN

    def as_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "created_utc": self.created_utc,
            "workload": self.workload,
            "artifact": self.artifact,
            "target": self.target,
            "consumer": self.consumer,
            "operator_regime": self.operator_regime,
            "preflight": self.preflight,
            "candidate_results": [c.as_dict() for c in self.candidate_results],
            "selected_codec": self.selected_codec,
            "selected_parameters": self.selected_parameters,
            "selection": self.selection,
            "expected_cost": self.expected_cost,
            "expected_quality": (
                self.expected_quality.as_dict() if self.expected_quality else None
            ),
            "certificate_requirement": self.certificate_requirement,
            "fallback_policy": self.fallback_policy,
            "evidence": self.evidence,
            "environment": self.environment,
        }

    def explain(self) -> str:
        """The record rendered for a person: what won, what it beat, and why."""
        lines = [
            f"# compression plan  target={self.target}  "
            f"consumer={self.consumer.get('metric')}",
        ]
        floor = self.workload.get("floor")
        if floor:
            lines.append(
                f"floor: {self.consumer.get('metric')} "
                f"{'>=' if self.consumer.get('higher_is_better') else '<='} "
                f"{floor['minimum']} at {floor['confidence']:.0%} one-sided"
            )
        budget = self.workload.get("budget") or {}
        stated = ", ".join(f"{k}={v}" for k, v in budget.items() if v is not None)
        lines.append(f"budget: {stated or 'unconstrained'}")
        for flag in self.preflight.get("flags", []):
            lines.append(f"  ! preflight: {flag}")
        header = (
            f"{'codec':<22}{'bits':>5}  {'bytes/vec':>10}  "
            f"{'consumer (bound)':>18}  verdict"
        )
        lines.append("")
        lines.append(header)
        lines.append("-" * len(header))
        for c in self.candidate_results:
            q = c.holdout_quality or c.quality
            bits = "" if c.declared_bits is None else f"{c.declared_bits:g}"
            bpv = "" if c.bytes_per_vector is None else f"{c.bytes_per_vector:.1f}"
            val = "" if q is None else f"{q.bound:.4f}"
            lines.append(
                f"{c.codec + ('/' + str(c.config.get('bits')) if c.config else ''):<22}"
                f"{bits:>5}  {bpv:>10}  {val:>18}  {c.verdict}"
            )
        lines.append("")
        if self.abstained:
            lines.append(f"selected: {ABSTAIN}")
            lines.append(f"  reason: {self.selection.get('reason')}")
        else:
            lines.append(f"selected: {self.selected_codec} {self.selected_parameters}")
            lines.append(f"  rule:   {self.selection.get('rule')}")
            lines.append(f"  reason: {self.selection.get('reason')}")
        for c in self.candidate_results:
            fc = c.false_clear or {}
            if fc.get("false_clear"):
                lines.append(
                    f"  ! {c.codec}{c.config}: the cheap metric cleared it and the "
                    f"consumer did not "
                    f"(P(consumer fails | cosine cleared) = "
                    f"{fc.get('false_clear_given_cleared', float('nan')):.2f})"
                )
        req = self.certificate_requirement
        if req:
            lines.append(f"certificate: {req.get('status')}  {req.get('note', '')}")
        fb = self.fallback_policy
        if fb:
            lines.append(f"runtime fallback: {fb.get('action')}  {fb.get('reason')}")
        return "\n".join(lines)


def plan_schema() -> dict:
    """The JSON-schema description of a plan record, for external validators."""
    from .schemas import load_schema

    return load_schema("compression_plan.schema.json")


# ------------------------------------------------------------------ #
# The planner                                                         #
# ------------------------------------------------------------------ #


class CompressionPlanner:
    """Given a consumer, a budget and a registry of codecs, decide and explain.

    The class is deliberately thin: every measurement it makes belongs to a
    module that can be used on its own, and the planner's contribution is the
    order, the uncertainty and the record.
    """

    def __init__(self, spec: WorkloadSpec):
        self.spec = spec

    # -- public ---------------------------------------------------------
    def plan(self, artifact: Any, **context: Any) -> CompressionPlan:
        """Measure every candidate against the consumer and choose one."""
        art = artifact if isinstance(artifact, Artifact) else Artifact(artifact)
        if context:
            art.context = {**art.context, **context}
        spec = self.spec

        consumer, consumer_info, abstain = self._resolve_consumer()
        pre = preflight(art.data, seed=spec.seed)
        if abstain is not None:
            return self._abstain_plan(art, consumer_info, pre, abstain)

        hints = _shape_hints(art)
        cands = _enumerate_candidates(spec, hints)
        live = [c for c in cands if c.verdict == "candidate"]
        if not live:
            return self._abstain_plan(
                art,
                consumer_info,
                pre,
                "no registered codec supports this target; install a plugin or "
                "widen --candidates",
                cands,
            )

        cal_idx, hold_idx = self._split(art)
        self._prune_on_priors(live)
        live = [c for c in live if c.verdict == "candidate"]

        if live:
            self._halving(live, art, cal_idx, consumer)
            live = [c for c in live if c.verdict == "candidate"]

        frontier = _frontier(live)
        for c in live:
            if c not in frontier and c.verdict == "candidate":
                c.verdict = "reject"
                c.left_at = "frontier"
                c.reason = "dominated on both stored bytes and consumer quality"

        ordered = _order(frontier, spec)
        for cand in ordered:
            self._verify(cand, art, hold_idx, consumer)
        passed = [c for c in ordered if c.verdict == "selected"]
        selected = _order(passed, spec)[0] if passed else None
        for cand in passed:
            if cand is not selected:
                cand.verdict = "candidate"
                cand.reason = (
                    "cleared the floor on held-out data but was not the "
                    f"objective's choice ({spec.objective})"
                )

        return self._record(art, consumer, consumer_info, pre, cands, selected)

    # -- stages ---------------------------------------------------------
    def _resolve_consumer(self):
        """Look the consumer up; an unknown one is an abstention, not a guess."""
        from . import consumers as consumers_mod

        spec = self.spec
        info = {
            "requested": spec.consumer,
            "config": dict(spec.consumer_config),
        }
        try:
            cspec = consumers_mod.get_consumer(spec.consumer)
        except KeyError as e:
            return None, info, str(e)
        if spec.target not in cspec.targets:
            return (
                None,
                info,
                f"consumer {spec.consumer!r} is defined for targets "
                f"{sorted(cspec.targets)}, not {spec.target!r}",
            )
        try:
            consumer = consumers_mod.create_consumer(
                spec.consumer, **spec.consumer_config
            )
        except Exception as e:  # noqa: BLE001
            return None, info, f"consumer {spec.consumer!r} failed to build: {e}"
        info.update(
            {
                "metric": consumer.name,
                "higher_is_better": bool(consumer.higher_is_better),
                "exact": bool(cspec.exact),
                "evidence_kind": cspec.evidence_kind,
                "description": cspec.description,
            }
        )
        return consumer, info, None

    def _split(self, art: Artifact):
        """Calibration and held-out row splits; the held-out one is used once."""
        n = art.n_rows
        rng = np.random.default_rng(self.spec.seed)
        perm = rng.permutation(n)
        n_hold = max(1, int(round(n * self.spec.holdout_fraction)))
        n_hold = min(n_hold, n - 1) if n > 1 else n
        return perm[n_hold:], perm[:n_hold]

    def _prune_on_priors(self, live: list) -> None:
        """Drop what cannot fit before anything is compressed."""
        b = self.spec.budget
        if b.max_bits is None:
            return
        for c in live:
            if c.declared_bits is not None and c.declared_bits > b.max_bits:
                c.verdict = "infeasible"
                c.left_at = "prior"
                c.reason = (
                    f"declared {c.declared_bits:g} bits exceeds the "
                    f"{b.max_bits:g}-bit budget"
                )

    def _rows(self, art: Artifact, idx: np.ndarray) -> np.ndarray:
        flat = art.data.reshape(-1, art.dim)
        return np.ascontiguousarray(flat[idx])

    def _evaluate(
        self,
        cand: CandidateResult,
        art: Artifact,
        rows: np.ndarray,
        consumer: Any,
        *,
        confidence: float,
        stage: str,
    ):
        """Compress ``rows``, then read the consumer's metric and the cheap one."""
        try:
            q = _create_codec(cand.codec, cand.config, _shape_hints(art))
            block = self._shape_for(cand, rows, art)
            container = q.compress(np.asarray(block, dtype=np.float32))
            recon = np.asarray(q.decompress(container)).reshape(rows.shape)
        except Exception as e:  # noqa: BLE001 - a broken codec loses, quietly
            cand.verdict = "unsupported"
            cand.left_at = stage
            cand.error = f"{type(e).__name__}: {e}"
            cand.reason = f"codec failed on this artifact: {cand.error}"
            return None, None

        acct = container_bytes(container)
        cand.byte_accounting = acct
        cand.bytes_per_vector = acct["total_bytes"] / max(rows.shape[0], 1)

        try:
            per_item = np.asarray(
                consumer.per_item(rows, recon, **art.context), dtype=np.float64
            )
        except Exception as e:  # noqa: BLE001
            cand.verdict = "unsupported"
            cand.left_at = stage
            cand.error = f"{type(e).__name__}: {e}"
            cand.reason = f"consumer metric could not read this codec: {cand.error}"
            return None, None

        est = _estimate(
            per_item,
            metric=consumer.name,
            higher_is_better=bool(consumer.higher_is_better),
            confidence=confidence,
            n_boot=self.spec.n_boot,
            seed=self.spec.seed,
        )
        self._diagnose(cand, consumer, art, rows, recon, per_item, stage)
        return est, (rows, recon, per_item)

    def _diagnose(
        self,
        cand: CandidateResult,
        consumer: Any,
        art: Artifact,
        orig: np.ndarray,
        recon: np.ndarray,
        per_item: np.ndarray,
        split: str,
    ) -> None:
        """Score the cheap metric on the consumer's own items and compare.

        Attached to every evaluation, not only the winner's, because "cosine
        cleared it and the consumer did not" is a fact about a candidate and a
        reader needs it for the codecs that lost as much as for the one that
        won.
        """
        from . import consumers as consumers_mod
        from .false_clear import false_clear_from_scores

        spec = self.spec
        nominal = consumers_mod.nominal_per_item(consumer, orig, recon, **art.context)
        if nominal is None:
            return
        if nominal.size != per_item.size:
            cand.nominal = {
                "metric": "reconstruction_cosine",
                "mean": float(np.mean(nominal)),
                "role": "diagnostic only",
                "false_clear": (
                    "unavailable: the cheap and consumer metrics score "
                    "different populations"
                ),
                "split": split,
            }
            return
        cand.nominal = {
            "metric": "reconstruction_cosine",
            "mean": float(np.mean(nominal)),
            "threshold": spec.nominal_threshold,
            "role": "diagnostic only; never an acceptance signal",
            "split": split,
        }
        if spec.floor is None:
            return
        report = false_clear_from_scores(
            nominal,
            per_item,
            nominal_threshold=spec.nominal_threshold,
            consumer_threshold=spec.floor.minimum,
            nominal_higher_is_better=True,
            consumer_higher_is_better=bool(consumer.higher_is_better),
        )
        d = report.to_dict()
        d["false_clear"] = bool(report.false_clear_rate > 0.0)
        d["split"] = split
        d["note"] = (
            "items the cheap cosine cleared that the consumer rejected, "
            "measured on the same items"
        )
        cand.false_clear = d

    def _shape_for(self, cand: CandidateResult, rows: np.ndarray, art: Artifact):
        """KV codecs read ``(B=1, H, S, D)``; retrieval codecs read ``(N, D)``.

        The artifact keeps its own rank, so a key block stays a key block and
        a codec is never handed a shape its kernels do not expect.
        """
        if self.spec.target in ("kv_key", "kv_value"):
            return rows.reshape(1, 1, rows.shape[0], rows.shape[1])
        return rows

    def _halving(
        self, live: list, art: Artifact, cal_idx: np.ndarray, consumer: Any
    ) -> None:
        """Measure cheaply first; promote only what still looks competitive.

        Survivors are kept on the conservative end of their interval, and a
        candidate is only cut when a survivor's interval lies wholly above its
        own. Cutting on means is how a noisy winner beats a real one.
        """
        spec = self.spec
        n = int(cal_idx.size)
        rounds = max(1, int(spec.halving_rounds))
        sizes = []
        for r in range(rounds):
            frac = 0.5 ** (rounds - 1 - r)
            sizes.append(max(min(spec.min_sample, n), int(round(n * frac))))
        sizes = sorted(set(min(s, n) for s in sizes))
        survivors = list(live)
        for round_i, size in enumerate(sizes):
            rows = self._rows(art, cal_idx[:size])
            for cand in survivors:
                est, _ = self._evaluate(
                    cand,
                    art,
                    rows,
                    consumer,
                    confidence=spec.floor.confidence if spec.floor else 0.95,
                    stage=f"halving round {round_i + 1} (n={size})",
                )
                if est is not None:
                    cand.quality = est
            survivors = [c for c in survivors if c.verdict == "candidate"]
            if round_i == len(sizes) - 1 or len(survivors) <= 1:
                break
            keep = max(1, math.ceil(len(survivors) / 2))
            best = max(survivors, key=lambda c: c.quality.score).quality
            ranked = _order(survivors, spec)
            for cand in ranked[keep:]:
                if cand.quality.clears(spec.floor) and spec.floor is not None:
                    # It already meets the declared bar. Quality above the floor
                    # is a tiebreak, not the bar, and under `min_cost` the
                    # cheapest candidate that clears is the answer -- cutting it
                    # here for being second-best would decide the plan before
                    # cost was ever consulted.
                    continue
                if _strictly_worse(cand.quality, best):
                    cand.verdict = "reject"
                    cand.left_at = f"halving round {round_i + 1} (n={size})"
                    cand.reason = (
                        f"consumer metric {cand.quality.mean:.4f} "
                        f"(bound {cand.quality.bound:.4f}) is below the round's "
                        f"best interval [{best.ci_low:.4f}, {best.ci_high:.4f}] "
                        "and does not clear the floor"
                    )
            survivors = [c for c in survivors if c.verdict == "candidate"]

    def _verify(
        self, cand: CandidateResult, art: Artifact, hold_idx: np.ndarray, consumer: Any
    ) -> None:
        """One evaluation on data that played no part in the search.

        This is also where the cheap metric is scored, on the same items, so
        the plan can say whether cosine would have cleared a candidate the
        consumer rejected.
        """
        spec = self.spec
        rows = self._rows(art, hold_idx)
        est, materials = self._evaluate(
            cand,
            art,
            rows,
            consumer,
            confidence=spec.floor.confidence if spec.floor else 0.95,
            stage="verification",
        )
        if est is None:
            return
        cand.holdout_quality = est

        if materials is not None and spec.target == "embedding":
            orig, recon, _ = materials
            cand.certificate = _certificate_block(orig, recon, spec.seed)

        budget = spec.budget
        if (
            budget.max_bytes_per_vector is not None
            and cand.bytes_per_vector is not None
            and cand.bytes_per_vector > budget.max_bytes_per_vector
        ):
            cand.verdict = "infeasible"
            cand.left_at = "verification"
            cand.reason = (
                f"{cand.bytes_per_vector:.1f} stored bytes/vector exceeds the "
                f"{budget.max_bytes_per_vector:g} budget"
            )
            return
        if not est.clears(spec.floor):
            cand.verdict = "reject"
            cand.left_at = "verification"
            cand.reason = (
                f"held-out {est.metric} bound {est.bound:.4f} does not clear the "
                f"floor {spec.floor.minimum:g} at {est.confidence:.0%}"
                if spec.floor
                else "held-out verification failed"
            )
            return
        cand.verdict = "selected"
        cand.left_at = "verification"
        cand.reason = "held-out consumer metric clears the floor within budget"

    # -- record ---------------------------------------------------------
    def _abstain_plan(
        self,
        art: Artifact,
        consumer_info: dict,
        pre: dict,
        reason: str,
        cands: list | None = None,
    ) -> CompressionPlan:
        return CompressionPlan(
            target=self.spec.target,
            consumer=consumer_info,
            operator_regime=_operator_regime(self.spec, consumer_info),
            candidate_results=list(cands or []),
            selected_codec=ABSTAIN,
            selected_parameters={},
            expected_cost={},
            expected_quality=None,
            certificate_requirement={
                "status": "not_established",
                "note": "no plan was selected, so nothing is certified",
            },
            fallback_policy=_fallback_policy(self.spec, None, None),
            evidence=[],
            selection={
                "rule": "abstain",
                "reason": reason,
                "note": (
                    "the control plane prefers ABSTAIN to a recommendation in a "
                    "regime it has not measured"
                ),
            },
            workload=self.spec.as_dict(),
            preflight=pre,
            environment=_environment(),
            artifact=_artifact_block(art),
        )

    def _record(
        self,
        art: Artifact,
        consumer: Any,
        consumer_info: dict,
        pre: dict,
        cands: list,
        selected: CandidateResult | None,
    ) -> CompressionPlan:
        spec = self.spec
        if selected is None:
            reasons = [
                f"{c.codec}{c.config}: {c.reason}"
                for c in cands
                if c.verdict in {"reject", "infeasible"} and c.reason
            ]
            plan = self._abstain_plan(
                art,
                consumer_info,
                pre,
                "no candidate cleared the floor within budget on held-out data"
                + (("; " + "; ".join(reasons[:4])) if reasons else ""),
                cands,
            )
            return plan

        runner_up = _runner_up(cands, selected)
        evidence = _evidence(selected, runner_up, consumer_info, spec)
        return CompressionPlan(
            target=spec.target,
            consumer=consumer_info,
            operator_regime=_operator_regime(spec, consumer_info),
            candidate_results=cands,
            selected_codec=selected.codec,
            selected_parameters=dict(selected.config),
            expected_cost={
                "bytes_per_vector": selected.bytes_per_vector,
                "declared_bits": selected.declared_bits,
                "accounting": selected.byte_accounting.get("accounting"),
                "measured_on": "held-out split",
            },
            expected_quality=selected.holdout_quality,
            certificate_requirement=_certificate_requirement(selected),
            fallback_policy=_fallback_policy(spec, selected, consumer_info),
            evidence=evidence,
            selection=_selection_rule(selected, runner_up, spec, consumer_info),
            workload=spec.as_dict(),
            preflight=pre,
            environment=_environment(),
            artifact=_artifact_block(art),
        )


#: The issue's preferred name for the same object.
QuantizationControlPlane = CompressionPlanner


# ------------------------------------------------------------------ #
# Selection helpers                                                   #
# ------------------------------------------------------------------ #


def _strictly_worse(a: QualityEstimate, best: QualityEstimate) -> bool:
    """True only when ``a``'s interval lies wholly below ``best``'s."""
    if a.higher_is_better:
        return a.ci_high < best.ci_low
    return a.ci_low > best.ci_high


def _frontier(live: list) -> list:
    """Non-dominated on (consumer quality bound, stored bytes), ties kept.

    A candidate is dropped only when another is at least as cheap *and* its
    quality interval lies wholly above; equal-quality ties survive and are
    reported rather than broken silently.
    """
    keep = []
    for c in live:
        if c.quality is None:
            continue
        dominated = False
        for o in live:
            if o is c or o.quality is None:
                continue
            cheaper = (o.bytes_per_vector or math.inf) <= (
                c.bytes_per_vector or math.inf
            )
            if cheaper and _strictly_worse(c.quality, o.quality):
                dominated = True
                break
        if not dominated:
            keep.append(c)
    return keep or list(live)


def _order(frontier: list, spec: WorkloadSpec) -> list:
    """The order verification tries frontier candidates in."""
    if spec.objective == "min_cost":
        return sorted(
            frontier,
            key=lambda c: (
                c.bytes_per_vector if c.bytes_per_vector is not None else math.inf,
                -(c.quality.score if c.quality else -math.inf),
            ),
        )
    return sorted(
        frontier,
        key=lambda c: (
            -(c.quality.score if c.quality else -math.inf),
            c.bytes_per_vector if c.bytes_per_vector is not None else math.inf,
        ),
    )


def _runner_up(cands: list, selected: CandidateResult):
    others = [
        c
        for c in cands
        if c is not selected and c.quality is not None and c.verdict != "unsupported"
    ]
    if not others:
        return None
    return max(others, key=lambda c: c.quality.score)


def _selection_rule(
    selected: CandidateResult,
    runner_up: CandidateResult | None,
    spec: WorkloadSpec,
    consumer_info: dict,
) -> dict:
    q = selected.holdout_quality or selected.quality
    if q is None:  # pragma: no cover - a selected candidate is always measured
        raise ValueError("a selected candidate must carry a quality estimate")
    rule = (
        "among frontier candidates meeting every budget, maximise the "
        "consumer metric's conservative bound, then minimise stored bytes"
        if spec.objective == "max_quality"
        else "among frontier candidates clearing the floor, minimise stored "
        "bytes, then maximise the consumer metric's conservative bound"
    )
    parts = [
        f"{consumer_info.get('metric')} bound {q.bound:.4f} on held-out items "
        f"at {q.confidence:.0%}",
    ]
    if spec.floor is not None:
        parts.append(f"floor {spec.floor.minimum:g}")
    if selected.bytes_per_vector is not None:
        parts.append(f"{selected.bytes_per_vector:.1f} stored bytes/vector")
    beat = None
    if runner_up is not None:
        rq = runner_up.holdout_quality or runner_up.quality
        beat = {
            "codec": runner_up.codec,
            "config": dict(runner_up.config),
            "metric_bound": rq.bound if rq else None,
            "margin": (q.bound - rq.bound) if rq else None,
            "source": runner_up.source,
        }
    return {
        "rule": rule,
        "objective": spec.objective,
        "reason": "; ".join(parts),
        "beat": beat,
        "selected_source": selected.source,
        "third_party": selected.source != "in-tree",
        "decided_on": consumer_info.get("metric"),
        "not_decided_on": "reconstruction cosine (diagnostic only)",
    }


def _certificate_block(orig: np.ndarray, recon: np.ndarray, seed: int) -> dict:
    from .rank_certificate import certificate_from_embeddings

    try:
        cert = certificate_from_embeddings(orig, recon, metric="cosine", seed=seed)
    except Exception as e:  # noqa: BLE001 - evidence is optional, never fatal
        return {"error": f"{type(e).__name__}: {e}"}
    d = cert.as_dict()
    d["metric"] = "cosine"
    return d


def _certificate_requirement(selected: CandidateResult) -> dict:
    cert = selected.certificate or {}
    if not cert or "error" in cert:
        return {
            "status": "not_established",
            "note": cert.get("error", "no rank certificate applies to this target"),
        }
    if cert.get("vacuous"):
        return {
            "status": "exact_rerank_required",
            "tau_floor": cert.get("tau_floor"),
            "note": (
                "the distribution-free floor is vacuous on this corpus: "
                "single-stage rank fidelity is not certifiable and an exact "
                "rerank stage is required"
            ),
        }
    return {
        "status": "certified",
        "tau_floor": cert.get("tau_floor"),
        "spearman_floor": cert.get("spearman_floor"),
        "note": "distribution-free rank floor holds for this corpus and codec",
    }


def _fallback_policy(
    spec: WorkloadSpec,
    selected: CandidateResult | None,
    consumer_info: dict | None,
) -> dict:
    """What to do at runtime when the plan's assumptions stop holding.

    The actions are :mod:`turboquant_pro.runtime_policy`'s, not a parallel
    vocabulary, and the triggers are the measurements that would show the plan
    going stale.
    """
    from . import runtime_policy as rp

    policy = rp.TQPRuntimePolicy()
    if selected is None:
        action = policy.unknown_operator_action
        reason = "no plan was selected; keep the conservative representation"
    else:
        cert = selected.certificate or {}
        if cert.get("vacuous"):
            action = rp.REQUIRE_EXACT_RERANK
            reason = "the rank certificate is vacuous on this corpus"
        elif spec.target in ("kv_key",):
            action = rp.PER_CHANNEL_OR_FP16
            reason = "keys are the target whose consumer failure is silent"
        else:
            action = rp.RERANK_MORE
            reason = "cheap first stage, exact rerank when boundary gaps close"
    return {
        "action": action,
        "reason": reason,
        "thresholds": {
            "retrieval_gap_floor": policy.retrieval_gap_floor,
            "rerank_oversample": policy.rerank_oversample,
            "min_tau_floor": policy.min_tau_floor,
            "radial_drift_floor": policy.radial_drift_floor,
            "basis_drift_floor": policy.basis_drift_floor,
        },
        "triggers": [
            "held-out consumer metric on canary queries falls below the floor",
            "rank certificate becomes vacuous on refreshed data",
            "(A2) tangential fraction or radial drift beyond the monitor's floors",
            "the consumer changes: a new read operator invalidates this plan",
        ],
        "module": "turboquant_pro.runtime_policy.TQPRuntimePolicy",
    }


def _evidence(
    selected: CandidateResult,
    runner_up: CandidateResult | None,
    consumer_info: dict,
    spec: WorkloadSpec,
) -> list:
    """Every claim in a plan names the kind of evidence behind it."""
    items = []
    q = selected.holdout_quality
    if q is not None:
        items.append(
            {
                "kind": "statistical",
                "claim": (
                    f"{q.metric} on held-out items is at least {q.bound:.4f} "
                    f"at {q.confidence:.0%} one-sided"
                ),
                "measurement": q.as_dict(),
                "population": "held-out split, unused during search",
                "method": f"percentile bootstrap, {spec.n_boot} resamples",
            }
        )
    cert = selected.certificate or {}
    if cert and "error" not in cert:
        items.append(
            {
                "kind": "certificate",
                "claim": (
                    f"Kendall tau >= {cert.get('tau_floor'):.4f} "
                    f"(distribution-free, cosine ranking)"
                    if cert.get("tau_floor") is not None
                    else "rank certificate measured"
                ),
                "measurement": cert,
                "vacuous": bool(cert.get("vacuous")),
            }
        )
    if selected.bytes_per_vector is not None:
        items.append(
            {
                "kind": "measured_cost",
                "claim": (f"{selected.bytes_per_vector:.2f} stored bytes per vector"),
                "measurement": selected.byte_accounting,
            }
        )
    if selected.false_clear:
        items.append(
            {
                "kind": "diagnostic",
                "claim": (
                    "reconstruction cosine agrees with the consumer on "
                    f"{selected.false_clear.get('agreement', float('nan')):.2f} "
                    "of held-out items"
                ),
                "measurement": selected.false_clear,
                "note": "measured, never used to accept",
            }
        )
    if runner_up is not None:
        rq = runner_up.holdout_quality or runner_up.quality
        items.append(
            {
                "kind": "comparison",
                "claim": (
                    f"beat {runner_up.codec}{runner_up.config} on "
                    f"{consumer_info.get('metric')}"
                ),
                "measurement": {
                    "selected_bound": q.bound if q else None,
                    "runner_up_bound": rq.bound if rq else None,
                    "runner_up_source": runner_up.source,
                },
                "note": (
                    "the two were measured on different splits when the runner-up "
                    "did not reach verification"
                    if runner_up.holdout_quality is None
                    else "both measured on the held-out split"
                ),
            }
        )
    return items


def _operator_regime(spec: WorkloadSpec, consumer_info: dict) -> dict:
    """Which read operator the plan was judged through, and how it was got."""
    cfg = spec.consumer_config or {}
    name = cfg.get("operator")
    if spec.consumer == "read_operator" and name:
        from .read_operators import get_read_operator

        try:
            rspec = get_read_operator(name)
        except KeyError:
            return {"operator": name, "status": "unregistered"}
        return {
            "operator": name,
            "exact": bool(rspec.exact),
            "description": rspec.description,
            "distortion": "tr(P_C Sigma_delta), per item",
        }
    return {
        "operator": "implicit",
        "consumer": consumer_info.get("metric"),
        "note": (
            "the consumer metric is measured directly, so no read operator had "
            "to be recovered"
        ),
    }


def _artifact_block(art: Artifact) -> dict:
    return {
        "name": art.name,
        "identity": art.identity,
        "shape": list(art.data.shape),
        "dtype": str(art.data.dtype),
        "n_rows": art.n_rows,
        "dim": art.dim,
        "context_keys": sorted(art.context),
    }


def _environment() -> dict:
    from .certify_report import _certify_environment

    env = _certify_environment()
    try:
        from . import plugins

        env["registered_codecs"] = sorted(plugins.available_plugins())
    except Exception:  # noqa: BLE001
        pass
    return env


# ------------------------------------------------------------------ #
# Replay                                                              #
# ------------------------------------------------------------------ #


def replay_plan(record: dict, artifact: Any, **context: Any) -> dict:
    """Re-run a plan's verification and report whether it still holds.

    Reproducibility here is a measurement, not an assertion: the same spec and
    the same artifact identity should give the same held-out numbers, and the
    report says by how much they differ when they do not. A different artifact
    identity is reported rather than silently replayed, because a plan is a
    statement about particular bytes.
    """
    art = artifact if isinstance(artifact, Artifact) else Artifact(artifact)
    if context:
        art.context = {**art.context, **context}
    w = record["workload"]
    floor = w.get("floor")
    spec = WorkloadSpec(
        target=w["target"],
        consumer=w["consumer"],
        consumer_config=dict(w.get("consumer_config") or {}),
        budget=Budget(**(w.get("budget") or {})),
        floor=QualityFloor(**floor) if floor else None,
        candidates=(
            (record["selected_codec"],) if record["selected_codec"] != ABSTAIN else None
        ),
        objective=w.get("objective", "max_quality"),
        seed=int(w.get("seed", 0)),
        holdout_fraction=float(w.get("holdout_fraction", 0.3)),
        n_boot=int(w.get("n_boot", 512)),
        nominal_threshold=float(w.get("nominal_threshold", NOMINAL_COSINE_CLEAR)),
    )
    identity_match = record.get("artifact", {}).get("identity") == art.identity
    fresh = CompressionPlanner(spec).plan(art)
    before = record.get("expected_quality") or {}
    after = fresh.expected_quality.as_dict() if fresh.expected_quality else {}
    delta = None
    if before.get("mean") is not None and after.get("mean") is not None:
        delta = float(after["mean"]) - float(before["mean"])
    return {
        "schema": "turboquant-pro/compression-plan-replay",
        "schema_version": 1,
        "created_utc": _now_utc(),
        "artifact_identity_matches": bool(identity_match),
        "recorded_codec": record.get("selected_codec"),
        "replayed_codec": fresh.selected_codec,
        "codec_agrees": record.get("selected_codec") == fresh.selected_codec,
        "recorded_quality": before,
        "replayed_quality": after,
        "delta_mean": delta,
        "reproduced": bool(
            identity_match
            and record.get("selected_codec") == fresh.selected_codec
            and (delta is None or abs(delta) < 1e-9)
        ),
        "note": (
            "a replay on a different artifact reports the mismatch rather than "
            "claiming reproduction; a plan is a statement about particular bytes"
        ),
    }

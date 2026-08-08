# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Read-operator provider protocol + registry.

The second extension point, symmetric with :mod:`turboquant_pro.plugins`.
That one plugs in **quantizers**, the things being certified. This one plugs
in the **read operator** they are certified *against*.

Until now the consumer side was closed. ``a2_probe`` offers a fixed menu of
named consumers (``cosine``, ``l2``, ``attention_logits``) whose scores it
computes internally, and ``operator_trace`` infers a regime *label* from the
model graph. Both are useful and neither yields the object the theory is
written in: a positive semi-definite ``P_C`` such that the distortion a
consumer actually feels is

    D = tr(P_C @ Sigma_delta)

with ``Sigma_delta`` the error covariance a codec introduces. With a provider
in hand that quantity is computable, so it can be an acceptance gate rather
than a slogan.

**Why a registry rather than a function.** There is more than one defensible
way to obtain ``P_C``, they disagree materially, and which one a number was
computed against changes what it means. Two reasonable references for a
single attention head differ by roughly 0.3 in subspace overlap, measured.
So the provider is named in the result and travels with it.

Three providers ship in tree, registered through the same interface an
out-of-tree package would use, which is the dogfood requirement the quantizer
plugins already carry:

``identity``
    Every direction read equally. The null provider, and the honest default
    when nothing is known about the consumer: reconstruction error in
    disguise, made explicit so a report can say so.

``declared``
    A matrix the caller supplies. For consumers whose read operator is known
    analytically or was measured elsewhere.

``attention_analytic``
    Closed form for a softmax attention head with respect to one key. The
    read subspace of such a head is spanned by its queries, weighted by how
    much the softmax responds along each, which makes this exact rather than
    estimated.

Out of tree, ``tqp-readscope`` adds a blind provider that recovers ``P_C``
from a consumer's outputs by finite differences, for consumers with no closed
form.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ENTRY_POINT_GROUP",
    "ReadOperator",
    "ReadOperatorSpec",
    "available_read_operators",
    "consumer_distortion",
    "create_read_operator",
    "error_covariance",
    "get_read_operator",
    "load_entry_point_read_operators",
    "register_read_operator",
]

ENTRY_POINT_GROUP = "turboquant_pro.read_operators"


@runtime_checkable
class ReadOperator(Protocol):
    """The minimal contract: activations in, a PSD read operator out."""

    def operator(self, activations: np.ndarray, **context: Any) -> np.ndarray:
        """Return the ``(D, D)`` operator the consumer applies.

        ``activations`` is ``(N, D)`` or any shape whose last axis is the
        channel axis. Extra keyword context is provider specific; a provider
        must ignore context it does not understand rather than fail on it, so
        that callers can pass a superset.
        """
        ...


@dataclass(frozen=True)
class ReadOperatorSpec:
    """A named provider factory, discoverable like a quantizer plugin."""

    name: str
    factory: Callable[..., Any]
    description: str = ""
    exact: bool = False
    """True when the operator is a closed form rather than an estimate."""

    requires: tuple[str, ...] = ()
    """Import names the provider needs, checked at creation."""

    metadata: dict[str, Any] = field(default_factory=dict)


_REGISTRY: dict[str, ReadOperatorSpec] = {}
_ENTRY_POINTS_LOADED = False


def register_read_operator(
    spec: ReadOperatorSpec, *, overwrite: bool = False
) -> ReadOperatorSpec:
    """Register a provider under its name."""
    if spec.name in _REGISTRY and not overwrite:
        raise ValueError(f"read operator {spec.name!r} already registered")
    _REGISTRY[spec.name] = spec
    return spec


def get_read_operator(name: str) -> ReadOperatorSpec:
    """Look up a provider, loading entry points on first miss."""
    if name not in _REGISTRY:
        load_entry_point_read_operators()
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown read operator {name!r}; available: "
            f"{sorted(available_read_operators())}"
        )
    return _REGISTRY[name]


def create_read_operator(name: str, **config: Any) -> Any:
    """Instantiate a provider, checking its declared imports first."""
    spec = get_read_operator(name)
    missing = []
    for mod in spec.requires:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise ImportError(
            f"read operator {name!r} requires {', '.join(missing)}; "
            f"install it or choose another provider"
        )
    return spec.factory(**config)


def available_read_operators() -> dict[str, ReadOperatorSpec]:
    """Every registered provider, entry points included."""
    load_entry_point_read_operators()
    return dict(_REGISTRY)


def load_entry_point_read_operators(*, force: bool = False) -> list[str]:
    """Discover out-of-tree providers via the entry-point group."""
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED and not force:
        return []
    _ENTRY_POINTS_LOADED = True
    found: list[str] = []
    try:
        eps = metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # noqa: BLE001 - importlib shape varies by version
        return found
    for ep in eps:
        try:
            spec = ep.load()
            if isinstance(spec, ReadOperatorSpec):
                register_read_operator(spec, overwrite=True)
                found.append(spec.name)
        except Exception as exc:  # noqa: BLE001 - a bad plugin must not break
            logger.warning(
                "read-operator entry point %r failed to load: %s", ep.name, exc
            )
    return found


# ---------------------------------------------------------------------------
# The quantity the whole extension point exists to make computable
# ---------------------------------------------------------------------------


def error_covariance(original: np.ndarray, reconstructed: np.ndarray):
    """``Sigma_delta``, the across-item covariance of a codec's error.

    Both arrays are flattened to ``(N, D)`` on their last axis. The mean is
    removed, because a shared offset is a different failure from a spread and
    the consumer-relative distortion is about the spread.
    """
    a = np.asarray(original, dtype=np.float64)
    b = np.asarray(reconstructed, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("original and reconstructed must have equal shape")
    d = a.shape[-1]
    delta = (b - a).reshape(-1, d)
    delta = delta - delta.mean(axis=0, keepdims=True)
    return delta.T @ delta / max(delta.shape[0], 1)


def consumer_distortion(P: np.ndarray, sigma_delta: np.ndarray) -> float:
    """``tr(P @ Sigma_delta)``, the distortion the consumer actually feels.

    Error piled into a direction the consumer does not read costs nothing
    here, which is the entire point and is what reconstruction error cannot
    express.
    """
    A = np.asarray(P, dtype=np.float64)
    B = np.asarray(sigma_delta, dtype=np.float64)
    if A.shape != B.shape:
        raise ValueError(f"P {A.shape} and Sigma_delta {B.shape} must have equal shape")
    return float(np.einsum("ij,ji->", A, B))


# ---------------------------------------------------------------------------
# In-tree providers, registered through the public interface (dogfood)
# ---------------------------------------------------------------------------


def _channels(activations: np.ndarray) -> tuple[np.ndarray, int]:
    a = np.asarray(activations, dtype=np.float64)
    d = a.shape[-1]
    return a.reshape(-1, d), d


class IdentityReadOperator:
    """Every direction read equally: reconstruction error, made explicit."""

    def operator(self, activations: np.ndarray, **_: Any) -> np.ndarray:
        _, d = _channels(activations)
        return np.eye(d)


class DeclaredReadOperator:
    """A caller-supplied operator, symmetrised and validated once."""

    def __init__(self, matrix: np.ndarray):
        m = np.asarray(matrix, dtype=np.float64)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("declared read operator must be square")
        m = 0.5 * (m + m.T)
        if float(np.linalg.eigvalsh(m).min()) < -1e-9:
            raise ValueError("declared read operator is not PSD")
        self._m = m

    def operator(self, activations: np.ndarray, **_: Any) -> np.ndarray:
        _, d = _channels(activations)
        if d != self._m.shape[0]:
            raise ValueError(
                f"declared operator is {self._m.shape[0]}-dimensional but "
                f"activations have {d} channels"
            )
        return self._m


class AttentionAnalyticReadOperator:
    """Closed form for a softmax attention head with respect to one key.

    For ``z_i = q_i . k_s / sqrt(d)`` and ``p_i = softmax(z_i)``,

        d p_i / d k_s = p_i * (e_s - p_{i,s}) (q_i / sqrt(d))

    so accumulating the Jacobian Gram over a set of probed keys gives

        P_C = sum_s sum_i a_{i,s} q_i q_i^T / d,
        a_{i,s} = || p_i * (e_s - p_{i,s} 1) ||^2

    The read subspace of the head is therefore spanned by its queries,
    weighted by how much the softmax actually responds along each. Exact, no
    estimation, and it needs the queries: pass them as ``queries=``.
    """

    def __init__(self, n_probe_keys: int = 32, seed: int = 0):
        self.n_probe_keys = int(n_probe_keys)
        self.seed = int(seed)

    def operator(self, activations: np.ndarray, **context: Any) -> np.ndarray:
        keys, d = _channels(activations)
        queries = context.get("queries")
        if queries is None:
            raise ValueError(
                "attention_analytic needs the head's queries; pass "
                "queries=<(n_q, D) array>"
            )
        q, dq = _channels(queries)
        if dq != d:
            raise ValueError(
                f"queries are {dq}-dimensional but keys are {d}-dimensional"
            )

        z = (q @ keys.T) / np.sqrt(d)
        z = z - z.max(axis=-1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=-1, keepdims=True)

        s_all = keys.shape[0]
        n = min(self.n_probe_keys, s_all)
        rng = np.random.default_rng(self.seed)
        probe_idx = rng.choice(s_all, size=n, replace=False)

        M = np.zeros((d, d))
        for s in probe_idx:
            e = np.zeros(s_all)
            e[s] = 1.0
            w = p * (e[None, :] - p[:, s : s + 1])
            a = (w**2).sum(axis=1)
            M += (q * a[:, None]).T @ q / d
        return 0.5 * (M + M.T)


register_read_operator(
    ReadOperatorSpec(
        name="identity",
        factory=lambda **_: IdentityReadOperator(),
        description="every direction read equally; reconstruction error",
        exact=True,
        metadata={"consumer": "none"},
    )
)

register_read_operator(
    ReadOperatorSpec(
        name="declared",
        factory=lambda matrix, **_: DeclaredReadOperator(matrix),
        description="a caller-supplied PSD operator",
        exact=True,
        metadata={"consumer": "caller-defined"},
    )
)

register_read_operator(
    ReadOperatorSpec(
        name="attention_analytic",
        factory=lambda **cfg: AttentionAnalyticReadOperator(**cfg),
        description="closed-form Jacobian Gram of a softmax attention head",
        exact=True,
        metadata={"consumer": "attention_softmax", "needs": "queries"},
    )
)

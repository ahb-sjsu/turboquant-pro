# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Quantizer plugin protocol + registry (P0 of ``docs/DESIGN_hardware_and_plugins.md``).

turboquant-pro's role in a world of production quantization recipes
(TensorRT-LLM's FP8/NVFP4/GPTQ/AWQ, bitsandbytes' NF4/LLM.int8) is the
*certification and fused-decode substrate* they plug into. This module defines
the contract that makes that possible without those recipes living in this
tree:

- :class:`Quantizer` -- the minimal protocol (``compress``/``decompress``).
  A quantizer that satisfies it is certifiable by every instrument in the
  toolkit (rank certificates, the (A2) probe, ``behavioral_agreement``) on
  day one, because the instruments are consumer-metric tools: they need
  round-trips, not internals.
- **The affine capability** -- optional methods ``grid_params(c)`` /
  ``codes(c)`` / ``outlier_csr(c)`` expressing the container as
  ``dequant = mu + weight * grid[code]`` (+ optional sparse fp16 overlay)
  under the ``(B=1, H, S, D)`` block convention. A key format that exposes
  them inherits the M4 fused compute-on-codes decode (see
  ``turboquant_pro.kv_fused_pck``) from the one existing kernel; a format
  that does not (or returns ``None``) gets decompress-then-attend -- still
  correct, still certified.
- **The native-dtype capability** -- optional ``native_dtype()`` naming a
  hardware dtype (e.g. ``"float8_e4m3fn"``) for passthrough execution where
  the silicon fuses the dequant itself.
- :class:`PluginSpec` + the registry -- named factories, discoverable via the
  ``turboquant_pro.plugins`` entry-point group so out-of-tree packages
  (``tqp-bnb``, ``tqp-trtllm``) register without touching this repo.

The executable side of the contract lives in
:mod:`turboquant_pro.plugin_conformance`; in-tree formats are registered
below through the same interface they would use externally (the dogfood
requirement).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "TARGET_KV_KEY",
    "TARGET_KV_VALUE",
    "TARGET_WEIGHT",
    "TARGET_EMBEDDING",
    "Quantizer",
    "PluginSpec",
    "register",
    "get_plugin",
    "create",
    "available_plugins",
    "load_entry_point_plugins",
    "resolve_plugins",
    "affine_params",
    "affine_codes",
    "outlier_csr",
    "native_dtype",
    "PerChannelKVQuantizer",
]

# Quantization targets, as plain strings so out-of-tree packages never import
# a moving enum. They correspond to `operator_trace.QuantTarget` where both
# exist: "weight" <-> QuantTarget.WEIGHT, "kv_key"/"kv_value" are the split of
# QuantTarget.KV_ACTIVATION that the keys finding forces (the two sides have
# opposite disciplines), and "embedding" is the retrieval track.
TARGET_KV_KEY = "kv_key"
TARGET_KV_VALUE = "kv_value"
TARGET_WEIGHT = "weight"
TARGET_EMBEDDING = "embedding"

_VALID_TARGETS = frozenset(
    {TARGET_KV_KEY, TARGET_KV_VALUE, TARGET_WEIGHT, TARGET_EMBEDDING}
)

_VALID_TIERS = frozenset({"stable", "beta", "experimental"})

ENTRY_POINT_GROUP = "turboquant_pro.plugins"


@runtime_checkable
class Quantizer(Protocol):
    """The minimal quantizer surface every plugin instance must provide.

    ``compress`` returns an opaque container; ``decompress`` inverts it to a
    float array of the original shape. Everything else -- the affine
    capability (``grid_params``/``codes``/``outlier_csr``), hardware dtypes
    (``native_dtype``), packing options -- is optional and probed with the
    module-level helpers (:func:`affine_params` etc.), never required.
    """

    def compress(self, x: np.ndarray, **kwargs: Any) -> Any: ...

    def decompress(self, c: Any) -> np.ndarray: ...


@dataclass(frozen=True)
class PluginSpec:
    """A registered quantizer factory.

    ``factory(**config)`` must return an object satisfying :class:`Quantizer`.
    ``tier`` follows ``docs/api-stability.md`` ("stable" / "beta" /
    "experimental"); out-of-tree plugins enter as "experimental" and promote
    on conformance + one public-data validation (DESIGN doc section 5).
    """

    name: str
    factory: Callable[..., Any]
    targets: frozenset[str]
    tier: str = "experimental"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name:
            raise ValueError(f"invalid plugin name: {self.name!r}")
        bad = set(self.targets) - _VALID_TARGETS
        if bad:
            raise ValueError(
                f"unknown targets {sorted(bad)}; valid: {sorted(_VALID_TARGETS)}"
            )
        if self.tier not in _VALID_TIERS:
            raise ValueError(f"tier must be one of {sorted(_VALID_TIERS)}")


_REGISTRY: dict[str, PluginSpec] = {}
_ENTRY_POINTS_LOADED = False


def register(spec: PluginSpec, *, overwrite: bool = False) -> PluginSpec:
    """Register a plugin. Raises on name collision unless ``overwrite=True``."""
    if spec.name in _REGISTRY and not overwrite:
        raise ValueError(
            f"plugin {spec.name!r} already registered "
            f"(pass overwrite=True to replace)"
        )
    _REGISTRY[spec.name] = spec
    return spec


def get_plugin(name: str) -> PluginSpec:
    """Look up a plugin by name (loads entry points on first miss)."""
    if name not in _REGISTRY:
        load_entry_point_plugins()
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"no quantizer plugin {name!r}; available: {known}") from None


def create(name: str, **config: Any) -> Any:
    """Instantiate a registered quantizer, e.g.
    ``create("per_channel", nf4_asym=True, ...)``."""
    q = get_plugin(name).factory(**config)
    if not isinstance(q, Quantizer):
        raise TypeError(
            f"plugin {name!r} factory returned {type(q).__name__}, which does "
            "not provide compress/decompress"
        )
    return q


def available_plugins(*, target: str | None = None) -> dict[str, PluginSpec]:
    """All registered plugins (entry points included), optionally by target."""
    load_entry_point_plugins()
    specs = dict(sorted(_REGISTRY.items()))
    if target is not None:
        specs = {n: s for n, s in specs.items() if target in s.targets}
    return specs


def load_entry_point_plugins(*, force: bool = False) -> list[str]:
    """Discover out-of-tree plugins via the ``turboquant_pro.plugins`` group.

    Each entry point may resolve to a :class:`PluginSpec`, an iterable of
    them, or a zero-argument callable returning either. A plugin package that
    fails to load is logged and skipped -- one broken plugin must not take
    down the registry. Returns the names newly registered.
    """
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED and not force:
        return []
    _ENTRY_POINTS_LOADED = True
    new: list[str] = []
    try:
        eps = metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # pragma: no cover - importlib.metadata quirks
        logger.exception("entry-point discovery failed")
        return new
    for ep in eps:
        try:
            obj = ep.load()
            if callable(obj) and not isinstance(obj, PluginSpec):
                obj = obj()
            specs = [obj] if isinstance(obj, PluginSpec) else list(obj)
            for spec in specs:
                register(spec, overwrite=False)
                new.append(spec.name)
        except Exception:
            logger.exception("skipping broken plugin entry point %r", ep.name)
    return new


# ------------------------------------------------------------------ #
# Optional-capability probes                                          #
# ------------------------------------------------------------------ #


def affine_params(q: Any, c: Any):
    """``(mu (H,D), weight (H,D), grid (L,))`` if ``q`` exposes the affine
    contract for container ``c``, else ``None``. ``None`` means
    decompress-then-attend (the graceful degrade), never an error."""
    fn = getattr(q, "grid_params", None)
    if fn is None:
        return None
    try:
        return fn(c)
    except NotImplementedError:
        return None


def affine_codes(q: Any, c: Any):
    """Unpacked code array ``(B, H, S, D) uint8`` for the affine contract,
    or ``None`` when the capability is absent."""
    fn = getattr(q, "codes", None)
    if fn is None:
        return None
    try:
        return fn(c)
    except NotImplementedError:
        return None


def outlier_csr(q: Any, c: Any):
    """Token-major CSR ``(row_ptr, cols, deltas)`` of sparse-overlay score
    deltas, ``None`` if the capability is absent or the container has none."""
    fn = getattr(q, "outlier_csr", None)
    if fn is None:
        return None
    return fn(c)


def native_dtype(q: Any) -> str | None:
    """Hardware dtype name for passthrough execution, or ``None``."""
    fn = getattr(q, "native_dtype", None)
    return None if fn is None else fn()


# ------------------------------------------------------------------ #
# In-tree formats, registered through the same interface (dogfood)    #
# ------------------------------------------------------------------ #


def _per_channel_factory(**config: Any):
    return PerChannelKVQuantizer(**config)


def _polar_factory(**config: Any):
    from .core import TurboQuantKV

    return TurboQuantKV(**config)


class PerChannelKVQuantizer:
    """:class:`~turboquant_pro.per_channel_kv.PerChannelKV` wearing the full
    plugin capability surface.

    The affine methods delegate to the M4 fused-decode helpers, so this class
    is also the reference implementation of the contract: ``grid_params`` +
    ``codes`` (+ ``outlier_csr``) are exactly what
    :class:`turboquant_pro.kv_fused_pck.PreparedPCKBlock` consumes. nuq
    (data-fit quantile) grids return ``None`` from ``grid_params`` -- the
    documented degrade to decompress-then-attend.
    """

    def __init__(self, **config: Any) -> None:
        from .per_channel_kv import PerChannelKV

        self._q = PerChannelKV(**config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._q, name)

    def compress(self, x: np.ndarray, **kwargs: Any) -> Any:
        return self._q.compress(x, **kwargs)

    def decompress(self, c: Any) -> np.ndarray:
        return self._q.decompress(c)

    def grid_params(self, c: Any):
        from .kv_fused_pck import _grid_params

        try:
            return _grid_params(self._q, c)
        except NotImplementedError:
            return None

    def codes(self, c: Any) -> np.ndarray:
        from .kv_fused_pck import _codes

        return _codes(c)

    def outlier_csr(self, c: Any):
        from .kv_fused_pck import build_outlier_csr

        if self.grid_params(c) is None:
            return None
        return build_outlier_csr(self._q, c)

    def native_dtype(self) -> str | None:
        return None


register(
    PluginSpec(
        name="per_channel",
        factory=_per_channel_factory,
        targets=frozenset({TARGET_KV_KEY}),
        tier="beta",
        description=(
            "PerChannelKV keys: per-channel uniform/NF4/asym-NF4, zero-point "
            "modes, dense-sparse fp16 outliers; full affine contract (fused "
            "decode) except nuq grids"
        ),
    )
)

register(
    PluginSpec(
        name="polar",
        factory=_polar_factory,
        targets=frozenset({TARGET_KV_VALUE}),
        tier="stable",
        description=(
            "PolarQuant (TurboQuantKV): rotation + norm/direction codes. "
            "Correct for VALUES; catastrophic for keys (v1.2.0 finding). "
            "Not per-channel affine -- fused decode via the dedicated M2 "
            "kernel, not the affine contract"
        ),
    )
)


# ------------------------------------------------------------------ #
# operator_trace -> named plugins (the P4 model-in, recipe-out demo)  #
# ------------------------------------------------------------------ #

_FAMILY_PLUGINS: dict[str, list[str]] = {
    # discipline family -> registered plugin names, best-evidenced first
    "per_channel": ["per_channel", "bnb_llm_int8", "nvfp4_kv"],
    "symmetric": ["gptq", "awq", "bnb_nf4", "fp8_kv"],
    "polar": ["polar"],
    "keep_fp": [],
}


def resolve_plugins(model, target: str = "weight", example_inputs=None):
    """Model in, named recipe out: trace operator regimes
    (:func:`turboquant_pro.operator_trace.recommend_quantization`), then map
    each tensor's (A2) discipline to the plugin names actually present in the
    registry (in-tree or entry-point discovered). Returns
    ``{tensor_name: {"family": ..., "sensitivity": ..., "plugins": [...]}}``;
    an empty list means keep full precision or install a plugin for that
    family. This is the human-out-of-the-loop hand-off: the discipline comes
    from measurement, the names come from the registry, nothing is guessed.
    """
    from .operator_trace import recommend_quantization

    disciplines = recommend_quantization(
        model, target=target, example_inputs=example_inputs
    )
    registered = available_plugins()
    return {
        name: {
            "family": d.family,
            "protect_dc": d.protect_dc,
            "sensitivity": d.sensitivity,
            "plugins": [
                p for p in _FAMILY_PLUGINS.get(d.family, []) if p in registered
            ],
        }
        for name, d in disciplines.items()
    }

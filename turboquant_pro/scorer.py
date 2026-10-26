"""Which scorer produced a ranking, named so that a result can say so.

A compressed search has two scoring stages, and each stage runs one of a few
scorers. They are not interchangeable at the top-k boundary (issue #171,
``docs/DESIGN_fast_adc.md`` section 3b), so every result that is recorded, replayed or
compared carries the identity of the scorers that made it.

First stage, the ADC scan over the codes:

* ``exact-float``: numpy, full float precision. This is the **reference
  semantics**: bit for bit reproducible across RAM, memory-mapped and blocked
  storage.
* ``kernel-uint8-lut``: the compiled kernel's fast path (AVX2 ``pshufb``, and
  the pruned scan on any build). Its per-dim lookup table is quantized to 255
  levels of one query-global scale. Its scores sit within that
  resolution of the reference, and it only reorders neighbours whose reference
  scores lie inside it. **Approximate by construction.**
* ``kernel-float-lut``: the compiled kernel's unpruned scan on a build without
  AVX2. Float tables, but
  float32 accumulation in a different order from numpy, so it agrees with the
  reference to rounding, not bit for bit. Treated as approximate.

Second stage, the optional rerank of ``k * rerank`` candidates:

* against the stored originals it is exact in the index's metric, and it is
  what reconciles the first stage's boundary region;
* against a reconstruction from the codes it is itself approximate.

Callers choose with ``mode``: ``"exact"`` demands the reference scorer,
``"fast"`` takes the kernel where it can run and says when it could not.
"""

from __future__ import annotations

from . import _adc

EXACT = "exact"
FAST = "fast"
MODES = (EXACT, FAST)

EXACT_FLOAT = "exact-float"
KERNEL_UINT8 = "kernel-uint8-lut"
KERNEL_FLOAT = "kernel-float-lut"

# The kernel's lookup table: 255 levels of one query-global scale (adc_scan.cpp,
# build_lut). Stated here so that provenance does not depend on reading C++.
LUT_LEVELS = 255
LUT_SCALE = "query-global"


def resolve_mode(mode: str | None, exact: bool) -> str:
    """The requested mode from ``mode`` and the older ``exact`` flag.

    ``exact=True`` is kept as a synonym for ``mode="exact"``. Asking for both
    ``exact=True`` and ``mode="fast"`` is a contradiction and raises."""
    if mode is None:
        return EXACT if exact else FAST
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if exact and mode != EXACT:
        raise ValueError("exact=True contradicts mode='fast'; pass one of them")
    return mode


def kernel_info(kernel=None) -> dict | None:
    """What the compiled kernel is, or ``None`` when none is built.

    ``isa`` is ``"avx2"`` or ``"scalar"`` when the build reports it (kernel
    version 4 and later), else ``None``: an older build does not say."""
    k = _adc.load() if kernel is None else kernel
    if k is None:
        return None
    simd = getattr(k, "SIMD", None)
    return {
        "version": int(getattr(k, "VERSION", 0)),
        "isa": None if simd is None else ("avx2" if simd else "scalar"),
        "lut_levels": LUT_LEVELS,
        "lut_scale": LUT_SCALE,
    }


def kernel_scorer(kernel) -> str:
    """The first-stage scorer name the compiled kernel's unpruned scan runs as.

    A build that does not report its ISA is assumed to be the AVX2 one: that
    is the uint8 case, so the assumption never overstates exactness."""
    info = kernel_info(kernel)
    if info is not None and info["isa"] == "scalar":
        return KERNEL_FLOAT
    return KERNEL_UINT8


def provenance(
    mode: str,
    scorer: str,
    reason: str | None,
    kernel=None,
    rerank_width: int = 0,
    rerank_basis: str | None = None,
) -> dict:
    """One search's scorer provenance: what was asked, what ran, and why.

    ``reason`` says why the scorer that ran is not the one the mode prefers
    (``None`` when it is)."""
    first = {
        "scorer": scorer,
        "semantics": "reference" if scorer == EXACT_FLOAT else "approximate",
    }
    if scorer != EXACT_FLOAT:
        first["kernel"] = kernel_info(kernel)
    rerank = None
    if rerank_width:
        rerank = {
            "width": int(rerank_width),
            "basis": rerank_basis,
            "semantics": "exact" if rerank_basis == "originals" else "approximate",
        }
    return {
        "mode": mode,
        "first_stage": first,
        "rerank": rerank,
        "fallback_reason": reason,
    }


def describe() -> dict:
    """This installation's scorers, for certificates and benchmark artifacts.

    Records the reference scorer, whether a kernel is built and what it is, and
    therefore what ``mode="fast"`` runs here when it can."""
    k = _adc.load()
    return {
        "reference": EXACT_FLOAT,
        "fast": EXACT_FLOAT if k is None else kernel_scorer(k),
        "kernel": kernel_info(k),
    }

# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Executable conformance kit for quantizer plugins (P0 of
``docs/DESIGN_hardware_and_plugins.md`` section 2.2).

This is how "out-of-tree stays trustworthy": the plugin contract is not prose,
it is this suite. A plugin package runs it in its own CI::

    from turboquant_pro.plugin_conformance import assert_conformance
    assert_conformance(my_quantizer, x)          # x: (1, H, S, D) float32

Checks (each returns pass / skip-with-reason / FAIL-with-detail):

1.  **roundtrip** -- ``decompress(compress(x))`` has the right shape, is
    finite float, and lands inside a relative-error envelope.
2.  **packed** -- if ``compress`` accepts ``packed=True``, the packed and
    unpacked containers decompress identically.
3.  **affine** -- if the plugin exposes the affine contract
    (``grid_params`` + ``codes``), then
    ``mu + weight * grid[codes]`` (+ sparse overlay) must reproduce
    ``decompress`` exactly. This is the gate that makes the M4 fused decode
    safe to inherit: score-level equality follows algebraically from
    container-level equality because keys enter attention only through
    ``q . k``.
4.  **csr** -- if ``outlier_csr`` is exposed and non-``None``: structural
    validity (monotone ``row_ptr``, consistent lengths, in-range columns).
5.  **serialization** -- if the container (or plugin) exposes
    ``to_bytes``/``from_bytes``, a byte round-trip must decompress
    identically. Skipped otherwise (TQE1 envelope integration is tracked in
    the design doc).

Instrument smoke (certificate + probe) is deliberately *not* here: those need
corpus-shaped inputs and consumer declarations; they stay in the instrument
modules. Conformance is about the container contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .plugins import affine_codes, affine_params, outlier_csr

__all__ = [
    "ConformanceReport",
    "run_conformance",
    "assert_conformance",
]


@dataclass
class ConformanceReport:
    """Outcome of one conformance run. ``results`` maps check name to
    ``"pass"``, ``"skip: <reason>"``, or ``"FAIL: <detail>"``."""

    results: dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def failures(self) -> dict[str, str]:
        return {k: v for k, v in self.results.items() if v.startswith("FAIL")}

    def __str__(self) -> str:
        lines = [f"  {k}: {v}" for k, v in self.results.items()]
        head = "conformance PASSED" if self.passed else "conformance FAILED"
        return "\n".join([head, *lines])


def _rel_err(got: np.ndarray, want: np.ndarray) -> float:
    denom = float(np.linalg.norm(want))
    return float(np.linalg.norm(got - want)) / max(denom, 1e-30)


def run_conformance(
    quantizer: Any,
    x: np.ndarray,
    *,
    rel_err_max: float = 0.75,
    affine_atol: float = 5e-3,
    compress_kwargs: dict[str, Any] | None = None,
) -> ConformanceReport:
    """Run every applicable check against one quantizer + input block.

    ``x`` follows the block convention of the affine contract when the
    quantizer targets KV: ``(B=1, H, S, D) float32``. ``rel_err_max`` is a
    coarse sanity envelope (a 2-bit quantizer on adversarial data still sits
    well under it), not a quality metric -- quality is the instruments' job.
    ``compress_kwargs`` is forwarded to every ``compress`` call (e.g.
    ``position_start`` for bias-mode zero-points).
    """
    kw = dict(compress_kwargs or {})
    x = np.asarray(x, dtype=np.float32)
    report = ConformanceReport()

    # 1. roundtrip -----------------------------------------------------
    try:
        c = quantizer.compress(x, **kw)
        recon = np.asarray(quantizer.decompress(c))
        if recon.shape != x.shape:
            report.results["roundtrip"] = (
                f"FAIL: shape {recon.shape} != input {x.shape}"
            )
            return report
        if not np.all(np.isfinite(recon)):
            report.results["roundtrip"] = "FAIL: non-finite values in decompress"
            return report
        err = _rel_err(recon, x)
        if err > rel_err_max:
            report.results["roundtrip"] = (
                f"FAIL: relative error {err:.3f} > envelope {rel_err_max}"
            )
            return report
        report.results["roundtrip"] = f"pass (rel err {err:.4f})"
    except Exception as e:  # noqa: BLE001 - conformance reports, never raises
        report.results["roundtrip"] = f"FAIL: {type(e).__name__}: {e}"
        return report

    # 2. packed --------------------------------------------------------
    try:
        cp_ = quantizer.compress(x, packed=True, **kw)
    except TypeError:
        report.results["packed"] = "skip: compress() has no packed option"
        cp_ = None
    except Exception as e:  # noqa: BLE001
        report.results["packed"] = f"FAIL: {type(e).__name__}: {e}"
        cp_ = None
    if cp_ is not None:
        recon_p = np.asarray(quantizer.decompress(cp_))
        if np.allclose(recon_p, recon, atol=1e-6):
            report.results["packed"] = "pass"
        else:
            report.results["packed"] = (
                "FAIL: packed decompress != unpacked decompress "
                f"(max diff {float(np.abs(recon_p - recon).max()):.2e})"
            )

    # 3. affine contract -------------------------------------------------
    params = affine_params(quantizer, c)
    codes = affine_codes(quantizer, c)
    if params is None or codes is None:
        report.results["affine"] = (
            "skip: no affine contract (decompress-then-attend path)"
        )
    else:
        try:
            mu, weight, grid = params
            grid = np.asarray(grid, dtype=np.float32)
            dense = (
                np.asarray(mu, dtype=np.float32)[None, :, None, :]
                + np.asarray(weight, dtype=np.float32)[None, :, None, :]
                * grid[np.asarray(codes)]
            )
            csr = outlier_csr(quantizer, c)
            if csr is not None:
                row_ptr, cols, deltas = csr
                _, H, S, _ = dense.shape
                rows = np.repeat(np.arange(H * S), np.diff(np.asarray(row_ptr)))
                dense[
                    0, rows // S, rows % S, np.asarray(cols, dtype=np.int64)
                ] += np.asarray(deltas, dtype=np.float32)
            adiff = float(np.abs(dense - recon).max())
            if adiff <= affine_atol:
                report.results["affine"] = f"pass (max |diff| {adiff:.2e})"
            else:
                report.results["affine"] = (
                    f"FAIL: affine reconstruction differs from decompress by "
                    f"{adiff:.2e} > {affine_atol} -- the fused decode would be "
                    "wrong for this format"
                )
        except Exception as e:  # noqa: BLE001
            report.results["affine"] = f"FAIL: {type(e).__name__}: {e}"

    # 4. CSR structure ---------------------------------------------------
    csr = outlier_csr(quantizer, c) if params is not None else None
    if csr is None:
        report.results["csr"] = "skip: no sparse overlay"
    else:
        row_ptr, cols, deltas = (np.asarray(a) for a in csr)
        _, H, S, D = x.shape
        problems = []
        if row_ptr.shape[0] != H * S + 1:
            problems.append(f"row_ptr length {row_ptr.shape[0]} != H*S+1")
        if np.any(np.diff(row_ptr) < 0):
            problems.append("row_ptr not monotone")
        if not (row_ptr[-1] == len(cols) == len(deltas)):
            problems.append("row_ptr[-1] != len(cols) != len(deltas)")
        if len(cols) and int(cols.max()) >= D:
            problems.append(f"column {int(cols.max())} out of range D={D}")
        report.results["csr"] = (
            "pass" if not problems else "FAIL: " + "; ".join(problems)
        )

    # 5. serialization -----------------------------------------------------
    if not (hasattr(c, "to_bytes") and hasattr(type(c), "from_bytes")):
        report.results["serialization"] = "skip: container has no to_bytes/from_bytes"
    else:
        try:
            c2 = type(c).from_bytes(c.to_bytes())
            recon2 = np.asarray(quantizer.decompress(c2))
            report.results["serialization"] = (
                "pass"
                if np.allclose(recon2, recon, atol=1e-6)
                else "FAIL: byte round-trip changed decompress output"
            )
        except Exception as e:  # noqa: BLE001
            report.results["serialization"] = f"FAIL: {type(e).__name__}: {e}"

    return report


def assert_conformance(
    quantizer: Any, x: np.ndarray, **kwargs: Any
) -> ConformanceReport:
    """pytest-friendly wrapper: raises ``AssertionError`` with the full report
    on any FAIL; returns the report otherwise."""
    report = run_conformance(quantizer, x, **kwargs)
    assert report.passed, str(report)
    return report

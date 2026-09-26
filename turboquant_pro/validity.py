# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Certificates expire: what a certificate records so it can later be found
no longer applicable, and the checks that decide it.

A rank certificate says that a statement was true for an observer under an
environment: these inputs, this read operator, this calibration sample. When
the observer's read geometry changes, or the data drifts out of the
calibration's coverage, the certificate is not false; it is no longer
applicable. That is a different thing from monitoring, which reports drift:
this is invalidation, a status with a reason and an action, computed from
what the certificate itself recorded at issue (issue #177, phase 1;
``docs/DESIGN_certificate_expiry.md``).

Two sketches travel in the certificate's additive ``validity`` section:

- an **operator sketch**, the top eigenvectors of the reference read operator
  and the fraction of its trace they carry, so a later operator's overlap
  with the certified read subspace can be measured, not only its hash
  compared;
- a **coverage sketch**, per-channel mean and variance of the certified
  sample and its row count, so later data can be tested against the
  calibration's coverage without storing a ``D x D`` covariance.

The thresholds the status is decided against are recorded beside them.

Phase 2 adds a third, a **strata sketch**: the areas of a STRATA area map
(or of given labels) as centroids, with each area's row count in the
certified sample and a radius, the Mondrian conformal ``q``-quantile of its
rows' distances to their nearest centroid. A new row is *uncovered* when it
lands in an area the certificate saw too thinly (fewer than ``n_min`` rows)
or beyond its area's radius. For rows exchangeable with the certified sample
the uncovered fraction is at most ``(thin + 1) / (n + 1) + (1 - q)``, the
baseline recorded at issue; the check calls the certificate stale only when
a Wilson interval on the new sample's uncovered fraction lies wholly above
baseline plus tolerance, valid when it lies wholly below, and abstains in
between (uncertain means no verdict). It catches what the moment check
cannot: a small new region, which moves the global means and variances very
little.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_MIN_OVERLAP = 0.85
DEFAULT_MAX_DIVERGENCE = 0.5
DEFAULT_SKETCH_CAP = 8
MIN_SKETCH_TRACE = 0.8
DEFAULT_STRATA_N_MIN = 100
DEFAULT_RADIUS_QUANTILE = 0.99
DEFAULT_UNCOVERED_TOLERANCE = 0.05
WILSON_Z = 1.96
NOISE_FRACTION = 0.2  # a check whose own noise reaches this share of its bar abstains
VALID, STALE, UNCHECKED = "VALID", "STALE", "UNCHECKED"
INCONCLUSIVE = "INCONCLUSIVE"

__all__ = [
    "DEFAULT_MIN_OVERLAP",
    "DEFAULT_MAX_DIVERGENCE",
    "DEFAULT_SKETCH_CAP",
    "operator_sketch",
    "coverage_sketch",
    "validity_section",
    "operator_overlap",
    "coverage_divergence",
    "coverage_noise_floor",
    "strata_sketch",
    "strata_coverage",
    "check_validity",
    "validity_summary",
]


def _round(a: np.ndarray, digits: int = 6) -> list:
    return [float(f"{v:.{digits}g}") for v in np.asarray(a, dtype=np.float64).ravel()]


def operator_sketch(P: np.ndarray, cap: int = DEFAULT_SKETCH_CAP) -> dict:
    """The top ``r`` eigenvectors of a PSD read operator, ``r`` the ceiling of
    its effective rank up to ``cap``, with the fraction of the trace they
    carry. A flat operator (identity, or nearly) gets a sketch that carries
    little of the trace, and the overlap check says so instead of testing
    against arbitrary directions."""
    A = np.asarray(P, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("a read operator must be a square matrix")
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    order = np.argsort(vals)[::-1]
    vals, vecs = np.clip(vals[order], 0.0, None), vecs[:, order]
    trace = float(vals.sum())
    s2 = float((vals**2).sum())
    eff = (trace * trace / s2) if s2 > 0 else 0.0
    r = int(min(max(1, int(np.ceil(eff))), cap, vals.size))
    carried = float(vals[:r].sum() / trace) if trace > 0 else 0.0
    return {
        "dim": int(A.shape[0]),
        "rank": r,
        "effective_rank": eff,
        "trace": trace,
        "trace_fraction": carried,
        "eigenvalues": _round(vals[:r]),
        "basis": [_round(vecs[:, i]) for i in range(r)],
    }


def coverage_sketch(sample: np.ndarray) -> dict:
    """Per-channel mean and variance of the certified sample."""
    x = np.asarray(sample, dtype=np.float64)
    x = x.reshape(-1, x.shape[-1])
    return {
        "rows": int(x.shape[0]),
        "dim": int(x.shape[1]),
        "mean": _round(x.mean(axis=0)),
        "variance": _round(x.var(axis=0)),
    }


def coverage_noise_floor(sketch: dict, sample: np.ndarray) -> float:
    """The divergence :func:`coverage_divergence` reports from sampling alone.

    For a sample exchangeable with the certified one, the plug-in Jeffreys
    divergence of a channel has expectation about
    ``(1/n + 1/m) * (1 + (kappa - 1) / 2)``: the mean term contributes
    ``1/n + 1/m`` and the variance term ``(kappa - 1)/2`` times that, with
    ``kappa`` the channel's kurtosis (3 for a Gaussian, so ``2 (1/n + 1/m)``),
    estimated here from the new sample. Averaged over channels like the
    divergence itself.
    """
    x = np.asarray(sample, dtype=np.float64)
    x = x.reshape(-1, x.shape[-1])
    m, n = x.shape[0], int(sketch["rows"])
    if m < 2 or n < 1:
        return float("inf")
    c = x - x.mean(axis=0)
    v = (c**2).mean(axis=0)
    kappa = np.where(v > 0, (c**4).mean(axis=0) / np.maximum(v * v, 1e-300), 3.0)
    kappa = np.maximum(kappa, 1.0)
    return float(((1.0 / n + 1.0 / m) * (1.0 + (kappa - 1.0) / 2.0)).mean())


def _assign(x: np.ndarray, centroids: np.ndarray, block: int = 4096):
    """Nearest centroid and the Euclidean distance to it, per row."""
    c2 = (centroids**2).sum(axis=1)
    area = np.empty(len(x), dtype=np.int64)
    dist = np.empty(len(x), dtype=np.float64)
    for s in range(0, len(x), block):
        xb = x[s : s + block]
        d2 = (xb**2).sum(axis=1)[:, None] - 2.0 * xb @ centroids.T + c2[None, :]
        j = d2.argmin(axis=1)
        area[s : s + block] = j
        dist[s : s + block] = np.sqrt(np.maximum(d2[np.arange(len(xb)), j], 0.0))
    return area, dist


def strata_sketch(
    sample: np.ndarray,
    area_map=None,
    *,
    labels=None,
    n_min: int = DEFAULT_STRATA_N_MIN,
    quantile: float = DEFAULT_RADIUS_QUANTILE,
    tolerance: float = DEFAULT_UNCOVERED_TOLERANCE,
) -> dict:
    """The certified sample's areas, so later data can be placed in them.

    ``area_map`` is a :class:`turboquant_pro.strata.AreaMap` built on this
    very sample (its corpus fingerprint is checked; a map of another corpus
    is refused), or ``labels`` names each row's area directly.

    The sample is split. The even rows and their labels fit the centroids
    (recorded, rounded); the odd rows are then assigned to their nearest
    centroid exactly as a check will assign a new row, and set each area's
    row count and radius. Given the centroids, an odd row and a new
    exchangeable row are exchangeable, so the radius is a conformal quantile
    and the baseline a bound, not an estimate. Fitting and calibrating on
    the same rows would bias the radii small.
    """
    if (area_map is None) == (labels is None):
        raise ValueError("give exactly one of area_map and labels")
    if not 0 < quantile < 1:
        raise ValueError("quantile must lie strictly between 0 and 1")
    raw = np.asarray(sample)
    x = np.asarray(raw, dtype=np.float64).reshape(-1, raw.shape[-1])
    if area_map is not None:
        from turboquant_pro.anatomy import _fingerprint

        if area_map.profile.corpus_fingerprint != _fingerprint(
            np.ascontiguousarray(raw)
        ):
            raise ValueError(
                "the area map was built on another corpus than the certified "
                "sample; its areas do not describe this certificate"
            )
        labels, digest = area_map.labels, area_map.digest
    else:
        digest = None
    lab = np.asarray([str(v) for v in labels])
    if len(lab) != len(x):
        raise ValueError(f"{len(lab)} labels for {len(x)} rows")
    fit_x, fit_lab, cal = x[0::2], lab[0::2], x[1::2]
    names = sorted(set(fit_lab.tolist()))
    cent = np.asarray(
        [_round(fit_x[fit_lab == nm].mean(axis=0)) for nm in names],
        dtype=np.float64,
    )
    area, dist = _assign(cal, cent)
    areas, thin = [], 0
    for j, nm in enumerate(names):
        dj = np.sort(dist[area == j])
        n = int(dj.size)
        rank = int(np.ceil((n + 1) * quantile))  # conformal: the rank-th smallest
        covered = n >= n_min and rank <= n
        thin += 0 if covered else n
        areas.append(
            {
                "name": nm,
                "rows": n,
                "covered": bool(covered),
                "radius": float(dj[rank - 1]) if covered else None,
                "centroid": cent[j].tolist(),
            }
        )
    n_cal = len(cal)
    baseline = (thin + 1) / (n_cal + 1) + (1.0 - quantile)
    return {
        "area_map_sha256": digest,
        "assignment": "nearest-centroid",
        "split": "centroids from even rows, radii from odd rows",
        "rows": n_cal,
        "n_min": int(n_min),
        "radius_quantile": float(quantile),
        "baseline_uncovered": float(baseline),
        "tolerance": float(tolerance),
        "areas": areas,
    }


def _wilson(u: int, m: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``u / m``."""
    if m == 0:
        return 0.0, 1.0
    p = u / m
    den = 1.0 + z * z / m
    mid = (p + z * z / (2 * m)) / den
    half = z * np.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / den
    return float(max(mid - half, 0.0)), float(min(mid + half, 1.0))


def strata_coverage(sketch: dict, data: np.ndarray) -> dict:
    """Place ``data`` in the certified areas and decide coverage.

    Returns the uncovered fraction with its Wilson interval, the limit
    (baseline plus tolerance), a status (``ok``, ``FAIL`` or ``abstain``) and
    the areas the uncovered rows came from.
    """
    x = np.asarray(data, dtype=np.float64)
    x = x.reshape(-1, x.shape[-1])
    ar = sketch["areas"]
    cent = np.asarray([a["centroid"] for a in ar], dtype=np.float64)
    if x.shape[1] != cent.shape[1]:
        raise ValueError(
            f"sample has {x.shape[1]} channels; the strata sketch has {cent.shape[1]}"
        )
    area, dist = _assign(x, cent)
    covered = np.asarray([a["covered"] for a in ar])
    radius = np.asarray([a["radius"] if a["covered"] else 0.0 for a in ar])
    thin = ~covered[area]
    beyond = covered[area] & (dist > radius[area])
    miss = thin | beyond
    m, u = len(x), int(miss.sum())
    lo, hi = _wilson(u, m)
    limit = float(sketch["baseline_uncovered"] + sketch["tolerance"])
    status = "FAIL" if lo > limit else "ok" if hi <= limit else "abstain"
    counts = np.bincount(area[miss], minlength=len(ar))
    where = [
        {
            "area": ar[j]["name"],
            "rows": int(counts[j]),
            "why": "beyond radius" if ar[j]["covered"] else "thin at issue",
        }
        for j in np.argsort(-counts, kind="stable")[:3]
        if counts[j] > 0
    ]
    return {
        "status": status,
        "uncovered": u / m if m else 0.0,
        "interval": [lo, hi],
        "limit": limit,
        "rows": m,
        "from": where,
    }


def validity_section(
    *,
    observer_sha256: str | None = None,
    reference: dict | None = None,
    operator: np.ndarray | None = None,
    sample: np.ndarray | None = None,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    max_divergence: float = DEFAULT_MAX_DIVERGENCE,
    sketch_cap: int = DEFAULT_SKETCH_CAP,
    area_map=None,
    labels=None,
) -> dict:
    """The ``validity`` section of a certificate. With ``area_map`` (or
    ``labels``) and a ``sample``, it also records the strata sketch."""
    issued: dict[str, Any] = {"observer_sha256": observer_sha256}
    if reference:
        issued["reference_provider"] = reference.get("provider")
        issued["operator_sha256"] = reference.get("operator_sha256")
    out: dict[str, Any] = {
        "issued_for": issued,
        "thresholds": {
            "min_operator_overlap": float(min_overlap),
            "max_coverage_divergence": float(max_divergence),
        },
        "operator_sketch": (
            operator_sketch(operator, sketch_cap) if operator is not None else None
        ),
        "coverage_sketch": coverage_sketch(sample) if sample is not None else None,
    }
    if sample is not None and (area_map is not None or labels is not None):
        out["strata_sketch"] = strata_sketch(sample, area_map, labels=labels)
    return out


def operator_overlap(sketch: dict, P_new: np.ndarray) -> float:
    """``tr(U^T P' U) / tr(P')``: the fraction of the new operator's sensitivity
    that lies inside the certified read subspace ``U``."""
    U = np.asarray(sketch["basis"], dtype=np.float64).T  # (D, r)
    A = np.asarray(P_new, dtype=np.float64)
    A = 0.5 * (A + A.T)
    if A.shape != (U.shape[0], U.shape[0]):
        raise ValueError(
            f"new operator is {A.shape}; the sketch is {U.shape[0]}-dimensional"
        )
    t = float(np.trace(A))
    if t <= 0:
        return 0.0
    return float(np.clip(np.trace(U.T @ A @ U) / t, 0.0, 1.0))


def coverage_divergence(sketch: dict, sample: np.ndarray, ridge: float = 1e-9) -> float:
    """Diagonal Jeffreys divergence per channel between the certified sample's
    moments and a new sample's, averaged over channels: zero when the moments
    agree, growing with a mean shift measured in either variance and with a
    variance mismatch in either direction."""
    x = np.asarray(sample, dtype=np.float64)
    x = x.reshape(-1, x.shape[-1])
    m1 = np.asarray(sketch["mean"], dtype=np.float64)
    v1 = np.asarray(sketch["variance"], dtype=np.float64) + ridge
    if x.shape[1] != m1.size:
        raise ValueError(f"sample has {x.shape[1]} channels; the sketch has {m1.size}")
    m2 = x.mean(axis=0)
    v2 = x.var(axis=0) + ridge
    per = 0.5 * (v1 / v2 + v2 / v1 - 2.0) + 0.5 * (m1 - m2) ** 2 * (1.0 / v1 + 1.0 / v2)
    return float(per.mean())


def _rebuild_operator(doc: dict, contract, data: np.ndarray, queries):
    """The observer's operator on new data: from the contract when one is
    given, else from the certificate's recorded reference provider."""
    if contract is not None:
        from turboquant_pro.refinement import observer_operator

        return observer_operator(contract, data, queries=queries), "contract"
    ref = doc.get("reference")
    if ref and ref.get("provider"):
        from turboquant_pro.read_operators import create_read_operator

        op = create_read_operator(ref["provider"], **(ref.get("config") or {}))
        x = np.asarray(data, dtype=np.float64).reshape(-1, np.shape(data)[-1])
        return (
            np.asarray(op.operator(x, queries=queries), dtype=np.float64),
            "reference",
        )
    return None, None


def check_validity(
    doc: dict,
    *,
    contract=None,
    data: np.ndarray | None = None,
    queries: np.ndarray | None = None,
    inputs_ok: bool | None = None,
) -> dict:
    """Decide whether a certificate is still applicable.

    ``inputs_ok`` is the recompute's hash verdict when one ran. ``contract``
    is the observer contract the certificate should have been issued for
    (its hash is compared with ``validity.issued_for`` and the top-level
    ``observer`` section). ``data`` is a sample of the current serving
    distribution, with ``queries`` for a retrieval consumer's operator.
    """
    v = doc.get("validity") or {}
    checks: dict[str, Any] = {}
    reasons: list[str] = []
    actions: list[str] = []

    checks["source_artifact"] = (
        {"status": "not_checked"}
        if inputs_ok is None
        else {"status": "ok" if inputs_ok else "FAIL"}
    )
    if inputs_ok is False:
        reasons.append("certified inputs changed")
        actions.append("RECERTIFY")

    if contract is not None:
        recorded = (doc.get("observer") or {}).get("sha256") or (
            v.get("issued_for") or {}
        ).get("observer_sha256")
        ok = recorded == contract.digest()
        checks["observer_contract"] = {
            "status": "ok" if ok else "FAIL",
            "recorded": recorded,
            "given": contract.digest(),
        }
        if not ok:
            reasons.append(
                "certificate names no observer"
                if not recorded
                else "observer contract changed"
            )
            actions.append("REPLAN")
    else:
        checks["observer_contract"] = {"status": "not_checked"}

    sk = v.get("operator_sketch")
    th = v.get("thresholds") or {}
    if data is not None and sk:
        if sk.get("trace_fraction", 0.0) < MIN_SKETCH_TRACE:
            checks["operator_overlap"] = {
                "status": "not_checked",
                "reason": (
                    f"the certified operator is not low-rank (its top {sk['rank']} "
                    f"directions carry {sk['trace_fraction']:.2f} of the trace); "
                    "overlap is not a meaningful test"
                ),
            }
        else:
            P_new, source = _rebuild_operator(doc, contract, data, queries)
            if P_new is None:
                checks["operator_overlap"] = {
                    "status": "not_checked",
                    "reason": "no contract and no recorded reference provider",
                }
            else:
                ov = operator_overlap(sk, P_new)
                lim = float(th.get("min_operator_overlap", DEFAULT_MIN_OVERLAP))
                ok = ov >= lim
                checks["operator_overlap"] = {
                    "status": "ok" if ok else "FAIL",
                    "overlap": ov,
                    "min": lim,
                    "operator_from": source,
                }
                if not ok:
                    reasons.append("consumer read geometry changed")
                    actions.append("REPLAN")
    else:
        checks["operator_overlap"] = {"status": "not_checked"}

    cs = v.get("coverage_sketch")
    if data is not None and cs:
        div = coverage_divergence(cs, data)
        floor = coverage_noise_floor(cs, data)
        lim = float(th.get("max_coverage_divergence", DEFAULT_MAX_DIVERGENCE))
        if floor > NOISE_FRACTION * lim:
            st = "abstain"  # the sample's own noise could reach the bar
        else:
            st = "ok" if div <= lim else "FAIL"
        checks["data_coverage"] = {
            "status": st,
            "divergence": div,
            "noise_floor": floor,
            "max": lim,
            "rows": int(np.shape(data)[0]),
        }
        if st == "FAIL":
            reasons.append("data outside calibration coverage")
            actions.append("RECERTIFY")
    else:
        checks["data_coverage"] = {"status": "not_checked"}

    ss = v.get("strata_sketch")
    if data is not None and ss:
        sc = strata_coverage(ss, data)
        checks["strata_coverage"] = sc
        if sc["status"] == "FAIL":
            reasons.append("data in strata the certificate did not cover")
            actions.append("RECERTIFY")
    else:
        checks["strata_coverage"] = {
            "status": "not_checked",
            **({} if ss else {"reason": "no strata sketch recorded at issue"}),
        }

    statuses = [c["status"] for c in checks.values()]
    if "FAIL" in statuses:
        status = STALE
    elif "abstain" in statuses:
        # uncertain means no verdict: neither stale nor shown to apply
        status = INCONCLUSIVE
        undecided = [
            {"data_coverage": "data", "strata_coverage": "strata"}[k]
            for k, c in checks.items()
            if c["status"] == "abstain"
        ]
        reasons.append(f"too few rows to decide {' and '.join(undecided)} coverage")
    elif "ok" in statuses:
        status = VALID
    else:
        status = UNCHECKED
    action = None
    if actions:
        # REPLAN subsumes RECERTIFY: a changed observer needs a new plan first
        action = "REPLAN" if "REPLAN" in actions else "RECERTIFY"
    return {
        "status": status,
        "applicable": status != STALE,
        "reason": "; ".join(reasons) if reasons else None,
        "action": action,
        "checks": checks,
    }


def validity_summary(result: dict) -> str:
    labels = {
        "source_artifact": "source artifact unchanged",
        "observer_contract": "observer contract unchanged",
        "operator_overlap": "observer read geometry",
        "data_coverage": "data within calibration coverage",
        "strata_coverage": "strata coverage",
    }
    lines = ["CERTIFICATE STATUS"]
    for key, c in result["checks"].items():
        st = c["status"]
        detail = ""
        if key == "operator_overlap" and "overlap" in c:
            detail = f" (overlap {c['overlap']:.2f}, min {c['min']:.2f})"
        elif key == "data_coverage" and "divergence" in c:
            detail = (
                f" (divergence {c['divergence']:.3f}, noise floor "
                f"{c['noise_floor']:.3f}, max {c['max']:.2f})"
            )
        elif key == "strata_coverage" and "uncovered" in c:
            lo, hi = c["interval"]
            detail = (
                f" (uncovered {c['uncovered']:.3f} [{lo:.3f}, {hi:.3f}], "
                f"limit {c['limit']:.3f})"
            )
            if c["from"]:
                detail += " from " + ", ".join(
                    f"{w['area']} ({w['rows']}, {w['why']})" for w in c["from"]
                )
        elif st == "not_checked" and c.get("reason"):
            detail = f" ({c['reason']})"
        lines.append(f"  {labels[key]:<34} {st}{detail}")
    tail = f"STATUS: {result['status']}"
    if result.get("reason"):
        tail += f"   reason: {result['reason']}"
    if result.get("action"):
        tail += f"   action: {result['action']}"
    lines.append(tail)
    return "\n".join(lines)

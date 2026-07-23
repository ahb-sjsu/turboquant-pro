# TurboQuant Pro: STRATA Phase 1 — stratified anatomy & area-scoped gates
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""STRATA Phase 1: stratified measurement instruments (docs/STRATA_RFC.md §1-2).

Global scalars hide regional pathology: this module partitions measurement by
*area* so verdicts are taken as min-over-strata and thin strata ABSTAIN
instead of passing by silence. Governing rules inherited from the RFC:
uncertain ⇒ no verdict; an incomplete area-map profile matches nothing,
including itself; unregistered verdict/warning causes raise ``KeyError``;
two artifacts computed under different area-map digests MUST NOT be
compared, merged, or gated together — tools refuse, not warn.

Report schema: ``tqp-strata-report/1``. Schema fields, cause IDs, and class
names are API — closed registry, additions only (same treatment as the
connector's miss causes and the anatomy report fields).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Closed registries. An unregistered cause or class is a KeyError at emit
# time, not a new value quietly entering the artifact.
# --------------------------------------------------------------------------

STRATA_CAUSES = (
    "stratum_insufficient_n",
    "stratum_anti_hub_gap",
    "transit_concentration",
    "area_map_mismatch",
)

AREA_CLASSES = ("backbone", "hub", "NSHA", "stub", "plain")

_REPORT_SCHEMA = "tqp-strata-report/1"
_MAP_PROFILE = "tqp-area-map/1"


def _cause(cause_id: str) -> str:
    if cause_id not in STRATA_CAUSES:
        raise KeyError(
            f"unregistered strata cause {cause_id!r}; registry: {STRATA_CAUSES}"
        )
    return cause_id


def _canonical(obj: Any) -> str:
    """Identity-module conventions: sorted keys, tight separators, no NaN."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


# --------------------------------------------------------------------------
# Area map (RFC §1.1): content-addressed configuration identity.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AreaMapProfile:
    """The configuration layer of an area map, ``tqp-area-map/1``.

    Every field participates in the digest; ``None`` means unknown, and
    unknown is contagious — an incomplete profile matches nothing,
    including itself (the KV identity rule, inherited verbatim).
    """

    algorithm_id: str | None = None  # e.g. "kmeans", "metadata-key", "labels-file"
    params: str | None = None  # canonical JSON of algorithm parameters
    seed: int | None = None
    corpus_fingerprint: str | None = None
    assignment_rule: str | None = None  # e.g. "single", "metadata-key language"
    query_assignment: str | None = None  # e.g. "corpus->corpus", "metadata-key"
    overlap_policy: str | None = None  # "none" | "multi-assign" | "adjacent-probe"
    software_version: str | None = None
    profile: str = _MAP_PROFILE

    @property
    def is_complete(self) -> bool:
        return all(v is not None for v in asdict(self).values())

    def digest(self) -> str:
        return hashlib.sha256(_canonical(asdict(self)).encode()).hexdigest()

    def compatible(self, other: AreaMapProfile) -> bool:
        """Digest equality of two COMPLETE profiles; anything else is False."""
        if not (self.is_complete and other.is_complete):
            return False
        return self.digest() == other.digest()


@dataclass(frozen=True)
class AreaMap:
    """A resolved area map: labels per corpus row + the profile that made them.

    Phase-1 scope: single assignment per row (``overlap_policy`` recorded;
    a partition map is a non-conforming configuration for §5 certificates,
    which are Phase-4 anyway — the report says what it is).
    """

    labels: tuple[str, ...]
    profile: AreaMapProfile

    @property
    def digest(self) -> str:
        return self.profile.digest()

    @property
    def areas(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.labels)))

    def to_json(self) -> str:
        return json.dumps(
            {
                "profile": asdict(self.profile),
                "digest": self.digest,
                "labels": list(self.labels),
            }
        )

    @staticmethod
    def from_json(text: str) -> AreaMap:
        doc = json.loads(text)
        profile = AreaMapProfile(**doc["profile"])
        m = AreaMap(labels=tuple(doc["labels"]), profile=profile)
        if doc.get("digest") != m.digest:
            raise ValueError(
                "area-map artifact digest does not match its recomputed profile "
                "digest; refusing to load a tampered or hand-edited map"
            )
        return m


def require_same_map(digest_a: str, digest_b: str) -> None:
    """The replication predicate, enforced: refuse, not warn (RFC §1.1)."""
    if digest_a != digest_b:
        raise ValueError(
            f"{_cause('area_map_mismatch')}: artifacts were computed under "
            f"different area-map digests ({digest_a[:12]}… vs {digest_b[:12]}…) "
            "and MUST NOT be compared, merged, or gated together"
        )


def _kmeans_labels(x: np.ndarray, n_areas: int, seed: int) -> np.ndarray:
    """Small deterministic Lloyd k-means (numpy only), for --strata kmeans:N."""
    rng = np.random.default_rng(seed)
    x64 = np.ascontiguousarray(x, dtype=np.float64)
    centers = x64[rng.choice(len(x64), size=n_areas, replace=False)]
    labels = np.zeros(len(x64), dtype=np.int64)
    for _ in range(25):
        d2 = ((x64[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new = d2.argmin(1)
        if (new == labels).all():
            break
        labels = new
        for j in range(n_areas):
            mask = labels == j
            if mask.any():
                centers[j] = x64[mask].mean(0)
    return labels


def build_area_map(
    x: np.ndarray,
    spec: str,
    *,
    seed: int = 0,
    labels: np.ndarray | None = None,
    assignment_rule: str = "single",
    query_assignment: str = "corpus->corpus",
) -> AreaMap:
    """Build a fingerprinted area map from a clustering spec or given labels.

    ``spec`` is ``kmeans:N`` (computed here, then fingerprinted) or a
    descriptive id for externally supplied ``labels`` (e.g. ``by:language``).
    """
    from . import __version__
    from .anatomy import _fingerprint

    if labels is None:
        if not spec.startswith("kmeans:"):
            raise ValueError(f"unknown strata spec {spec!r} (expected kmeans:N)")
        n_areas = int(spec.split(":", 1)[1])
        lab = _kmeans_labels(x, n_areas, seed)
        algorithm_id, params = "kmeans", _canonical({"n_areas": n_areas, "iters": 25})
    else:
        lab = np.asarray(labels)
        if len(lab) != len(x):
            raise ValueError(f"labels length {len(lab)} != corpus length {len(x)}")
        algorithm_id, params = spec, _canonical({"n_labels": int(len(lab))})
    profile = AreaMapProfile(
        algorithm_id=algorithm_id,
        params=params,
        seed=int(seed),
        corpus_fingerprint=_fingerprint(np.ascontiguousarray(x)),
        assignment_rule=assignment_rule,
        query_assignment=query_assignment,
        overlap_policy="none",
        software_version=__version__,
    )
    return AreaMap(labels=tuple(str(v) for v in lab), profile=profile)


# --------------------------------------------------------------------------
# Stratified anatomy (RFC §1.3-1.5, §2).
# --------------------------------------------------------------------------


def _area_class(
    s_k: float, tau_mean: float, max_nk: float, k: int, thresholds: dict
) -> str:
    """§1.4 classification. Thresholds are FIELDS of the report, not constants."""
    if max_nk < thresholds["N"] and tau_mean < thresholds["tau"] and s_k < 1.0:
        return "stub"
    if tau_mean >= thresholds["tau"] and max_nk >= thresholds["N"]:
        return "backbone"
    if s_k >= 1.0 and tau_mean < thresholds["tau"]:
        return "NSHA"  # local hubs, no global transit (the licensed pun)
    if s_k >= 1.0:
        return "hub"
    return "plain"


def stratified_anatomy(
    base: np.ndarray,
    area_map: AreaMap,
    *,
    queries: np.ndarray | None = None,
    query_labels: list[str] | None = None,
    k: int = 10,
    n_min: int = 2000,
    q_min: int = 500,
    tau_threshold: float = 0.5,
    nk_threshold: float | None = None,
    seed: int | None = None,
) -> dict:
    """Per-area hubness anatomy: the ``tqp-strata-report/1`` artifact.

    Per area: count skew ``S_k``, ``max_Nk``, mean transit fraction, the
    anatomy correlations restricted to the area (mechanism attribution is
    per-area — §7's non-identifiability holds per-area too), size-stable
    companions, and a verdict: ABSTAIN below ``n_min`` corpus rows or
    ``q_min`` queries (registered cause ``stratum_insufficient_n``);
    ABSTAIN is not a pass.
    """
    from . import __version__
    from .anatomy import _fingerprint, _rankcorr, knn_exact

    base = np.ascontiguousarray(base, dtype=np.float32)
    if len(area_map.labels) != len(base):
        raise ValueError("area map labels do not cover the corpus")
    self_mode = queries is None
    q = base if self_mode else np.ascontiguousarray(queries, dtype=np.float32)
    q_labels = list(area_map.labels) if self_mode else query_labels
    if q_labels is None or len(q_labels) != len(q):
        raise ValueError("query labels required (and must cover queries)")

    d, idx = knn_exact(base, q, k, exclude_self=self_mode)
    counts = np.bincount(idx[:, :k].ravel(), minlength=len(base)).astype(np.float64)
    labels = np.asarray(area_map.labels)
    qlab = np.asarray(q_labels)
    # Intra vs transit per corpus row (§1.3): a hit is intra when the query's
    # area matches the neighbour row's area.
    intra = np.zeros(len(base), dtype=np.float64)
    hit_rows = idx[:, :k]
    same = qlab[:, None] == labels[hit_rows]
    np.add.at(intra, hit_rows.ravel(), same.ravel().astype(np.float64))
    tau = (counts - intra) / np.maximum(counts, 1.0)

    if self_mode:
        d_bb = d
    else:
        d_bb, _ = knn_exact(base, base, k, exclude_self=True)
    dk = d_bb[:, k - 1]
    central = -np.linalg.norm(base - base.mean(0), axis=1).astype(np.float64)

    thresholds = {
        "tau": float(tau_threshold),
        "N": float(nk_threshold if nk_threshold is not None else 5 * k),
        "n_min": int(n_min),
        "q_min": int(q_min),
    }
    areas = []
    for area in area_map.areas:
        rows = labels == area
        n_i = int(rows.sum())
        n_q = int((qlab == area).sum())
        entry: dict[str, Any] = {"id": area, "n": n_i, "n_queries": n_q}
        if n_i < n_min or n_q < q_min:
            entry["verdict"] = "ABSTAIN"
            entry["cause"] = _cause("stratum_insufficient_n")
            areas.append(entry)
            continue
        c = counts[rows]
        sd = c.std()
        s_k = float(((c - c.mean()) ** 3).mean() / max(sd**3, 1e-12))
        tau_mean = float(tau[rows].mean())
        entry.update(
            {
                "verdict": "pass",
                "count_skew": s_k,
                "max_Nk": float(c.max()),
                "tau_mean": tau_mean,
                "robin_hood_index": float(
                    0.5 * np.abs(c - c.mean()).sum() / max(c.sum(), 1e-12)
                ),
                "frac_above_2k": float((c > 2 * k).mean()),
                "corr_neg_dk": _rankcorr(c, -dk[rows]),
                "corr_centrality": _rankcorr(c, central[rows]),
                "class": _area_class(s_k, tau_mean, float(c.max()), k, thresholds),
            }
        )
        areas.append(entry)

    eligible = [a for a in areas if a["verdict"] != "ABSTAIN"]
    return {
        "schema": _REPORT_SCHEMA,
        "instrument": "anatomy",
        "provenance": {
            "area_map_digest": area_map.digest,
            "corpus_fingerprint": _fingerprint(base),
            "n": int(len(base)),
            "n_queries": int(len(q)),
            "k": int(k),
            "seed": seed,
            "estimator": "exact_knn_on_given_vectors",
            "battery": "corpus->corpus" if self_mode else "query->corpus",
            "software_version": __version__,
        },
        "thresholds": thresholds,
        "areas": areas,
        "summary": {
            "n_areas": len(areas),
            "n_abstain": len(areas) - len(eligible),
            "only_abstain": not eligible,
        },
    }


# --------------------------------------------------------------------------
# Stratified hubdiff (RFC §2.1): gates become min over eligible strata.
# --------------------------------------------------------------------------


def stratified_hub_differential(
    exact_idx: np.ndarray,
    approx_idx: np.ndarray,
    n_base: int,
    area_map: AreaMap,
    query_labels: list[str],
    *,
    k: int = 10,
    anti_quantile: float = 0.10,
    min_anti_recall: float | None = None,
    n_min: int = 2000,
    q_min: int = 500,
) -> dict:
    """Per-stratum differential oracle; verdict = min over eligible strata.

    Strata are assigned per QUERY (``query_labels``). Per eligible stratum:
    recall@k, p05 per-query recall, anti-hub recall (anti-hub set computed
    on the GLOBAL count distribution — a stratum cannot vote itself out of
    having anti-hubs). ``stratum_anti_hub_gap`` is emitted per failing
    stratum when ``min_anti_recall`` is gated.
    """
    exact_idx = np.asarray(exact_idx)[:, :k]
    approx_idx = np.asarray(approx_idx)[:, :k]
    if exact_idx.shape != approx_idx.shape:
        raise ValueError("exact/approx shapes differ")
    qlab = np.asarray(query_labels)
    if len(qlab) != len(exact_idx):
        raise ValueError("query labels must cover the query set")

    counts = np.bincount(exact_idx.ravel(), minlength=n_base).astype(np.float64)
    anti_rows = counts <= np.quantile(counts, anti_quantile)
    per_q_recall = np.array(
        [
            len(set(e.tolist()) & set(a.tolist())) / k
            for e, a in zip(exact_idx, approx_idx)
        ]
    )
    anti_q = anti_rows[exact_idx[:, 0]]

    thresholds: dict[str, Any] = {
        "n_min": int(n_min),
        "q_min": int(q_min),
        "anti_quantile": float(anti_quantile),
    }
    if min_anti_recall is not None:
        thresholds["min_anti_recall"] = float(min_anti_recall)

    areas = []
    for area in sorted(set(qlab.tolist())):
        rows = qlab == area
        n_q = int(rows.sum())
        entry: dict[str, Any] = {"id": area, "n_queries": n_q}
        if n_q < q_min:
            entry["verdict"] = "ABSTAIN"
            entry["cause"] = _cause("stratum_insufficient_n")
            areas.append(entry)
            continue
        r = per_q_recall[rows]
        a_mask = rows & anti_q
        anti_recall = float(per_q_recall[a_mask].mean()) if a_mask.any() else None
        entry.update(
            {
                "recall_at_k": float(r.mean()),
                "p05_recall": float(np.quantile(r, 0.05)),
                "anti_hub_recall": anti_recall,
                "anti_hub_query_frac": float(a_mask.sum() / max(n_q, 1)),
            }
        )
        failed = (
            min_anti_recall is not None
            and anti_recall is not None
            and anti_recall < min_anti_recall
        )
        entry["verdict"] = "fail" if failed else "pass"
        if failed:
            entry["cause"] = _cause("stratum_anti_hub_gap")
        areas.append(entry)

    eligible = [a for a in areas if a["verdict"] != "ABSTAIN"]
    from . import __version__

    return {
        "schema": _REPORT_SCHEMA,
        "instrument": "hubdiff",
        "provenance": {
            "area_map_digest": area_map.digest,
            "n_base": int(n_base),
            "n_queries": int(len(exact_idx)),
            "k": int(k),
            "estimator": "id_arrays",
            "software_version": __version__,
        },
        "thresholds": thresholds,
        "areas": areas,
        "summary": {
            "n_areas": len(areas),
            "n_abstain": len(areas) - len(eligible),
            "n_failed": sum(1 for a in eligible if a["verdict"] == "fail"),
            "only_abstain": not eligible,
            "min_over_strata_anti_hub_recall": min(
                (
                    a["anti_hub_recall"]
                    for a in eligible
                    if a.get("anti_hub_recall") is not None
                ),
                default=None,
            ),
        },
    }


def report_exit_code(report: dict, *, abstain_fails: bool = False) -> int:
    """§2.1 exit semantics: 0 pass · 1 a stratum failed · 3 only-ABSTAIN."""
    if report["summary"]["only_abstain"]:
        return 1 if abstain_fails else 3
    if report["summary"].get("n_failed", 0) > 0:
        return 1
    return 0

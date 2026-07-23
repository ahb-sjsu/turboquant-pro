"""Hub anatomy and the compression differential oracle.

Scalar hubness (skewness of the reverse-kNN occurrence counts) is
**non-identifying**: two corpora can share the same reading produced by opposite
mechanisms — a smooth density-driven tail versus a few centrality super-hubs —
with opposite ANN behaviour (openvector-bench, ``spec/BOND_METRIC.md``). The
identifying report is the **anatomy vector**: the count distribution's tail
plus what the hubs *are* (their centrality, local density, and nearest-pair
structure relative to the population).

``hub_differential`` is the retrieval-space version of this project's
coherence rule. Aggregate recall is a mean, and means hide tails: compression
can preserve overall recall while silently re-ranking the hub set or
collapsing on anti-hub queries — the same blindness reconstruction cosine has
for ranking (``docs/KV_KEYS_FINDING.md``), one level up. The oracle therefore
reports, alongside recall@k: hub-rank divergence (does the compressed index
agree about *which rows are hubs*?) and anti-hub recall (the queries whose
true neighbours are the least-visited rows — the first casualties of an
over-regularised index).
"""

from __future__ import annotations

import numpy as np

__all__ = ["knn_exact", "hub_anatomy", "hub_differential"]


def knn_exact(
    base: np.ndarray,
    queries: np.ndarray,
    k: int,
    *,
    exclude_self: bool = False,
    block: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-``k`` Euclidean neighbours, query-blocked.

    Returns ``(dists, idx)``. With ``exclude_self=True`` the first neighbour of
    each query is dropped (use when ``queries`` are rows of ``base``).
    """
    base = np.ascontiguousarray(base, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    kk = k + 1 if exclude_self else k
    kk = min(kk, len(base))
    bsq = (base * base).sum(1)
    D = np.empty((len(queries), kk), dtype=np.float32)
    Ix = np.empty((len(queries), kk), dtype=np.int64)
    for s in range(0, len(queries), block):
        q = queries[s : s + block]
        d2 = bsq[None, :] - 2.0 * (q @ base.T) + (q * q).sum(1, keepdims=True)
        part = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        dd = np.take_along_axis(d2, part, 1)
        order = np.argsort(dd, 1)
        Ix[s : s + block] = np.take_along_axis(part, order, 1)
        D[s : s + block] = np.sqrt(np.maximum(np.take_along_axis(dd, order, 1), 0.0))
    if exclude_self:
        D, Ix = D[:, 1:], Ix[:, 1:]
    return D, Ix


def _rankcorr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (rank-transform, then Pearson on ranks).

    Deliberate: reverse-count distributions are heavy-tailed, where Pearson
    on raw values is fragile. Every ``corr_*`` field in this module is
    Spearman, and the reports say so (``corr_method``).
    """
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def _fingerprint(x: np.ndarray) -> str:
    """Cheap, deterministic dataset fingerprint for report provenance.

    shape:dtype:sha256-prefix over a strided byte sample (full-corpus hashing
    would dominate runtime at scale; the stride still catches any global
    substitution). Reports that carry results must carry this — a number
    without its dataset is not evidence.
    """
    import hashlib

    x = np.ascontiguousarray(x)
    raw = x.view(np.uint8).reshape(-1)
    stride = max(1, len(raw) // (1 << 20))  # sample <= ~1 MiB
    h = hashlib.sha256(bytes(raw[::stride])).hexdigest()[:16]
    return f"{'x'.join(map(str, x.shape))}:{x.dtype}:{h}"


def _mechanism_prescription(
    corr_central: float, corr_neg_dk: float, hub_c: float, all_c: float
) -> tuple[str, str]:
    """Mechanism classification -> remedy. The anatomy is a prescription pad.

    Centrality super-hubs respond to centering / localized centering;
    density-driven hubs respond to CSLS / mutual-proximity rescaling — the
    latter needing one precomputed scalar per record (a natural optional
    TQE1 trailer). Thresholds are heuristics and are visible here on
    purpose; the classification quotes its own inputs in the report.
    """
    centrality = corr_central > 0.3 and hub_c > all_c * 1.1
    density = corr_neg_dk > 0.4
    if centrality and not density:
        return (
            "centrality",
            "centering / localized centering (hubs are pipeline-central; "
            "check for mean shift or collapsed subspace upstream)",
        )
    if density and not centrality:
        return (
            "density",
            "CSLS / mutual-proximity rescaling (one precomputed scalar per "
            "record — optional TQE1 trailer candidate)",
        )
    if density and centrality:
        return ("mixed", "apply both: center first, then mutual-proximity rescale")
    return ("unclear", "no dominant mechanism; re-measure with real queries")


def hub_anatomy(
    base: np.ndarray,
    queries: np.ndarray | None = None,
    *,
    k: int = 10,
    hub_quantile: float = 0.99,
) -> dict:
    """The hub anatomy vector A_hub for one corpus.

    With ``queries=None`` the battery is corpus->corpus (each base row queries
    the rest); otherwise query->corpus. Reported: reverse-count skew and max,
    top-hub mass share, rank correlations of the count with centrality, local
    density (-d_k) and nearest-pair distance (-d_1), and hub-vs-population
    medians for each — the components that distinguish density-driven hubs
    from centrality super-hubs at equal scalar skew.
    """
    base = np.ascontiguousarray(base, dtype=np.float32)
    self_mode = queries is None
    q = base if self_mode else np.ascontiguousarray(queries, dtype=np.float32)
    d, idx = knn_exact(base, q, k, exclude_self=self_mode)
    counts = np.bincount(idx[:, :k].ravel(), minlength=len(base)).astype(np.float64)
    sd = counts.std()
    skew = float(((counts - counts.mean()) ** 3).mean() / max(sd**3, 1e-12))
    mdir = base.mean(0)
    mdir /= max(float(np.linalg.norm(mdir)), 1e-12)
    central = base @ mdir
    # Per-base-row local scale needs base->base neighbours regardless of battery.
    if self_mode:
        d_bb = d
    else:
        d_bb, _ = knn_exact(base, base, k, exclude_self=True)
    dk, d1 = d_bb[:, k - 1], d_bb[:, 0]
    hubs = counts >= np.quantile(counts, hub_quantile)
    top = np.sort(counts)[-10:][::-1]
    corr_central = _rankcorr(counts, central)
    corr_neg_dk = _rankcorr(counts, -dk)
    hub_c, all_c = float(np.median(central[hubs])), float(np.median(central))
    mechanism, prescription = _mechanism_prescription(
        corr_central, corr_neg_dk, hub_c, all_c
    )
    from turboquant_pro import __version__

    # NOTE: every field name below is API (same closed-registry treatment as
    # the connector's miss causes) — additions only, never renames.
    return {
        "battery": "corpus->corpus" if self_mode else "query->corpus",
        "k": k,
        "n_base": int(len(base)),
        "n_queries": int(len(q)),
        # -- provenance: the meter meters itself ---------------------------
        "estimator": "exact_knn_on_given_vectors",
        "corr_method": "spearman",
        "dataset_fingerprint": _fingerprint(base),
        "query_fingerprint": None if self_mode else _fingerprint(q),
        "tool_version": __version__,
        # -- the count tail ------------------------------------------------
        "count_skew": skew,
        "count_max": float(counts.max()),
        "top10_counts": [float(t) for t in top],
        "top_hub_mass_share": float(counts[hubs].sum() / max(counts.sum(), 1e-12)),
        # Size-stable companions: skew GROWS with n, so cross-n comparisons
        # need statistics that do not (see the primer's cross-n note).
        "robin_hood_index": float(
            0.5 * np.abs(counts - counts.mean()).sum() / max(counts.sum(), 1e-12)
        ),
        "frac_above_2k": float((counts > 2 * k).mean()),
        # -- what the hubs ARE --------------------------------------------
        "corr_count_centrality": corr_central,
        "corr_count_neg_dk": corr_neg_dk,
        "corr_count_neg_d1": _rankcorr(counts, -d1),
        "hub_vs_all_median_centrality": [hub_c, all_c],
        "hub_vs_all_median_dk": [float(np.median(dk[hubs])), float(np.median(dk))],
        "hub_vs_all_median_d1": [float(np.median(d1[hubs])), float(np.median(d1))],
        # -- the prescription pad -------------------------------------------
        "mechanism": mechanism,
        "prescription": prescription,
    }


def hub_differential(
    exact_idx: np.ndarray,
    approx_idx: np.ndarray,
    n_base: int,
    *,
    k: int = 10,
    hub_quantile: float = 0.99,
    anti_quantile: float = 0.10,
    mode: str = "id_arrays",
) -> dict:
    """Differential oracle between an exact and an approximate/compressed search.

    Inputs are the two neighbour-id arrays for the SAME query set (shape
    ``(n_queries, >=k)``). Reported:

    * ``recall_at_k`` — the aggregate everyone already looks at;
    * ``hub_rank_corr`` / ``hub_set_jaccard`` — do the two systems agree about
      which rows are hubs? (rank correlation of reverse-count vectors; Jaccard
      of the top-``hub_quantile`` hub sets);
    * ``anti_hub_recall`` — recall restricted to queries whose exact nearest
      neighbour is an anti-hub (bottom ``anti_quantile`` of exact counts):
      the tail where over-regularised indexes fail first while the mean recall
      stays green;
    * ``recall_p05`` — the 5th percentile of per-query recall, the
      distributional version of the same warning.
    """
    e = np.asarray(exact_idx)[:, :k]
    a = np.asarray(approx_idx)[:, :k]
    if e.shape != a.shape:
        raise ValueError(f"shape mismatch: exact {e.shape} vs approx {a.shape}")
    per_q = np.fromiter(
        (len(set(er) & set(ar)) / k for er, ar in zip(e, a)),
        dtype=np.float64,
        count=len(e),
    )
    ce = np.bincount(e.ravel(), minlength=n_base).astype(np.float64)
    ca = np.bincount(a.ravel(), minlength=n_base).astype(np.float64)
    he = ce >= np.quantile(ce, hub_quantile)
    ha = ca >= np.quantile(ca, hub_quantile)
    inter, union = int((he & ha).sum()), int((he | ha).sum())
    anti_rows = ce <= np.quantile(ce, anti_quantile)
    anti_q = anti_rows[e[:, 0]]
    # The full recall-vs-N_k-percentile curve: the anti-hub gate is a knob;
    # the curve is the fact it summarizes — stored so the threshold choice
    # is auditable after the fact.
    nn_counts = ce[e[:, 0]]
    edges = np.quantile(nn_counts, np.linspace(0, 1, 11))
    curve = []
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        sel = (
            (nn_counts >= lo) & (nn_counts <= hi)
            if i == 9
            else (nn_counts >= lo) & (nn_counts < hi)
        )
        curve.append(float(per_q[sel].mean()) if sel.any() else float("nan"))
    from turboquant_pro import __version__

    # NOTE: field names + warning strings are API (closed registry;
    # additions only).
    return {
        "k": k,
        "n_queries": int(len(e)),
        # -- provenance ----------------------------------------------------
        "mode": mode,  # "id_arrays" (primary: the REAL neighbor graphs) or
        # "reconstructed_vectors" (convenience PROXY — see the CLI warning)
        "corr_method": "spearman",
        "tool_version": __version__,
        # -- the aggregate everyone already looks at -----------------------
        "recall_at_k": float(per_q.mean()),
        "recall_p05": float(np.percentile(per_q, 5)),
        # -- the tail ------------------------------------------------------
        "hub_rank_corr": _rankcorr(ce, ca),
        "hub_set_jaccard": float(inter / union) if union else float("nan"),
        "anti_hub_recall": (
            float(per_q[anti_q].mean()) if anti_q.any() else float("nan")
        ),
        "anti_hub_query_frac": float(anti_q.mean()),
        "recall_by_count_decile": curve,
        "exact_count_skew": float(
            ((ce - ce.mean()) ** 3).mean() / max(ce.std() ** 3, 1e-12)
        ),
        "approx_count_skew": float(
            ((ca - ca.mean()) ** 3).mean() / max(ca.std() ** 3, 1e-12)
        ),
    }

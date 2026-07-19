# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""
(A2) consumer-metric probe: is the angular quotient safe for *your* metric?

The v1.2.0 KV-keys incident (docs/KV_KEYS_FINDING.md) had a signature that
cosine could not see: per-vector polar quantization of attention keys scored
0.095 mean reconstruction error while perplexity exploded to ~10^4, because
softmax(Q.K^T) depends on per-channel scale structure that per-vector
normalization deletes. The general lesson, made precise as condition (A2) of
the angular-transfer theorem in the companion theory paper
(https://github.com/ahb-sjsu/the-angular-observer): a quotient that discards
scale is safe exactly when the *consumer's* metric is carried by the
tangential part of the displacement,

    tangential_fraction(x, y) = (|x - y|^2 - (|x| - |y|)^2) / |x - y|^2,

the part that survives row-normalization. When displacement becomes
radial/norm-dominated (tangential fraction falling), angular quantization
will destroy ranking while cosine still reads fine.

There are two distinct failure classes, and this module instruments both:

* **Radial-displacement failure** (norm-dominated variation, rising
  hubness): the tangential fraction falls, and the angular quotient becomes
  blind to what distinguishes vectors. Guarded continuously by the
  streaming statistic (:func:`displacement_decomposition` and the
  :class:`~turboquant_pro.monitor.QualityMonitor` integration).
* **Direction-concentration failure** (the actual v1.2.0 keys regime): a
  large shared per-channel offset leaves displacements fully tangential --
  the tangential fraction reads ~1! -- but squeezes every direction into a
  tiny cone, so the informative angular displacement sits *below the
  angular quantizer's cell size* and rounds away. The streaming fraction
  cannot see this; the ``median_unit_displacement`` statistic and,
  decisively, the end-to-end probe can.

* :func:`probe_quotient` / :func:`recommend_key_quantizer` -- the
  calibration-time probe: quantize a sample batch under each candidate
  family's information quotient and measure rank agreement of the *declared
  consumer's* scores (cosine ranking, L2 ranking, attention logits), then
  recommend the family whose preserved component tracks the consumer. Being
  end-to-end, it catches both failure classes.

The probe quantizers here are deliberate *proxies* that isolate each
family's invariant (what it preserves vs deletes) at matched bit budget;
they are not the shipped kernels. Use them to pick a family, then validate
the shipped path end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backend import to_numpy

__all__ = [
    "tangential_fraction",
    "tangential_fractions",
    "displacement_decomposition",
    "probe_quotient",
    "recommend_key_quantizer",
    "A2ProbeResult",
]


# ---------------------------------------------------------------------------
# The (A2) quantity
# ---------------------------------------------------------------------------


def tangential_fraction(x: np.ndarray, y: np.ndarray) -> float:
    """Fraction of the displacement x - y that survives row-normalization.

    Returns (|x - y|^2 - (|x| - |y|)^2) / |x - y|^2, in [0, 1]. Values near 1
    mean the pair differs in direction (angular quantization can see it);
    values near 0 mean the pair differs mostly in norm (angular quantization
    is blind to it). Returns NaN for coincident vectors.
    """
    x = np.asarray(to_numpy(x), dtype=np.float64).ravel()
    y = np.asarray(to_numpy(y), dtype=np.float64).ravel()
    d2 = float(((x - y) ** 2).sum())
    if d2 <= 0.0:
        return float("nan")
    dr = float(np.linalg.norm(x)) - float(np.linalg.norm(y))
    frac = (d2 - dr * dr) / d2
    return float(min(max(frac, 0.0), 1.0))


def tangential_fractions(
    batch: np.ndarray,
    n_pairs: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Tangential fractions over sampled row pairs of ``batch``."""
    batch = np.asarray(to_numpy(batch), dtype=np.float64)
    n = batch.shape[0]
    if n < 2:
        return np.empty(0)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    diff = batch[i] - batch[j]
    d2 = (diff**2).sum(axis=1)
    dr = np.linalg.norm(batch[i], axis=1) - np.linalg.norm(batch[j], axis=1)
    ok = d2 > 0
    frac = (d2[ok] - dr[ok] ** 2) / d2[ok]
    return np.clip(frac, 0.0, 1.0)


def displacement_decomposition(
    batch: np.ndarray,
    n_pairs: int = 2000,
    seed: int = 0,
) -> dict:
    """Summary statistics of the tangential/radial displacement split.

    Returns a dict with median/mean tangential fraction and the radial
    complement. A falling median under data drift (norm-dominated variation,
    rising hubness) is the early-warning signature of angular-quantization
    ranking damage.
    """
    frac = tangential_fractions(batch, n_pairs=n_pairs, seed=seed)
    if len(frac) == 0:
        nan = float("nan")
        return {
            "median_tangential_fraction": nan,
            "mean_tangential_fraction": nan,
            "median_radial_fraction": nan,
            "n_pairs": 0,
        }
    med = float(np.median(frac))
    return {
        "median_tangential_fraction": med,
        "mean_tangential_fraction": float(np.mean(frac)),
        "median_radial_fraction": 1.0 - med,
        "n_pairs": int(len(frac)),
    }


# ---------------------------------------------------------------------------
# Consumer scores and rank agreement (scipy-free)
# ---------------------------------------------------------------------------

_CONSUMERS = ("cosine", "l2", "attention_logits")


def _consumer_scores(
    data: np.ndarray,
    queries: np.ndarray,
    consumer: str,
) -> np.ndarray:
    """Scores (n_queries, n_data) under the declared consumer metric."""
    if consumer == "cosine":
        qn = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-30)
        dn = data / np.maximum(np.linalg.norm(data, axis=1, keepdims=True), 1e-30)
        return qn @ dn.T
    if consumer == "l2":
        sq_q = (queries**2).sum(axis=1)[:, None]
        sq_d = (data**2).sum(axis=1)[None, :]
        return -(sq_q + sq_d - 2.0 * (queries @ data.T))
    if consumer == "attention_logits":
        return queries @ data.T
    raise ValueError(f"consumer must be one of {_CONSUMERS}, got {consumer!r}")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, ordinal ranks, scipy-free."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------
# Family-invariant proxy quantizers (matched bit budget)
# ---------------------------------------------------------------------------


def _polar_proxy(batch: np.ndarray, bits: int, rng: np.random.Generator):
    """Per-vector polar quotient: store norm, quantize the unit direction.

    Isolates the polar family's invariant: per-vector scale is preserved
    exactly (fp32 norm), the direction is uniformly scalar-quantized per
    dimension after a random rotation, matching the shipped polar flow's
    information split.
    """
    n, d = batch.shape
    norms = np.maximum(np.linalg.norm(batch, axis=1, keepdims=True), 1e-30)
    unit = batch / norms
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    rot = unit @ q
    levels = 2**bits
    lo, hi = rot.min(), rot.max()
    step = (hi - lo) / max(levels - 1, 1)
    codes = np.round((rot - lo) / max(step, 1e-30))
    deq = (codes * step + lo) @ q.T
    deq_norms = np.maximum(np.linalg.norm(deq, axis=1, keepdims=True), 1e-30)
    return (deq / deq_norms) * norms


def _per_channel_proxy(batch: np.ndarray, bits: int):
    """Per-channel asymmetric quotient: per-channel scale and zero-point.

    Isolates the per-channel family's invariant: each channel's affine
    structure (min/scale) is preserved, per-vector norms are not specially
    protected.
    """
    levels = 2**bits
    mn = batch.min(axis=0, keepdims=True)
    mx = batch.max(axis=0, keepdims=True)
    scale = np.maximum((mx - mn) / max(levels - 1, 1), 1e-30)
    codes = np.round((batch - mn) / scale)
    return codes * scale + mn


def _zca_whiten_fit(batch: np.ndarray):
    """Fit a symmetric (ZCA) whitening transform on ``batch``.

    Returns ``(mu, W, W_inv)`` where the whitening map is ``(x - mu) @ W`` and
    its inverse is ``z @ W_inv + mu``. A small ridge (relative to the mean
    eigenvalue) keeps the covariance invertible for rank-deficient or
    low-sample batches.
    """
    mu = batch.mean(axis=0)
    centered = batch - mu
    n = max(centered.shape[0], 1)
    cov = (centered.T @ centered) / n
    ridge = 1e-6 * (np.trace(cov) / cov.shape[0] + 1e-30)
    cov = cov + ridge * np.eye(cov.shape[0])
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-12)
    w = (evecs * (1.0 / np.sqrt(evals))) @ evecs.T
    w_inv = (evecs * np.sqrt(evals)) @ evecs.T
    return mu, w, w_inv


def _whitened_polar_proxy(batch: np.ndarray, bits: int, rng: np.random.Generator):
    """Polar quotient applied in a ZCA-whitened basis, then mapped back.

    Whitening removes the shared offset AND the cross-channel correlation that
    define the direction-concentration regime (post-RoPE keys; anisotropic
    sentence embeddings), so the informative directions expand out of the cone
    where a per-vector polar code can resolve them. This is the third candidate
    family: the per-channel proxy removes per-channel scale but leaves
    cross-channel correlation, which ZCA whitening removes.
    """
    mu, w, w_inv = _zca_whiten_fit(batch)
    whitened = (batch - mu) @ w
    return _polar_proxy(whitened, bits, rng) @ w_inv + mu


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------


@dataclass
class A2ProbeResult:
    """Outcome of a consumer-metric probe over candidate quantizer families.

    Attributes:
        consumer: The declared downstream metric.
        spearman_polar: Rank agreement of consumer scores after the polar
            (per-vector norm + direction) quotient at the probe bit budget.
        spearman_per_channel: Same for the per-channel affine quotient.
        spearman_whitened: Same for the ZCA-whitened polar quotient, or None
            unless ``include_whitened=True`` was passed. Whitening removes
            cross-channel correlation the per-channel (diagonal) code leaves,
            and can win in the correlated direction-concentration regime.
        median_tangential_fraction: The (A2) statistic of the batch itself
            (guards the radial-displacement failure class).
        median_unit_displacement: Median chordal distance between the
            row-normalized directions of sampled pairs. Values near sqrt(2)
            = isotropic directions; values near 0 = direction concentration
            (the DC-offset keys regime, where angular quantization cells
            swamp the informative displacement even though the tangential
            fraction reads ~1).
        recommendation: "polar", "per_channel", or (when included) "whitened"
            -- the family whose preserved component tracks the consumer metric
            best. Ties favour the cheaper code (polar, then per_channel).
        margin: Absolute Spearman gap between the best and second-best family.
    """

    consumer: str
    spearman_polar: float
    spearman_per_channel: float
    median_tangential_fraction: float
    median_unit_displacement: float
    recommendation: str
    margin: float
    spearman_whitened: float | None = None

    def as_dict(self) -> dict:
        return {
            "consumer": self.consumer,
            "spearman_polar": self.spearman_polar,
            "spearman_per_channel": self.spearman_per_channel,
            "spearman_whitened": self.spearman_whitened,
            "median_tangential_fraction": self.median_tangential_fraction,
            "median_unit_displacement": self.median_unit_displacement,
            "recommendation": self.recommendation,
            "margin": self.margin,
        }


def probe_quotient(
    batch: np.ndarray,
    consumer: str = "cosine",
    queries: np.ndarray | None = None,
    bits: int = 4,
    n_queries: int = 32,
    seed: int = 0,
    include_whitened: bool = False,
) -> A2ProbeResult:
    """Probe whether the polar or per-channel quotient preserves the consumer.

    Applies each family's proxy quantizer to ``batch`` at the same bit
    budget, computes the declared consumer's scores against ``queries``
    (sampled from the batch when not supplied), and compares Spearman rank
    agreement with the exact scores. The family with higher agreement is
    recommended.

    This is the check that would have caught the v1.2.0 keys catastrophe at
    calibration time: for ``consumer="attention_logits"`` on post-RoPE-like
    key statistics (large shared per-channel offsets), the polar quotient's
    rank agreement collapses while its reconstruction cosine stays high.

    Args:
        batch: (n, d) sample of the tensors to be quantized.
        consumer: "cosine", "l2", or "attention_logits".
        queries: Optional (m, d) query sample; defaults to batch rows.
        bits: Probe bit budget per dimension (default 4).
        n_queries: Queries sampled from the batch when ``queries`` is None.
        seed: Determinism seed for rotation/query sampling.
        include_whitened: When True, also evaluate the ZCA-whitened polar
            family and let it win the recommendation. Off by default so the
            two-family verdict is unchanged; the whitened proxy uses a
            dedicated RNG, so the polar/query/unit-displacement stream is
            byte-for-byte identical to the two-family path.
    """
    batch = np.asarray(to_numpy(batch), dtype=np.float64)
    if batch.ndim != 2 or batch.shape[0] < 4:
        raise ValueError("batch must be (n, d) with n >= 4")
    rng = np.random.default_rng(seed)
    if queries is None:
        take = rng.choice(
            batch.shape[0], size=min(n_queries, batch.shape[0]), replace=False
        )
        queries = batch[take]
    queries = np.asarray(to_numpy(queries), dtype=np.float64)

    exact = _consumer_scores(batch, queries, consumer)
    polar = _consumer_scores(_polar_proxy(batch, bits, rng), queries, consumer)
    perch = _consumer_scores(_per_channel_proxy(batch, bits), queries, consumer)

    sp_polar = float(
        np.nanmean([_spearman(exact[q], polar[q]) for q in range(len(queries))])
    )
    sp_perch = float(
        np.nanmean([_spearman(exact[q], perch[q]) for q in range(len(queries))])
    )
    families = {"polar": sp_polar, "per_channel": sp_perch}
    sp_whit: float | None = None
    if include_whitened:
        # Dedicated RNG: leaves the polar/query/unit-displacement stream above
        # unchanged, so the two-family verdict stays byte-for-byte identical.
        whit = _consumer_scores(
            _whitened_polar_proxy(batch, bits, np.random.default_rng(seed + 7)),
            queries,
            consumer,
        )
        sp_whit = float(
            np.nanmean([_spearman(exact[q], whit[q]) for q in range(len(queries))])
        )
        families["whitened"] = sp_whit
    decomp = displacement_decomposition(batch, seed=seed)

    # Direction-concentration statistic: chordal displacement of the
    # row-normalized batch (small = the DC-offset keys regime).
    unit = batch / np.maximum(np.linalg.norm(batch, axis=1, keepdims=True), 1e-30)
    pi = rng.integers(0, batch.shape[0], size=2000)
    pj = rng.integers(0, batch.shape[0], size=2000)
    keep = pi != pj
    unit_disp = float(
        np.median(np.linalg.norm(unit[pi[keep]] - unit[pj[keep]], axis=1))
    )

    # Recommend the highest-agreement family; the fixed order makes ties favour
    # the cheaper code (polar, then per_channel, then whitened) via stable sort.
    present = [f for f in ("polar", "per_channel", "whitened") if f in families]
    ranked = sorted(present, key=lambda f: families[f], reverse=True)
    rec = ranked[0]
    margin = families[ranked[0]] - families[ranked[1]] if len(ranked) > 1 else 0.0
    return A2ProbeResult(
        consumer=consumer,
        spearman_polar=sp_polar,
        spearman_per_channel=sp_perch,
        spearman_whitened=sp_whit,
        median_tangential_fraction=decomp["median_tangential_fraction"],
        median_unit_displacement=unit_disp,
        recommendation=rec,
        margin=abs(margin),
    )


def recommend_key_quantizer(
    keys: np.ndarray,
    queries: np.ndarray | None = None,
    bits: int = 4,
    seed: int = 0,
    include_whitened: bool = False,
) -> A2ProbeResult:
    """Calibration-time family recommendation for attention keys.

    Thin wrapper over :func:`probe_quotient` with
    ``consumer="attention_logits"``: keys are consumed through Q.K^T, so the
    probe measures logit-rank preservation, the quantity reconstruction
    metrics provably miss (docs/KV_KEYS_FINDING.md).

    ``keys``/``queries`` are (n, head_dim) samples (flatten heads first).
    """
    return probe_quotient(
        keys,
        consumer="attention_logits",
        queries=queries,
        bits=bits,
        seed=seed,
        include_whitened=include_whitened,
    )

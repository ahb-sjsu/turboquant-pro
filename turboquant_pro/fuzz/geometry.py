"""NumPy-only, deterministic geometry profiling for retrieval fuzzing.

The profile is fitted once to an immutable corpus.  Later fuzzing phases use
its regularized whitening transform and frozen quantiles without refitting.
"""

from __future__ import annotations

import hashlib

import numpy as np


_SCHEMA = "turboquant-pro/fuzz-geometry-profile"
_SCHEMA_VERSION = 1
_EIGENVALUE_FLOOR_RELATIVE = 1e-8
_EIGENVALUE_FLOOR_ABSOLUTE = 1e-12


def _array_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic average ranks, including tied values."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts, dtype=np.float64) - counts
    ranks = starts + (counts - 1) / 2.0
    return ranks[inverse]


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left_rank = _average_ranks(left).astype(np.float64)
    right_rank = _average_ranks(right).astype(np.float64)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = float(np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
    if denominator == 0.0:
        return None
    return float(left_rank @ right_rank / denominator)


def _quantiles(values: np.ndarray) -> list[float]:
    quantile_values = np.quantile(values, np.linspace(0.0, 1.0, 11))
    return [float(value) for value in quantile_values]


def _point_records(
    ids: np.ndarray,
    radii: np.ndarray,
    counts: np.ndarray,
) -> list[dict[str, float | int]]:
    return [
        {
            "id": int(row),
            "mahalanobis_radius": float(radii[row]),
            "reverse_knn_count": int(counts[row]),
        }
        for row in ids
    ]


def _exact_sampled_knn(
    corpus: np.ndarray,
    sample_ids: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact sampled-query search with deterministic self exclusion.

    This deliberately uses the immutable fp32 corpus directly.  The profile's
    reverse-kNN and margin signals are estimates from the sampled query rows,
    not claims about an approximate index.
    """
    queries = corpus[sample_ids]
    n = len(corpus)
    query_count = len(queries)
    distances = np.empty((query_count, k), dtype=np.float64)
    neighbours = np.empty((query_count, k), dtype=np.int64)
    corpus_norm = np.einsum("ij,ij->i", corpus, corpus)
    identifiers = np.arange(n)
    for query_row, (query, self_id) in enumerate(zip(queries, sample_ids)):
        squared = corpus_norm - 2.0 * (corpus @ query) + float(query @ query)
        squared[int(self_id)] = np.inf
        # lexsort makes equal-distance results portable rather than depending
        # on an unstable partition implementation.
        order = np.lexsort((identifiers, squared))[:k]
        neighbours[query_row] = order
        distances[query_row] = np.sqrt(np.maximum(squared[order], 0.0))
    return distances, neighbours


def profile_geometry(
    embeddings: np.ndarray,
    *,
    k: int = 10,
    sample: int | None = None,
    seed: int = 0,
) -> dict:
    """Fit a regularized Mahalanobis profile for an immutable 2-D corpus.

    Covariance estimation is float64 and uses a fixed-seed sample when the
    requested sample is smaller than the corpus.  A relative-or-absolute
    eigenvalue floor always defines the whitening transform.  ``singular``
    reports the unregularized covariance, so callers never mistake the floor
    for evidence of full rank.
    """
    corpus = np.ascontiguousarray(embeddings, dtype=np.float64)
    if corpus.ndim != 2:
        raise ValueError(f"embeddings must have shape (n, d), got {corpus.shape}")
    n, dimension = corpus.shape
    if n < 3:
        raise ValueError("embeddings must contain at least three rows")
    if dimension < 1:
        raise ValueError("embeddings must contain at least one dimension")
    if not np.isfinite(corpus).all():
        raise ValueError("embeddings must be finite")
    if not 1 <= k < n:
        raise ValueError(f"k must be in [1, {n - 1}], got {k}")
    sample_size = n if sample is None else int(sample)
    if not 2 <= sample_size <= n:
        raise ValueError(f"sample must be in [2, {n}], got {sample_size}")

    rng = np.random.default_rng(seed)
    sample_ids = (
        np.arange(n, dtype=np.int64)
        if sample_size == n
        else np.sort(rng.choice(n, size=sample_size, replace=False))
    )
    fit = corpus[sample_ids]
    mean = fit.mean(axis=0)
    covariance = np.atleast_2d(np.cov(fit, rowvar=False, ddof=1))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest_eigenvalue = float(max(float(eigenvalues[-1]), 0.0))
    floor = max(
        _EIGENVALUE_FLOOR_ABSOLUTE,
        largest_eigenvalue * _EIGENVALUE_FLOOR_RELATIVE,
    )
    unregularized_rank = int(np.linalg.matrix_rank(covariance))
    singular = unregularized_rank < dimension
    regularized_eigenvalues = np.maximum(eigenvalues, floor)
    regularized = bool(np.any(eigenvalues < floor))
    inverse_sqrt = (eigenvectors / np.sqrt(regularized_eigenvalues)) @ eigenvectors.T
    whitened = (corpus - mean) @ inverse_sqrt.T
    radii = np.linalg.norm(whitened, axis=1)

    # One extra neighbor is necessary for the k / k+1 exact boundary margin.
    search_k = min(k + 1, n - 1)
    distances, neighbours = _exact_sampled_knn(corpus, sample_ids, search_k)
    reverse_counts = np.bincount(neighbours[:, :k].ravel(), minlength=n)
    margins = (
        distances[:, k] - distances[:, k - 1]
        if search_k > k
        else np.zeros(sample_size)
    )

    central_cutoff, peripheral_cutoff = np.quantile(radii, [0.25, 0.75])
    hub_cutoff = np.quantile(reverse_counts, 0.95)
    anti_hub_cutoff = np.quantile(reverse_counts, 0.10)
    central = np.flatnonzero(radii <= central_cutoff)
    peripheral = np.flatnonzero(radii >= peripheral_cutoff)
    hubs = np.flatnonzero(reverse_counts >= hub_cutoff)
    anti_hubs = np.flatnonzero(reverse_counts <= anti_hub_cutoff)
    central_hubs = np.intersect1d(central, hubs, assume_unique=True)
    peripheral_hubs = np.intersect1d(peripheral, hubs, assume_unique=True)
    central_anti_hubs = np.intersect1d(central, anti_hubs, assume_unique=True)

    total_variance = float(eigenvalues.clip(min=0.0).sum())
    normalized_spectrum = (
        eigenvalues.clip(min=0.0) / total_variance
        if total_variance > 0.0
        else np.zeros_like(eigenvalues)
    )
    effective_rank = float(
        np.exp(-np.sum(normalized_spectrum * np.log(normalized_spectrum + 1e-300)))
    )
    order_by_hubness = np.lexsort((np.arange(n), -reverse_counts))
    from turboquant_pro import __version__

    return {
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "tool_version": __version__,
        "metric": "euclidean",
        "centrality": "regularized_mahalanobis_radius",
        "corpus": {
            "shape": [int(n), int(dimension)],
            "dtype": str(np.asarray(embeddings).dtype),
            "sha256": _array_sha256(np.asarray(embeddings)),
        },
        "sample": {
            "size": int(sample_size),
            "seed": int(seed),
            "ids": [int(row) for row in sample_ids],
        },
        "covariance": {
            "estimator": "sample_covariance_ddof1",
            "calculation_dtype": "float64",
            "singular": singular,
            "regularized": regularized,
            "regularized_dimensions": int((eigenvalues < floor).sum()),
            "eigenvalue_floor": float(floor),
            "eigenvalue_floor_relative": _EIGENVALUE_FLOOR_RELATIVE,
            "eigenvalue_floor_absolute": _EIGENVALUE_FLOOR_ABSOLUTE,
            "unregularized_rank": unregularized_rank,
            "effective_rank": effective_rank,
            "eigenvalues": [float(value) for value in eigenvalues],
            "regularized_eigenvalues": [
                float(value) for value in regularized_eigenvalues
            ],
            "explained_variance_ratio": [
                float(value) for value in normalized_spectrum
            ],
            "mean": [float(value) for value in mean],
            "whitening_matrix": inverse_sqrt.tolist(),
        },
        "mahalanobis_radius": {"quantiles": _quantiles(radii)},
        "reverse_knn": {
            "k": int(k),
            "estimator": "sampled_queries_exact",
            "query_sample_size": int(sample_size),
            "counts_quantiles": _quantiles(reverse_counts),
            "centrality_hubness_spearman": _spearman(-radii, reverse_counts),
            "top_hubs": _point_records(
                order_by_hubness[:10], radii, reverse_counts
            ),
        },
        "exact_neighbor_margin": {
            "definition": "distance_at_k_plus_1_minus_distance_at_k",
            "quantiles": _quantiles(margins),
        },
        "strata": {
            "central_hubs": _point_records(central_hubs, radii, reverse_counts),
            "peripheral_hubs": _point_records(
                peripheral_hubs, radii, reverse_counts
            ),
            "central_anti_hubs": _point_records(
                central_anti_hubs, radii, reverse_counts
            ),
        },
    }

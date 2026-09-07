"""Pure query mutators in a frozen regularized whitened space.

The fuzzer only mutates queries.  A geometry profile supplies the immutable
mean and whitening matrix that define the coordinate system, so these helpers
never fit data or modify the corpus or index.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MutationRecord:
    """Stable, JSON-ready description of a pure query mutation."""

    name: str
    parameters: tuple[tuple[str, float], ...]

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "parameters": dict(self.parameters)}


@dataclass(frozen=True)
class MutationResult:
    """Mutated queries and the exact operation needed to reproduce them."""

    queries: np.ndarray
    record: MutationRecord


def _profile_coordinates(
    queries: np.ndarray,
    profile: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate a profile and return queries, mean, and its whitening matrix."""
    values = np.ascontiguousarray(queries, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("queries must have shape (n, d) with at least one row")
    if not np.isfinite(values).all():
        raise ValueError("queries must be finite")
    try:
        covariance = profile["covariance"]
        if not isinstance(covariance, Mapping):
            raise TypeError
        mean = np.asarray(covariance["mean"], dtype=np.float64)
        whitening = np.asarray(covariance["whitening_matrix"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "profile lacks a valid covariance whitening transform"
        ) from error
    dimension = values.shape[1]
    if mean.shape != (dimension,) or whitening.shape != (dimension, dimension):
        raise ValueError("profile whitening transform does not match query dimension")
    if not np.isfinite(mean).all() or not np.isfinite(whitening).all():
        raise ValueError("profile whitening transform must be finite")
    if np.linalg.matrix_rank(whitening) != dimension:
        raise ValueError("profile whitening transform must be invertible")
    return values, mean, whitening


def whiten_queries(queries: np.ndarray, profile: Mapping[str, object]) -> np.ndarray:
    """Map queries into the profile's frozen regularized whitened space."""
    values, mean, whitening = _profile_coordinates(queries, profile)
    return (values - mean) @ whitening.T


def radial_mutation(
    queries: np.ndarray,
    profile: Mapping[str, object],
    *,
    alpha: float,
) -> MutationResult:
    """Scale every whitened query by a positive radial factor.

    Positive scaling preserves each nonzero whitened direction exactly while
    multiplying its regularized Mahalanobis radius by ``alpha``.
    """
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("radial alpha must be finite and greater than zero")
    values, mean, whitening = _profile_coordinates(queries, profile)
    whitened = (values - mean) @ whitening.T
    mutated = np.linalg.solve(whitening, (alpha * whitened).T).T + mean
    return MutationResult(
        queries=mutated,
        record=MutationRecord("radial", (("alpha", float(alpha)),)),
    )


def _perpendicular_unit(vector: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a deterministic random unit vector perpendicular to ``vector``."""
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    unit = vector / norm
    candidate = rng.standard_normal(vector.shape[0])
    candidate -= unit * float(candidate @ unit)
    candidate_norm = float(np.linalg.norm(candidate))
    if candidate_norm == 0.0:
        # The probability of an exact projection is negligible, but this fallback
        # makes the transform total and deterministic on every platform.
        axis = np.zeros_like(vector)
        axis[int(np.argmin(np.abs(unit)))] = 1.0
        candidate = axis - unit * float(axis @ unit)
        candidate_norm = float(np.linalg.norm(candidate))
    return candidate / candidate_norm


def shell_mutation(
    queries: np.ndarray,
    profile: Mapping[str, object],
    *,
    angle: float,
    rng: np.random.Generator,
) -> MutationResult:
    """Rotate queries in whitened space while preserving each whitened norm.

    Zero-radius queries are unchanged because their angular direction is
    undefined.  A supplied Generator makes the operation deterministic once a
    campaign has derived its stream from the campaign seed.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("shell rng must be a numpy.random.Generator")
    if not np.isfinite(angle):
        raise ValueError("shell angle must be finite")
    values, mean, whitening = _profile_coordinates(queries, profile)
    if values.shape[1] < 2:
        raise ValueError("shell mutation requires at least two dimensions")
    whitened = (values - mean) @ whitening.T
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    rotated = np.empty_like(whitened)
    for row, vector in enumerate(whitened):
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            rotated[row] = vector
            continue
        rotated[row] = norm * (
            cosine * (vector / norm) + sine * _perpendicular_unit(vector, rng)
        )
    mutated = np.linalg.solve(whitening, rotated.T).T + mean
    return MutationResult(
        queries=mutated,
        record=MutationRecord("shell", (("angle", float(angle)),)),
    )

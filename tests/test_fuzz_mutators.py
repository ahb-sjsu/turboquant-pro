"""Mathematical invariants for frozen-space retrieval-fuzzing mutators."""

from __future__ import annotations

import numpy as np

from turboquant_pro.fuzz.geometry import profile_geometry
from turboquant_pro.fuzz.mutators import (
    radial_mutation,
    shell_mutation,
    whiten_queries,
)


def _profile_and_queries() -> tuple[dict, np.ndarray]:
    rng = np.random.default_rng(9)
    corpus = rng.normal(size=(20, 3)).astype(np.float32)
    queries = corpus[:4] + np.array([0.1, -0.2, 0.3])
    return profile_geometry(corpus, k=3, seed=2), queries


def test_radial_mutation_scales_whitened_radius_and_preserves_direction():
    profile, queries = _profile_and_queries()

    result = radial_mutation(queries, profile, alpha=1.75)
    before = whiten_queries(queries, profile)
    after = whiten_queries(result.queries, profile)

    np.testing.assert_allclose(after, before * 1.75, rtol=1e-12, atol=1e-12)
    assert result.record.as_dict() == {
        "name": "radial",
        "parameters": {"alpha": 1.75},
    }


def test_shell_mutation_preserves_whitened_norm_and_is_seed_deterministic():
    profile, queries = _profile_and_queries()

    first = shell_mutation(
        queries, profile, angle=np.pi / 3, rng=np.random.default_rng(47)
    )
    second = shell_mutation(
        queries, profile, angle=np.pi / 3, rng=np.random.default_rng(47)
    )
    before = whiten_queries(queries, profile)
    after = whiten_queries(first.queries, profile)

    np.testing.assert_allclose(
        np.linalg.norm(after, axis=1),
        np.linalg.norm(before, axis=1),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(first.queries, second.queries, rtol=0.0, atol=0.0)
    assert first.record.as_dict() == {
        "name": "shell",
        "parameters": {"angle": float(np.pi / 3)},
    }

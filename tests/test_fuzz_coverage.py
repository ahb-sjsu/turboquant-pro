"""Frozen coverage retention contracts for retrieval fuzzing."""

from __future__ import annotations

import pytest

from turboquant_pro.fuzz.coverage import (
    CoverageCase,
    FrozenQuantileCoverage,
    retain_cases,
)


@pytest.fixture
def coverage():
    return FrozenQuantileCoverage(
        {
            "mahalanobis_centrality": [0.0, 1.0, 2.0],
            "reverse_knn_hubness": [0.0, 3.0, 6.0],
            "exact_neighbor_margin": [0.0, 0.5, 1.0],
            "consumer_error": [0.0, 0.25, 1.0],
        }
    )


def test_frozen_coverage_clamps_and_assigns_boundaries_to_higher_cells(coverage):
    assert coverage.signature(
        {
            "mahalanobis_centrality": -1.0,
            "reverse_knn_hubness": 3.0,
            "exact_neighbor_margin": 1.0,
            "consumer_error": 2.0,
        }
    ) == (0, 1, 1, 1)


def test_coverage_loads_only_the_profile_frozen_schema():
    profile = {
        "coverage": {
            "schema": "turboquant-pro/fuzz-frozen-coverage",
            "signals": {
                "mahalanobis_centrality": [0.0, 1.0],
                "reverse_knn_hubness": [0.0, 1.0],
                "exact_neighbor_margin": [0.0, 1.0],
                "consumer_error": [0.0, 1.0],
            },
        }
    }

    assert FrozenQuantileCoverage.from_geometry(profile).signature(
        {
            "mahalanobis_centrality": 0.5,
            "reverse_knn_hubness": 0.5,
            "exact_neighbor_margin": 0.5,
            "consumer_error": 0.5,
        }
    ) == (0, 0, 0, 0)


def test_coverage_retention_is_order_independent_and_keeps_new_classes(coverage):
    shared = {
        "mahalanobis_centrality": 1.2,
        "reverse_knn_hubness": 4.0,
        "exact_neighbor_margin": 0.2,
        "consumer_error": 0.5,
    }
    cases = [
        CoverageCase("weaker", shared, 0.2, "consumer_regression"),
        CoverageCase("stronger", shared, 0.8, "consumer_regression"),
        CoverageCase("certificate", shared, 0.1, "certified_violation"),
        CoverageCase(
            "novel",
            {**shared, "mahalanobis_centrality": 0.1},
            0.1,
            "stress_discovery",
        ),
    ]

    forward = retain_cases(coverage, cases)
    backward = retain_cases(coverage, list(reversed(cases)))

    assert [case.case_id for case in forward] == [case.case_id for case in backward]
    assert {case.case_id for case in forward} == {
        "stronger",
        "certificate",
        "novel",
    }


def test_coverage_rejects_unknown_classifications(coverage):
    with pytest.raises(ValueError, match="unsupported classification"):
        retain_cases(
            coverage,
            [
                CoverageCase(
                    "bad",
                    {
                        "mahalanobis_centrality": 1.0,
                        "reverse_knn_hubness": 1.0,
                        "exact_neighbor_margin": 1.0,
                        "consumer_error": 1.0,
                    },
                    0.0,
                    "unexpected",
                )
            ],
        )

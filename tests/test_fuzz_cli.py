"""Public CLI coverage for the first geometry-aware fuzzing workflow."""

from __future__ import annotations

import json

import numpy as np

from turboquant_pro.cli import main


def test_geometry_profile_cli_reports_regularized_singular_geometry(
    capsys, tmp_path
):
    """A rank-deficient corpus is profiled deterministically through ``tqp``."""
    embeddings = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 2.0], [2.0, 4.0, 4.0], [3.0, 6.0, 6.0]],
        dtype=np.float32,
    )
    source = tmp_path / "corpus.npy"
    output = tmp_path / "geometry.json"
    np.save(source, embeddings)

    rc = main(
        [
            "geometry",
            "profile",
            "--embeddings",
            str(source),
            "--k",
            "2",
            "--sample",
            "4",
            "--centrality",
            "mahalanobis",
            "--seed",
            "7",
            "--out",
            str(output),
            "--format",
            "json",
        ]
    )

    assert rc == 0
    doc = json.loads(output.read_text(encoding="utf-8"))
    assert doc["schema"] == "turboquant-pro/fuzz-geometry-profile"
    assert doc["schema_version"] == 1
    assert doc["covariance"]["singular"] is True
    assert doc["covariance"]["regularized"] is True
    assert doc["sample"]["seed"] == 7
    assert doc["reverse_knn"]["estimator"] == "sampled_queries_exact"
    assert len(doc["strata"]["central_hubs"]) >= 1
    assert "geometry profile" in capsys.readouterr().out

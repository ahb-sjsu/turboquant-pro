"""Public CLI coverage for the first geometry-aware fuzzing workflow."""

from __future__ import annotations

import json

import numpy as np

from turboquant_pro.cli import main
from turboquant_pro.fuzz import load_replay_bundle, profile_geometry
from turboquant_pro.index import TQEIndex


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


def test_fuzz_retrieval_cli_mutates_queries_and_writes_replayable_cases(
    capsys, tmp_path
):
    """The public command uses fresh truth and archives immutable index state."""
    corpus = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]],
        dtype=np.float32,
    )
    queries = np.array([[0.8, 0.1], [0.1, 0.8]], dtype=np.float32)
    index_path = tmp_path / "corpus.tqe"
    queries_path = tmp_path / "queries.npy"
    geometry_path = tmp_path / "geometry.json"
    output = tmp_path / "fuzz-run"
    TQEIndex.create(corpus, bits=2, seed=3, metric="l2").save(index_path)
    np.save(queries_path, queries)
    geometry_path.write_text(
        json.dumps(profile_geometry(corpus, k=1, sample=len(corpus), seed=9)),
        encoding="utf-8",
    )

    rc = main(
        [
            "fuzz",
            "retrieval",
            "--index",
            str(index_path),
            "--queries",
            str(queries_path),
            "--geometry",
            str(geometry_path),
            "--mutators",
            "radial,shell",
            "--budget",
            "2",
            "--seed",
            "11",
            "--k",
            "1",
            "--out",
            str(output),
            "--format",
            "json",
        ]
    )

    assert rc == 0
    campaign = json.loads((output / "campaign.json").read_text(encoding="utf-8"))
    assert campaign["evaluated"] == 2
    assert campaign["truth_policy"] == "fresh_exact_recomputed_for_every_mutated_query"
    assert campaign["retained_cases"]
    bundle = load_replay_bundle(output / "cases" / campaign["retained_cases"][0])
    assert set(bundle["arrays"]) == {"corpus", "index_bytes", "queries"}
    assert "fuzz-retrieval-campaign" in capsys.readouterr().out

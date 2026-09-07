"""Deterministic query-only retrieval campaign execution.

The MVP never changes the corpus or index.  It records a complete snapshot of
the single-file TQE index with each retained case so a later replay command can
evaluate precisely the same compressed retrieval path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import write_replay_bundle
from .coverage import CoverageCase, FrozenQuantileCoverage, retain_cases
from .mutators import radial_mutation, shell_mutation, whiten_queries
from .oracles import exact_top_k, exact_vs_tqp

CAMPAIGN_SCHEMA = "turboquant-pro/fuzz-retrieval-campaign"
CAMPAIGN_SCHEMA_VERSION = 1


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_geometry(path: str | Path) -> dict[str, Any]:
    try:
        profile = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read geometry profile: {error}") from error
    if (
        not isinstance(profile, dict)
        or profile.get("schema") != "turboquant-pro/fuzz-geometry-profile"
        or profile.get("schema_version") != 1
    ):
        raise ValueError("geometry profile has an unsupported schema")
    return profile


def _live_corpus(index: object) -> tuple[np.ndarray, np.ndarray]:
    """Extract immutable original vectors and external IDs from a TQE index."""
    originals = getattr(index, "_originals", None)
    ids = getattr(index, "_ids", None)
    tombstones = getattr(index, "_tomb", None)
    if originals is None:
        raise ValueError("fuzz retrieval needs an index created with stored originals")
    corpus = np.asarray(originals)
    all_ids = np.asarray(ids, dtype=np.int64)
    tomb = np.asarray(tombstones, dtype=np.uint8)
    live = tomb == 0
    if corpus.ndim != 2 or all_ids.shape != (len(corpus),) or not np.any(live):
        raise ValueError("index has invalid or empty stored originals")
    return np.ascontiguousarray(corpus[live]), np.ascontiguousarray(all_ids[live])


def _candidate_signals(
    profile: dict[str, Any],
    corpus: np.ndarray,
    queries: np.ndarray,
    corpus_ids: np.ndarray,
    result: dict[str, object],
    k: int,
    metric: str,
) -> dict[str, float]:
    whitened = whiten_queries(queries, profile)
    exact_ids, exact_scores = exact_top_k(
        corpus,
        queries,
        k=min(k + 1, len(corpus)),
        metric=metric,
        corpus_ids=corpus_ids,
    )
    del exact_ids
    if len(corpus) > k:
        margin = float(np.mean(exact_scores[:, k - 1] - exact_scores[:, k]))
    else:
        margin = 0.0
    # The sampled reverse-kNN estimator is a corpus signal.  Associate a query
    # with its exact nearest live row and use that row's recorded occurrence
    # count.  Old profiles without per-row counts remain usable with a neutral
    # value, preserving their frozen quantile semantics rather than refitting.
    counts = profile.get("reverse_knn", {}).get("counts")
    by_id = {
        int(row["id"]): int(row["reverse_knn_count"])
        for row in profile.get("reverse_knn", {}).get("top_hubs", [])
    }
    if isinstance(counts, list) and len(counts) == profile["corpus"]["shape"][0]:
        by_id = {row: int(value) for row, value in enumerate(counts)}
    nearest = result["exact_top_k"]
    hubness = float(np.mean([by_id.get(int(row[0]), 0) for row in nearest]))
    return {
        "mahalanobis_centrality": float(np.mean(np.linalg.norm(whitened, axis=1))),
        "reverse_knn_hubness": hubness,
        "exact_neighbor_margin": margin,
        "consumer_error": 1.0 - float(result["metrics"]["recall_at_k"]),
    }


def run_retrieval_campaign(
    *,
    index_path: str | Path,
    queries: np.ndarray,
    geometry_path: str | Path,
    output: str | Path,
    budget: int,
    seed: int,
    k: int,
    mutators: tuple[str, ...],
    rerank: int = 0,
    truth_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a bounded, deterministic query-only fuzz campaign.

    ``truth_path`` is provenance-only.  It is deliberately never loaded because
    every mutation needs freshly recomputed exact truth.
    """
    from turboquant_pro import __version__
    from turboquant_pro.index import TQEIndex

    source = Path(index_path)
    target = Path(output)
    if not source.is_file():
        raise ValueError(
            "fuzz retrieval currently supports one regular TQE index file"
        )
    if target.exists():
        raise ValueError(f"fuzz output already exists: {target}")
    if budget < 1:
        raise ValueError("budget must be at least one")
    if not mutators or set(mutators).difference({"radial", "shell"}):
        raise ValueError("MVP mutators must be radial and/or shell")
    profile = _load_geometry(geometry_path)
    index = TQEIndex.open(str(source))
    corpus, corpus_ids = _live_corpus(index)
    if profile["corpus"].get("sha256") != _array_sha256(corpus):
        raise ValueError("geometry profile corpus hash does not match index originals")
    q = np.ascontiguousarray(queries, dtype=np.float32)
    if (
        q.ndim != 2
        or q.shape[1] != corpus.shape[1]
        or not len(q)
        or not np.isfinite(q).all()
    ):
        raise ValueError(
            "queries must be a non-empty finite (n, d) array matching index"
        )
    if not 1 <= k <= len(corpus):
        raise ValueError(f"k must be in [1, {len(corpus)}]")
    coverage = FrozenQuantileCoverage.from_geometry(profile)
    metric = str(index.stats()["metric"])
    candidates: list[
        tuple[CoverageCase, dict[str, Any], np.ndarray, dict[str, Any]]
    ] = []
    for number in range(budget):
        name = mutators[number % len(mutators)]
        rng = np.random.default_rng(np.random.SeedSequence([seed, number]))
        if name == "radial":
            mutation = radial_mutation(
                q, profile, alpha=float(rng.uniform(0.5, 1.5))
            )
        else:
            mutation = shell_mutation(
                q,
                profile,
                angle=float(rng.uniform(-np.pi, np.pi)),
                rng=rng,
            )
        result = exact_vs_tqp(
            corpus,
            mutation.queries,
            index,
            k=k,
            metric=metric,
            corpus_ids=corpus_ids,
            search_kwargs={"rerank": rerank},
        )
        signals = _candidate_signals(
            profile, corpus, mutation.queries, corpus_ids, result, k, metric
        )
        case_id = f"case-{number:06d}"
        coverage_case = CoverageCase(
            case_id=case_id,
            signals=signals,
            severity=float(
                signals["consumer_error"] + result["metrics"]["rank_disagreement"]
            ),
            classification=str(result["classification"]),
        )
        candidates.append(
            (
                coverage_case,
                result,
                mutation.queries,
                {
                    "name": mutation.record.name,
                    "parameters": dict(mutation.record.parameters),
                },
            )
        )
    winners = {
        case.case_id
        for case in retain_cases(coverage, [item[0] for item in candidates])
    }
    target.mkdir()
    cases_dir = target / "cases"
    cases_dir.mkdir()
    source_bytes = np.frombuffer(source.read_bytes(), dtype=np.uint8).copy()
    retained: list[str] = []
    for coverage_case, result, mutated_queries, mutation_record in candidates:
        if coverage_case.case_id not in winners:
            continue
        retained.append(coverage_case.case_id)
        case = {
            "schema": "turboquant-pro/fuzz-retrieval-case",
            "schema_version": 1,
            "case_id": coverage_case.case_id,
            "seed": int(seed),
            "mutator": mutation_record,
            "coverage_signature": list(coverage.signature(coverage_case.signals)),
            "coverage_signals": dict(coverage_case.signals),
            "severity": coverage_case.severity,
            "classification": coverage_case.classification,
            "oracle": result,
            "index": {
                "sha256": _file_sha256(source),
                "metric": metric,
                "rerank": rerank,
            },
            "tool_version": __version__,
            "tolerances": {"atol": 1e-6, "rtol": 1e-6},
        }
        write_replay_bundle(
            cases_dir / coverage_case.case_id,
            case=case,
            geometry=profile,
            arrays={
                "queries": mutated_queries,
                "corpus": corpus,
                "index_bytes": source_bytes,
            },
            documents={
                "expected_exact.json": {
                    "top_k": result["exact_top_k"],
                    "scores": result["exact_scores"],
                },
                "observed_tqp.json": {
                    "top_k": result["observed_tqp_top_k"],
                    "scores": result["observed_tqp_scores"],
                },
            },
        )
    doc = {
        "schema": CAMPAIGN_SCHEMA,
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "tool_version": __version__,
        "seed": int(seed),
        "budget": int(budget),
        "evaluated": len(candidates),
        "retained_cases": retained,
        "index": {"sha256": _file_sha256(source), "metric": metric},
        "truth_path": str(truth_path) if truth_path is not None else None,
        "truth_policy": "fresh_exact_recomputed_for_every_mutated_query",
    }
    (target / "campaign.json").write_text(
        json.dumps(doc, allow_nan=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return doc

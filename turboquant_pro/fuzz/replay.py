"""Fail-closed replay for deterministic query-only retrieval findings."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .artifacts import ReplayBundleError, canonical_json_bytes, load_replay_bundle
from .oracles import exact_vs_tqp

REPLAY_SCHEMA = "turboquant-pro/fuzz-retrieval-replay"
REPLAY_SCHEMA_VERSION = 1
_CASE_SCHEMA = "turboquant-pro/fuzz-retrieval-case"
_GEOMETRY_SCHEMA = "turboquant-pro/fuzz-geometry-profile"


class ReplayMismatchError(ReplayBundleError):
    """A valid bundle whose current retrieval result differs from its record."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _live_corpus(index: object) -> tuple[np.ndarray, np.ndarray]:
    originals = getattr(index, "_originals", None)
    ids = getattr(index, "_ids", None)
    tombstones = getattr(index, "_tomb", None)
    if originals is None:
        raise ReplayBundleError("replay index does not retain original vectors")
    corpus = np.asarray(originals)
    all_ids = np.asarray(ids, dtype=np.int64)
    tomb = np.asarray(tombstones, dtype=np.uint8)
    live = tomb == 0
    if corpus.ndim != 2 or all_ids.shape != (len(corpus),) or not np.any(live):
        raise ReplayBundleError("replay index has invalid or empty stored originals")
    return np.ascontiguousarray(corpus[live]), np.ascontiguousarray(all_ids[live])


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReplayBundleError(f"replay bundle {name} must be an object")
    return value


def _tolerances(case: Mapping[str, Any]) -> tuple[float, float]:
    value = _require_mapping(case.get("tolerances"), "tolerances")
    try:
        atol = float(value["atol"])
        rtol = float(value["rtol"])
    except (KeyError, TypeError, ValueError) as error:
        raise ReplayBundleError("replay bundle has invalid tolerances") from error
    if not np.isfinite([atol, rtol]).all() or atol < 0.0 or rtol < 0.0:
        raise ReplayBundleError("replay bundle has invalid tolerances")
    return atol, rtol


def _same_value(
    recorded: object,
    actual: object,
    *,
    atol: float,
    rtol: float,
    path: str,
) -> None:
    if isinstance(recorded, Mapping):
        if not isinstance(actual, Mapping) or set(recorded) != set(actual):
            raise ReplayMismatchError(f"recorded fields differ at {path}")
        for key in sorted(recorded):
            _same_value(
                recorded[key],
                actual[key],
                atol=atol,
                rtol=rtol,
                path=f"{path}.{key}",
            )
        return
    if isinstance(recorded, list):
        if not isinstance(actual, list) or len(recorded) != len(actual):
            raise ReplayMismatchError(f"recorded sequence differs at {path}")
        for number, (expected, observed) in enumerate(zip(recorded, actual)):
            _same_value(
                expected,
                observed,
                atol=atol,
                rtol=rtol,
                path=f"{path}[{number}]",
            )
        return
    if isinstance(recorded, float):
        if not isinstance(actual, (float, int)) or not np.isclose(
            recorded, actual, atol=atol, rtol=rtol
        ):
            raise ReplayMismatchError(f"recorded numeric value differs at {path}")
        return
    if recorded != actual:
        raise ReplayMismatchError(f"recorded value differs at {path}")


def replay_retrieval_bundle(source: str | Path) -> dict[str, Any]:
    """Validate and reproduce one retained retrieval case without refitting.

    All content and semantic checks happen before the temporary index is opened.
    The geometry profile is evidence only during replay: queries were already
    mutated in its frozen whitened space and are never re-mutated or refitted.
    """
    from turboquant_pro import __version__
    from turboquant_pro.index import TQEIndex

    bundle = load_replay_bundle(source)
    documents = _require_mapping(bundle["documents"], "documents")
    arrays = _require_mapping(bundle["arrays"], "arrays")
    case = _require_mapping(documents.get("case.json"), "case.json")
    geometry = _require_mapping(documents.get("geometry.json"), "geometry.json")
    expected = _require_mapping(
        documents.get("expected_exact.json"), "expected_exact.json"
    )
    observed = _require_mapping(
        documents.get("observed_tqp.json"), "observed_tqp.json"
    )
    if case.get("schema") != _CASE_SCHEMA or case.get("schema_version") != 1:
        raise ReplayBundleError("unsupported retrieval case schema")
    if (
        geometry.get("schema") != _GEOMETRY_SCHEMA
        or geometry.get("schema_version") != 1
    ):
        raise ReplayBundleError("unsupported replay geometry schema")
    if case.get("tool_version") != __version__:
        raise ReplayBundleError("replay bundle was created by an incompatible version")
    if geometry.get("tool_version") != case.get("tool_version"):
        raise ReplayBundleError("replay geometry version does not match the case")
    if set(arrays) != {"corpus", "index_bytes", "queries"}:
        raise ReplayBundleError("replay bundle has unexpected array payloads")
    corpus = np.ascontiguousarray(np.asarray(arrays["corpus"], dtype=np.float32))
    queries = np.ascontiguousarray(np.asarray(arrays["queries"], dtype=np.float32))
    index_array = np.asarray(arrays["index_bytes"])
    if index_array.dtype != np.dtype(np.uint8) or index_array.ndim != 1:
        raise ReplayBundleError(
            "replay index bytes must be a one-dimensional uint8 array"
        )
    index_bytes = index_array.tobytes()
    if (
        corpus.ndim != 2
        or queries.ndim != 2
        or not len(corpus)
        or not len(queries)
        or corpus.shape[1] != queries.shape[1]
    ):
        raise ReplayBundleError("replay arrays have incompatible shapes")
    geometry_corpus = _require_mapping(
        geometry.get("corpus"), "geometry corpus"
    )
    if geometry_corpus.get("sha256") != hashlib.sha256(
        corpus.view(np.uint8)
    ).hexdigest():
        raise ReplayBundleError("replay corpus does not match its geometry profile")
    geometry_metadata = _require_mapping(case.get("geometry"), "case geometry")
    if geometry_metadata.get("sha256") != _sha256(canonical_json_bytes(geometry)):
        raise ReplayBundleError("replay geometry does not match the case record")
    index_metadata = _require_mapping(case.get("index"), "case index")
    codec = _require_mapping(index_metadata.get("codec"), "case codec")
    if index_metadata.get("sha256") != _sha256(index_bytes):
        raise ReplayBundleError("replay index bytes do not match the case record")
    oracle = _require_mapping(case.get("oracle"), "case oracle")
    if expected != {
        "top_k": oracle.get("exact_top_k"),
        "scores": oracle.get("exact_scores"),
    } or observed != {
        "top_k": oracle.get("observed_tqp_top_k"),
        "scores": oracle.get("observed_tqp_scores"),
    }:
        raise ReplayBundleError("replay evidence does not match the case oracle")
    atol, rtol = _tolerances(case)
    try:
        k = int(oracle["k"])
        metric = str(oracle["metric"])
        rerank = int(index_metadata["rerank"])
    except (KeyError, TypeError, ValueError) as error:
        raise ReplayBundleError(
            "replay case has invalid retrieval parameters"
        ) from error
    if k < 1 or rerank < 0 or metric not in {"cosine", "l2"}:
        raise ReplayBundleError("replay case has invalid retrieval parameters")

    with tempfile.TemporaryDirectory(prefix="tqp-fuzz-replay-") as directory:
        index_path = Path(directory) / "index.tqe"
        index_path.write_bytes(index_bytes)
        try:
            index = TQEIndex.open(str(index_path))
            live_corpus, corpus_ids = _live_corpus(index)
            if not np.array_equal(live_corpus, corpus):
                raise ReplayBundleError("replay index originals do not match corpus")
            if index.stats()["metric"] != metric:
                raise ReplayBundleError("replay index metric does not match case")
            _same_value(
                codec,
                index.stats(),
                atol=0.0,
                rtol=0.0,
                path="codec",
            )
            result = exact_vs_tqp(
                corpus,
                queries,
                index,
                k=k,
                metric=metric,
                corpus_ids=corpus_ids,
                search_kwargs={"rerank": rerank},
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise ReplayBundleError(
                f"could not evaluate replay bundle: {error}"
            ) from error
    _same_value(oracle, result, atol=atol, rtol=rtol, path="oracle")
    return {
        "schema": REPLAY_SCHEMA,
        "schema_version": REPLAY_SCHEMA_VERSION,
        "case_id": case.get("case_id"),
        "status": "reproduced",
        "classification": result["classification"],
        "metrics": result["metrics"],
        "tolerances": {"atol": atol, "rtol": rtol},
        "tool_version": __version__,
        "bundle_sha256": _sha256(canonical_json_bytes(bundle["manifest"])),
    }

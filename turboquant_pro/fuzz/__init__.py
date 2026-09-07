"""Deterministic geometry-aware retrieval fuzzing primitives."""

from .artifacts import (
    ReplayBundleError,
    canonical_json_bytes,
    load_replay_bundle,
    write_replay_bundle,
)
from .coverage import CoverageCase, FrozenQuantileCoverage, retain_cases
from .geometry import profile_geometry
from .mutators import (
    MutationRecord,
    MutationResult,
    radial_mutation,
    shell_mutation,
    whiten_queries,
)
from .oracles import CLASSIFICATIONS, exact_top_k, exact_vs_tqp

__all__ = [
    "MutationRecord",
    "MutationResult",
    "CLASSIFICATIONS",
    "CoverageCase",
    "FrozenQuantileCoverage",
    "ReplayBundleError",
    "canonical_json_bytes",
    "exact_top_k",
    "exact_vs_tqp",
    "load_replay_bundle",
    "profile_geometry",
    "radial_mutation",
    "retain_cases",
    "shell_mutation",
    "whiten_queries",
    "write_replay_bundle",
]

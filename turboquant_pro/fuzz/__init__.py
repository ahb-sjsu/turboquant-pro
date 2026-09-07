"""Deterministic geometry-aware retrieval fuzzing primitives."""

from .artifacts import (
    ReplayBundleError,
    canonical_json_bytes,
    load_replay_bundle,
    write_replay_bundle,
)
from .geometry import profile_geometry
from .mutators import (
    MutationRecord,
    MutationResult,
    radial_mutation,
    shell_mutation,
    whiten_queries,
)

__all__ = [
    "MutationRecord",
    "MutationResult",
    "ReplayBundleError",
    "canonical_json_bytes",
    "load_replay_bundle",
    "profile_geometry",
    "radial_mutation",
    "shell_mutation",
    "whiten_queries",
    "write_replay_bundle",
]

"""Deterministic geometry-aware retrieval fuzzing primitives."""

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
    "profile_geometry",
    "radial_mutation",
    "shell_mutation",
    "whiten_queries",
]

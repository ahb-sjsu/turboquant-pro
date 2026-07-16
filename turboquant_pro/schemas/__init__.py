# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Shipped JSON Schemas for TurboQuant Pro artifacts.

The schemas are versioned data files, packaged so a consumer can validate a
`tqp`-emitted artifact without depending on this project's internals::

    from importlib.resources import files
    schema = files("turboquant_pro.schemas") / "rank_certificate.schema.json"

See ``docs/CERTIFICATE_SPEC.md`` for the certificate compatibility promise.
"""

from __future__ import annotations

from importlib.resources import files

__all__ = ["schema_path", "load_schema"]


def schema_path(name: str):
    """Return a traversable path to a shipped schema file (e.g.
    ``"rank_certificate.schema.json"``)."""
    return files(__name__) / name


def load_schema(name: str) -> dict:
    """Load and parse a shipped JSON Schema by file name."""
    import json

    return json.loads(schema_path(name).read_text(encoding="utf-8"))

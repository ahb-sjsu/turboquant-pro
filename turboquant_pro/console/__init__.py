"""The TurboQuant Pro console (docs/DESIGN_console.md, Phase 1): a local, read-only
instrument panel over the telemetry contract. Start it with ``tqp console``."""

from __future__ import annotations

from .server import ConsoleServer, Workload, demo_index

__all__ = ["ConsoleServer", "Workload", "demo_index"]

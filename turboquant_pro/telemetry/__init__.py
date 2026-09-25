"""Telemetry contract for the console (docs/DESIGN_console.md, Phase 0).

Three parts, each with a JSON Schema in ``turboquant_pro/schemas``:

- :mod:`.metrics`: every metric the console can show is registered with its
  unit, aggregation, window, source and kind (measured, estimated, sampled,
  derived), and every reading carries them. A panel cannot show an unregistered
  metric.
- :mod:`.trace`: query traces with stage timings, the scan path actually taken
  and, for the first query of a batch, approximate vs exact scores and rank
  movement. Off by default; when off, an instrumented search pays one truth
  test per stage.
- :mod:`.api`: version and capability negotiation for clients.
"""

from __future__ import annotations

from .api import API_VERSION, capabilities
from .metrics import REGISTRY, MetricSpec, reading
from .trace import Scope, Tracer, active, begin, disable, enable, scope

__all__ = [
    "API_VERSION",
    "REGISTRY",
    "MetricSpec",
    "Scope",
    "Tracer",
    "active",
    "begin",
    "capabilities",
    "disable",
    "enable",
    "reading",
    "scope",
]

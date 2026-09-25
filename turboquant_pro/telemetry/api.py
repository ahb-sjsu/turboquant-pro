"""Version and capability negotiation (requirements API-008)."""

from __future__ import annotations

API_VERSION = "0.1"

SCHEMAS = {
    "turboquant-pro/query-trace": 1,
    "turboquant-pro/metric-reading": 1,
}


def capabilities() -> dict:
    """What this runtime can report. A client hides a panel whose capability is absent
    rather than showing an empty or zero-filled one."""
    try:
        import psutil  # noqa: F401

        resources = True
    except ImportError:
        resources = False
    from turboquant_pro import __version__
    from turboquant_pro.telemetry.metrics import REGISTRY

    return {
        "api_version": API_VERSION,
        "tool_version": __version__,
        "schemas": dict(SCHEMAS),
        "metrics": sorted(REGISTRY),
        "features": {
            "query_trace": True,
            "stage_timing": True,
            "approx_vs_exact": True,
            "process_resources": resources,
            "gpu_utilization": False,
            "operator_actions": False,
        },
    }

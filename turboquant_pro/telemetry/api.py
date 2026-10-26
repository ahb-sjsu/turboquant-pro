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
    from turboquant_pro.schemas import KINDS
    from turboquant_pro.telemetry.metrics import REGISTRY

    return {
        "api_version": API_VERSION,
        "tool_version": __version__,
        "schemas": dict(SCHEMAS),
        # every artifact kind the package writes, and whether a JSON Schema ships
        # for it (the console validates only those; the rest are "no schema")
        "artifact_kinds": {
            k.id: {"title": k.title, "schema_shipped": k.schema_file is not None}
            for k in KINDS
        },
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

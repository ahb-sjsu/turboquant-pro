# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Shipped JSON Schemas for TurboQuant Pro artifacts, and the registry of every
artifact kind the package writes.

The schemas are versioned data files, packaged so a consumer can validate a
`tqp`-emitted artifact without depending on this project's internals::

    from importlib.resources import files
    schema = files("turboquant_pro.schemas") / "rank_certificate.schema.json"

See ``docs/CERTIFICATE_SPEC.md`` for the certificate compatibility promise.

**The registry.** :data:`REGISTRY` maps each artifact kind's id (the value of
its ``schema`` field) to what it is, what writes it and its JSON Schema, when
one ships. Most kinds have no schema yet, and the registry says so rather than
implying one: :func:`validate` answers ``no schema shipped``, never ``valid``,
for them. A few outputs carry no ``schema`` field at all; :func:`identify`
recognises them by the fields they always have and reports that it did.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

__all__ = [
    "ArtifactKind",
    "KINDS",
    "REGISTRY",
    "identify",
    "load_schema",
    "schema_path",
    "validate",
]


def schema_path(name: str):
    """Return a traversable path to a shipped schema file (e.g.
    ``"rank_certificate.schema.json"``)."""
    return files(__name__) / name


def load_schema(name: str) -> dict:
    """Load and parse a shipped JSON Schema by file name."""
    import json

    return json.loads(schema_path(name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArtifactKind:
    """One kind of document the package writes."""

    id: str
    """The document's ``schema`` value (or, for a kind without one, a name
    this registry gives it, and :attr:`fields` to recognise it by)."""
    title: str
    emitted_by: str
    schema_file: str | None = None
    """Shipped JSON Schema file name, or None: no schema ships for this kind."""
    fields: tuple[str, ...] = ()
    """For a kind whose documents carry no ``schema`` field: fields they always
    have. Empty for kinds identified by their ``schema`` field."""


_K = ArtifactKind
KINDS: tuple[ArtifactKind, ...] = (
    # certificates and their checks
    _K("turboquant-pro/rank-certificate", "rank certificate", "tqp certify",
       "rank_certificate.schema.json"),
    _K("turboquant-pro/index-certificate", "index rank certificate",
       "tqp index certify"),
    _K("turboquant-pro/verification", "certificate verification", "tqp verify"),
    _K("turboquant-pro/composition-certificate", "composed certificate",
       "tqp compose"),
    _K("turboquant-pro/capability-report", "capability report", "tqp capabilities"),
    _K("turboquant-pro/feasibility-report", "feasibility report", "tqp feasibility"),
    # plans
    _K("turboquant-pro/embedding-plan", "embedding plan", "tqp plan embeddings",
       "embedding_plan.schema.json"),
    _K("turboquant-pro/kv-plan", "KV plan", "tqp plan kv", "kv_plan.schema.json"),
    _K("turboquant-pro/compression-plan", "compression plan", "tqp plan run",
       "compression_plan.schema.json"),
    _K("turboquant-pro/compression-plan-replay", "plan replay", "tqp plan replay"),
    _K("turboquant-pro/refinement-report", "refinement report", "tqp plan refine"),
    _K("turboquant-pro/compatibility-matrix", "observer compatibility matrix",
       "tqp plan compat"),
    _K("tqp.weight_cost_table/1", "weight cost table",
       "turboquant_pro.weight_plan", "weight_cost_table.schema.json"),
    _K("tqp.weight_plan/1", "weight plan", "tqp plan weights",
       "weight_plan.schema.json"),
    _K("tqp.weight_encoding/1", "weight encoding manifest",
       "tqp plan encode-weights"),
    _K("tqp.packed_weights/1", "packed weights (TQPW binary; meta section)",
       "tqp plan encode-weights"),
    # observers and workload
    _K("turboquant-pro/observer-contract", "observer contract",
       "tqp observer init", "observer_contract.schema.json"),
    _K("turboquant-pro/workload-summary", "workload summary", "tqp observer learn"),
    _K("turboquant-pro/adaptive-policy", "adaptive rerank policy",
       "turboquant_pro.adaptive_rerank.calibrate", "adaptive_policy.schema.json"),
    # indexes and queries
    _K("turboquant-pro/index-shards", "sharded index manifest",
       "turboquant_pro.sharded_index"),
    _K("turboquant-pro/index-search", "index search results", "tqp index search"),
    _K("turboquant-pro/index-drift", "PCA basis drift report", "tqp index drift"),
    _K("turboquant-pro/index-info", "index container summary", "tqp index info"),
    _K("turboquant-pro/query-catalog", "workload statistics catalog",
       "tqp query ANALYZE"),
    _K("turboquant-pro/query-plan", "query plan", "tqp query EXPLAIN"),
    _K("turboquant-pro/query-result", "query result", "tqp query SELECT"),
    # geometry
    _K("turboquant-pro/hub-anatomy", "hub anatomy", "tqp anatomy"),
    _K("turboquant-pro/hub-differential", "hub differential", "tqp hubdiff"),
    _K("tqp-strata-report/1", "stratified report",
       "tqp anatomy --strata, tqp hubdiff --strata"),
    _K("tqp-area-map/1", "area map", "tqp anatomy --save-map",
       fields=("profile", "digest", "labels")),
    # claims and the console
    _K("turboquant-pro/replay-report", "claim replay report", "tqp replay"),
    _K("turboquant-pro/query-trace", "query trace", "turboquant_pro.telemetry",
       "query_trace.schema.json"),
    _K("turboquant-pro/metric-reading", "metric reading", "turboquant_pro.telemetry",
       "metric_reading.schema.json",
       fields=("name", "unit", "aggregation", "window_s", "kind", "value")),
    _K("turboquant-pro/console-setup", "console setup", "tqp console (S)",
       "console_setup.schema.json"),
    _K("turboquant-pro/console-export", "console session export", "tqp console (e)"),
    _K("turboquant-pro/machine-snapshot", "machine snapshot (/proc, /sys, NVML)",
       "tqp console --machine"),
    _K("turboquant-pro/dht-snapshot", "BitTorrent DHT snapshot (from tqp-dht)",
       "tqp console --dht"),
    # the NATS fabric
    _K("turboquant-pro/fabric-snapshot", "NATS fabric snapshot", "tqp fabric",
       "fabric_snapshot.schema.json"),
    # KV connector state
    _K("tqp-kv-identity/1", "KV identity profile", "turboquant_pro.connectors"),
    _K("tqp-kv-store-state/2", "KV block store state", "turboquant_pro.connectors"),
    # outputs without a schema field
    _K("turboquant-pro/quality-monitor", "quality monitor metrics",
       "tqp monitor --format json",
       fields=("turboquant_quality_mean_cosine", "turboquant_quality_is_healthy")),
    _K("turboquant-pro/a2-probe", "consumer-metric probe", "tqp probe --json",
       fields=("consumer", "spearman_polar", "spearman_per_channel",
               "recommendation")),
)  # fmt: skip
del _K

REGISTRY: dict[str, ArtifactKind] = {k.id: k for k in KINDS}
if len(REGISTRY) != len(KINDS):  # pragma: no cover - a registry edit error
    raise RuntimeError("duplicate artifact kind id in schemas.KINDS")


def identify(doc) -> tuple[ArtifactKind | None, str | None]:
    """``(kind, how)``: the registered kind of ``doc`` and how it was known,
    ``"schema"`` (its ``schema`` field) or ``"fields"`` (the shape of a kind
    that carries no ``schema`` field); ``(None, None)`` when unrecognised. An
    unregistered ``schema`` value is unrecognised, never matched by shape."""
    if not isinstance(doc, dict):
        return None, None
    sid = doc.get("schema")
    if isinstance(sid, str):
        k = REGISTRY.get(sid)
        return (k, "schema") if k else (None, None)
    for k in KINDS:
        if k.fields and all(f in doc for f in k.fields):
            return k, "fields"
    return None, None


VALID, INVALID = "valid", "invalid"
NO_SCHEMA, UNRECOGNIZED = "no schema shipped", "unrecognized"
NOT_VALIDATED = "not validated (jsonschema is not installed)"


def validate(doc) -> dict:
    """Identify ``doc`` and validate it against its shipped schema.

    ``status`` is one of ``valid``, ``invalid`` (with ``errors``, each a path
    and a message), ``no schema shipped`` (the kind is known, but nothing
    checks its structure), ``unrecognized``, or ``not validated`` when the
    optional ``jsonschema`` package is missing. Only ``valid`` means valid."""
    kind, how = identify(doc)
    out = {
        "kind": kind.id if kind else None,
        "identified_by": how,
        "schema_file": kind.schema_file if kind else None,
        "status": UNRECOGNIZED,
        "errors": [],
    }
    if kind is None:
        return out
    if kind.schema_file is None:
        out["status"] = NO_SCHEMA
        return out
    try:
        import jsonschema
    except ImportError:
        out["status"] = NOT_VALIDATED
        return out
    schema = load_schema(kind.schema_file)
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path))
    out["errors"] = [
        {"path": "/" + "/".join(str(p) for p in e.absolute_path), "message": e.message}
        for e in errors
    ]
    out["status"] = INVALID if errors else VALID
    return out

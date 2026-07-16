# Rank-certificate artifact specification

`tqp certify` emits a **rank certificate**: a small, provenance-stamped JSON
document that records what was measured, on what inputs, and whether a
distribution-free rank-fidelity floor could be certified. This page is the
compatibility promise for that artifact so downstream tooling can depend on it.

- **Schema:** [`turboquant_pro/schemas/rank_certificate.schema.json`](../turboquant_pro/schemas/rank_certificate.schema.json)
  (JSON Schema, Draft 2020-12), shipped with the package.
- **Current version:** `schema_version = 1`.
- **Golden fixture:** [`tests/fixtures/rank_certificate_golden.json`](../tests/fixtures/rank_certificate_golden.json),
  regenerated and compared in `tests/test_certificate_schema.py`.

## Example

```json
{
  "schema": "turboquant-pro/rank-certificate",
  "schema_version": 1,
  "tool_version": "1.8.0.dev0",
  "created_utc": "2026-07-16T18:04:11.686000+00:00",
  "inputs": {
    "original":      { "path": "emb.npy",   "shape": [1000, 768], "dtype": "float32", "sha256": "…" },
    "reconstructed": { "path": "emb_q.npy", "shape": [1000, 768], "dtype": "float32", "sha256": "…" }
  },
  "params": { "metric": "cosine", "n_anchors": 200, "seed": 0 },
  "certificate": {
    "kappa": 1.0148,
    "mu_hat": 0.0664,
    "tau_floor": 0.8671,
    "spearman_floor": 0.8006,
    "n_pairs": 19900,
    "max_certifiable_kappa": 1.83,
    "vacuous": false
  },
  "interpretation": "certifies Kendall tau >= 0.8671, Spearman rho >= 0.8006 (distribution-free)",
  "passed": true
}
```

## Field reference

| Field | Meaning |
|---|---|
| `schema` | Constant format identifier `"turboquant-pro/rank-certificate"`. |
| `schema_version` | Integer format version (see promise below). |
| `tool_version` | turboquant-pro version that produced the certificate. |
| `created_utc` | ISO-8601 UTC timestamp. |
| `inputs.{original,reconstructed}` | `path`, `shape`, `dtype`, and content `sha256` of each input — a certificate is bound to the exact bytes it certifies. |
| `params` | `metric` (`cosine`/`l2`), `n_anchors`, `seed` — enough to reproduce the measurement. |
| `certificate.kappa` | Robust multiplicative distortion of pairwise distances. |
| `certificate.mu_hat` | Corpus distance-ratio concentration at `kappa`. |
| `certificate.tau_floor` | Guaranteed Kendall τ floor, `1 − 2·mu_hat`. |
| `certificate.spearman_floor` | Guaranteed Spearman ρ floor, `1 − 3·mu_hat`. |
| `certificate.n_pairs` | Number of anchor pairs measured. |
| `certificate.max_certifiable_kappa` | Largest distortion that still certifies a positive floor. |
| `certificate.vacuous` | `true` when no finite distortion certifies rank — **exact reranking is required**. |
| `interpretation` | Human-readable one-line verdict. |
| `passed` | Gate outcome (positive floor, or `tau_floor >= --min-tau` when given). |

## The `null` convention

The certificate floats — `kappa`, `mu_hat`, `tau_floor`, `spearman_floor`,
`max_certifiable_kappa` — are **non-finite on a degenerate corpus** (e.g. a NaN
distortion when every pairwise distance is zero). That is a real, meaningful
outcome, so it is serialized as JSON **`null`**, and the certificate is marked
`vacuous: true`. Bare `NaN` / `Infinity` (invalid JSON) are **never** emitted —
every `tqp` JSON artifact is spec-valid and parses under a strict reader.

## Compatibility promise

Within a major version:

1. **`schema_version` bumps only for a breaking change** — removing or renaming a
   field, or changing a field's type or meaning.
2. **Additive changes do not bump it.** New optional fields may appear; consumers
   must ignore unknown fields. (The shipped schema is strict —
   `additionalProperties: false` — but validate with the schema matching the
   `schema_version` you read.)
3. **`schema` and the `schema_version` constant are stable** for the life of the
   format identifier.
4. **Provenance is guaranteed:** every certificate carries input hashes, tool
   version, params, and a UTC timestamp, so it is reproducible and auditable.
5. **`null` means non-finite**, per the convention above — not "missing".

## Validating a certificate

```python
import json
from importlib.resources import files
import jsonschema  # pip install turboquant-pro[dev]  (or jsonschema)

schema = json.loads(
    (files("turboquant_pro.schemas") / "rank_certificate.schema.json").read_text()
)
cert = json.load(open("certificate.json"))
jsonschema.validate(cert, schema)  # raises on drift
```

# Design — observer contracts

Issue #173, the first of the Observation Theory family (#183). Phase 1 shipped
on `feat/observer-contracts`; the later phases are listed at the end.

## 0. Thesis

A consumer-relative distortion is not interpretable on its own. Two defensible
read operators for one attention head differ by about 0.3 in subspace
overlap, which is why a certificate records the provider and the hash of the
operator it was computed against (`docs/CERTIFICATE_SPEC.md`, `reference`).
The planner already declares one consumer per workload; the strata layer
already names its area map by content hash. What was missing was the observer
itself as an object: one artifact that says who reads the data, how, under
what population, with what guarantees, that any plan, certificate or monitor
report can name exactly.

Observation Theory's statement is that meaning and distortion are properties
of the observation, not of the object. The contract is that statement as a
file.

## 1. The artifact

Profile `tqp-observer/1`, schema `turboquant-pro/observer-contract`, shipped
as `turboquant_pro/schemas/observer_contract.schema.json`. YAML or JSON; the
file extension `.tqo` is a convention, not a format.

```yaml
schema: turboquant-pro/observer-contract
profile: tqp-observer/1
observer: retrieval-prod-v3
target: embedding
source:
  embedding_model: text-embedding-3-large
  dim: 1536
consumers:
  - name: retrieval
    metric: topk_cosine          # a registered consumer metric
    config: {k: 10}
    weight: 0.85
  - name: reader
    metric: read_operator
    config: {provider: attention_analytic}   # a registered read operator
    weight: 0.15
population:
  strata: language-region-map.tqa
  strata_sha256: ...
  calibration_sha256: ...
requirements:
  floor: {metric: recall@10, minimum: 0.995, confidence: 0.95}
  worst_stratum_minimum: 0.98
budget:
  max_bytes_per_vector: 128
fallback:
  action: exact_rerank
```

Rules, each with its reason.

- **Consumers are registry names.** A `metric` is a consumer registered in
  `turboquant_pro.consumers`; a `read_operator` consumer's `config.provider` is
  a provider registered in `turboquant_pro.read_operators`. The contract binds
  names that already carry their own evidence and defines no new metric. A
  contract that names an unknown consumer is refused, for the same reason the
  planner abstains on one: guessing what a reader means is how a certificate
  ends up answering a different question.
- **Content-addressed.** `digest()` is the sha256 of the canonical JSON form
  (sorted keys, tight separators, ASCII, no NaN), the convention the area map
  uses. Key order, whitespace and YAML-versus-JSON do not change it; any
  declared value does. A plan record or certificate carries
  `{profile, observer, sha256, target, consumers, primary_consumer}`, enough
  to find the contract and to refuse a mismatch.
- **Provenance is free-form, requirements are not.** `source` and
  `population` take any keys, because what was embedded and how the strata
  were made vary by deployment. `requirements`, `budget` and `fallback` have
  the keys the planner reads and nothing else, so a typo cannot become a
  silently ignored guarantee.
- **Validation never weakens by host.** With `jsonschema` installed the
  shipped schema is the check; without it the same rules run in plain Python.
  The tests pin the two to agree.

## 2. What reads it

| command | reads | writes |
|---|---|---|
| `tqp observer validate C` | structure, then the registries | exit 0 valid, 1 invalid with each problem named, 2 unreadable |
| `tqp observer show C` | | the contract for a person, or `--format json` with its hash |
| `tqp observer hash C` | | the digest |
| `tqp observer init --name N --out C` | | a one-consumer retrieval contract to edit |
| `tqp plan run --observer C` | target, primary consumer, floor, budget | the plan record with an `observer` section |
| `tqp certify --observer C` | | the certificate with an `observer` section (additive, `schema_version` stays 1) |
| `tqp verify CERT --observer C` | the certificate's `observer.sha256` | `checks.observer` with `match`; verification fails when the certificate names no observer or a different one |

`ObserverContract.to_workload_spec()` hands the planner the **primary**
consumer, the largest weight, first on a tie. The planner scores one consumer
today; the record says which one it scored and lists the others by name. The
weighted mixture is Phase 2.

## 3. What Phase 1 does not do

- No weighted objective across consumers (the planner's search is single
  metric; the mixture needs the compatibility matrix, #179).
- No refinement layers (#174) and no expiry (#177). The contract carries the
  fields those will read, `fallback` and `population`, so the artifact does
  not change shape when they arrive.
- No learning from traces (#180). `tqp observer init` writes a template; the
  contract is authored.
- `worst_stratum_minimum` is recorded and validated, not enforced: enforcement
  needs the STRATA gates to read a contract, which is the monitor's part.

## 4. Phases

1. **The artifact and the three commands that name it.** Shipped: `observer.py`,
   the schema, `tqp observer`, `--observer` on `plan run`, `certify`, `verify`,
   the `observer` section in both artifact schemas. Tests in
   `tests/test_observer.py`.
2. **The monitor.** `tqp monitor --observer C` and `QualityMonitor` report
   against the contract's floor and strata, and `tqp verify --observer` gains
   the coverage checks that #177 turns into expiry.
   *Status 2026-11-26:* `QualityMonitor(certificate=, contract=)` and the coverage checks
   shipped with #177. `tqp monitor --observer/--certificate` now exposes them. The contract's
   floor is reported but **not enforced** by the monitor, which measures cosine, not the floor's
   metric (recall@10 and the like); enforcing it needs a consumer-metric stream. Not done:
   enforcing `worst_stratum_minimum` (Phase 3, with the STRATA gates).
3. **The mixture.** The planner scores every consumer in the contract and
   combines by weight; `worst_stratum_minimum` enforced through the STRATA
   gates.
4. **Learned contracts** (#180) and **capabilities** (#178), once #174 and #177
   exist to name what a representation can do.

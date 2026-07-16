# TurboQuant Pro: Phased Roadmap to the Next Level

Prepared: 2026-07-16

## Executive summary

TurboQuant Pro already has many of the ingredients of a serious operator-aware quantization and compression platform: rank certificates, the A2 probe, operator tracing, auto-compression, auto-config, production monitoring, plugin conformance, a TQE format spec, and claim/reproduction discipline.

The next step is not to add many more quantizers. The next step is to make the existing components compose into one visible workflow:

```text
trace -> plan -> compress -> certify -> replay -> monitor
```

The north-star product should be:

> An operator-aware quantization and compression certification layer: trace the consumer operator, choose the safe quotient, certify preserved geometry, replay the claim, and monitor drift in production.

This aligns the theory and practice: the theory says useful compression comes from preserving the geometry that matters after quotienting nuisance coordinates; the software should make that task geometry observable, testable, and certifiable.

## Scope and assumptions

This roadmap assumes the current known hygiene issues have been addressed or are being addressed first:

- release-state clarity between PyPI release and master/unreleased features;
- stale documentation counts and roadmap items;
- torch optional dependency or install-message mismatch;
- plugin conformance design-vs-implementation mismatch;
- `torch_decode` wording around host-side reconstruction versus true device-native operation;
- gate-routing wording clarified as margin/order sensitivity rather than raw magnitude alone.

The roadmap also assumes that the central validated product track remains embedding/vector-DB compression and compressed-domain retrieval, while KV-cache and operator-aware inference features are treated as powerful but more hardware/model-dependent.

## Roadmap at a glance

| Phase | Theme | Timeline | Primary outcome |
|---|---:|---:|---|
| Phase 0 | Stabilize the release surface | 1 week | External reviewers can tell what is stable, beta, experimental, released, and master-only. |
| Phase 1 | Unify instruments into one CLI | 2-3 weeks | A `tqp` command exposes trace, probe, plan, certify, replay, monitor, and plugin workflows. |
| Phase 2 | Certificate schema | 3-4 weeks | Every important artifact can emit a machine-readable `certificate.json`. |
| Phase 3 | Claim replay | 3-4 weeks | `CLAIMS.md` becomes executable through `claims.yaml` and `tqp replay`. |
| Phase 4 | Productize the planner | 4-6 weeks | Existing auto-compression and auto-config become a task-aware recipe compiler. |
| Phase 5 | Prove the plugin ecosystem | 4-6 weeks | At least one out-of-tree plugin passes conformance and participates in certification. |
| Phase 6 | Production vector-index lifecycle | 6-8 weeks | TQE becomes usable for real corpus update, migration, compaction, and drift workflows. |
| Phase 7 | Real-model operator validation | 6-10 weeks | Operator-aware claims are validated on real attention, MoE, and SSM/recurrent models. |
| Phase 8 | Runtime safe fallback | 6-8 weeks | The runtime can escalate precision or reranking when operator margins are fragile. |
| Phase 9 | Documentation and paper packaging | Parallel, then 2 weeks | The project is legible as a coherent certification system. |

## Existing ingredients to consolidate

These components appear to be the core building blocks that should be unified rather than replaced:

| Ingredient | Role in the future product |
|---|---|
| `rank_certificate` | Provides distribution-free rank-fidelity guarantees or principled vacuity signals. |
| `a2_probe` | Tests whether a proposed quotient preserves the task-relevant tangential component. |
| `operator_trace` | Maps tensors to consumer operators and quantization disciplines. |
| `auto_compress` | Searches embedding compression recipes against quality/compression targets. |
| `AutoConfig` | Selects KV-cache settings from model architecture, preset, and hardware assumptions. |
| `QualityMonitor` | Tracks reconstruction quality, distribution drift, and A2/tangential-fraction streams. |
| plugin protocol and conformance kit | Makes external quantization recipes testable and certifiable. |
| TQE format spec | Provides the durable artifact boundary for compressed embeddings and related formats. |
| `CLAIMS.md` and reproduction docs | Provide the evidence ladder that should become executable claim replay. |
| operator-sensitivity notes | Extend the task-geometry principle beyond attention to gates and recurrence. |

## Phase 0: Stabilize the release surface

**Timeline:** 1 week

**Goal:** Make the repository state unambiguous to a first-time reviewer, PyPI user, and potential contributor.

### Deliverables

| Item | Concrete action | Acceptance check |
|---|---|---|
| Release-state banner | Add a README table separating latest PyPI release, master branch, and unreleased/dev features. | A user can tell in 10 seconds what is stable versus master-only. |
| API tier audit | Ensure all public components are marked Stable, Beta, Experimental, or Internal. | README, API docs, and changelog agree. |
| Test-count de-drift | Replace pinned prose counts with generated or command-derived counts. | No stale test-count mismatch remains. |
| Release checklist | Add a release checklist covering tag, GitHub release, PyPI upload, changelog, docs, and claim table. | Every future release follows the same checklist. |
| Issue labels | Add labels such as `P0-release`, `P1-certification`, `P2-planner`, `P3-hardware`, and `research-validation`. | Roadmap items can be mapped directly to GitHub issues. |

### Exit criterion

A first-time reviewer can distinguish:

- safe to depend on;
- beta but usable;
- research preview;
- internal or implementation detail;
- released versus master-only.

## Phase 1: Unify existing instruments into one CLI

**Timeline:** 2-3 weeks

**Goal:** Surface the power that already exists. The library has more capability than the product surface suggests. Add a top-level command, ideally `tqp`, while keeping the existing script as a compatibility alias.

### Proposed CLI

```bash
tqp trace ...
tqp probe ...
tqp plan ...
tqp certify ...
tqp replay ...
tqp monitor ...
tqp plugin ...
```

### Proposed subcommands

| Command | Wraps existing capability | Output |
|---|---|---|
| `tqp trace model` | `operator_trace` | Tensor-to-operator discipline table. |
| `tqp probe a2` | `a2_probe` | Quantizer-family safety recommendation. |
| `tqp plan embeddings` | `auto_compress` plus rank certificate preview | Pareto frontier and recommended recipe. |
| `tqp plan kv` | `AutoConfig` plus operator trace | KV key/value policy and risk flags. |
| `tqp certify embeddings` | `rank_certificate`, reconstruction metrics, optional recall harness | `certificate.json`. |
| `tqp certify kv` | operator trace, behavioral agreement, task metrics | KV safety report. |
| `tqp replay claim-id` | claim metadata plus benchmark scripts/notebooks | reproducibility report. |
| `tqp plugin list` | plugin registry | installed plugins and capability table. |
| `tqp plugin conformance` | conformance kit | pass/skip/fail report. |
| `tqp monitor serve` | `QualityMonitor` | JSON/Prometheus-compatible metrics. |

### Example workflow

```bash
tqp plan embeddings \
  --data embeddings.npy \
  --target-recall 0.995 \
  --max-bytes-per-vector 64 \
  --hardware cpu-avx2 \
  --out plan.json

# Build/compress using the chosen plan.
tqp certify embeddings \
  --original embeddings.npy \
  --compressed index.tqe \
  --queries eval_queries.npy \
  --out certificate.json
```

### Exit criterion

A user can get a plan and a certificate without writing Python.

## Phase 2: Create a unified certificate schema

**Timeline:** 3-4 weeks

**Goal:** Turn measurement into a durable certification artifact.

### Design principle

A certificate should not claim more than it measured. It should say what artifact was tested, what task geometry was declared, what metrics were measured, what pass/fail thresholds were used, and what risks remain.

### Example `certificate.json`

```json
{
  "schema": "tqp.certificate.v1",
  "artifact": {
    "kind": "embedding_index",
    "path": "index.tqe",
    "sha256": "...",
    "format": "TQE",
    "format_version": 2
  },
  "environment": {
    "turboquant_pro": "1.8.0.dev0",
    "python": "3.12",
    "hardware": "...",
    "git_commit": "..."
  },
  "task": {
    "kind": "retrieval",
    "metric": "cosine",
    "target": "recall@10 >= 0.995"
  },
  "geometry": {
    "rank_certificate": {
      "kappa": 2.13,
      "mu_hat": 0.04,
      "kendall_tau_floor": 0.92,
      "spearman_floor": 0.88,
      "rerank_required": false
    },
    "a2": {
      "median_tangential_fraction": 0.97,
      "recommended_family": "polar",
      "margin": 0.31
    }
  },
  "decision": {
    "status": "pass",
    "warnings": []
  }
}
```

### Required certificate sections

| Section | Purpose |
|---|---|
| `schema` | Version the certificate format. |
| `artifact` | Identify and hash the compressed artifact. |
| `environment` | Record package, Python, dependency, hardware, and git state. |
| `task` | Declare the downstream consumer and target metric. |
| `operator_trace` | For model/tensor workflows, record the inferred consumer operator. |
| `geometry` | Store rank certificate, A2 probe, margin, or behavioral-agreement results. |
| `metrics` | Store recall, QPS, build time, perplexity, LongBench, or task-specific metrics. |
| `decision` | Emit `pass`, `warn`, `fail`, or `inconclusive`. |
| `limitations` | Record scope, missing data, stochasticity, or hardware dependence. |

### Acceptance checks

| Check | Requirement |
|---|---|
| JSON schema | `certificate.schema.json` validates all generated certificates. |
| Golden files | Tiny deterministic fixtures have committed expected certificates. |
| Hashing | Artifact hash, sample/data hash, code version, and config hash are included. |
| CLI support | `tqp certify ... --out certificate.json` works for embeddings first. |
| Human report | Optional `--html report.html` renders a readable report. |
| Machine status | CI can parse pass/warn/fail/inconclusive. |

### Exit criterion

Every important benchmark and claim can emit a durable certificate artifact.

## Phase 3: Make claim replay executable

**Timeline:** 3-4 weeks

**Goal:** Convert the claims/evidence discipline into a reproducibility product.

### Add `claims.yaml`

Example:

```yaml
claims:
  embedding_27x_high_recall:
    track: embedding
    status: reproducible
    dataset: glove-100-angular
    command: python benchmarks/canonical_embedding.py --small
    full_command: python benchmarks/canonical_embedding.py --full
    hardware: cpu
    expected:
      recall_at_10_min: 0.99
      compression_min: 27
    outputs:
      - results.json
      - certificate.json
```

### CLI

```bash
tqp replay embedding_27x_high_recall --small
tqp replay kv_asym_nf4_qwen --gpu
tqp replay all --track embedding
```

### Deliverables

| Item | Concrete action |
|---|---|
| `claims.yaml` | One entry per claim row. |
| Replay runner | Executes scripts or notebooks through a shared harness. |
| Expected ranges | Use ranges rather than exact values for stochastic metrics and timing. |
| Result normalization | Every run writes `results.json`; eligible runs also write `certificate.json`. |
| CI split | Small CPU replay in CI; full replay manual/nightly; GPU replay opt-in. |
| Failure mode docs | A failed replay explains whether it is environment, data, metric, or code drift. |

### Exit criterion

An external reviewer can run one command and reproduce the central Track 1 claim on public data.

## Phase 4: Productize the planner

**Timeline:** 4-6 weeks

**Goal:** Turn auto-compression and auto-config into a task-aware recipe compiler.

The planner should not only pick the most compressed configuration. It should choose a recipe that preserves the downstream task geometry under declared constraints.

### Embedding planner example

```bash
tqp plan embeddings \
  --embeddings embeddings.npy \
  --target-recall 0.995 \
  --max-bytes-per-vector 64 \
  --hardware cpu-avx2 \
  --out plan.json
```

### KV planner example

```bash
tqp plan kv \
  --model Qwen/Qwen2.5-7B \
  --context 32768 \
  --target balanced \
  --hardware l40s \
  --out kv_plan.json
```

### Planner output

| Section | Contents |
|---|---|
| Recommended recipe | PCA dimension, bit width, rotation, norm policy, ADC/rerank, KV key/value policy. |
| Alternatives | Pareto frontier, not just one result. |
| Certificate preview | Expected rank-cert behavior or reason rerank is required. |
| Risk flags | OOD anisotropy, high concentration, small margins, unknown operators, stale basis. |
| Runtime fallback | When to rerank, preserve fp16, escalate bits, or disable a quotient. |
| Reproduction command | Exact `tqp certify` command to validate the chosen plan. |

### Exit criterion

A non-expert can get a safe first recipe without knowing PCA-Matryoshka, A2, KV key DC offsets, GQA, rank certificates, or operator regimes.

## Phase 5: Prove the plugin ecosystem

**Timeline:** 4-6 weeks

**Goal:** Demonstrate that the plugin protocol works outside the main repo.

### Recommended first external plugin

Choose one:

1. `tqp-bnb`: bitsandbytes NF4/FP4 wrapper.
2. `tqp-reference-plugin`: tiny pure-NumPy package used only to prove packaging and conformance.
3. `tqp-faiss-opq`: FAISS OPQ/PQ adapter for embedding comparisons.

### Deliverables

| Item | Acceptance check |
|---|---|
| External plugin package | Installable independently from the main repo. |
| Entry-point registration | `tqp plugin list` discovers it. |
| Conformance CI | External repo runs `tqp plugin conformance`. |
| Certification integration | Plugin can be used by `tqp certify`. |
| Capability table | Plugin declares target tensors, affine support, hardware assumptions, and maturity. |
| Tutorial | A user can write a toy plugin in 20 minutes. |

### Exit criterion

At least one plugin outside the main tree passes conformance and participates in certification.

## Phase 6: Harden the production vector-index lifecycle

**Timeline:** 6-8 weeks

**Goal:** Make Track 1 production-grade rather than benchmark-only.

### Proposed commands

```bash
tqp index create embeddings.npy --out index.tqe
tqp index add index.tqe new_embeddings.npy
tqp index delete index.tqe ids.txt
tqp index compact index.tqe --out compacted.tqe
tqp index migrate old.tqe --to-version 2 --out new.tqe
tqp index certify index.tqe --queries queries.npy
```

### Features

| Feature | Why it matters |
|---|---|
| Append | Real corpora grow. |
| Delete/tombstone | Real corpora change. |
| Compact | Storage remains honest after deletes. |
| Migrate | TQE versioning is tested under real upgrades. |
| Memmap/shard | Large indexes become practical. |
| Drift detection | Stale PCA basis or distribution shift is caught. |
| Corruption/fuzz tests | The format becomes trustworthy. |
| Exact rerank compatibility | High-recall retrieval remains safe under compression. |

### Exit criterion

Track 1 supports a realistic RAG corpus lifecycle:

```text
ingest -> search -> update -> compact -> migrate -> certify -> monitor
```

## Phase 7: Real-model validation for operator sensitivity

**Timeline:** 6-10 weeks

**Goal:** Turn synthetic/operator probes into model-family evidence.

### Experiment matrix

| Regime | Model target | Metrics |
|---|---|---|
| Attention keys | Qwen, Llama, Mistral | perplexity, LongBench, attention top-k overlap, behavioral agreement. |
| MoE routing | Mixtral or Qwen-MoE | routing flip rate, low-margin token sensitivity, downstream task score. |
| SSM/recurrence | Mamba-family or RetNet-like model | state drift, perplexity/task score, long-context degradation. |
| Weight PTQ | Qwen, Gemma, Mistral | behavioral agreement, flip rate over noise floor, task score. |
| RAG retrieval | public embedding corpus | recall, QPS, build time, rerank cost, answer-level quality. |

### Deliverables

| Item | Acceptance check |
|---|---|
| Model cards | One `docs/model_cards/*.md` per model family. |
| Operator traces | Saved trace output for each model. |
| Certificates | `certificate.json` for each key run. |
| Claim rows | Validated claims promoted into `claims.yaml`. |
| Negative cases | Failures are preserved and explained. |

### Exit criterion

Operator-aware quantization is supported by at least one real attention validation, one real MoE validation, and one real SSM/recurrent validation.

## Phase 8: Runtime safe fallback

**Timeline:** 6-8 weeks

**Goal:** Make the system robust under uncertain or fragile inputs.

Static quantization is brittle. TurboQuant Pro can become more serious by knowing when to back off.

### Example policy API

```python
policy = TQPRuntimePolicy(
    attention_margin_floor=0.02,
    retrieval_gap_floor=0.01,
    radial_drift_floor=0.15,
    unknown_operator="per_channel_or_fp16",
)
```

### Fallback examples

| Situation | Action |
|---|---|
| Retrieval top-k score gap is small | Rerank more candidates. |
| Rank certificate is vacuous | Require exact rerank or refuse certification. |
| KV-key operator is unknown | Use per-channel plus zero-point, or fp16. |
| MoE routing margin is tiny | Keep router/gate higher precision. |
| SSM decay channel is slow | Use log-time-constant basis or fp16. |
| A2 tangential fraction drifts down | Recalibrate or disable the angular/polar quotient. |
| Encoder distribution drifts | Refit PCA basis or migrate index. |

### Exit criterion

Compression becomes adaptive: cheap where margins are large, conservative where the operator is fragile.

## Phase 9: Documentation and paper packaging

**Timeline:** parallel work, then 2 weeks of polish

**Goal:** Make the work readable as a serious project rather than a collection of clever modules.

### Canonical documents

1. **User guide:** Compress embeddings safely in 15 minutes.
2. **Operator-aware quantization guide:** Why keys, values, gates, decays, and weights need different quotients.
3. **Certification guide:** What a TurboQuant Pro certificate means and what it does not mean.
4. **Plugin guide:** How to write, test, and certify a plugin.
5. **Claim replay guide:** How to reproduce the headline numbers.
6. **Production lifecycle guide:** How to maintain a mutable compressed vector index.

### Architecture diagram

```text
artifact/model
  -> operator trace
  -> task geometry schema
  -> planner
  -> compressor/plugin
  -> certificate
  -> claim replay
  -> production monitor
```

### Exit criterion

A reviewer can understand the whole project from:

```text
README -> quickstart -> certificate example -> claim replay
```

## Release milestone sequence

| Release | Theme | Headline |
|---|---|---|
| 1.8.0 | Cohesion release | `tqp` CLI, certificate schema, claim replay skeleton, plugin protocol released. |
| 1.9.0 | Planner release | Task-aware planner for embeddings and KV; certificates attached to plans. |
| 2.0.0 | Production Track 1 | Mutable TQE index, migration, drift detection, central embedding claims replayable. |
| 2.1.0 | Operator validation | Real attention, MoE, and SSM/recurrent model cards and certificates. |
| 2.2.0 | Ecosystem release | First external plugin, hardware matrix, stable plugin API. |
| 2.3.0 | Runtime safety | Safe fallback policies and monitoring integrations. |

## Suggested immediate issue tickets

These are concrete GitHub issues that could be opened directly.

### P0 release-surface issues

1. Add README release-state table for PyPI versus master.
2. Replace stale test-count prose with generated command references.
3. Audit and update API stability labels.
4. Add release provenance checklist.
5. Mark all master-only 1.8 features clearly.

### P1 certification issues

1. Draft `certificate.schema.json`.
2. Implement `tqp certify embeddings` with rank-certificate output.
3. Add golden certificate fixtures.
4. Add artifact/config/environment hashing.
5. Add human-readable certificate rendering.

### P1 CLI issues

1. Add `tqp` console script.
2. Add `tqp trace model` wrapper.
3. Add `tqp probe a2` wrapper.
4. Add `tqp plan embeddings` wrapper.
5. Add `tqp plugin list` and `tqp plugin conformance` wrappers.

### P2 claim replay issues

1. Create `claims.yaml` from `CLAIMS.md`.
2. Implement replay runner for one central embedding claim.
3. Normalize benchmark outputs into `results.json`.
4. Add CI path for small CPU replay.
5. Document hardware-specific replay classes.

### P2 planner issues

1. Define `plan.json` schema.
2. Integrate rank-certificate preview into embedding planner.
3. Integrate operator trace into KV planner.
4. Add risk-flag output.
5. Add emitted `tqp certify` command to every plan.

### P3 production-index issues

1. Add `tqp index create`.
2. Add append/delete/tombstone support.
3. Add compact and migrate commands.
4. Add memmap/shard design doc.
5. Add encoder-drift detection tied to index certification.

### P3 operator-validation issues

1. Add real attention-key validation model card.
2. Add real MoE routing validation model card.
3. Add real SSM/recurrent validation model card.
4. Promote passing validations into `claims.yaml`.
5. Preserve and document negative cases.

## Task-geometry schema proposal

A small schema can unify operator tracing, planning, certification, and monitoring.

```yaml
artifact: kv_cache.layer_12.keys
consumer_operator: attention_scores
preserved_quantity:
  type: bilinear_score
  expression: Q @ K.T
fragility:
  metric: attention_topk_margin
  threshold: 0.02
allowed_quantizers:
  - per_channel_affine
  - nf4_with_outliers
forbidden_quantizers:
  - per_vector_polar
certification:
  metric: attention_topk_overlap
  threshold: 0.98
fallback:
  if_margin_below_threshold: fp16_keys
```

For embeddings:

```yaml
artifact: vector_index.embeddings
consumer_operator: nearest_neighbor_retrieval
preserved_quantity:
  type: angular_rank_with_norm_side_channel
fragility:
  metric: nearest_neighbor_gap
  threshold: 0.01
allowed_quantizers:
  - pca_matryoshka_scalar
  - polar_with_norm
  - adc_index
certification:
  metric: recall_at_10
  threshold: 0.995
fallback:
  if_gap_small: exact_rerank
```

## Key engineering rules

1. **Do not add many new quantizers yet.** Make the existing components compose.
2. **Every headline claim should replay.** If it cannot replay, mark it archival, GPU-only, or research-preview.
3. **Every compression recipe should name its consumer operator.** Unknown operators should default conservative.
4. **Every artifact should be hashable and certifiable.** Results without provenance should not become claims.
5. **Every failure should become a guardrail.** Polar keys, small routing margins, stale PCA bases, and vacuous rank certificates are valuable negative controls.
6. **Prefer pass/warn/fail/inconclusive over vague success language.** Certification should make uncertainty explicit.
7. **Keep Track 1 production-grade.** Embedding/vector-DB compression is the clearest validated path to practical adoption.
8. **Keep Track 2 operator-aware and evidence-labeled.** KV/inference work is powerful, but model and hardware dependence must stay visible.

## Final target state

The project should feel like a coherent system rather than a toolkit:

```text
1. Trace the operator.
2. Declare the task geometry.
3. Plan a quantization recipe.
4. Compress using a built-in or plugin quantizer.
5. Certify preserved geometry.
6. Replay the claim.
7. Monitor drift and fallback in production.
```

The deepest version of TurboQuant Pro is not merely better compression. It is the reference implementation of operator-aware quantization safety.

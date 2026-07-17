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

This roadmap assumes the current known hygiene issues have been addressed or are being addressed first. Status as of 2026-07-16:

- release-state clarity between PyPI release and master/unreleased features — **in progress** (this status pass; a README release-state banner is the remaining Phase 0 item);
- stale documentation counts and roadmap items — **addressed** (test count is command-derived, this roadmap now carries live status);
- torch optional dependency or install-message mismatch — **addressed** (`torch` and `yaml` extras added; imports are lazy);
- plugin conformance design-vs-implementation mismatch — **addressed** (design doc reconciled with `plugin_conformance.py`);
- `torch_decode` wording around host-side reconstruction versus true device-native operation — **addressed**;
- gate-routing wording clarified as margin/order sensitivity rather than raw magnitude alone — **addressed**.

The roadmap also assumes that the central validated product track remains embedding/vector-DB compression and compressed-domain retrieval, while KV-cache and operator-aware inference features are treated as powerful but more hardware/model-dependent.

## Roadmap at a glance

> **Status update (2026-07-16).** Much of the early roadmap is **built on master**
> (unreleased — ahead of the latest PyPI release; see [Release milestone
> sequence](#release-milestone-sequence)). Phases **1 and 4 are shipped**; Phases
> **2 and 3 are partial** — the artifact ships, the hardening (JSON Schema + golden
> fixtures, a fully-executable public claim) is the open work; Phase 0 is mostly
> done. The next release is the **v1.8.0 coherence release** that packages exactly
> this shipped surface. Legend: ✅ shipped · ◑ partial · ○ not started.

| Phase | Theme | Status | Primary outcome |
|---|---|---|---|
| Phase 0 | Stabilize the release surface | ◑ mostly done | External reviewers can tell what is stable, beta, experimental, released, and master-only. |
| Phase 1 | Unify instruments into one CLI | ✅ shipped | The `tqp` command exposes trace, probe, plan, certify, replay, monitor, and plugin workflows. |
| Phase 2 | Certificate schema | ✅ shipped | `tqp certify` emits a schema-locked, golden-tested, provenance-stamped `certificate.json` with a documented compatibility promise; the canonical GloVe claim ships a durable artifact bundle. |
| Phase 3 | Claim replay | ✅ shipped | `claims.yaml` + `tqp replay` gate executable claims; the canonical public GloVe recall claim (`embedding_glove_recall`) is executable end-to-end and CI-gated on a hermetic subset. |
| Phase 4 | Productize the planner | ✅ shipped | `tqp plan embeddings` / `plan kv` emit a Pareto frontier, rank-certificate preview, and risk flags. |
| Phase 5 | Prove the plugin ecosystem | ✅ shipped | `tqp-reference-plugin` — a package **outside this repo** — registers via the entry point, passes `tqp plugin conformance` (roundtrip/packed/affine/serialization), and participates in `tqp certify`. Exit criterion met. |
| Phase 6 | Production vector-index lifecycle | ✅ shipped | `tqp index create/add/delete/compact/migrate/search/certify/drift/info` over the versioned, CRC-checked TQIX container — the full ingest→search→update→compact→migrate→certify→monitor loop. |
| Phase 7 | Real-model operator validation | ✅ shipped | Three regimes validated on real weights and promoted to `docs/model_cards/` + `claims.yaml`: attention keys (Llama/Mistral/Qwen), MoE routing (OLMoE-1B-7B), SSM decay (Mamba-790m). |
| Phase 8 | Runtime safe fallback | ✅ shipped | `TQPRuntimePolicy` reads every Phase 1–7 fragility signal and returns a back-off action; `TQEIndex.search(policy=...)` escalates to exact rerank adaptively. |
| Phase 9 | Documentation and paper packaging | ✅ shipped | A documentation hub (`docs/`) with an architecture diagram + six canonical guides; the reviewer path README → quickstart → certificate → replay is legible end to end. |

Per-phase timelines (for the not-yet-shipped phases) are noted in each section below.

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

> ✅ **Shipped on master (v1.8.0.dev).** The `tqp` console script exposes
> `version`, `plugin list`/`conformance`, `trace`, `probe`, `plan embeddings`/`kv`,
> `certify`, `replay`, and `monitor`. Every subcommand is real (no stubs) and
> covered by `tests/test_cli.py`. Acceptance signals are rank-fidelity / (A2) /
> certificate throughout — see [`CLI.md`](CLI.md). The remaining items below are
> historical planning context.

**Timeline:** 2-3 weeks (delivered)

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

> ✅ **Shipped — exit criterion met.** `tqp certify` emits a durable
> `certificate.json`: provenance-stamped (input sha256, tool version, UTC
> timestamp, params), distribution-free κ / μ̂ / τ-floor + pass/vacuous decision,
> **schema-locked** by a committed [`rank_certificate.schema.json`](../turboquant_pro/schemas/rank_certificate.schema.json),
> **golden-tested** (a committed fixture regenerated and compared in CI), and
> covered by a documented **compatibility promise**
> ([CERTIFICATE_SPEC.md](CERTIFICATE_SPEC.md)) so the format cannot drift. The
> canonical GloVe claim ships a full **artifact bundle** (results + certificate +
> environment + exact command) under
> [`benchmarks/artifacts/embedding_glove_recall/`](../benchmarks/artifacts/embedding_glove_recall/).
> A richer multi-section envelope (a `task`/`environment`/`limitations`-shaped
> superset, an optional `--html` report) is a possible future enhancement, not
> gating.

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

### Exit criterion — met

Every important benchmark and claim can emit a durable certificate artifact:
`tqp certify --out certificate.json` produces the schema-locked, provenance-stamped
artifact, and the canonical GloVe claim ships a committed bundle
([`benchmarks/artifacts/embedding_glove_recall/`](../benchmarks/artifacts/embedding_glove_recall/)).

## Phase 3: Make claim replay executable

> ◑ **Partial.** `claims.yaml` and `tqp replay` ship: `track1_recall_smoke` runs
> end-to-end through a shared harness, checks `results.json` against `expected`
> ranges, and gates the exit code. **Open work:** the major claims are still
> references to notebooks/scripts — the next serious step is making the canonical
> **GloVe** claim executable through `tqp replay` (even if 1M-scale timing remains
> "local hardware").

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

> ✅ **Shipped on master (v1.8.0.dev).** `tqp plan embeddings` runs `auto_compress`
> (now ranking the frontier on a **measured `recall@k`** target, default
> `recall@10 >= 0.90`) and attaches a rank-certificate preview, Pareto
> `alternatives` with bytes/vector, `risk_flags`, and a `tqp certify` reproduction
> command; `tqp plan kv` does the `AutoConfig` + operator-trace policy. The
> deeper task-geometry compiler below (hardware-aware constraints, richer fallback
> policy) is the remaining ambition.

**Timeline:** 4-6 weeks (core delivered; task-geometry compiler ongoing)

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

> ✅ **Shipped — exit criterion met.** The plugin protocol + conformance kit ship,
> and the proof is now closed by a package that lives **outside this repo**:
> [`tqp-reference-plugin`](https://github.com/ahb-sjsu/tqp-reference-plugin) (pure
> NumPy). Installed into a fresh environment alongside turboquant-pro, it is
> discovered purely through the `turboquant_pro.plugins` entry point (no import
> from this tree), passes `tqp plugin conformance` on every applicable check —
> roundtrip, packed, **affine** (the fused-decode gate), and serialization — and
> its reconstruction is certified by the same rank certificate `tqp certify` uses.
> The in-tree `plugins/tqp-bnb`, `tqp-gptq-awq`, `tqp-trtllm` remain *incubators*
> that dogfood the same contract. With an external plugin passing conformance, the
> `plugin` / `plugin_conformance` API can promote from **Experimental** on its next
> stability review.

**Timeline:** 4-6 weeks

**Goal:** Demonstrate that the plugin protocol works outside the main repo.

### First external plugin (shipped)

`tqp-reference-plugin` (option 2 — a tiny pure-NumPy package that exists only to
prove packaging + conformance) was chosen as the cleanest, dependency-light
proof. `tqp-faiss-opq` and a full `tqp-bnb` release remain good future additions.

### Deliverables

| Item | Acceptance check | Status |
|---|---|---|
| External plugin package | Installable independently from the main repo. | ✅ `pip install` in a fresh env, no main-tree code |
| Entry-point registration | `tqp plugin list` discovers it. | ✅ discovered via `turboquant_pro.plugins`, no direct import |
| Conformance CI | External repo runs `tqp plugin conformance`. | ✅ GitHub Actions + verified on a clean venv |
| Certification integration | Plugin can be used by `tqp certify`. | ✅ `certificate_from_embeddings` + `tqp certify` on its output |
| Capability table | Plugin declares target tensors, affine support, hardware assumptions, and maturity. | ✅ in the plugin README |
| Tutorial | A user can write a toy plugin in 20 minutes. | ✅ "Write your own plugin in 20 minutes" in the README |

### Exit criterion — met

At least one plugin outside the main tree passes conformance and participates in
certification: `tqp-reference-plugin` does both, verified from a fresh install
where it is reachable only through the entry point.

## Phase 6: Harden the production vector-index lifecycle

> ✅ **Shipped.** `turboquant_pro.index.TQEIndex` + the `tqp index` command group
> give Track 1 a full production lifecycle over the versioned, corruption-checkable
> **TQIX** container (`turboquant_pro.index_file`): a persisted compressed-domain
> ADC index that grows, forgets, compacts, migrates, certifies, and self-checks
> for drift. Every container section is CRC32-guarded and writes are atomic.

**Goal:** Make Track 1 production-grade rather than benchmark-only.

### Commands (shipped)

```bash
tqp index create --embeddings embeddings.npy --out index.tqe
tqp index add index.tqe --embeddings new_embeddings.npy
tqp index delete index.tqe --ids ids.txt
tqp index compact index.tqe --out compacted.tqe
tqp index migrate old.tqe --to-version 2 --out new.tqe
tqp index search index.tqe --queries queries.npy --k 10 --rerank 10
tqp index certify index.tqe --min-tau 0.5
tqp index drift index.tqe --embeddings recent.npy
tqp index info index.tqe
```

### Features

| Feature | Why it matters | Status |
|---|---|---|
| Append | Real corpora grow. | ✅ `add` (same basis, no refit) |
| Delete/tombstone | Real corpora change. | ✅ `delete` (O(1) by external id) |
| Compact | Storage remains honest after deletes. | ✅ `compact` (physical drop, ids preserved) |
| Migrate | TQE versioning is tested under real upgrades. | ✅ `migrate` v1→v2 (+ round-trip test) |
| Drift detection | Stale PCA basis or distribution shift is caught. | ✅ `drift` (retained-variance + mean-shift) |
| Corruption/fuzz tests | The format becomes trustworthy. | ✅ per-section CRC32 + single-byte-flip fuzzer |
| Exact rerank compatibility | High-recall retrieval remains safe under compression. | ✅ metric-correct two-stage rerank |
| Memmap/shard | Large indexes become practical. | ◑ `read_directory` enables it; not yet wired to search |

### Exit criterion — met

Track 1 supports a realistic RAG corpus lifecycle, exercised end to end by
`tests/test_index.py::test_full_lifecycle`:

```text
ingest -> search -> update -> compact -> migrate -> certify -> monitor
```

## Phase 7: Real-model validation for operator sensitivity

> ✅ **Shipped — exit criterion met.** Three operator regimes are validated on
> real model weights and promoted to `docs/model_cards/` + `claims.yaml`:
>
> - **Attention keys** (`SOFTMAX_SCORE`) — real Llama-2-7B/13B, Mistral-7B,
>   Qwen2.5-7B/1.5B perplexity + LongBench; the PolarQuant collapse (12.24 →
>   10643) and symmetric-NF4 GQA collapse (43.8 → 4.7) preserved as negative
>   cases. [`attention_keys.md`](model_cards/attention_keys.md).
> - **MoE routing** (`GATE_SELECTION`) — real **OLMoE-1B-7B**: a controlled
>   differential-logit perturbation at the margin scale flips low-margin tokens
>   **~1740×** more than high-margin (top-8), **~1256×** at the argmax; naive
>   4-bit gate quant reshuffles 92% of top-8 sets.
>   [`moe_routing.md`](model_cards/moe_routing.md) +
>   [`validate_olmoe_routing.py`](../benchmarks/validate_olmoe_routing.py).
> - **SSM decay** (`STATE_DECAY`) — real **Mamba-790m**: 3-bit linear decay quant
>   collapses WikiText-2 ppl 11.65 → **1.01×10¹⁰**; the native A_log basis keeps
>   it at **14.44** (~7×10⁸ gap). [`ssm_decay.md`](model_cards/ssm_decay.md) +
>   [`validate_mamba_decay.py`](../benchmarks/validate_mamba_decay.py).

**Goal:** Turn synthetic/operator probes into model-family evidence.

### Deliverables (status)

| Item | Acceptance check | Status |
|---|---|---|
| Model cards | One `docs/model_cards/*.md` per model family. | ✅ 3 cards + index |
| Claim rows | Validated claims promoted into `claims.yaml`. | ✅ `kv_keys_per_channel`, `moe_routing_margin`, `ssm_decay_basis` (track `operator`) |
| Negative cases | Failures are preserved and explained. | ✅ PolarQuant/NF4 collapse; OLMoE top-8 saturation nuance; linear-basis SSM collapse |
| Reproduction scripts | Real numbers backed by committed code + data. | ✅ `benchmarks/validate_{olmoe_routing,mamba_decay}.py` + `results_*.json` |
| Certificates | Acceptance by the consumer metric. | ✅ perplexity / expert-set flip rate — **not** reconstruction cosine (the coherence rule; a rank certificate on key/gate *reconstruction* would be actively misleading here) |

**Weight PTQ** and **RAG retrieval** rows from the original matrix are already
covered elsewhere (`experiments/results_matched_bit/*`, `embedding_glove_recall`)
and are not gating for this phase.

### Exit criterion — met

Operator-aware quantization is supported by **one real attention validation
(Llama/Mistral/Qwen keys), one real MoE validation (OLMoE-1B-7B), and one real
SSM/recurrent validation (Mamba-790m)** — all with committed reproduction scripts,
raw data, and preserved negative cases.

### Follow-up (not gating)

The companion paper (`paper/foundational/main.tex`) cites specific **Mixtral-8x7B**
top-2 numbers whose backing run was never committed; the OLMoE validation confirms
the mechanism on a real router but on a top-8 model where the effect saturates at
coarse bit-depths. Backing the paper's exact Mixtral top-2 figures with a committed
run remains a documented open item.

## Phase 8: Runtime safe fallback

> ✅ **Shipped — exit criterion met.** `turboquant_pro.runtime_policy.TQPRuntimePolicy`
> is the policy layer that reads every fragility signal built in Phases 1–7 and
> returns a conservative back-off action where the operator is fragile, while
> letting the cheap path run where margins are wide. `TQEIndex.search(policy=...)`
> makes retrieval **adaptive** — single-pass by default, escalating to exact rerank
> only when the top-k boundary is tied.

**Goal:** Make the system robust under uncertain or fragile inputs.

Static quantization is brittle. TurboQuant Pro is more serious for knowing when to back off.

### Policy API

```python
from turboquant_pro import TQPRuntimePolicy
policy = TQPRuntimePolicy(
    retrieval_gap_floor=0.01,
    routing_margin_floor=0.02,
    decay_slow_fraction_ceiling=0.02,
    radial_drift_floor=0.15,
    basis_drift_floor=0.05,
)
d = policy.evaluate_routing(gate_logits, k=8)     # -> RuntimeDecision
if d.conservative:                                 # d.action == "keep_router_fp16"
    ...
policy.evaluate_all(regime="unknown", decays=..., drift_report=...)  # many at once
```

### Fallback map (each backed by a shipped instrument)

| Situation | Signal (instrument) | Action | Evaluator |
|---|---|---|---|
| Retrieval top-k score gap is small | boundary gap | Rerank more candidates. | `evaluate_retrieval` |
| Rank certificate is vacuous | `RankCertificate.vacuous` | Require exact rerank. | `evaluate_certificate` |
| KV-key operator is unknown | regime / `a2_probe` | Per-channel + zero-point, or fp16. | `evaluate_kv_keys` |
| MoE routing margin is tiny | `routing_sensitivity` p10 | Keep router higher precision. | `evaluate_routing` |
| SSM decay channel is slow | `state_decay_sensitivity` | Log-time-constant basis or fp16. | `evaluate_decay` |
| A2 tangential fraction drifts down | `a2_probe` | Recalibrate / disable polar quotient. | `evaluate_a2` |
| Encoder distribution drifts | `index.drift` | Refit PCA basis or migrate index. | `evaluate_index_drift` |

### Exit criterion — met

Compression is adaptive: `TQEIndex.search(policy=...)` is cheap where margins are
wide and escalates to exact rerank where the ranking is tied (see
`tests/test_runtime_policy.py::test_adaptive_search_escalates_when_tied`), and the
policy surfaces the conservative action for every fragile-operator situation.

## Phase 9: Documentation and paper packaging

> ✅ **Shipped — exit criterion met.** [`docs/`](.) is now a documentation hub with
> a rendered architecture diagram and the reviewer path front-and-centre; the six
> canonical guides live under [`docs/guides/`](guides/) (+ the existing
> [`PLUGINS.md`](PLUGINS.md)). The main README links the hub and states the
> 15-minute path.

**Goal:** Make the work readable as a serious project rather than a collection of clever modules.

### Canonical documents (shipped)

1. **User guide** — compress embeddings safely in 15 minutes → [`guides/user_guide.md`](guides/user_guide.md).
2. **Operator-aware quantization** — why keys, values, gates, decays, and weights need different quotients → [`guides/operator_aware_quantization.md`](guides/operator_aware_quantization.md).
3. **Certification** — what a certificate means and does not mean → [`guides/certification.md`](guides/certification.md).
4. **Plugin guide** — write, test, certify a plugin → [`PLUGINS.md`](PLUGINS.md) (+ the external [`tqp-reference-plugin`](https://github.com/ahb-sjsu/tqp-reference-plugin) `docs/GUIDE.md`).
5. **Claim replay** — reproduce the headline numbers → [`guides/claim_replay.md`](guides/claim_replay.md).
6. **Production lifecycle** — maintain a mutable compressed index → [`guides/production_lifecycle.md`](guides/production_lifecycle.md).

### Architecture diagram

Rendered (mermaid) at the top of [`docs/README.md`](README.md):
`artifact → operator trace → planner → compressor/plugin → certificate → claim
replay → production monitor + index`, with the `TQPRuntimePolicy` back-off loop.

### Exit criterion — met

A reviewer can understand the whole project from **README → [docs hub](README.md) →
[quickstart](guides/user_guide.md) → [certificate example](guides/certification.md) →
[claim replay](guides/claim_replay.md)**.

## Release milestone sequence

| Release | Theme | Headline |
|---|---|---|
| 1.8.0 | **Coherence release** (next) | **`tqp`: trace, probe, plan, certify, replay, and monitor with rank / (A2) / certificate-first acceptance.** Bundles the plugin protocol + conformance kit, `claims.yaml`, TQE v2, provenance-stamped certificate JSON, the (A2)-gated monitor, and the soundness-audit fixes. fp8 / nvfp4 / Hopper / QLoRA stay labeled *experimental / incubator results on master* until their artifacts, commands, and reproduction notes are fully packaged. |
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

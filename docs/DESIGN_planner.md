# Design — the planner: a control plane for lossy representation

**Status: P0 LANDED 2026-09-15** (issue #169), design otherwise DRAFT. Written while three registered experiments run
(`docs/PREREG_rabitq_public.md`, `docs/PREREG_consumer_basis.md`, `docs/PREREG_pruned_scan.md`).
Sections marked *pending* change when their results land. Status marks follow
`POSITIONING_2.0.md`: 🟢 shipped, 🟡 partial, ⚪ designed.

## 0. Thesis

turboquant-pro does not need the best quantizer. It needs to know which representation is best
for a given consumer, under a given budget, on given hardware, and to show why.

"Best" here has three parts, and a plan is incomplete without all three:

1. **Consumer-relative.** Quality is measured through what the consumer does with the vectors
   (a top-k ranking, attention logits, a rerank stage), never through reconstruction error alone.
2. **Budget-constrained.** Bytes per vector, memory, latency or throughput on the target hardware,
   and build time are constraints, measured rather than assumed.
3. **Proven.** Every recommendation carries evidence of a declared kind (section 2.6), recorded so a
   third party can replay it.

The planner is to lossy representation what a query optimizer is to a database: a catalog of
interchangeable operators, cost models, a search over plans, and an explain output. The existing
modules are its operators and instruments. What is missing is the contract that joins them.

**Non-goals.** A new quantizer. A vector database. Guarantees the evidence cannot support
(`POSITIONING_2.0.md` bans guarantee language, and so does this design).

## 1. Requirements, each traced to a measured failure

Every requirement below comes from something the 2026-09-15 campaign found. None is hypothetical.

| # | requirement | evidence |
|---|---|---|
| R1 | **Judge through the consumer.** Accept a plan only on the consumer's metric against an exact reference. | The v1 AVX2 kernel wrapped its uint16 score sums above 257 dims (recall@10 0.848 vs 0.905 exact on 1536-d rows). Reconstruction checks and the test suite never saw it; a ranking comparison did (`3d96506`). |
| R2 | **Check data preconditions before choosing a metric or operator.** | NYTimes has 239 all-zero rows. L2 on normalized vectors, safe everywhere else, dropped an exact IVFFlat scan to recall 0.43 (Amendment 1, `a51c689`). |
| R3 | **No universal winner; search the space per workload.** | Flat vs IVF RaBitQ differ by 7.4 points single-pass at 2 bits on GloVe and 2.6 on NYTimes; rabitqlib searches 2–6× faster than tq-pro while tq-pro builds faster than OPQ; pruning pays at d' = 1024 and not at 256 (interim measurements, 2026-09-15). |
| R4 | **Costs are measured on the target, not modelled once.** | Nibble packing and streaming top-k changed wall time by −6% to +10%; the scan is compute-bound (v1-vs-v2 benchmark). A memory model overestimated real peaks by 1.2–4× (`footprints.py` calibration). |
| R5 | **Choose on a calibration split, verify on held-out data.** | Pruning parameters picked in-sample looked 70–80% cheaper; the registration re-chooses them on calibration queries and evaluates once on held-out queries (`PREREG_pruned_scan.md`). |
| R6 | **Account for every stored byte, and say what is excluded.** | RaBitQ's per-vector correction factors and tq-pro's norm were both omitted in the canonical harness (`933a4d7`); tq-pro holds unpacked uint8 codes in memory, more than it stores. |
| R7 | **External backends are first-class candidates.** | The strongest search in the campaign so far is rabitqlib's, not tq-pro's. A planner that only ranks its own operators is a recommender with a conflict of interest. |
| R8 | **Every decision is replayable.** | The claims ledger and `tqp replay` already make results rerunnable; the planner's output must be a ledger-shaped record, not a log line. |

## 2. Architecture

```
          ┌────────────────────────── workload spec ──────────────────────────┐
          │ data sample · query sample or read operator · consumer metric      │
          │ budget (bytes, RAM, latency on H, build time) · constraints        │
          └───────────────┬────────────────────────────────────────────────────┘
                          ▼
   (1) preflight ── data preconditions, spectrum, query/corpus mismatch, hubness
                          ▼
   (2) candidate space ── transforms × quantizers × search operators × backends
                          ▼
   (3) cost + quality models ── analytic priors, refined by measurement
                          ▼
   (4) search ── dominance pruning, successive halving, frontier under uncertainty
                          ▼
   (5) verification ── held-out evaluation, certificates, measured cost on H
                          ▼
   (6) plan record ── choice, frontier, evidence, environment, replay command
                          ▼
   (7) runtime loop ── drift and canary monitoring, re-plan triggers, safe fallback
```

### 2.1 The workload spec (the planner's intermediate representation)

One document declares the problem; every later stage reads only this and the measurements it
produces.

| field | content | notes |
|---|---|---|
| `data` | a sample of the corpus, its size N, dimension, content hash | the sample is what is measured; N sets costs |
| `consumer` | one of: `topk_inner_product(k)`, `topk_l2(k)`, a registered read operator `P_C` (`read_operators`), a model-level metric (`behavioral_agreement`) | default `topk_inner_product(10)` against exact search |
| `queries` | a query sample, or none | none means the corpus stands in for the query distribution, and the plan says so |
| `budget` | any of: bytes/vector, total memory, p50/p95 latency or QPS at batch B, build time | hard constraints; unmet ones make a plan infeasible, not worse |
| `hardware` | a target fingerprint (CPU model, SIMD level, threads, GPU if any) | costs are only valid for this fingerprint |
| `floor` | minimum acceptable quality on the consumer metric, with a confidence level | e.g. recall@10 ≥ 0.95 at 95% one-sided |

### 2.2 Preflight (R2)

Checks that change what is legal or likely, run before any candidate is built:

- all-zero and non-finite rows; norm spread (whether normalization changes rankings);
- duplicate and near-duplicate rate (ties inflate or deflate recall);
- spectral concentration: variance retained at d/8, d/4, d/2 (truncatability, as in `RESULTS_glove.md`);
- query/corpus mismatch `1 − ⟨S, C⟩/(‖S‖‖C‖)` when queries are given (*pending*: whether it predicts
  the consumer-basis gain, `PREREG_consumer_basis.md`);
- hubness and the (A2) tangential fraction (`a2_probe`, `monitor`).

A failed precondition either removes candidates (L2 search with zero rows) or rewrites them (move zero
rows out of the unit sphere), and the plan record lists which.

### 2.3 Candidate space (R3, R7)

A candidate is a pipeline of passes, each implementing an existing protocol:

| stage | options | protocol | status |
|---|---|---|---|
| transform | identity · corpus PCA · consumer basis S or O (*pending*) · random rotation | `PCAMatryoshka`, a new `Transform` protocol | 🟡 |
| quantizer | TurboQuant scalar b bits · learned codebooks · PQ / OPQ · RaBitQ (faiss, rabitqlib) | `plugins.Quantizer` | 🟡 (faiss/rabitqlib adapters ⚪) |
| bit allocation | uniform · reverse water-filling against `P_C` | `read_allocation` | 🟢 |
| search | flat ADC (v2 kernel) · pruned scan (*pending*) · IVF routing · HNSW · rerank depth r | a new `SearchOperator` protocol | 🟡 |
| backend | tq-pro · faiss · rabitqlib · pgvector/tqvector | conformance kit (`plugin_conformance`) | 🟡 |

External backends enter through adapters that pass the same conformance kit as tq-pro's own
operators: the round-trip contract for quantizers, and for search operators a new contract of
top-k output, stored-byte accounting (R6) and a declared cost function.

### 2.4 Cost and quality models (R4, R6)

Each operator declares **analytic priors**, which are cheap and used only to rule candidates out:

- stored bytes per vector, exact (R6: codes plus every per-vector scalar; shared tables listed separately);
- in-memory bytes during search (they differ: `ADCIndex` holds unpacked codes);
- a lookup-count model for search cost, and a build-cost model in N and d.

**Measured quantities** replace priors before any decision is final:

- quality on the consumer metric against exact search, per query, on the calibration split, with a
  percentile bootstrap interval (the scorer in `benchmarks/rabitq_public/score.py`);
- search latency and throughput on the target fingerprint, minimum of repeated timings;
- build time and peak anonymous memory (the sampler in `benchmarks/rabitq_public/cell.py`).

Measured costs are keyed by the hardware fingerprint and the data scale. A cost measured on a sample
is extrapolated to N only through a declared scaling law, and the plan labels the extrapolation.

### 2.5 Search over plans

1. **Dominance pruning by priors.** Drop candidates that cannot meet the byte or memory budget, and
   candidates dominated on stored bytes by a strictly cheaper candidate of the same family.
2. **Successive halving on sample size.** Evaluate survivors on a small sample; promote only those
   whose quality interval overlaps the frontier; repeat at larger samples. This is where most
   compute is saved.
3. **Frontier under uncertainty.** Keep a candidate unless another is better on every budgeted axis
   *and* its quality interval lies wholly above. Ties are kept and reported, not broken silently.
4. **Choice.** Among frontier candidates that meet every budget with the quality floor's lower
   confidence bound, choose by the declared objective (default: maximize quality lower bound; then
   minimize stored bytes; then minimize latency).

### 2.6 Verification (R1, R5)

The chosen plan and its nearest competitors are re-evaluated once on a held-out split that played no
part in the search. Evidence comes in three declared kinds, and every claim in a plan names its kind:

| kind | what it establishes | source | status |
|---|---|---|---|
| **certificate** | a distribution-free floor on rank agreement for the given data | `rank_certificate`, `tqp certify` / `tqp verify` | 🟢 |
| **statistical** | quality on held-out queries with a one-sided confidence bound; paired comparison to the runner-up | bootstrap scorer | 🟡 (to lift from the benchmark) |
| **measured cost** | latency, throughput, memory on the stated fingerprint | timing harness | 🟡 |

A plan whose held-out result falls below the floor is rejected. The record keeps the failure, and the
planner falls back to the next frontier candidate, which is then verified in turn.

### 2.7 The plan record (R8)

The output is a JSON document in the family of `rank_certificate.schema.json`:

- workload spec and content hashes of the data and query samples;
- preflight results and the precondition actions taken;
- every candidate considered, with the stage at which it left (prior, halving round, frontier);
- the frontier with intervals, the choice, and the rule that chose it;
- evidence entries, each with its kind (certificate, statistical, measured cost);
- environment: package and backend versions, hardware fingerprint, seeds;
- a replay command: `tqp plan replay <record>` re-runs verification and reports agreement.

`tqp plan explain <record>` renders it for a person: what was chosen, what it beat, by how much, and
what would change the answer (the nearest budget boundary).

### 2.8 The runtime loop

A plan is valid for the distribution it was measured on. In service:

- **canaries:** a small stream of sampled production queries is also answered exactly, offline, and
  recall against exact is tracked with an interval (R1 in production);
- **drift:** `monitor`'s cosine floor and (A2) tangential fraction, plus the query/corpus mismatch
  index on recent queries;
- **triggers:** a canary interval below the floor, or drift beyond the preflight bands, marks the plan
  stale and schedules a re-plan; `runtime_policy` supplies the conservative action meanwhile
  (larger rerank depth, a more precise tier).

## 2.9 What P0 shipped

`turboquant_pro/planner.py` and `turboquant_pro/consumers.py`, with
`tqp plan run | explain | replay | consumers` and the record schema
`turboquant_pro/schemas/compression_plan.schema.json`. The pieces map onto the
architecture above as follows.

| stage | where it lives | note |
|---|---|---|
| workload spec | `planner.WorkloadSpec`, `Budget`, `QualityFloor`, `Artifact` | the artifact carries a content hash and the context the consumer reads with |
| preflight | `planner.preflight` | zero rows, non-finite rows, norm spread, spectral concentration; flags travel in the record |
| candidate space | `planner._enumerate_candidates`, `plugins.capabilities` | from the registry, never a hard-coded list; a codec that cannot be built is recorded as `unsupported`, not dropped |
| consumer metric | `consumers` (retrieval top-k, attention softmax, read-operator distortion, declared) | its own entry-point group, so a consumer can arrive out of tree |
| cost | `planner.container_bytes` | every array and buffer reachable in the container, with a breakdown (R6) |
| search | `_prune_on_priors`, `_halving`, `_frontier` | a candidate is cut only when a survivor's interval lies wholly above its own |
| verification | `CompressionPlanner._verify` | held-out split, used once, on the conservative end of the bootstrap interval |
| false clear | `_diagnose` + `false_clear` | attached to **every** evaluation, not just the winner's, and scored on the consumer's own items |
| record | `CompressionPlan.as_dict`, `explain` | schema-validated; `replay_plan` re-runs verification and reports agreement |
| runtime | `_fallback_policy` | actions and thresholds come from `runtime_policy.TQPRuntimePolicy`, not a parallel vocabulary |

Not yet done, and named here rather than implied: measured latency and
throughput evidence (only stored bytes are measured, so `measured_cost` covers
size and not time), transforms and search operators as candidate stages (the
candidate is a codec, not yet a pipeline), faiss and rabitqlib adapters, the
runtime loop of section 2.8, and the P0 exit test of section 5 — the planner has
not yet been scored for regret against the RaBitQ campaign's exhaustive grid.
Until that runs, this is a working control plane, not a validated one.

## 3. What exists, what is missing

| capability | module | status |
|---|---|---|
| quantizer contract and conformance | `plugins`, `plugin_conformance` | 🟢 |
| consumer read operators, bit allocation against them | `read_operators`, `read_allocation` | 🟢 |
| distribution-free certificates, verify/replay | `rank_certificate`, `certify_report`, `tqp certify/verify` | 🟢 |
| consumer-blind-spot probe, drift monitor | `a2_probe`, `monitor` | 🟢 |
| conservative runtime decisions | `runtime_policy` | 🟢 |
| Pareto sweeps over PCA dim × bits | `autotune`, `auto_compress` | 🟡 tq-pro operators only, reconstruction and recall, no held-out verification |
| claims ledger and replay | `claims.yaml`, `tqp replay` | 🟢 for claims; plans not yet |
| bootstrap scorer, stored-byte accounting, memory sampling | `benchmarks/rabitq_public/` | 🟡 benchmark code, to lift into the library |
| workload spec, preflight, candidate search, plan record, `tqp plan run` | `planner` | 🟢 P0 |
| consumer-metric registry (retrieval, attention, read operator) | `consumers` | 🟢 P0 |
| measured latency / throughput evidence, hardware fingerprinting | — | ⚪ |
| search-operator protocol; faiss / rabitqlib adapters | — | ⚪ |
| consumer bases, pruned scan as operators | `benchmarks/consumer_basis/`, `search_pruned` | ⚪ *pending registered results* |

The planner absorbs `autotune` and `auto_compress`. Both become frontends that build a workload spec
and call the planner; neither keeps its own search loop.

## 4. Milestones

- **P0 — offline planner for top-k retrieval.** Workload spec; preflight (zero rows, norm spread,
  spectrum); candidates: identity/PCA × TurboQuant bits × flat ADC with rerank depth, plus faiss PQ,
  OPQ and RaBitQ through adapters; priors for bytes; successive halving; held-out verification with
  statistical evidence; plan record; `tqp plan`, `tqp plan explain`, `tqp plan replay`.
  *Exit test:* section 5.
- **P1 — consumer-relative and faster operators.** Consumer bases and the pruned scan enter the
  candidate space only with the verdicts of their preregistrations (*pending*); rabitqlib adapter;
  query/corpus mismatch in preflight; measured-cost evidence with hardware fingerprints; IVF routing.
- **P2 — runtime loop.** Canaries, drift triggers, re-plan scheduling, `runtime_policy` actions.
- **P3 — beyond retrieval.** The same contract for KV-cache compression, where `autoconfig` and
  `runtime_policy` already hold the operators and `behavioral_agreement` is the consumer metric.

## 5. How the planner itself is judged

A planner that recommends is making a claim, so it is tested like one. The RaBitQ public campaign is
its first ground truth: an exhaustive, seeded, byte-accounted grid on six public arms.

- **Regret.** For a budget, regret is the best feasible quality in the exhaustive grid minus the
  quality of the planner's choice, both on held-out queries. A good planner has small regret at a
  fraction of the grid's compute.
- **Cost of planning.** Candidate-evaluations and core-hours spent, against the exhaustive grid's.
- **Calibration of its evidence.** Across budgets, how often a statistical claim's held-out bound is
  violated on a fresh split; it should match the declared confidence.

The P0 exit test is preregistered before the planner runs on the campaign's arms: budgets chosen in
advance, regret and planning-cost thresholds fixed, and the grid results treated as sealed until the
planner's choices are recorded.

## 6. Risks and prior art

- **Prior art.** Automatic index and parameter selection exists: faiss's autotuning of operating
  points, automatic index choices in managed vector databases, and cost-based query optimizers in
  databases generally. The defensible difference is the objective and the evidence — consumer-relative
  quality, certificates, held-out verification, replayable records — not automation by itself. A
  proper survey precedes any claim of novelty.
- **Measurement integrity is the product.** Every failure listed in section 1 would have produced a
  confident wrong plan. The planner inherits the campaign's discipline or it is a recommender.
- **Scope.** The planner is the reason to consolidate, not a new front. Features that do not become
  operators, evidence, or records wait.
- **Search cost.** Exhaustive grids do not scale; if successive halving and priors cannot keep
  planning cost well below the grid's, P0 fails its exit test and the design changes.

## 7. Open questions

1. Is recall against exact search the right default consumer metric, or should the default be a rank
   certificate target, which needs no query sample?
2. How should a plan state quality when no query sample is given? Corpus-as-queries is an explicit
   assumption the consumer-basis results may show to be wrong for real queries.
3. Which hardware fingerprint granularity makes measured costs transferable: CPU model, SIMD level
   and thread count, or finer?
4. Should the planner optimize a single plan, or return a small frontier for a person to choose from,
   with `explain` making the trade-offs legible?

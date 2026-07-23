# TurboQuant Pro 2.0 — the platform release (v2)

> **Thesis.** TurboQuant Pro 2.0 turns certified tensor compression into a **safe
> storage and interchange layer for inference caches and vector indexes:
> engine-integrated, format-stable, observable, and failure-safe.**
> "Observable" and "failure-safe" are in the thesis on purpose — they are what
> separate infrastructure from a library.

This v2 incorporates an external evaluation in full
([`ROADMAP_2.0-eval.txt`](ROADMAP_2.0-eval.txt)); its two governing corrections:

1. **2.0 GA proves ONE complete production path** — TQE1 freeze + a production
   vLLM connector + one stable analytical query surface — instead of partially
   completing every integration. SGLang production support moves to **2.1**;
   the native PostgreSQL access method moves to **2.2**.
2. **The missing work is contracts, not algorithms**: identity, failure
   semantics, observability, security, compatibility, transactional lifecycle.

Three kinds of value, named separately because customers weight them
differently: **capacity** (more prefixes retained), **durability** (reuse
across restarts), **portability** (movement across engines and nodes).

Positioning rule: compression numbers are **measured targets, not promises** —
"up to 4–5× in currently validated configurations; exact effective capacity is
model-, layout-, and workload-dependent" (and end-to-end accounting includes
record metadata, padding, fragmentation, store indexes, and integrity fields).

Status legend: 🟢 shipping on `master` · 🟡 scaffolded · ⚪ designed.

---

## Pillar 1 — the vLLM KV connector 🟡

**Scope sentence, stated so nobody over-assumes:** *TurboQuant 2.0 compresses
KV blocks outside the active GPU-resident tier. Active attention behavior and
engine-native GPU cache layout remain under the serving engine's control.*
The connector is the offload/persistence tier (save on evict, restore on
scheduler match), enabled by configuration alone via vLLM's V1 connector
interface — which upstream labels **experimental**, so we ship a formal
compatibility policy (below), not just a CI lane.

**Shipped (🟡 scaffold):** `turboquant_pro.connectors.vllm_v1` — protocol
surface (scheduler + worker roles), engine-agnostic `TurboQuantBlockStore`
quantizing with the (A2)-correct disciplines through the public plugin
registry, `register()` for `KVConnectorFactory`, protocol/round-trip tests
without vLLM. **Current safety scope: in-process, request-id-keyed, no
persistence — so no wrong-prefix risk exists yet.** Every milestone below is
a precondition for turning persistence on.

### P1-M1 · KV identity & compatibility contract ⚪ (the gate for everything)
A persisted KV block is bound to a canonical, content-addressed **identity
profile**: model repo + revision + weight fingerprint; tokenizer fingerprint;
token IDs (not source text); architecture + layer; LoRA/adapter identity;
RoPE config + scaling; attention backend + KV-layout version; TP/PP config;
KV dtype; block/page size; GQA/MQA/MLA config; sliding-window/hybrid config;
quantization discipline + parameters; TurboQuant encoder version.
**Governing rule: uncertain compatibility ⇒ cache miss and recomputation** —
never best-effort decode.

### P1-M2 · Failure semantics ⚪ (own milestone, before any beta)
Defined behavior for: truncated record, checksum failure, load timeout,
partial-layer load, worker death mid-save, store-full, concurrent
evict/request, cross-node transfer failure, unknown future profile, absent or
stale certification record. Production invariants: writes atomic-or-ignored;
partial records never visible; corruption ⇒ miss; timeout ⇒ recompute;
connector failure never fails the request unless configured to; bounded
backpressure; cache operations cannot deadlock scheduling.

### P1-M3 · Observability & operational controls ⚪
First-class metrics: logical/physical hit rates, partial-prefix hits, bytes
saved, effective cache expansion, save/load throughput, p50/p95/p99 load
latency, TTFT and inter-token deltas, dequant time, recompute-fallback count,
integrity failures, compatibility misses, probe verdict + age, eviction and
admission counts, queue depth/backpressure, spill/host-memory utilization.
Controls: per-model quotas, per-tenant namespaces, flush/invalidate commands,
read-only mode, max load latency, max storage, selectable fallback.

### P1-M4 · Break-even admission policy ⚪
Reuse only when `T_lookup + T_read + T_transfer + T_dequant <
T_recompute` in expectation — accounting for prefix length, storage tier,
queue depth, bandwidth, prefill load, ratio, and expected future reuse. A
simple admission model is a practical differentiator; report the break-even
region, don't hide it. **Save-side corollary (from the beta backpressure
work):** shed-newest under overload has a locality tension — the newest
prefix is often the likeliest re-request — so admission logic should
eventually inform *what* to shed, not merely whether to save.

### P1-M5 · Fused restore path ⚪ (corrected wording)
**Fused transfer + dequantization directly into the engine-native GPU KV
block layout, without an intermediate decompressed host or GPU buffer** — so
restored prefixes stay live for subsequent decode steps. (Not "dequant into
one attention call", which would not populate the engine cache.)

### P1-M6 · Coverage & quality gates ⚪
Model-family matrix (MHA, GQA, MQA, MLA where supported, RoPE scaling,
sliding window, TP, LoRA, speculative decoding, multimodal prefixes, chunked
prefill) — unsupported combinations **detected and rejected**, not assumed.
Quality acceptance beyond perplexity/agreement: first-token agreement, full
greedy-sequence agreement, logit divergence at restored positions,
long-context needle tests, repeated save/load cycles, worst-case (not only
average) degradation, multiple models and seeds. The (A2) verdict is stored
with model fingerprint, probe dataset + version, discipline, confidence
bounds, date, software/hardware versions — and **expires automatically** on
relevant configuration change.

### Compatibility policy (applies to every pillar)
Published, version-pinned matrix; **"supported" means "in CI"**, not
"expected to work": exact vLLM minors (the connector API is experimental),
exact SGLang releases (2.1), tested PyTorch/CUDA/Python ranges, tested GPU
architectures, explicit model families, supported attention layouts, tested
PostgreSQL majors and DuckDB versions, accepted TQE format/profile versions.

---

## Pillar 2 — TQE1 as the interchange format 🟡

Shipped 🟢: golden corpus + dependency-free single-file Python reader,
conformance-tested (`contrib/tqe1_reader.py`, `tests/golden/tqe1/`).

### P2-M1 · Versioning terminology, fixed before the RFC ⚪
Four explicit dimensions, never conflated again: **container format** (TQE1) ·
**specification revision** (1.0) · **record profile** (`embedding`,
`kv_block`, …) · **codec ID** (`polar-v3`, `key-channel-v2`, …). The
bit-packing generation is never called bare "v3" in normative text.

### P2-M2 · The RFC draft ⚪, including (each its own section):
- **Canonical encoding**: two conforming writers, same input + parameters ⇒
  byte-identical output (field ordering, float representation, NaN/Inf
  policy, padding, parameter normalization, exact hash coverage) — required
  for hashing, dedup, golden tests, content-addressed caching.
- **Random access & recovery**: optional record index/footer, block lookup,
  append behavior, truncation recovery, atomic commit marker, tombstones,
  compaction, concurrent-reader rules, append-while-reading policy.
- **Integrity vs identity**: separate fields for the fast corruption
  checksum, the cryptographic content hash, and the semantic
  identity/fingerprint — three different problems.
- **Extension behavior**: unknown optional field ⇒ skip; unknown mandatory
  feature ⇒ reject; unknown record profile ⇒ enumerate, don't decode;
  unknown codec ⇒ reject the record, not the file. Feature bits + reserved
  ranges.
- **Parser limits** (security): max tensor rank/dims, max record size,
  checked arithmetic, bounded allocation, depth limits — plus a
  malformed-file fuzz corpus in CI.

### P2-M3 · Interoperability as a release requirement ⚪
GA requires: Python writer → Rust reader; Rust writer → Python reader; old
reader → new writer (optional extensions only); big-endian simulation or
explicit rejection; fuzz suite green; golden files reproduced independently;
at least one reader living outside the main package. "Standard" is not
claimed prominently before an external adopter exists.

### P2-M4 · The compatibility promise, made affordable ⚪
Replaces "readers MUST open every 1.9+ file forever": TQE1 readers always
understand all **finalized TQE1 core records**; deprecated codecs stay
readable for a documented minimum period; `tqp format migrate` upgrades older
experimental records; **pre-freeze 1.9 files are legacy**, not accidental
permanent contracts.

---

## Pillar 3 — SQL surfaces, honestly named

### 3a · DuckDB 🟢 (the GA analytical surface)
`turboquant_pro.duckdb_ext` — compressed-domain blocked ADC search registered
as a joinable relation; streaming Arrow scans reconstructed block-at-a-time
(RAM bounded by batch, tombstones skipped). **GA ships on today's stable
relation/Arrow surface** — prettier SQL-native table-function syntax is
additive later and never a GA dependency on an API outside our control.

### 3b · PostgreSQL **compressed-storage bridge** 🟢→⚪ (Track A, renamed)
What ships today (`turboquant_pro.pgvector` + `[pgvector]` extra):
TQE1-in-bytea storage with `insert_compressed`/`search_compressed` —
**storage in Postgres, scoring in Python**. It is deliberately *not* called
"direct SQL over compressed indexes." 2.0 upgrades: COPY-based batch
ingestion; an in-database calibration catalog; migration path from vanilla
pgvector columns.

### 3c · PostgreSQL **SQL-native compressed search** ⚪ (Track B → release 2.2)
The real access method is a database project, not a scoring project. Beyond
ADC + operator syntax it requires: MVCC visibility, transactional
insert/delete, WAL, crash recovery, replication, backup/restore, VACUUM,
concurrent/online (re)build, update handling, tombstone reclamation,
privilege model, planner costing, predicate pushdown, parallel query, exact
rerank, corruption detection, upgrade paths across PG majors. **Storage
decision recorded now:** PostgreSQL-managed index pages (not external
memmapped shards as the authoritative index) — external files create
backup/replication/transaction problems an operator class must not have.
The maturity bar is pgvector's operational envelope, not raw search speed.

### The recall contract (applies to `WITH (RECALL >= r)` everywhere) ⚪
A stored guarantee defines: ground truth + metric; whether r is an
expectation, percentile, or lower confidence bound (+ confidence level);
query distribution; dataset/index version; calibration sample size; staleness
rules after inserts/deletes; behavior when infeasible; monotonicity;
recalibration triggers. Catalog keyed by *(index fingerprint, query
population, metric, operating-point family, software version)* — a guarantee
that can go stale silently is not a guarantee.

---

## Release sequence (replaces the old three-line sketch)

**2.0.0-alpha** — TQE1 draft spec · golden corpus 🟢 · Python reader 🟢 ·
vLLM connector functional smoke · **canonical KV identity profile** ·
safe corruption/miss fallback · version-pinned compatibility matrix.

**2.0.0-beta** — production vLLM save/evict/reload · async I/O +
backpressure · metrics + tracing · crash/timeout/corruption test suite ·
cross-restart persistence · Rust TQE reader · DuckDB stable surface (done) ·
Postgres Track A batch ingestion + calibration catalog.

**2.0.0-rc** — TQE1 freeze candidate · cross-language interop · fuzz +
malformed-record suite · model-family matrix · cold/warm/cross-restart
benchmark suite · quality certification records · migration + rollback docs ·
security review · zero open correctness/data-loss bugs.

**2.0.0 GA** — TQE1 v1.0 · production-supported vLLM connector · stable
Python/Rust readers · DuckDB integration · Postgres Track A. SGLang and
Postgres Track B ship **clearly labeled preview** unless they independently
meet the GA gates.

**2.1** — production SGLang backend (via HiCache's *public* configurable
storage-backend surface — module path + class name — with its own pinned
lane; not private offload hooks) · fused restore into native cache layouts ·
distributed backing stores.

**2.2** — native PostgreSQL type/operator/access method with full database
semantics · SQL-native recall-constrained planning.

## Quantitative GA gates

| Area | GA gate |
|---|---|
| Correctness | no wrong-prefix reuse anywhere in the compatibility matrix |
| Failure safety | all corruption/timeout paths fall back to recomputation |
| Quality | registered bounds per model × discipline, with provenance + expiry |
| TTFT | warm-hit benefit demonstrated at realistic prefix lengths |
| Tail latency | no unacceptable p99 regression at low hit rates |
| Throughput | positive break-even region **reported**, not hidden |
| Memory | end-to-end bytes include metadata + fragmentation |
| Durability | crash during write ⇒ old record or safe miss |
| Interoperability | Python and Rust cross-read all golden files |
| Compatibility | exact supported engine/version matrix, enforced in CI |
| Security | namespace isolation + malformed-input fuzzing |
| Operations | metrics, quotas, invalidation, inspect/repair tools |
| Reproducibility | public benchmark commands + result artifacts |

Per-model acceptance envelopes and trade-off curves — no universal percentage
until measurements exist.

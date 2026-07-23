# Changelog

## Unreleased

## 2.0.0a2 (2026-07-23)

> Second 2.0 pre-release: the **beta slice** — the connector grows production
> semantics (metrics, async saves, durable persistence), Postgres Track A
> lands, and the anatomy instruments learn to meter themselves. Hardened
> through three external review rounds; every finding dispositioned with a
> test. Pre-release: APIs under `connectors/` may still move before 2.0.0.

### Added
- **Connector metrics (`connectors.metrics`).** Closed-registry counters with
  Prometheus text exposition (`_total` suffixes, HELP/TYPE); miss causes are
  registered labels (`empty` / `corrupt` / `incompatible` / `timeout` /
  `declined`) — an unregistered cause is a `KeyError`, not a new time series.
  Save/load latency reported over an explicit last-1024-ops window.
- **Async saves with bounded backpressure.** Background writer with
  **copy-on-enqueue** (enqueued tensors are snapshotted so later in-place
  reuse cannot corrupt certified bytes — regression-tested by mutating the
  source after enqueue), a bounded queue with declared block/drop policy, and
  a worker-death fallback: if the writer dies, `flush()` drains inline and
  counts `worker_fallbacks` rather than silently losing saves.
- **Durable, pickle-free persistence.** `save_to_dir`/`load_from_dir` with
  atomic write + fsync discipline (COMMIT marker written last; file and — on
  POSIX — directory fsynced), blobs in the self-versioned `tqp-kv-npz/1`
  interim codec (declared legacy-with-migrate ahead of TQE1), zip hardening
  (size caps, dtype whitelist, `[0-9a-f]{24}\.rec` manifest regex vs path
  traversal), per-record sha256, `allow_pickle=False` throughout. Any decode
  failure ⇒ cache miss, never an exception in the serving path.
- **Postgres Track A (`pgvector` additions).** `insert_compressed_copy`
  (COPY-based bulk ingestion, chunked commits), a calibration catalog keyed
  by the full recall contract (index fingerprint, query population, metric,
  ground truth, operator family, software version), and
  `plan_operating_point`/`search_compressed_planned` — plans from measured
  `recall_ci_low`, refuse with the best achievable point named when the
  target is infeasible, `LookupError` on an uncalibrated fingerprint.
- **Anatomy self-metering.** `tqp anatomy`/`tqp hubdiff` reports now carry
  provenance (estimator, Spearman declared, dataset fingerprint — which
  declares its own sampling stride: `s1` = exact hash, `sN` = probabilistic,
  and is layout-normalized so views and copies hash identically),
  size-stable cross-n companions (Robin Hood index, `frac_above_2k`), the
  auditable `recall_by_count_decile` curve, and a `mechanism` →
  `prescription` classification (centrality ⇒ centering, density ⇒
  CSLS/mutual proximity). `--exact/--approx` id arrays are blessed as the
  primary hubdiff mode; the decompressed-vector mode warns that it is a
  proxy. Report field names are API (closed registry).
- **Mechanism classifier confusion matrix** (`tests/test_anatomy.py`): four
  planted fixtures, one per prescription cell (centrality / density / mixed
  / unclear-abstain). Building it killed two classifier designs and forced
  the shipped one: corpus-wide `corr(count, -d_k)` is mechanically
  rank-coupled to the count in any corpus (>0.8 on an isotropic Gaussian)
  and cannot discriminate; the projection-based centrality statistic
  misread non-normalized corpora (now negative distance-to-mean). The
  classifier now types each hub by population percentile, hierarchically
  (centrality first — a point close to everything is mechanically dense),
  reports its inputs (`hub_frac_central`, `hub_frac_dense_noncentral`), and
  abstains entirely when there is no material hub tail (`count_max <
  2.5k`). Earlier reports could prescribe CSLS to corpora whose hubs were
  pipeline-central; treat pre-a2 `mechanism` fields as suspect.
- **STRATA RFC draft** (`docs/STRATA_RFC.md`): stratified anatomy and
  area-scoped guarantees — min-over-strata verdicts, ABSTAIN semantics,
  per-area remedies, bounded certificate blast radius. Plus the first
  measured-run pre-registration (`docs/PREREG_multilingual_strata.md`):
  multilingual per-language hubness, transit backbone, and
  compression-damage concentration, predictions frozen before measurement.
- **TQE1 freeze policy ratified** (RFC §9a): the registry freezes before
  the profiles. Trailer `0x10` (`hubness-scalar/1`, the CSLS scalar) and
  the `tqp-area-map/1` record-metadata fields are allocated with syntax
  frozen and semantics provisional; the `kv_block` profile ships
  labeled-experimental under the 1.1-draft spec revision and freezes only
  after a full persistence soak cycle.

### Changed
- vLLM conformance lane hardened: era-matched pins
  (vllm 0.9.2 + transformers 4.52.4), concurrency cancellation, CPU-runner
  fallback to HF config extraction (full engine execution remains the GPU
  beta gate).

### Fixed
- `tqp replay` error-propagation test was Windows-hostile (single-quoted
  shell command); replay itself was correct on both platforms.

## 2.0.0a1 (2026-07-23)

> First 2.0 pre-release: **every 2.0.0-alpha roadmap item**
> ([`docs/ROADMAP_2.0.md`](docs/ROADMAP_2.0.md), v2 — the externally-reviewed
> sequencing). The theme is contracts: identity, failure semantics, format
> conformance, pinned compatibility. Pre-release: APIs under `connectors/`
> may still move before 2.0.0.

### Added
- **KV identity profile (`connectors.identity`, `tqp-kv-identity/1`).**
  Canonical content-addressed configuration identity for KV reuse; the
  governing rule is structural — any unknown field ⇒ incomplete ⇒ compatible
  with nothing, including itself; prefix keys cannot be minted under
  uncertainty. `prefix_block_hashes` chains token-ID block hashes seeded by
  the profile digest.
- **vLLM V1 connector scaffold (`connectors.vllm_v1`)** with the
  engine-agnostic `TurboQuantBlockStore` ((A2)-correct disciplines through
  the public plugin registry), profile-gated `export_state`/`import_state`
  (mismatch ⇒ total refusal; per-record sha256, corrupt ⇒ skip), and
  load-path failure semantics (any decode error ⇒ miss, never an exception
  in the serving path). In-process scope only, by declared design.
- **Pinned vLLM conformance lane** (`vllm-lane.yml`: vllm 0.9.2 +
  transformers 4.52.4, weekly + on connector changes) — real-base
  subclassing, abstract-surface completeness, factory registration, identity
  extraction. Engine execution on GPU is the beta gate.
- **TQE1 RFC draft** (`docs/TQE1_RFC.md`): four-dimension versioning
  (container/spec-revision/profile/codec), canonical encoding,
  integrity-vs-identity separation, extension rules, parser limits, the
  affordable compatibility promise, interop requirements.
- **TQE1 golden corpus + dependency-free single-file reader**
  (`tests/golden/tqe1/`, `contrib/tqe1_reader.py`) enforced in CI — a writer
  change that alters golden bytes fails as a format break.
- **DuckDB surface** (`duckdb_ext`, `[duckdb]` extra): compressed-domain
  search as a joinable relation; streaming Arrow scans reconstructed
  block-at-a-time.
- **Compatibility matrix** (`docs/COMPATIBILITY_2.0.md`): supported = in CI.
- 2.0 roadmap v2 with both external reviews committed
  (`ROADMAP_2.0-eval.txt`, `ROADMAP_2.0-review2.txt`) and the honest-claims
  positioning page (`docs/POSITIONING_2.0.md`).

## 1.9.1 (2026-07-23)

> Patch release on the 1.9 line: the index becomes **queryable and
> self-describing** (`tqp query`), the toolkit becomes **agent-consumable**
> (`agent_tools` + LangChain/DSPy/MCP/GPT wrappers), and acceptance grows a
> **tail**: the hub anatomy vector and the anti-hub differential oracle
> (`tqp anatomy` / `tqp hubdiff`) extend the "never a blind aggregate"
> coherence rule from reconstruction cosine to mean recall itself. Plus
> resumable IVF builds and cell→shard source routing from the 100-billion-row
> fleet campaign. All additions are additive; no format or API changes.

### Added
- **`tqp query` — a SQL-ish workload interface over TQE indexes.**
  `ANALYZE INDEX 'x.tqe'` builds a statistics catalog (geometry + a measured
  recall/latency calibration sweep); `EXPLAIN SELECT ... WITH (RECALL >= 0.95)`
  shows the calibration-based plan; `SELECT id, score FROM 'x.tqe' ORDER BY
  COSINE(:q) LIMIT k WITH (RECALL >= 0.95[, CERTIFY])` executes it, planning
  operating points from the measured sweep — the declared target is the
  acceptance signal, per the coherence rule.
- **Hub anatomy + the anti-hub differential oracle
  (`turboquant_pro.anatomy`, `tqp anatomy`, `tqp hubdiff`).** Scalar hubness
  is non-identifying: two corpora can share a skewness reading produced by
  opposite mechanisms (density-driven tails vs. centrality super-hubs) with
  opposite ANN behaviour. `tqp anatomy` reports the identifying vector —
  count tail, correlations with centrality / local density / nearest-pair
  distance, hub-vs-population medians. `tqp hubdiff` compares an exact and a
  compressed (or any two) searches beyond aggregate recall: hub-rank
  correlation, hub-set Jaccard, 5th-percentile per-query recall, and
  **anti-hub recall** — the queries whose true neighbours are the
  least-visited rows, where compressed indexes fail first while mean recall
  stays green. Gate in CI with `--min-anti-recall` (exit 1). System-agnostic
  mode (`--exact/--approx` neighbour-id arrays) accepts output from any ANN
  system. See [`docs/HUBNESS_PRIMER.md`](docs/HUBNESS_PRIMER.md) for the
  newcomer's primer.
- **Resumable at-scale IVF builds.** `build_ivf` writes each shard as a
  self-contained, restartable block and `fleet_build` resumes at shard
  granularity, so 100-billion-row builds converge under preemption; IVF
  **source routing** persists cell→shard occupancy and scans only live shards.
  L2 support in the ADC scan unblocks the standard billion-scale benchmarks.
- **Agent tool surface (`turboquant_pro.agent_tools`).** JSON-in/JSON-out
  wrappers written for tool-calling models — `best_compression_at_recall`
  ("best ratio at a target recall"), `certify_ranking` (the distribution-free
  rank certificate), `recommend_kv_key_quantizer` (the (A2) key probe), and
  `list_tools` (a JSON-Schema manifest). All exported from the package root and
  delegating to the existing Beta certificate / (A2) / auto-compress paths. Each
  takes the **goal as a runtime argument** (target recall / consumer metric) and
  accepts against that goal — never reconstruction cosine. Ready-to-run
  LangChain, DSPy, MCP, and custom-GPT wrappers in `examples/agentic/`.
  Experimental (names/return schemas may change while the contract settles).

## 1.9.0 (2026-07-19)

> Minor release: **at-scale search + a smaller on-disk format**. Indexes now
> exceed RAM via memory-mapped, sharded, block-streamed search; format v3
> bit-packs codes for ~1.7× smaller `--no-originals` indexes; the (A2) probe
> gains an optional ZCA-whitened quantizer family; and `tqp certify` grows a
> richer (schema-compatible) certificate envelope. All additions are additive —
> v1/v2 indexes keep opening and the two-family (A2) verdict is byte-for-byte
> unchanged.

### Added
- **(A2) probe: optional ZCA-whitened quantizer family.** `probe_quotient(...,
  include_whitened=True)` and `recommend_key_quantizer(..., include_whitened=True)`
  add a third candidate — a per-vector polar code applied in a symmetric (ZCA)
  whitened basis — that removes the cross-channel correlation the per-channel
  (diagonal) code leaves. In the correlated direction-concentration regime
  (post-RoPE keys; anisotropic sentence embeddings) it can beat both polar and
  per-channel; the new `A2ProbeResult.spearman_whitened` field carries its rank
  agreement (`None` when not requested), and
  `TQPRuntimePolicy.evaluate_kv_keys(..., include_whitened=True)` surfaces the
  winner as the `WHITENED_KEYS` action. Off by default: the two-family verdict
  and its determinism are byte-for-byte unchanged (the whitened proxy uses a
  dedicated RNG). `tests/test_a2_probe.py` is now gated in CI.
- **Index format v3 — bit-packed codes; ~1.7× smaller `--no-originals` indexes.**
  The stored codes section packs sub-byte quantizer levels at slot granularity
  (2 codes/byte at 3–4 bits, 4/byte at 2-bit) instead of one byte per code, and
  arange-reconstructible ids + empty tombstones are elided from the file
  entirely (`ids_arange_start` in meta). IVF member sidecars shrink to
  `uint32`. A RAM open unpacks once (the AVX2 kernel and every mutation path
  see the same uint8/dim codes as before); a memmap open serves rows through a
  `PackedCodes` view that unpacks only the rows a probe gathers — so the
  packing also halves code I/O on the storage-bound search path. Packing is a
  lossless re-encoding: rankings are bit-identical to v2 (asserted by tests,
  not sampled). Measured with `bench_ivf_sharded.py` at 2M rows / 4-bit /
  `--no-originals`: **24.1 B/row all-in** (codes 12 + norms 8 + members 4) vs
  41 B/row for the same layout in v2. v1/v2 files keep opening; writers can
  pin `format_version=2` for old readers; `migrate(3)` upgrades in place.
- **Memory-mapped + sharded search — indexes larger than RAM (Phase 6 memmap).**
  `TQEIndex.open(path, mmap=True)` memory-maps the large arrays (codes, norms,
  originals, ids, tombstones) instead of loading them, and `search` gained a
  **blocked** path that streams the codes in row-blocks, so peak memory is
  `O(n_queries × block)` regardless of how many vectors the index holds — bounded
  RAM at any scale. A memmap-opened index is read/search only (mutations raise;
  reopen in RAM to modify). New `ShardedIndex` splits a corpus into `shard_size`-row
  shards that **share one PCA basis** (scores stay comparable) behind a JSON
  manifest; `search` fans out over the shards — each memory-mapped — and merges the
  per-shard top-k into a global top-k, so the fan-out parallelizes across cores or
  machines. CLI: `tqp index create --shard-size N` (writes a shard directory +
  manifest) and `tqp index search --mmap [--block B]` (a directory / `manifest.json`
  is searched as a shard set). Exported `ShardedIndex`; closes the last ◑ (memmap /
  shard) deliverable of the production-index phase.
- **Richer certificate envelope (optional, additive).** `tqp certify` gains an
  optional self-describing envelope that does **not** bump `schema_version` (it
  stays 1) and leaves base certificates unchanged: `--task "recall@10 >= 0.995"`
  records the declared downstream consumer, `--environment` stamps the run's
  provenance (tool / python / numpy / platform / git commit / hardware),
  `--limitation "…"` (repeatable) records scope caveats, and `--html report.html`
  writes a readable human report next to the machine-readable JSON. The shipped
  JSON Schema, `docs/CERTIFICATE_SPEC.md`, and the golden-fixture drift guard are
  extended to cover the new optional sections. Completes the Phase-2 certificate
  work.

## 1.8.1 (2026-07-17)

Patch release — **release metadata only, no API or behavior change** from 1.8.0.

### Fixed
- **PyPI long description.** The 1.8.0 sdist/wheel were built from the `v1.8.0`
  tag *before* the "released" documentation banners merged, so the PyPI project
  page rendered a stale description (still reading "Current package: 1.7.0" and
  "the `tqp` CLI … not yet on PyPI"). 1.8.1 rebuilds from the corrected README so
  the PyPI page, the repository README, and the release history all agree. First
  independent reproduction of `embedding_glove_recall` from the published wheel is
  recorded in `benchmarks/reproductions/`.

## 1.8.0 (2026-07-17)

> Released 2026-07-17 (git tag `v1.8.0` + GitHub Release + PyPI). This is the
> **certification-platform** release: the `tqp` CLI, the persisted-index and
> plugin certification lifecycle, and the portable Triton fused-decode kernels
> land here. `pip install turboquant-pro` now gives you `tqp`.

### Added
- **Certificate verification — `tqp verify`.** Re-check a `certificate.json`
  emitted by `tqp certify`: schema / self-consistency always (offline), plus an
  independent recompute when `--original`/`--reconstructed` are given (re-hash the
  inputs against the recorded `sha256`, re-run the certification with the
  certificate's own params, confirm the recorded floors reproduce within tolerance).
  Exit 0 verified / 1 malformed-or-mismatch / 2 read error. `certify` writes
  certificates; `verify` is how a third party trusts one.
- **Hardware & plugin widening (P0–P5).** Out-of-tree quantizer plugin protocol +
  entry-point registry + conformance kit; torch backend for reference paths;
  in-tree incubator plugins `tqp-bnb` (NF4 / LLM.int8 + QLoRA interop),
  `tqp-trtllm` (FP8 / NVFP4 KV), `tqp-gptq-awq`; `operator_trace` weight/KV
  recommendation. **P5 — a portable Triton port of the M2/M4 fused-decode
  kernels**: exact across Turing→Ampere→Ada→Hopper, its single-launch batched-page
  variant beats the CuPy RawKernel **1.3–9.4×** and stays exact to ~1e-7 through
  128k context (the CuPy RawKernel remains the CUDA reference/oracle). See
  `docs/DESIGN_hardware_and_plugins.md` and `benchmarks/RESULTS_p5_triton.md`.
- **Documentation hub + canonical guides (Phase 9).** [`docs/`](docs/) is now a
  front door: a rendered mermaid **architecture diagram** (artifact → trace → plan →
  compress → certify → replay → monitor, with the runtime-policy back-off loop), a
  curated map of every guide/spec/model-card, and the coherence rule stated up top.
  Six canonical guides under [`docs/guides/`](docs/guides/) (+ the existing
  `docs/PLUGINS.md`): user (15-minute quickstart), operator-aware quantization,
  certification (what a certificate means and does not), claim replay, and
  production lifecycle. The README links the hub and names the 15-minute reviewer
  path (README → docs hub → quickstart → certificate → replay). Closes Phase 9 —
  **all nine roadmap phases are now shipped.**
- **Runtime safe fallback (Phase 8).** `turboquant_pro.runtime_policy.TQPRuntimePolicy`
  turns the fragility measurements built across Phases 1–7 into one adaptive
  back-off decision layer: it reads a retrieval boundary gap, a rank certificate's
  vacuity, an unknown KV-key regime / the (A2) probe, a routing margin, an SSM
  slow-channel fraction, the (A2) tangential fraction, and index drift, and returns
  a `RuntimeDecision` (`conservative`, machine-readable `action`, `reason`,
  `measured`, `params`) — `rerank_more`, `require_exact_rerank`,
  `per_channel_or_fp16`, `keep_router_fp16`, `log_tau_or_fp16`,
  `recalibrate_or_disable_polar`, `refit_or_migrate`, or `proceed`. Each evaluator
  is a thin wrapper over the shipped instrument (`rank_certificate`, `a2_probe`,
  `operator_sensitivity`, `index.drift`); `evaluate_all(**inputs)` runs many at
  once. `TQEIndex.search(policy=...)` makes retrieval **adaptive** — single-pass by
  default, escalating to exact rerank only when the top-k boundary is tied (cheap
  where margins are wide, conservative where they are not). Exported:
  `TQPRuntimePolicy`, `RuntimeDecision`. Closes Phase 8 of the roadmap.
- **Real-model operator-sensitivity validation (Phase 7).** Three operator
  regimes are now validated on real model weights and promoted to
  `docs/model_cards/` + `claims.yaml` (track `operator`), each with committed
  reproduction scripts, raw data, and preserved negative cases:
  - **Attention keys** (`SOFTMAX_SCORE`) — real Llama-2-7B/13B, Mistral-7B,
    Qwen2.5-7B/1.5B perplexity + LongBench; PolarQuant collapse (Qwen2.5 ppl
    12.24 → 10643) and symmetric-NF4 GQA collapse (qasper 43.8 → 4.7) preserved
    (`docs/model_cards/attention_keys.md`).
  - **MoE routing** (`GATE_SELECTION`) — real **OLMoE-1B-7B** (64 experts,
    top-8): a controlled differential-logit perturbation at the margin scale
    flips low-margin tokens **~1740×** more than high-margin (top-8 set),
    **~1256×** at the argmax; naive 4-bit gate quant reshuffles 92% of top-8
    sets. `benchmarks/validate_olmoe_routing.py` + `results_olmoe_routing.json`,
    `docs/model_cards/moe_routing.md`.
  - **SSM decay** (`STATE_DECAY`) — real **Mamba-790m**: 3-bit linear
    quantization of the continuous decay collapses WikiText-2 ppl 11.65 →
    **1.01×10¹⁰**, while the native A_log (log-time-constant) basis keeps it at
    **14.44** (~7×10⁸ gap). `benchmarks/validate_mamba_decay.py` +
    `results_mamba_decay.json`, `docs/model_cards/ssm_decay.md`.

  Both runners feed real model data into the shipped
  `turboquant_pro.operator_sensitivity` primitives (`routing_sensitivity`,
  `state_decay_sensitivity`, `quantize_decay`). Acceptance is the consumer metric
  (perplexity / expert-set flip rate), never reconstruction cosine. Closes
  Phase 7 of the roadmap.
- **Production vector-index lifecycle (Phase 6).** `turboquant_pro.index.TQEIndex`
  and the `tqp index` command group make Track 1 production-grade: a persisted,
  compressed-domain ADC search index with `create / add / delete / compact /
  migrate / search / certify / drift / info`. It lives in the new **TQIX**
  container (`turboquant_pro.index_file`) — a versioned, section-based binary
  format where every section is **CRC32-checked** (a flipped byte is a clean
  `IndexCorruptionError`, never silent bad data) and writes are atomic
  (temp-file + `os.replace`). Highlights: append with no refit (ids stay stable
  and comparable); O(1) tombstone `delete` with byte-reclaiming `compact` that
  preserves external ids; format `migrate` (v1 positional ids → v2 explicit ids +
  tombstone bitmap); metric-correct exact **rerank**; a `certify` that emits a
  distribution-free rank certificate over stored originals (acceptance is the
  Kendall-τ floor, never reconstruction cosine); and PCA-basis **drift** detection
  (retained-variance drop + mean shift). Covered by an end-to-end lifecycle test
  (ingest→search→update→compact→migrate→certify→monitor), a container
  corruption-detection suite, and a single-byte-flip fuzzer whose invariant is
  "detected, or byte-identical behaviour — never silently corrupt." Exported:
  `TQEIndex`, `DriftReport`, `index_info`, `IndexCorruptionError`. Closes Phase 6
  of the roadmap.
- **Plugin ecosystem proven out-of-tree (Phase 5 exit criterion met).** The first
  genuine external plugin, [`tqp-reference-plugin`](https://github.com/ahb-sjsu/tqp-reference-plugin)
  (pure NumPy, its own repo), closes the open Phase-5 proof: installed into a fresh
  environment alongside turboquant-pro, it is discovered **only** through the
  `turboquant_pro.plugins` entry point (no import from this tree), passes
  `tqp plugin conformance` on every applicable check — roundtrip, packed, **affine**
  (the fused-decode gate), and serialization — and its reconstruction is certified by
  the same rank certificate `tqp certify` uses. The in-tree `plugins/tqp-bnb`,
  `tqp-gptq-awq`, `tqp-trtllm` remain incubators that dogfood the contract. With the
  gating condition met, the `plugins` / `plugin_conformance` API is now eligible to
  promote from Experimental to Beta at the next stability review (see
  `docs/api-stability.md` and Phase 5 of the roadmap).
- **Canonical GloVe recall claim is now executable and CI-gated.** The Track-1
  embedding result on real public data is reproduced by one command:
  `tqp replay embedding_glove_recall` runs `benchmarks/canonical_glove.py`
  (full-dimension PCA rotation + 3-bit TurboQuant → compressed-domain ADC search
  + exact rerank) and gates on **reranked recall@10** and compression — the
  metrics retrieval consumes, never reconstruction cosine. Measured on the full
  1.18M-vector glove-100-angular: **~9.6x compression at recall@10 ≈ 0.999**
  (single-pass ADC recall is only ~0.74 at the same ~0.98 cosine — rerank on the
  retained originals is what recovers retrieval). CI runs the hermetic `--small`
  path against a tiny **real** GloVe subset bundled at
  `benchmarks/fixtures/glove_tiny.npz` (numpy + core only, no network/GPU);
  `--full` is the Atlas/local 1.18M run. Regenerate the fixture with
  `benchmarks/make_glove_fixture.py`. The prior `embedding_27x_high_recall`
  claim is retained as the fuller notebook headline (where PCA truncation pays
  off on concentrated-spectrum encoders) and cross-referenced.
- **Hard schema under the rank certificate (Phase 2 hardening).** `tqp certify`'s
  JSON artifact is now drift-proof: a shipped, versioned JSON Schema
  (`turboquant_pro/schemas/rank_certificate.schema.json`, Draft 2020-12,
  importable via `importlib.resources`), a committed golden fixture regenerated
  and compared in CI, and a documented compatibility promise
  (`docs/CERTIFICATE_SPEC.md`). Every `tqp` JSON artifact is also now
  **spec-valid**: non-finite measurements (e.g. NaN κ on a degenerate corpus)
  serialize as JSON `null` instead of bare `NaN`, with `allow_nan=False` as a
  hard guard — so certificates never emit invalid JSON. `jsonschema` added to the
  `[dev]` extra.
- **Hopper fp8 compute measured (H100)** — boundary found: per-head
  _scaled_mm loops are launch-bound and lose to fp16 at every context
  (1.7–26 vs 41–122 TF/s); fp8 COMPUTE needs FA3-class fusion, while the
  fp8 STORAGE passthrough (half KV bytes, validated Ada+Hopper) remains
  the win at this layer. RESULTS_hopper_fp8_compute.md.
- **resolve_plugins** (P4 exit): model in, named recipe out —
  operator_trace disciplines resolved to registered plugin names per
  tensor and per target. On a Llama-shaped module: k_proj under
  kv_activation → per_channel (protect_dc), q_proj under weight →
  {gptq, awq, bnb_nf4}. Discipline from measurement, names from the
  registry, nothing guessed.
- **Native fp8 KV passthrough** (P3; tqp_trtllm.native.FP8NativeKV):
  keys/values stored as real torch.float8_e4m3fn tensors (half the
  bytes of fp16) with per-head fp32 scale, upcast on read — the
  hardware milestone of fp8_kv, honestly scoped to storage (fp8
  compute is Hopper/FA3 territory). The code-space plugin is the
  semantics oracle: torch RNE cast agrees with the 253-value grid to
  1e-6 on real-shaped keys. CUDA-gated device test for the Ada leg.
- **Keys comparison run — prediction refuted, boundary found**
  (P3; plugins/tqp-trtllm/examples/): on real Llama-3.2-3B keys, fp8
  (both scale modes, 8-bit float grid) is near-lossless — the
  pre-registered scale-granularity fragility does NOT apply to
  floating-point grids (every e4m3 value carries its own exponent); it
  is a fixed-point, matched-bit phenomenon. At matched 4 bits,
  per-channel asym-NF4 beats nvfp4 block-16 by ~2.2x attention KL.
  Reported as found; RESULTS_keys_comparison.md.
- **tqp-trtllm plugin incubator** (P3 milestone 1): fp8_kv (per-head
  e4m3, 253-entry grid — table verified against ml_dtypes bit-for-bit)
  and nvfp4_kv (block-16 e2m1, block-granular (H, S, D) weight) in CODE
  SPACE — correctness on any CPU, per design doc 4.3; both pass
  conformance with affine PASS, so both are fused-decode-eligible.
  fp8_kv declares native_dtype float8_e4m3fn for the Ada/Hopper
  passthrough milestone. Substrate for the pre-registered fp8-vs-nvfp4
  keys comparison.
- **QLoRA interop demo** (P2 exit criterion;
  `plugins/tqp-bnb/examples/qlora_interop.py`, run on NRP L40S): a
  bnb-4bit NF4 double-quant Llama-3.2-3B with real activations, per-layer
  post-RoPE KV captured into `TurboQuantKVCache` — attention on the
  compressed cache matches exact-KV attention at mean KL 2e-05
  (layers 0/13/27). Same job ran the tqp-bnb suite with bnb+CUDA present:
  10/10 incl. the `bnb.functional` cross-check. bnb weights + tqp cache
  compose; results in `examples/RESULTS_qlora_interop.md`.
- **`bnb_llm_int8` plugin** (P2): LLM.int8 mixed decomposition on the KV
  block convention — per-channel int8 absmax dense part + fp16 outlier
  *channels* (Dettmers-style emergent features) surfaced through the
  contract's `outlier_csr` as dense per-token columns. Conformance:
  affine **pass** and CSR **pass** — LLM.int8's decomposition is the
  dense-sparse overlay at column granularity, so the format is fully
  fused-decode-eligible with no kernel changes.
- **Block-granular affine contract** (P2 milestone 2, design doc §6):
  `grid_params` weight may now be token-block-granular `(H, S, D)` as
  well as per-channel `(H, D)`; conformance broadcasts both. `bnb_nf4`
  implements it — its conformance `affine` check now **passes** (block
  absmax expanded per element reproduces `decompress` exactly), so the
  bitsandbytes NF4 format is fused-decode-eligible; kernel-side compact
  block weights are follow-up work.
- **`tqp-bnb` plugin incubator** (P2 milestone 1, `plugins/tqp-bnb/`):
  the first external consumer of the P0 contract — bitsandbytes blockwise
  NF4 (QLoRA semantics: per-block absmax over the fixed NF4 table) as a
  separately-installable package registering via the
  `turboquant_pro.plugins` entry point. NumPy reference implementation;
  cross-check vs `bnb.functional` runs where bitsandbytes+CUDA exist.
  Passes conformance with `affine: skip` — blockwise scales vary along
  the token axis, so fused decode awaits the block-granular contract
  extension (milestone 2, design doc §6).
- **Torch backend, milestone 1** (P1 of `docs/DESIGN_hardware_and_plugins.md`):
  `turboquant_pro.backend` with `to_numpy` — the boundary adapter that lets
  every instrument (rank certificate, (A2) probe) accept torch tensors from
  **any device** (CUDA/ROCm/MPS/XPU) and CuPy arrays, with identical numbers
  to NumPy input — and `torch_decode`, a **torch attention reference after
  host-side reconstruction** (keys/values reconstructed on the host via the
  cache's public getters, then the attention math runs on-device in torch on
  any device; every key format supported; matches `fused_decode` to float
  tolerance). It is a correctness/portability reference, *not* a zero-copy or
  fused on-device dequant. Torch stays an optional, lazily-imported dependency
  (`pip install turboquant-pro[torch]`). Device-parametrized
  tests activate automatically on CUDA/MPS hosts. P1 slice 2: `backend.torch_xp` — a NumPy-signature shim (amax/keepdims
  naming, uint8-code promotion for torch's mask-indexing semantics) that
  runs the `kv_fused` / `kv_fused_pck` **reference einsum paths on any
  torch device** via the existing `xp=` seam, matching NumPy to 5e-5;
  suite-leg run green on NRP L40S (torch-CUDA legs all pass; issue #123
  filed for two pre-existing GPU-present/cupy-absent failures). Remaining
  for P1: MPS numbers (needs an Apple host); A100 leg deferred.
- **Quantizer plugin protocol + registry + conformance kit** (P0 of
  `docs/DESIGN_hardware_and_plugins.md`): `turboquant_pro.plugins` defines the
  minimal `Quantizer` protocol (compress/decompress — enough for every
  instrument), the optional **affine capability** (`grid_params`/`codes`/
  `outlier_csr` — a format that exposes it inherits the M4 fused decode from
  the existing kernel), `native_dtype` for hardware passthrough, and a
  `PluginSpec` registry discoverable via the `turboquant_pro.plugins`
  entry-point group so packages like `tqp-bnb`/`tqp-trtllm` register without
  touching this tree. `turboquant_pro.plugin_conformance` is the executable
  side of the contract (round-trip, packed equivalence, affine-reconstruction
  == decompress, CSR validity, serialization) reported as
  pass/skip/FAIL-with-detail. Dogfood: in-tree formats are registered through
  the same interface (`"per_channel"` with the full affine surface, `"polar"`
  as the non-affine degrade) and pass the same suite an external plugin would
  run. Author guide: `docs/PLUGINS.md`. Top-level exports:
  `available_plugins`, `create_quantizer`, `register_plugin`,
  `run_conformance`, `assert_conformance`.
- **`tqp` CLI** (Phases 1–4 of `docs/turboquant_pro_next_level_roadmap.md`): a
  single console entry point over the full `trace → plan → compress → certify →
  replay → monitor` pipeline — `version`; `plugin list` / `plugin conformance`
  (conformance kit over registered plugins on a canonical KV block or `--shape`
  sample); `trace <hf-model>` (operator-regime → (A2) discipline distributions,
  built on the **meta device** so a 7B model needs no download/RAM); `probe`
  (the (A2) consumer-metric quantizer-family probe on a `.npy` batch or a
  labeled `--demo` — the calibration-time check for the v1.2.0 KV-keys class);
  `plan embeddings` (auto_compress + Pareto frontier + a **rank-certificate
  preview**) and `plan kv` (`AutoConfig` key/value policy + risk flags);
  `certify` (a **distribution-free rank certificate** as a provenance-stamped
  `certificate.json`, gating on a positive or `--min-tau` floor); `replay`
  (executes claim reproductions from `claims.yaml` through a shared harness,
  checks `results.json` against `expected` ranges, gates the exit code); and
  `monitor` (`QualityMonitor` metrics as JSON / Prometheus / text). **Coherence
  rule:** every command's acceptance signal is rank fidelity / the (A2) consumer
  metric / a distribution-free certificate — never reconstruction cosine on its
  own; cosine appears only as a labelled secondary diagnostic, and where it is
  the base signal (`monitor`) it is guarded by the (A2) tangential-fraction /
  radial-drift statistics. Pure `argparse` — the core install stays numpy-only;
  `trace` needs `[torch]` + transformers and `replay` needs `[yaml]`, both
  imported lazily. Ships `claims.yaml` (the reproduction ledger) and
  `benchmarks/replay_smoke.py` (a CPU/seconds Track-1 recall@10 reproduction:
  ≥0.80 at >10× compression, the metric retrieval consumes — not cosine).
  Registered as the `tqp` script; `turboquant-pro` remains the AutoConfig entry
  point. Guide: `docs/CLI.md`. Covered by `tests/test_cli.py` (65 tests:
  dispatch, exit-code contracts, output content, arg parsing,
  plugin/probe/plan/certify/replay error paths, the probe's KV-keys
  recommendation, the plan's rank-over-cosine acceptance, certificate provenance
  + gating, replay verdicts, and meta-device `trace`).
- **M4 cache dispatch** — `TurboQuantKVCache.fused_decode` now routes
  per-channel key pages through the fused compute-on-codes path (previously
  decompress-then-attend). Each cold page gets a `PreparedPCKBlock`
  (`kv_fused_pck`) built **once per flush** — key codes unpacked, grid
  parameters and the token-major outlier CSR built, value codes/norms staged,
  device-resident on GPU — and reused every decode step; per call only the
  O(H·d) query projections remain. Exact vs decompress-then-attend across all
  zero-point modes, outlier fractions, and multi-page + hot-window merges
  (`tests/test_kv_fused_pck.py::TestCacheDispatch`). nuq grids, structured
  rotations, and off-envelope head dims fall back to reconstruction
  automatically. Measured on GV100 (H=8, d=128, NF4+2% outliers): steady-state
  decode **2.0× @2k / 5.6× @8k / 12.5× @32k** over decompress-then-attend,
  exact to ≤6e-8. See `docs/DESIGN_fused_kv_decode.md` §8.5.
- **`ModelCompressor.quantize_weights(rope_aware_k=True)`** — weight PTQ
  (uniform symmetric per-output-channel absmax over FFN + attention Linears)
  with the §2.3 mechanism of
  `docs/notes/projection_sensitivity_deconfounded.md` wired in as the
  default: the long-wavelength (DC) RoPE rows of `W^K` — read from the
  model's own `inv_freq` buffer (rope_scaling included), `rope_theta`
  fallback — are kept full-precision (`k_protect_bits=8` for mixed-precision
  instead). Measured basis: at 3-bit on Llama-3.2-3B, protecting the longest
  octile (12.5% of K rows, ~0.4% of attention weights) recovers 87% of the
  functional damage; the coupling replicates at 25× smaller amplitude on
  Mistral-7B. Helpers `rope_protected_rows` / `quantize_weight_rows`
  exported from `model_compress`; graceful degrade (warning, no protection)
  for non-rotary models. Incident → instrument, weight-space edition.
  **Validated end-to-end** through the shipped path
  (`experiments/validate_rope_aware_k.py`, run as an NRP batch Job on an
  L40S — cross-hardware vs the GV100 the mechanism was measured on):
  K-only 3-bit out_kl 0.0994 protected vs 0.7352 unprotected — **86.5%
  recovery against the pre-registered ~87%**, matching the probe to the
  third decimal. `k_protect_bits=8` is indistinguishable from
  full-precision protection (the fix costs ~0.16 bits/weight averaged over
  Q/K/V/O). Honest boundary: at whole-model 4-bit the option buys only
  ~2.5% (V/O dominate there, per the note's §2.2) — `rope_aware_k` earns
  its keep at ≤3-bit K.
  Results: `experiments/results_matched_bit/validate_rope_aware_k_Llama-3.2-3B.json`.
- **Hadamard rotation (opt-in).** `TurboQuantPGVector(..., rotation="hadamard")`
  and `PCAMatryoshka.with_quantizer(..., rotation="hadamard")` select a randomized
  Fast Walsh-Hadamard rotation — orthogonal, applied in `O(dim log dim)` instead of
  materializing and multiplying a `dim × dim` matrix. Requires a power-of-two `dim`
  (raises a clear `ValueError` otherwise). The default remains `rotation="qr"`, which
  is exactly the previous behavior. `ADCIndex` reproduces the exact reconstruct-cosine
  under the Hadamard rotation (verified to `< 2e-4`).
- **TQE format v2.** A self-describing `rotation` byte is recorded so a reader
  reconstructs the exact rotation with no out-of-band metadata. Writers emit v2 **only**
  for non-default rotations; the default `"qr"` still serializes as the byte-identical
  20-byte v1 record, so existing data and readers are unaffected. `decompress_embedding`
  now guards against a rotation mismatch instead of silently mis-decoding. Spec:
  [`docs/FORMAT_SPEC.md`](docs/FORMAT_SPEC.md).

### Changed
- **Release-surface docs de-drifted (v1.8.0 coherence-release prep).** The
  phased roadmap now carries **live status** — Phases 1 (the `tqp` CLI) and 4
  (the planner) are marked shipped-on-master, 2 (certificate schema) and 3 (claim
  replay) partial with the open hardening named, and Phase 5 explicitly *not*
  met because `tqp-bnb` is an **in-tree incubator**, not an external plugin — so
  the repo no longer reads as less finished than it is. `docs/api-stability.md`
  now lists all runtime extras (`gpu`/`faiss`/`pgvector`/`nats`/`vllm`/`fast`/
  `torch`/`yaml`) with tiers, and states the plugin API stays Experimental until
  a true out-of-tree plugin passes conformance.
- **Docs + `tqp plan embeddings` de-cosined for coherence.** A docs audit found
  the last acceptance-as-cosine framing outside the monitor: `tqp plan
  embeddings` defaulted `--target` to `cosine > 0.95` and its `plan.json` note
  called the cosine/ratio search "a library limitation" — both stale now that
  `auto_compress` measures real `recall@k`. The default is now
  `recall@10 >= 0.90`, the note explains the frontier ranks on the target metric
  (measured recall when it is a recall target), and `docs/CLI.md` was corrected
  (monitor exit code gates on `is_healthy` = cosine floor **and** (A2)
  noncollapse; plan example leads with recall). Also fixed a false
  cosine⇒ranking equivalence in the Atlas expansion plan. The remaining cosine
  mentions across the docs are legitimate diagnostics (benchmark columns, per-bit
  fidelity, the "cosine is not a recall proxy" warnings) and were left as-is.
- **Quality monitor health now incorporates the (A2) signal, not cosine alone.**
  `QualityMonitor.is_healthy` (and the alert, and the `tqp monitor` exit code) had
  gated purely on `mean_cosine >= quality_floor`, while the module *computed* the
  (A2) tangential-fraction stream it says catches "ranking damage cosine cannot
  see" — and never used it in the verdict. Health now also requires (A2)
  noncollapse: a self-calibrating guard trips on a significant *downward* drift of
  the tangential stream (norm-dominated regime), and an opt-in `tangential_floor`
  (constructor + `tqp monitor --tangential-floor`) adds a hard level gate. No-op
  when the statistic is disabled (`tangential_reservoir=0`) or too sparse. This
  closes the last acceptance path that gated on reconstruction cosine, so every
  `tqp` component now shares one coherent accept signal (rank fidelity / (A2) /
  distribution-free certificate).

### Fixed (Tier-0 soundness audit)
- **GPU fused rotate+quantize produced garbage codes.** The 2/3/4-bit fused
  kernels cached the wrong operand (each thread stored its own output column of
  `Pi_T` but read it as if indexed by the contraction variable), so every output
  column got the same bogus diagonal sum. Rewritten to tile the input row into
  shared memory and contract against `Pi_T[:, out_col]` correctly; GPU batch
  compression now matches the CPU path exactly (`test_cuda_compress.py` 25/25 on
  a GV100, was 18 failing). This was the default GPU compress path; the bug hid
  because the equality test needs CuPy + a GPU, absent from CI.
- **Acceptance metrics no longer alias reconstruction cosine.** `auto_compress`
  answered a `recall@k` target with mean cosine (`# approximate`) — the exact
  blind-metric failure the project warns about. Recall targets now measure *true*
  recall@k (exact vs. reconstructed rankings, via `compute_recall_at_k`), the
  Pareto frontier and unmet-target fallback rank on the target's own axis, and an
  unmeasured metric raises instead of silently substituting cosine.
- **TQE format seed decoupled from its record → silent wrong decode.** `pack`
  took `seed` as a free argument defaulting to 42, unrelated to the embedding it
  serialized, so `pack(ce)` could stamp a seed that rebuilds a *different*
  rotation/codebook on decode. `CompressedEmbedding` now carries `seed`, `pack`
  reads `ce.seed` by default, `unpack` returns it on the object, and both reject
  `bits ∉ {2,3,4}`.
- **`ADCIndex.add()` silently dropped earlier batches.** It assigned rather than
  appended, so `index.add(a).add(b)` kept only `b` despite the accumulating name
  and fluent return. Now concatenates; incremental indexing reproduces one-shot
  codes exactly.
- **`backend.to_numpy` crashed on bfloat16.** NumPy has no bf16 dtype, so
  `.numpy()` raised `TypeError`; now falls back to a lossless float32 upcast.
- **Activation-space weight compression crashed on non-square layers.** The basis
  spans a layer's *output* space, but the code computed `W @ V^T`, which requires
  `in == out` — i.e. it never worked on the FFN matrices it targets. Now projects
  output rows (`Vk^T (Vk W)`), skips mismatched bases, and its docstring/log no
  longer imply a size/latency reduction the dense writeback does not deliver. The
  rank-ratio denominator was also corrected from `max(shape)` to `min(shape)`
  (effective rank is bounded by the smaller dimension), which had over-recommended
  compression on non-square weights.
- **Plugin conformance tightened.** The affine check — an *algebraic* equality
  that gates fused-decode safety — used a flat `5e-3` absolute tolerance loose
  enough to pass a subtly-wrong grid mapping; it is now scale-relative
  (`max(1e-4, 1e-5·peak)`). The packed check no longer passes vacuously when a
  plugin silently ignores `packed=True` (it must produce a container that reports
  itself packed or serializes distinctly).
- **`rank_certificate` over-claim removed.** The docstrings said "no
  distributional assumptions whatsoever"; the default robust `measure_kappa`
  (2.5/97.5 trim) is *conditional* on trimming ~5% and holds for the central 95%,
  not unconditionally. Documented the strict-vs-robust regimes honestly.
- **`ADCIndex` silent mis-scoring under a whitened PCA.** When built from a
  `whiten=True` pipeline the database projection was whitened while the query terms and
  reconstruction norm were not, so scores were wrong. The scorer now restores the
  per-component `sqrt(eigenvalue)` factor and is exact for both `whiten` settings.
  `whiten=False` remains the recommended operating point for retrieval (whitening
  equalizes PCA modes and lowers recall).

### Changed (review-response hygiene)
- **Packaging:** added the missing `torch` optional extra
  (`pip install turboquant-pro[torch]`) that `operator_trace`'s error message and
  the backend docs already referenced; also included in `[all]`.
- **`torch_decode` documented precisely** as a *torch attention reference after
  host-side reconstruction* — the dequant is host NumPy and only the attention
  math runs on-device; it is not a zero-copy / fused on-device decode.
- **`operator_trace` routing wording** sharpened: `GATE_SELECTION` reads the
  relative order/margins of gate logits (common-mode shift is free; per-expert
  scale/offset error moves margins and flips selection), removing the apparent
  "margin vs magnitude" contradiction with the `operator_sensitivity` note.
- **Conformance contract** reconciled: `docs/DESIGN_hardware_and_plugins.md` now
  matches `plugin_conformance.py` — the core kit is the container contract only;
  instrument smoke (certificate/(A2) probe) is explicitly a separate
  corpus-shaped `instrument_conformance` concern.
- **Docs de-drifted:** `CODE_QUALITY.md` / `COMPREHENSIVE_ANALYSIS.md` stopped
  asserting a stale "397 tests" and checked off the now-shipped versioned format
  spec (`docs/FORMAT_SPEC.md`); the exact suite size is a `pytest --co` command,
  not a pinned number.
- **Release hygiene:** cut the **v1.7.0 GitHub Release** (marked *Latest*) from
  its existing tag — previously only tags v1.5.1/v1.6.0/v1.7.0 existed and the
  Releases UI still badged v1.4.3 as latest.

## 1.7.0 — 2026-07-15

Operator-dependent quantization for hybrid / state-space architectures: the
(A2) sensitivity analysis for the two regimes beyond attention — MoE routing
and SSM state decay. Every boundary below is *measured* on synthetic operators
before it is acted on (the formal derivations belong to the theory paper);
where the data said MSE was already optimal (consumer-weighted codebooks) we
reported no win rather than manufacture one.

### Added
- **`operator_sensitivity`** — the (A2) analysis for `GATE_SELECTION` and
  `STATE_DECAY` (the regimes `operator_trace` added for MoE/SSM):
  - **Routing (gates):** selection is carried by the **margin**, not the
    magnitude. `routing_margins` (top-k boundary gap), `differential_fraction`
    (the routing analog of `tangential_fraction` — common-mode logit error is
    free, only the differential component flips selection), and
    `routing_sensitivity` / `predict_routing_flips`. Measured: at 4-bit gate
    quantization, low-margin tokens flip ~88× more than high-margin ones.
  - **Recurrences (SSM decay):** the **slow (long-memory) channels are
    fragile** and the error **compounds over the sequence** — the recurrent
    analog of the RoPE-slow-channel key finding. `decay_gain` (`1/(1-a)`),
    `decay_time_constant`, `decay_sensitivity` (grows toward `a→1` and with
    seq-len), and `quantize_decay(basis="log_tau")` — quantizing the decay in
    the log-time-constant basis cuts state drift **5–6× vs linear at matched
    bits**, the SSM analog of NF4-for-keys. Measured: a fixed decay error is
    amplified ~28–52× more in slow than fast channels.
  - Write-up + honest scope: `docs/notes/operator_sensitivity_ssm_moe.md`.

## 1.6.0 — 2026-07-15

Toward human-out-of-the-loop, operator-dependent quantization: the (A2)
consumer no longer has to be *declared* — it is *inferred* from the model.

### Added
- **`operator_trace`** — an operator-regime classifier that maps every
  parameter tensor to the operator its output flows into (`SOFTMAX_SCORE`,
  `LINEAR_RESIDUAL`, `GATE_SELECTION`, `STATE_DECAY`, `NORM`) and, from that
  regime, to its (A2) quantization discipline — turning the manual "keys feed
  softmax, V/O write the residual" reasoning into a pass. Two combined
  front-ends: a **structural** classifier (module type + name; always
  available) and a best-effort **torch.fx** graph pass that backtraces the
  *sink* operators (softmax / topk / cumsum-scan) to the `Linear` layers that
  feed them — so it tags the score/gate/state tensors even when the names are
  obfuscated, the capability needed on an *unseen* architecture. The
  regime→discipline table encodes the operator-dependent flip: the same
  attention projection's **weights** are the robust, symmetric side under
  weight PTQ while its **cached keys** are the fragile per-channel+DC side
  under activation quant (`QuantTarget.WEIGHT` vs `KV_ACTIVATION`).
  `trace_operators` / `recommend_quantization` are the entry points; unknown
  tensors default to the conservative per-channel+zero-point discipline.
  `GATE_SELECTION`/`STATE_DECAY` regimes seed the SSM/MoE (item 3) work.
  Write-up: `docs/notes/operator_trace.md`.

## 1.5.1 — 2026-07-15

Theory-to-practice round: the companion paper's v0.8 results
([the-angular-observer](https://github.com/ahb-sjsu/the-angular-observer))
land as instruments, following the house pattern of turning each incident
into an instrument — now joined by a behavioral-equivalency metric answering
*The Illusion of Equivalency* (Rababah et al., arXiv:2607.08734).

(1.5.0 was an incomplete release built from a partial tree and is superseded/
yanked; 1.5.1 is the first release carrying the full tree below.)

### Added
- **`behavioral_agreement`** — a decision-level quantization-quality metric
  answering the *illusion of equivalency*: aggregate accuracy can be
  preserved while individual decisions churn. Ships `flip_rate` (symmetric
  McNemar split — regressions **and** recoveries), prediction-level
  `behavioral_agreement`, and a `noise_floor` control (churn between two
  near-lossless requantizations) so drift is reported as *excess over floor*
  with a z-score — fixing Correctness Agreement's two defects (bounded by
  `min(acc)`, blind to recoveries; no noise-floor control). scipy-free.
  The de-confounded companion experiment
  (`experiments/matched_bit_projection_sensitivity.py`) shows that at matched
  bits, weight-space drift is equal across Q/K/V/O while **V/O are
  functionally 2.3–6× more quantization-sensitive than Q/K** across three
  architectures (Qwen2.5-1.5B, Gemma-3-4B, and the paper's own Mistral-7B) —
  reversing the paper's ranking. Write-up:
  `docs/notes/projection_sensitivity_deconfounded.md`.
- **`rank_certificate`** — distribution-free rank-agreement floors for
  compressed retrieval (a shipped theorem): measured robust distortion κ,
  one-pass corpus concentration μ̂(κ), guaranteed floors Kendall τ ≥ 1−2μ̂ and
  Spearman ≥ 1−3μ̂ (Daniels), `max_certifiable_kappa` as the per-corpus
  vacuity threshold. `autotune` results now carry `kappa` / `mu_hat` /
  `tau_floor` / `rank_certified` per operating point — a vacuous certificate
  is the principled "exact reranking required" signal, derived per corpus
  instead of the fixed 2×/5×/10× menu.
- **`a2_probe`** — the (A2) consumer-metric probe: calibration-time
  quantizer-family selection against the declared consumer (cosine ranking,
  L2 ranking, attention logits). Reproduces the v1.2.0 KV-keys catastrophe
  on synthetic key statistics as a unit test; `recommend_key_quantizer`
  makes the keys→per-channel decision an instrument, not a memory. Also
  distinguishes and documents the two failure classes: radial-displacement
  (norm-dominated drift, guarded by the streaming statistic) vs
  direction-concentration (the actual keys regime, guarded by the
  end-to-end probe and the `median_unit_displacement` statistic).
- **`QualityMonitor`**: streaming (A2) tangential-fraction statistic over a
  reservoir of recorded originals, with `check_radial_drift()` (scipy-free
  KS), new `stats()` keys `median_tangential_fraction` /
  `radial_drift_detected`, and matching Prometheus gauges.
- `docs/KV_KEYS_FINDING.md`: "The general principle: condition (A2)"
  section, cross-cited with the theory paper (which cites this finding back
  as the scope boundary of the polar move).
- `docs/claims.md`: two new L5 rows (rank certificates; A2 probe/monitor).
- **Two pre-registered theory experiments** (CPU, deterministic, results
  committed): `benchmarks/heat_taper_experiment.py` — at matched total bits,
  an exponentially tapered bit schedule beats hard truncation on single-stage
  recall at constrained budgets (+1.6 pts at 512/768 bits) and certifies a
  non-vacuous tau floor at half the budget hard truncation needs, while raw
  pairwise Spearman marginally favors hard truncation (the boundary,
  documented) — `RESULTS_heat_taper.md`. `benchmarks/hubness_local_scaling.py`
  — a per-vector density quotient on ADCIndex candidates reproduces the
  theory paper's density dissociation in retrieval: +15.7 recall points
  (beating exact rerank) when the anisotropic mean is nuisance, −28 points
  when it is part of the truth — quotient density only when density is
  nuisance; `RESULTS_hubness_local_scaling.md`.
- **RoPE-offset frequency structure measured** (`benchmarks/
  rope_offset_frequency.py`, run on the atlas GPU host; `RESULTS_rope_offsets.md`):
  the per-channel DC offset that asym-NF4's zero-point absorbs is
  RoPE-frequency-structured — |offset| vs rotary wavelength at pooled Spearman
  0.91–0.98 across Qwen2.5-1.5B/7B/14B and Mistral-7B, with 96–99% of offset
  mass in channels whose wavelength exceeds the context window. Closes the
  GQA-collapse causal chain (structural DC + codebook waste + GQA error
  amplification) and motivates a deterministic, calibration-free
  frequency-derived zero-point. Folded into the TMLR draft
  (`paper/kv_tmlr/main.tex`, "Where the offset lives"). Includes a recorded
  config gotcha: transformers 5.x can hide `rope_theta` in the
  rope-parameters dict; the corrected read is in the script.
- **Deterministic zero-point validated** (`benchmarks/deterministic_zeropoint.py`,
  GPU host; results appended to `RESULTS_rope_offsets.md`): a zero-point
  computed purely from weights + config (k_proj bias through the
  position-averaged rotation) matches calibrated asym-NF4 on WikiText-2
  perplexity (12.533/9.179 vs 12.534/9.155 on Qwen2.5-1.5B/7B; fp16
  12.234/9.057; symmetric NF4 collapses to 30.96/241.3) — zero calibration
  data, zero stored per-channel metadata. Sparse zero-points on the
  config-identified DC channels alone slightly beat dense calibration at
  ~33% less metadata. Bias route is Qwen-family-specific (Mistral has no
  k_proj bias); folded into the TMLR draft's mechanism section.
- **LongBench confirmation of the deterministic zero-point** (GPU host, full
  200-sample qasper, published protocol, `tq_zp_lb_shard.py`): calibrated
  asym-NF4 anchor 42.35 (published 41.91 — harness parity), config-sparse
  zero-point 42.66, bias-derived (zero-calibration) zero-point **43.36**
  (fp16 43.77; symmetric NF4 4.69). Both calibration-lean variants beat
  dense calibration at the task level; folded into RESULTS_rope_offsets.md
  and the TMLR draft.
- **`PerChannelKV` zero-point modes shipped** (`zero_point=` "calibrated" |
  "sparse" | "bias"): the LongBench-validated calibration-lean zero-points
  are now library features, not just benchmark results. "bias" derives the
  zero-point from the model's k_proj bias through the position-averaged
  rotation — zero calibration data AND zero stored zero-point metadata
  (recomputed at decode; container carries zp_mode/rope_theta/position_start
  with backward-compatible defaults). "sparse" stores calibrated means only
  for the config-identified DC channels (~1/3 less zero-point metadata).
  Plumbed through `TurboQuantKVCache` (`key_zero_point` / `key_rope_theta` /
  `key_k_bias`), with cold flushes passing their absolute start position.
  New helpers `PerChannelKV.rope_averaged_bias` / `dc_channel_mask`.
  18 new tests (`tests/test_zero_point_modes.py`), synthetic ground truth
  built with an independent rotation implementation.
- **`AutoConfig`: "bias" zero-point is the Qwen-family default.** New
  `key_zero_point="auto"` field resolves to `"bias"` for Qwen-family models
  when the layer's k_proj bias is supplied to `build_cache(k_bias=...)`
  (and degrades gracefully to `"calibrated"` with an info log when it is
  not; explicit `"bias"` without a bias raises). New duck-typed helper
  `AutoConfig.extract_k_biases(model)` collects per-layer biases from an
  HF-style model without requiring torch. Non-Qwen families are unchanged.
- **Public-data replications of the two theory experiments** (BGE-M3 over
  WikiText-2, public model + text; `embed_wikitext_bge.py` regenerates the
  corpus). Heat-taper: the synthetic result does NOT replicate -- hard
  truncation wins every metric at every budget on the real spectrum, and the
  boundary is quantified (taper wins iff spectral effective rank << bit-head
  width: synthetic eff-rank 13.6 vs BGE 136.7) -- `heat_taper_public.py`,
  documented per the pre-registration in RESULTS_heat_taper.md. Hubness: the
  density dissociation CONFIRMS out-of-sample -- plain quotient -5.9 recall
  pts (observed-truth regime, as predicted before the run), while correcting
  only the compression-induced density shift (mu_recon - mu_orig, nuisance
  by construction) is safe (+0.3 pts) -- `hubness_public.py`,
  RESULTS_hubness_local_scaling.md; delta-centering is the variant to ship.
- **M4 fused decode: per-channel keys with dense-sparse outliers**
  (`kv_fused_pck.py` reference + `fused_decode_pck` CUDA kernel in
  `kv_kernel.py`; design in `docs/DESIGN_fused_kv_decode.md` section 8). Keys
  enter attention only through q.k, so the whole per-channel format becomes
  score terms: q.mu bias (all three zero-point modes fold here), per-channel
  weights x 16-entry grid (no shared LUT), and outliers as token-major-CSR
  score DELTAS applied by a warp-cooperative pass -- the dense loop never
  branches, divergence is bounded to ~3-entry rows. Exact vs
  decompress-then-attend across every key variant (15 tests, CPU + GPU;
  kernel max err 8e-8 on GV100) and 2-6x faster end-to-end than
  decompress-then-attend even with per-call CSR rebuild.
- README: component-map mermaid gains a "Guarantees & guardrails" subgraph
  (RankCertificate → autotune; a2_probe → keys family), How-It-Works gains
  the instrumented-boundary paragraph, Production/API/Highlights sections
  updated; `docs/api-stability.md` lists the new modules as Beta.

## v1.4.3

Documentation patch (no library API changes).

### Changed
- Removed meta-commentary from the README and docs (text about search indexing / release history /
  review process rather than the software itself).
- **Library-growth table**: the v1.4.x row now uses the same **pytest-collected item count** (514) as
  the earlier rows. A prior row briefly showed the raw `def test_` function count (~445), which looked
  like a regression but was just a different, smaller metric — no tests were removed (function count
  went 429 → 446 across 1.4.0 → 1.4.3). Footnote clarified.

## v1.4.2

Version-coherence and positioning patch (no library API changes).

### Fixed
- **Version coherence.** `CITATION.cff` was left at 1.4.0 after the 1.4.1 release; all current-version
  declarations now agree (`pyproject.toml`, `turboquant_pro/__init__.py`, `CITATION.cff`, PyPI). The
  "v1.1.0" seen in some search crawls is stale indexing of the historical "Release v1.1.0" commit — no
  current file declares 1.1.0.

### Changed
- README "Not to be confused with" states plainly that turboquant-pro is **distinct from the
  `turboquant` package focused on HuggingFace KV-cache compression** (the Google/ICLR TurboQuant
  KV-cache algorithm); the original TurboQuant paper is about online vector quantization, and here that
  quantizer is one component of a retrieval-first toolkit.

## v1.4.1

Documentation, reproducibility, and positioning release (no library API changes) — publishes the
three rounds of external review response to PyPI so the public page matches the repo.

### Added
- **`benchmarks/canonical_embedding.py`** — one tested harness running every embedding method
  (flat / PQ / OPQ / IVFPQ / RaBitQ / PCA-only / TQ-only / PCA+TQ / ADCIndex) at an **identical**
  rerank protocol on public ann-benchmarks data with provided ground-truth; bytes/vec computed
  analytically. Verified end-to-end (all 9 methods).
- **`notebooks/claims/`** — one Colab notebook per evidence-ladder claim (`00` flagship canonical
  SOTA table; `01`–`04` embedding/CPU incl. an OOD anisotropic/heavy-tailed stress test; `10`–`12`
  KV-cache/GPU), each embedding the verified harness. `_gen_notebooks.py` regenerates them.
- **`CLAIMS.md`** (repo root) — claim → reproduction table (Claim / Public reproduction? / Dataset /
  Command-or-notebook / Hardware / Status), Track 1 (central, CPU) vs Track 2 (GPU, experimental).
- **`docs/claims.md`** (evidence ladder, L1–L5) and **`docs/api-stability.md`** (Stable/Beta/
  Experimental tiers); **`benchmarks/RESULTS_canonical.md`** (protocol + one-command recipe).

### Fixed
- **`PCAMatryoshkaPipeline.estimate_storage()`** no longer reports hard-coded 1024→384 @ 3-bit
  regardless of configuration. It is now an instance method defaulting to the pipeline's real
  `input_dim` / `output_dim` / `bits` (explicit overrides still accepted); the dimension-agnostic
  form is available as the static `estimate_storage_for(...)`. Regression test in `tests/test_pca.py`.
  (Reported in external review #3.)

### Changed
- **README**: slimmed the dense headline claims paragraph (full list now in `CLAIMS.md` with
  reproduction status); added a **Version** block, an inline API-stability table, and a
  **"Not to be confused with"** section (incl. vLLM's TurboQuant integration) stating what
  turboquant-pro uniquely does; Benchmarks section now opens with the one-*Run all* SOTA pointer.
- **Test count** removed from static prose in favour of the CI badge (single source of truth).

## v1.4.0

### Added
- **Asymmetric (zero-point) NF4 keys — `nf4_asym` / `key_nf4_asym`.** Symmetric NF4 scales
  per channel by abs-max about *zero*; KV keys carry a large per-channel DC offset, so a
  symmetric grid wastes ~half its codes on the empty side. On high-ratio-GQA models this
  error exceeds attention's tolerance and generation **collapses** (Qwen2.5-7B: LongBench
  qasper 43.8 fp16 → **4.7**; WikiText-2 perplexity 7.46 → **74.7**, degenerate repetition).
  `nf4_asym` subtracts the per-channel mean, NF4-quantizes the residual, and adds the mean
  back — one calibration-free codebook robust across architectures.
- **`TurboQuantKVCache.robust()`** — recommended factory: asym-NF4 keys + 2% dense-sparse
  outliers, 4-bit K/V. Prefer it over passing `key_nf4=True` (the fragile symmetric grid).
- **Compact NF4/asym-NF4 storage** — store per-channel scalars (abs-max, and the mean for
  asym) instead of the expanded `(B,H,D,16)` level table. Compression ratio **6.3× → 7.9×**
  (on par with uniform 4-bit), no quality change.
- **Cross-model KV-quant matrix** in `benchmarks/kvquant_matrix/` (Llama-2-7B/13B, Mistral-7B,
  Qwen2.5-7B × fp16/NF4/uniform/asym-NF4; LongBench + WikiText-2), a decision guide, the
  reproducible harness, and a TMLR paper draft.

### Result
- asym-NF4 **ties** symmetric NF4 where NF4 already works (Llama-2-7B qasper 20.81≈20.82;
  Mistral 28.74; both *beat* NF4 on triviaqa and WikiText-2 ppl) and **rescues** the Qwen
  collapse to **41.9 qasper / 7.50 ppl** (+8 over asymmetric uniform; vs NF4's 4.7 / 74.7).
- **No universal best naive codebook:** plain NF4 wins on MHA/low-GQA but collapses at high
  GQA; asymmetric uniform never collapses but loses 5–9 qasper on MHA. asym-NF4 dominates both.
- **Honest limitation:** all 4-bit KV quant (asym-NF4 included) degrades on very-long-generation
  tasks (e.g. 512-token summarization, gov_report/multi_news) as the residual key error
  compounds across the decode — a generation-length effect that affects MHA *and* GQA models.

## v1.3.0

### Added
- **Calibration-free key-quality boosters for `PerChannelKV`** (the KVQuant recipe
  without its Fisher/K-means calibration):
  - `nf4=True` — NormalFloat-4 non-uniform levels (fixed codebook, per-channel abs-max
    scale). Better reconstruction than uniform or quantile-NUQ on Gaussian keys.
  - `outlier_frac` — **dense-and-sparse**: keep the top-`frac` magnitude entries per
    channel in fp16. Fixes the outlier-key-channel collapse that uniform low-bit
    quantization inflicts on attention (the LongBench `qasper` regime).
- **`TurboQuantKVCache`** exposes `key_nf4` and `key_outlier_frac` (default off → v1.2.0
  behavior unchanged).
- **Real LongBench task scores** in `benchmarks/RESULTS_longbench.md` (full 200-sample
  splits, Llama-2-7B-chat) + the runner `benchmarks/tq_enh_lb_shard.py`.

### Result
- On `qasper` (outlier-sensitive), per-channel **uniform 4-bit** scores 14.38 vs fp16
  22.06. **NF4 + 2% outliers + sink** recovers it to **20.82** — within 0.24 of KVQuant
  nuq4-1% (21.06) and **exceeding it on `triviaqa`** (83.32 vs 83.16), **with no
  calibration**. Outlier sweep 1%/2%/3% → qasper 20.23/20.82/20.67, so **2% is the sweet
  spot**; ship `outlier_frac=0.02`.

## v1.2.0

### Added
- **`PerChannelKV`** — per-channel quantizer for KV-cache **keys** (asymmetric uniform,
  optional non-uniform/NUQ, with 2/3/4-bit packing). This is the **correct architecture
  for keys**, replacing PolarQuant on the key path.

### Fixed (significant)
- **KV-cache keys no longer destroy generation.** PolarQuant's per-vector normalization
  is near-lossless for *values* but **catastrophic for keys**: it preserves each key's
  norm and quantizes its *direction*, discarding the per-channel scale that attention's
  `softmax(Q·Kᵀ)` depends on. Measured on Qwen2.5 (post-RoPE keys, perplexity):
  PolarQuant-K4 keys → ppl ≈ 10⁴ (yet reconstruction 0.095!), per-channel-K4 keys →
  ppl ≈ 15 (near fp16). Confirmed on 1.5B + 7B, at 512 + 4k context.
- **`TurboQuantKVCache` now uses `PerChannelKV` for keys by default** (`per_channel_keys=True`,
  values stay PolarQuant). Opt back to legacy with `per_channel_keys=False`.

### Note
- Reconstruction fidelity (cosine-sim / attention-output error) is *anti-correlated* with
  perplexity for keys, so the prior reconstruction-only benchmarks could not detect this.
  A **generation/perplexity** benchmark is added (`benchmarks/benchmark_kvcache_postrope.py`,
  `benchmarks/kv_quant_shootout.py`). See `docs/KV_KEYS_FINDING.md`.

## v1.1.0

### Added
- **`ADCIndex`** — fast compressed-domain search (asymmetric-distance scan over the
  packed codes). Optional AVX2 kernel (`turboquant_pro/_adc`, build with
  `pip install turboquant-pro[fast]` + `python -m turboquant_pro._adc`) with a correct
  numpy fallback. ~3700 qps at recall@10 0.9995 (7.9× over flat reconstruct).
- **Fused KV-decode kernel** — `turboquant_pro.kv_kernel` (split-K CUDA flash-decode)
  and `turboquant_pro.kv_fused` (reference + hot/cold online-softmax merge), wired into
  `TurboQuantKVCache.fused_decode`. Beats decompress-then-attend up to 13× at 32k
  context; exact to ≤4e-7.
- **`PCAMatryoshka.suggest_output_dim`** — variance-aware truncation-dim selection.
- **`turboquant_pro.format`** — TQE1 versioned, self-describing compressed-vector
  container (`pack`/`unpack`/`pack_batch`/`unpack_batch`); spec in `docs/FORMAT_SPEC.md`.
- Public ANN benchmarks across text **and vision** (GloVe-100, NYTimes-256,
  deep-image-96) with honest scope; `COMPREHENSIVE_ANALYSIS.md`.

### Changed
- **AutoConfig defaults keys to 4-bit.** The `compression` preset moves K3/V2 → K4/V2.
  Keys amplify quantization error through the attention softmax; 4-bit keys roughly
  halve the per-layer attention error vs 3-bit on real Qwen2.5-7B activations
  (`benchmarks/RESULTS_longbench.md`). Only `extreme` drops keys below 4-bit.
- Corrected citations repo-wide against arXiv (TurboQuant 2504.19874, PolarQuant
  2502.02617, QJL 2406.03482); removed a fabricated title.

### Notes
- The AVX2/CUDA kernels are optional accelerators with CPU/numpy fallbacks; the
  pure-Python wheel installs everywhere.

## v1.0.0
- Learned codebook fine-tuning (`LearnedQuantizer`), multi-modal presets
  (`ModalityPreset`), production observability (`QualityMonitor`).

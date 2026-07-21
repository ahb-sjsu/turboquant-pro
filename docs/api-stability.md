# API stability tiers

TurboQuant Pro spans research and production components. This page states the stability of
each so external users know what to depend on. Semantics: **Stable** = API frozen within a
major version, covered by tests, changes go through deprecation; **Beta** = usable and tested
but the API may change in a minor release; **Experimental** = research/preview, may change or
be removed without notice, may require optional dependencies or specific hardware.

> **Release note.** The `tqp` CLI and the certification platform (`tqp
> certify`/`verify`, the `tqp index` lifecycle, the plugin registry) ship in
> **1.8.0**; **1.9.0** is the current PyPI release — `pip install turboquant-pro`
> includes them. The tiers below describe the underlying APIs.

## Stable
- **`PCAMatryoshka`** — PCA-reordered dimension reduction.
- **Embedding compression** — the PCA + TurboQuant scalar-quantization pipeline and its
  compress/decompress + search entry points.
- **Basic `TurboQuantKV`** — the core key/value quantizers.
- **TQE1 serialized format** — the on-disk/wire format (versioned; see `docs/FORMAT_SPEC.md`).

## Beta
- **`ADCIndex`** — compressed-domain (asymmetric distance) search.
- **`TurboQuantKVCache`** — the drop-in KV-cache with per-channel keys / NF4 / asym-NF4.
- **`PerChannelKV` zero-point modes** (`"sparse"` / `"bias"`) — LongBench-validated; the
  container metadata fields (`zp_mode`, `rope_theta`, `position_start`) may evolve.
- **FAISS and pgvector wrappers** — integration adapters.
- **`rank_certificate`** (surfaced by `tqp certify` to emit and `tqp verify` to re-check a
  `certificate.json`) — distribution-free rank floors (the mathematics is frozen — it is
  a theorem — but the reporting surface / autotune fields may evolve in a minor release).
- **`a2_probe` + `QualityMonitor` tangential stream** — (A2) consumer-metric probe and the
  streaming radial-drift statistic; thresholds and result fields may evolve. An optional
  ZCA-whitened quantizer family (`probe_quotient(..., include_whitened=True)` /
  `recommend_key_quantizer(..., include_whitened=True)`, `A2ProbeResult.spearman_whitened`,
  and a `WHITENED_KEYS` runtime-policy action) is additive and off by default — the existing
  two-family verdict is byte-for-byte unchanged.
- **`TQEIndex` + `ShardedIndex` + `tqp index` lifecycle** — the persisted,
  corruption-checkable vector index (create/add/delete/compact/migrate/search/certify/drift)
  over the TQIX container, plus a memory-mapped read/search-only open (`TQEIndex.open(path,
  mmap=True)`) and block-streamed search (`search(..., block=B)`, bounded RAM), and
  `ShardedIndex` (shared-basis shards behind a JSON manifest, fanned across shards and merged
  to a global top-k) for indexes larger than RAM. On-disk **format v3** bit-packs sub-byte
  codes; it is a lossless re-encoding (rankings bit-identical to v2), v1/v2 files keep
  opening, and `TQEIndex.migrate(3)` upgrades in place. The container's magic/version/CRC
  layout and the shard manifest schema are stable within a format version. Method names and
  JSON fields may still evolve in a minor release.
- **`TQPRuntimePolicy` + `RuntimeDecision`** — the adaptive safe-fallback layer. The action
  vocabulary and evaluator names are stable; the default floors and `measured` fields may
  evolve as more real-model calibration lands.

## Experimental
- **Agent tool surface** (`agent_tools` + `examples/agentic/`) — JSON-in/JSON-out
  wrappers (`best_compression_at_recall`, `certify_ranking`,
  `recommend_kv_key_quantizer`, `list_tools`) over the Beta certificate / (A2) /
  auto-compress paths, plus LangChain / DSPy / MCP / custom-GPT adapters. New this
  cycle; the tool names and return schemas may change while the contract settles.
  The functions delegate to Beta components, so the underlying behaviour is as
  stable as those — it is the agent-facing shape that is Experimental.
- **Multi-node shard server** (`distributed.py`) — a server layer on top of `ShardedIndex`
  that partitions the shards across machines and fans search out over them. Newer and less
  settled than the single-process sharded path; API may change without notice.
- **`IVFIndex` coarse-partition search** (`ivf`) — sublinear compressed-domain search:
  k-means the quantized directions into `nlist` cells, probe best-first with a fixed
  `nprobe` or a weighted-A\* adaptive stop (`radius_scale`). Prototype toward
  trillion-scale *serving* (single-node substrate; composes with sharding + memmap).
  Recall/scan tradeoff measured in `benchmarks/RESULTS_ivf.md`. Also folded into
  `ShardedIndex` as an opt-in layer: `sh.build_ivf(nlist=...)` writes a global coarse
  quantizer + per-shard posting-list sidecars, and `sh.search(nprobe=...)` probes the
  best cells across shards. API and the coarse quantizer (kmeans, radius bound, sidecar
  format, manifest `ivf` block) may change without notice.
- **Quantizer plugin registry + conformance kit** (`plugins`, `plugin_conformance`)
  — the out-of-tree format contract (P0 of `docs/DESIGN_hardware_and_plugins.md`).
  The gating condition (a genuine external plugin passing conformance) is now
  **met** by `tqp-reference-plugin`, so this API is **eligible to promote to Beta**
  at the next stability review; it is held at Experimental for this release cycle
  while the protocol settles. Out-of-tree plugins themselves enter at this tier
  and promote per the design doc.
- **Online key calibration** (`calibrate_key_quantizer` / `PerChannelKV.calibrate`)
  — opt-in data-fit Lloyd-Max key codebook. Lowers reconstruction error but, on
  our attention proxy, does **not** beat the calibration-free default on
  softmax-KL (`benchmarks/RESULTS_calibration.md`); provided for users to measure
  on their own task. Experimental; the zero-calibration default stays recommended.
- **CUDA fused decode kernel** — requires a compatible GPU + build toolchain.
- **vLLM manager** (`TurboQuantKVManager`) — inference-server integration.
- **Model-weight compressor** — weight pruning/quantization path.
- **PostgreSQL extension** (`pgext/`) — server-side compressed search.
- **NATS transport** — compressed-artifact messaging.

## Optional dependencies
Core embedding + KV-cache functionality is **NumPy-only** — none of the extras below are
required for it. Each extra is lazily imported, only when the feature that needs it is used:

| Extra | Gates | Tier |
|---|---|---|
| `gpu` | `cupy-cuda12x` — CUDA fused decode + GPU ADC/compress | Experimental |
| `faiss` | `faiss-cpu` — FAISS index adapter | Beta |
| `pgvector` | `psycopg2-binary` — pgvector integration | Beta |
| `nats` | `nats-py` — compressed-artifact messaging | Experimental |
| `vllm` | `vllm` — inference-server KV manager | Experimental |
| `fast` | `pybind11` — build the AVX2 ADC kernel (numpy fallback otherwise) | Beta |
| `torch` | `torch` — cross-device backend adapter (`backend.py`) + operator tracer (`operator_trace.py`) | Beta / Experimental |
| `yaml` | `pyyaml` — `tqp replay` reads `claims.yaml` | Beta |

`pip install turboquant-pro[all]` pulls every runtime extra; `[dev]` adds the test/lint stack.
Missing optional fast kernels are surfaced explicitly rather than silently falling back
(tracked in `reviews/REVIEW_RESPONSE_1.md`).

**Plugin status.** The promotion condition — a genuine *out-of-tree* plugin that ships and
passes conformance — is **met**: [`tqp-reference-plugin`](https://github.com/ahb-sjsu/tqp-reference-plugin)
installs separately, is discovered only through the `turboquant_pro.plugins` entry point, passes
`tqp plugin conformance`, and participates in `tqp certify`. The registry + conformance kit are
therefore **eligible to promote to Beta** at the next review, held at Experimental for this cycle
while the protocol settles. `plugins/tqp-bnb`, `tqp-gptq-awq`, `tqp-trtllm` remain **in-tree
incubators** that dogfood the same contract — see Phase 5 of
`docs/turboquant_pro_next_level_roadmap.md`.

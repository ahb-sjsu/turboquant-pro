# API stability tiers

TurboQuant Pro spans research and production components. This page states the stability of
each so external users know what to depend on. Semantics: **Stable** = API frozen within a
major version, covered by tests, changes go through deprecation; **Beta** = usable and tested
but the API may change in a minor release; **Experimental** = research/preview, may change or
be removed without notice, may require optional dependencies or specific hardware.

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
- **`rank_certificate`** — distribution-free rank floors (the mathematics is frozen — it is
  a theorem — but the reporting surface / autotune fields may evolve in a minor release).
- **`a2_probe` + `QualityMonitor` tangential stream** — (A2) consumer-metric probe and the
  streaming radial-drift statistic; thresholds and result fields may evolve.

## Experimental
- **CUDA fused decode kernel** — requires a compatible GPU + build toolchain.
- **vLLM manager** (`TurboQuantKVManager`) — inference-server integration.
- **Model-weight compressor** — weight pruning/quantization path.
- **PostgreSQL extension** (`pgext/`) — server-side compressed search.
- **NATS transport** — compressed-artifact messaging.

## Optional dependencies
Extras `gpu`, `faiss`, `pgvector`, `nats`, `vllm` gate the Beta/Experimental integrations. Core
embedding + KV-cache functionality has no such requirement. Missing optional fast kernels should
be surfaced explicitly rather than silently falling back (tracked as a hygiene item in
`REVIEW_RESPONSE_1.md`).

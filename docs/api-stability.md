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
- **Quantizer plugin registry + conformance kit** (`plugins`, `plugin_conformance`)
  — the out-of-tree format contract (P0 of `docs/DESIGN_hardware_and_plugins.md`).
  The gating condition (a genuine external plugin passing conformance) is now
  **met** by `tqp-reference-plugin`, so this API is **eligible to promote to Beta**
  at the next stability review; it is held at Experimental for this release cycle
  while the protocol settles. Out-of-tree plugins themselves enter at this tier
  and promote per the design doc.
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
(tracked in `REVIEW_RESPONSE_1.md`).

**Plugin status.** The promotion condition — a genuine *out-of-tree* plugin that ships and
passes conformance — is **met**: [`tqp-reference-plugin`](https://github.com/ahb-sjsu/tqp-reference-plugin)
installs separately, is discovered only through the `turboquant_pro.plugins` entry point, passes
`tqp plugin conformance`, and participates in `tqp certify`. The registry + conformance kit are
therefore **eligible to promote to Beta** at the next review, held at Experimental for this cycle
while the protocol settles. `plugins/tqp-bnb`, `tqp-gptq-awq`, `tqp-trtllm` remain **in-tree
incubators** that dogfood the same contract — see Phase 5 of
`docs/turboquant_pro_next_level_roadmap.md`.

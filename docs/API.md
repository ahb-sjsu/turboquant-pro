# API / component reference

Public classes and functions of TurboQuant Pro, with a one-line purpose for each.

Part of [TurboQuant Pro](../README.md). Stability tiers: [api-stability.md](api-stability.md).

| Class | Purpose |
|-------|---------|
| `TurboQuantKV` | Stateless compress/decompress (PolarQuant) with optional bit-packing |
| `PerChannelKV` | Per-channel quantizer for KV-cache **keys** (correct key architecture) |
| `TurboQuantKVCache` | Streaming L1/L2 tiered cache (per-channel keys + PolarQuant values) |
| `TurboQuantKVManager` | Multi-layer KV cache manager (vLLM plugin) |
| `AutoConfig` | Model-aware selection of K/V bits, RoPE-awareness, and components |
| `PCAMatryoshka` / `PCAMatryoshkaPipeline` | PCA rotation + truncation; end-to-end PCA + TurboQuant |
| `LearnedQuantizer` | Data-fit Lloyd-Max codebooks (drop-in for the default) |
| `ADCIndex` / `CompressedHNSW` | Compressed-domain search; compressed HNSW graph index |
| `TQEIndex` / `ShardedIndex` | Persisted index (memory-mapped, block-streamed, bit-packed format v3); sharded larger-than-RAM index over a shared PCA basis (`distributed.py` fans it across machines) |
| `TurboQuantFAISS` | FAISS index wrapper with auto PCA compression |
| `TurboQuantPGVector` | Compress pgvector embeddings for PostgreSQL storage |
| `TurboQuantNATSCodec` | Encode/decode embeddings for NATS transport |
| `ModelCompressor` | SVD / activation-space analysis + low-rank compression of model weights |
| `QualityMonitor` | Drift detection (cosine + (A2) tangential) + Prometheus metrics |
| `RankCertificate` / `certificate_from_embeddings` | Distribution-free rank-agreement floors (κ, μ̂, τ-floor) + rerank-required signal — emit with `tqp certify`, re-check with `tqp verify` |
| `probe_quotient` / `recommend_key_quantizer` | (A2) consumer-metric probe: polar vs per-channel family selection |
| `behavioral_agreement` / `flip_rate` / `noise_floor` | Decision-level quantization-quality metric: symmetric flip rate + prediction agreement + noise-floor excess (z) |
| `trace_operators` / `recommend_quantization` | Operator-regime tracing: infer each tensor's (A2) consumer (softmax/residual/gate/state) via structural + `torch.fx` graph, and its per-target quantization discipline |
| `routing_sensitivity` / `differential_fraction` | MoE routing fragility: top-k margin distribution + the common-mode-free (differential) logit fraction that flips selection |
| `state_decay_sensitivity` / `quantize_decay` | SSM decay fragility: per-channel gain/compounding + log-time-constant quantization (5–6× less state drift than linear) |
| `run_autotune` / `auto_compress` | Sweep configs and recommend optimal compression (now with certificates) |
| `best_compression_at_recall` / `certify_ranking` / `recommend_kv_key_quantizer` / `list_tools` | Agent-facing tool surface (`agent_tools`): JSON-in/JSON-out wrappers for tool-calling models — "best ratio at target recall", the rank certificate, and the (A2) key probe. LangChain / DSPy / MCP / GPT wrappers in [`examples/agentic/`](../examples/agentic/) |

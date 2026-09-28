# Survey: the MODEL side of turboquant-pro, for the console's Models workspace

Read-only survey, 2026-09-26. Source: `origin/master` at `24d49ac` (exported with `git archive` to
`scratchpad/tqp_om/`; `C:\source\turboquant-pro` was not touched; the `C:\source\tqp-master-wt`
worktree is stale at `c8d7d2b` and was NOT used). Nothing was executed except `git`, `ast` outlines
and greps. Every statement about behaviour below comes from reading code. Where I state a
consequence (bytes, cost), it is arithmetic from the code, not a measurement, and is marked so.

Requirement ids refer to `readscope_req.txt` (OV, QI, RS, VS, IS, BM, UX, API, NFR, section 10).

---

## 0. Findings that should shape the Models workspace before any panel is drawn

These come from reading `hf_cache.py`, `per_channel_kv.py` and `core.py`. The first live source
you plan to use (a small HF model with `TurboQuantLayer` on a GPU) behaves differently from what
the docstrings imply. Each item is something the UI must show, or it will decorate a false picture.

1. **The HF drop-in compresses on the CPU, in numpy, even when the model is on a GPU.**
   `TurboQuantLayer.lazy_initialization` builds `TurboQuantKV(..., use_gpu=False)` and a numpy
   `PerChannelKV`. Every spill does `tensor.detach().to("cpu", float32).numpy()`, and every
   `update()` calls `_full()`, which decompresses **every** cold chunk in numpy, copies it back to the
   device and `torch.cat`s it with the hot window. None of the fused kernels (M1/M2/M4, K2, Triton)
   are reachable from `TurboQuantCache`. The fused path lives only on the numpy/CuPy
   `TurboQuantKVCache.fused_decode`. The Overview "mode" indicator (OV-006) must therefore say
   "decompress-then-attend, host codec", not "fused kernel".

2. **Device memory is not reduced by the HF drop-in; host memory is.** `_full()` materializes the
   full dense K and V for the layer on the device on every step (the compressed store stays on the
   host). Per-layer transient device memory is about fp16-full-KV, and PyTorch's caching allocator
   keeps it. A "KV bytes saved" bar must split *resident compressed store (host RAM)* from
   *transient dense materialization (device)*, or it will claim a VRAM saving that does not exist.

3. **Decode-time spills are one token long, which makes each key chunk larger than fp16 and lossless.**
   Once the hot window is full, each decode step overflows by one token and `_ingest` compresses a
   chunk with S=1. In `PerChannelKV._outliers`, `k = max(1, round(S*outlier_frac))` = 1 and
   `k >= S`, so **every entry becomes an fp16 outlier** (int32 index + fp16 value). The chunk also
   stores fp32 `nf4_scale` and fp32 `nf4_mean` per channel over a single token. By arithmetic
   (head_dim 128, 4-bit): about 64 B codes + 512 B scale + 512 B mean + 512 B outlier idx + 256 B
   outlier val, roughly 1.86 KB per head per token versus 256 B fp16, about 7x *larger*, and
   reconstruction is exact. The numpy `TurboQuantKVCache` avoids this (it flushes half the window
   at once). Values (PolarQuant) are fine per token. I did not run this; a unit test that
   compresses a (1,H,1,128) block and reads `nbytes()` would confirm it in milliseconds on Atlas.

4. **With the default `hot_window=512` a short demo compresses nothing.** If prompt plus generated
   tokens stay under 512, the cold store is empty and every KPI is fp16. A prompt longer than the
   window spills one large prefill chunk (compressed properly), then one-token chunks per step.
   The per-layer hot/cold token split is the first number the Models page must show.

5. **`_full()` cost grows with the number of cold chunks, not only tokens.** One Python-level
   decompress per chunk per layer per step. With one-token decode chunks, step time grows roughly
   linearly in generated tokens. A tokens/s timeline with a "cold chunks" overlay will show it.

6. **The policy knobs are not wired together.** `tqp plan kv` / `AutoConfig` (K4/V3 balanced,
   `key_zero_point`, RoPE boost) builds the numpy `TurboQuantKVCache`, not the HF cache.
   `TurboQuantCache` takes its own `key_bits/value_bits/outlier_frac` and always uses
   `zero_point="calibrated"` asym-NF4 (key_bits must be 4). `TQPRuntimePolicy` is advisory:
   nothing in the generation path calls it. `vllm_plugin.TurboQuantKVManager` builds the default
   (non-robust) `TurboQuantKVCache` (per-channel uniform keys, no asym-NF4, no outliers). The UI
   must show the *effective* configuration of the running cache, never the plan's.

7. **Telemetry today is search-only.** `telemetry.metrics.REGISTRY` has 12 metrics: `search.*`,
   `index.*`, `process.*`. `telemetry.api.capabilities()` reports `gpu_utilization: False`. There is
   no `model.*` or `kv.*` metric, no model entity, and nothing in `hf_cache.py`, `core.py` or the
   kernels imports `telemetry`. Every live Models panel needs new instrumentation (section 9).

8. **Memory accounting uses different baselines.** `TurboQuantKVCache.memory_stats()` compares against
   **fp32** (`* 4`), so its `effective_ratio` is about 2x the fp16 ratio. `CompressedKV.original_nbytes`
   uses the input dtype (float32 after `.to(float32)` in the HF bridge). `AutoConfig.estimate_memory`
   and `TurboQuantKV.compression_ratio` use fp16. A ratio tile must state its baseline.

9. **No llama.cpp / GGUF integration exists.** `examples/llama_integration.py` is a sketch that feeds
   random K/V (it says "conceptual"). `benchmarks/benchmark_llama.py` simulates access patterns.
   `benchmark_e2e.py --backend llama-cpp` runs a GGUF as a *baseline* (`gguf-q3_k`, explicitly not a
   TurboQuant result). **No NATS KV transport exists**: `nats_codec.py` is an embedding wire format
   and `nats_transport.py`/`nats_worker.py` serve sharded vector search. KV persistence is the vLLM
   V1 connector's `TurboQuantBlockStore` (`tqp-kv-store-state/2`, interim before a TQE1 `kv_block`
   profile).

10. **There is no `weight_observer` module in the package.** Weight observer work lives in
    `benchmarks/weight_observer/` (Part III campaign: predictor tables, atlas, single-matrix
    sensitivity, measured KL, plans). The package ships only `weight_plan.py` (exact MCKP solver)
    and `model_compress.py` (SVD analysis, fake quantization).

---

## 1. KV cache

Legend for "exists as": RV = return value, SD = stats dict, AR = artifact/schema, IN = internal
state only (readable but not exposed), NO = not computed anywhere. Effort: S (<1 day), M (days),
L (week+).

| # | Feature (files) | What it does | Data produced or exposable (exists as) | Live / artifact | Best visualization | Workspace, req ids | Hook without slowing generation, effort | Honesty notes |
|---|---|---|---|---|---|---|---|---|
| A1 | `TurboQuantCache`, `enable_turboquant_cache` (`hf_cache.py`) | One-line HF `Cache`: one `TurboQuantLayer` per layer, created lazily; patched `generate` injects a fresh cache | config: hot_window, key_bits, value_bits, outlier_frac, seed (attrs, IN); per-layer list `cache.layers` (IN) | live | Config strip in the Models header: effective K/V codec per tensor (keys: per-channel asym-NF4 + outliers, zp=calibrated; values: PolarQuant), hot window, seed, "host codec / decompress-then-attend" mode badge | Models header, Overview mode; OV-006, RS-001, RS-002 | Read attributes at attach time. S | Show the effective codec, not the kv_plan (finding 6). Label mode honestly (finding 1). |
| A2 | `TurboQuantLayer` tier state | Hot fp16 window (device) plus cold compressed chunks (host numpy) | per layer: `_seq_len`, hot len `_hot_keys.shape[-2]`, `_cold_lengths` list (chunk sizes), head_dim, n_kv_heads, device, dtype (IN) | live | Per-layer stacked horizontal bar: hot tokens vs cold tokens, with chunk-count label; small multiple per layer; plus a chunk-size histogram (reveals 1-token decode chunks) | Models; OV-001, IS-001 analogue | Poll attributes at 1-10 Hz from the console thread (reads of Python ints/lists; no sync). S | Hot/cold counts are exact. Say "0 cold" plainly when the window never filled (finding 4). |
| A3 | `TurboQuantLayer` KV bytes | Compressed containers know their size | per chunk: `CompressedPerChannelKV.nbytes()` split into codes / nf4_scale / nf4_mean / outlier_idx / outlier_val; `CompressedKV.nbytes()` = packed codes + fp32 norms (RV on the container); per layer sum: NO | live | Stacked bar per layer, **three baselines side by side**: fp16-equivalent, host compressed resident (split by component: codes, scale, mean, outlier idx, outlier val, value norms), device transient dense; a "bytes per cold token" sparkline per layer | Models, Overview KPI; OV-001 (compression ratio), section 10 Compression | Accumulate `nbytes()` at spill time inside `_ingest` (one call per spill, O(1) fields). Sum on read. S | Measured bytes of numpy arrays. Device transient bytes are not in the containers; take them from `torch.cuda.memory_allocated` deltas (sampled). The component split is what exposes finding 3. |
| A4 | Spill events (`TurboQuantLayer._ingest`) | Moves overflow tokens hot to cold, compresses keys and values | per spill: layer, tokens, position_start, compress time (NO), key/value bytes (derivable) | live | Event strip / waterfall: x = decode step, y = layer, a tick per spill colored by chunk size; tooltip with bytes and ms | Models, Overview events; OV-005 | Wrap `_ingest` with `perf_counter`; the `.cpu()` inside already synchronizes, so the timing is real wall time. Emit into a bounded ring buffer. S | Timing includes device-to-host copy and numpy compress; say so. |
| A5 | Cold materialization (`TurboQuantLayer._full`) | Decompresses all cold chunks, copies to device, concatenates with hot | per step per layer: decompress ms, H2D copy, bytes materialized (NO) | live | Per-step stacked time bar: attention-model compute vs cold decompress vs H2D copy, over steps (a flame-like strip), plus "cold chunks per layer" overlay | Models, System; OV-003 (pipeline stage times), NFR-003 | `perf_counter` around the decompress list and the `torch.cat`; decompress is host numpy so timing is honest; the cat is async on device (time with CUDA events if needed, sampled). S/M | This is the dominant cost at long context by code reading (finding 5). Measured, not estimated. |
| A6 | Key/value reconstruction error at spill | Not computed today | at spill the fp16 source block is in hand (`spill_k`, `spill_v` as numpy): per head relative L2 error, per channel error, max abs error, SNR in dB | live (sampled) | Layer x head heatmap of key error in dB (and a second for values); per-channel error strip for a selected head | Models; section 10 Quality, QI-005 | On every Nth spill (or sample rate r), decompress the chunk just compressed and compare to the source already on the host. One extra decompress per sampled spill. M | Reconstruction error is **diagnostic only, never acceptance** (repo rule; `KV_KEYS_FINDING.md`: reconstruction error is anti-correlated with perplexity for keys). Label "diagnostic". With 1-token chunks the key error is zero by construction (finding 3), which is a red flag, not a success. |
| A7 | Attention consumer quality (live) | Not computed during generation; `bench_kv_serving.py` does it offline | per sampled layer and step: KL(attn_fp16 || attn_quant) per head, attention top-k overlap, logit Spearman (the `attention_softmax` / `attention_topk` consumers) | live (sampled, needs reference) | Layer x head heatmap of attention KL (log scale) with a step slider; sparkline of mean KL per probe layer | Models, ReadScope; section 10 Quality, RS-004, QI-005 | Needs the query and an fp16 reference. Options: (a) keep a shadow fp16 K/V for 1-3 probe layers (bench_kv_serving uses [0, L/2, L-1]) and wrap the attention function (`eager_attention_forward` patch pattern from `benchmark_longbench_parity.py`, or the `AttentionInterface` registry) on sampled steps only; (b) teacher-forced paired run as an artifact. L | Requires a reference fp16 pass and eager attention (SDPA/flash do not return probabilities). Sampled layers and steps; state which. This is the metric the project accepts for keys. |
| A8 | Per-channel key metadata (`CompressedPerChannelKV`) | Stores per (head, channel) scale (`nf4_scale`), zero point (`nf4_mean`, or none for zp=bias), outlier indices/values | (H, D) scale and mean per chunk; outlier count per (head, channel) from `outlier_idx` (RV/IN on container) | live | Head x channel heatmap of |zero point| (the DC offset) with RoPE long-wavelength channels marked (mask from `PerChannelKV.dc_channel_mask`); head x channel outlier-count heatmap | Models, ReadScope; RS-001, RS-004 | Read from the prefill chunk (large S) once; decode chunks are degenerate (S=1). S | Values are exactly what the codec stored. The DC-offset heatmap is the picture behind the Qwen2.5 symmetric-NF4 collapse finding. |
| A9 | `TurboQuantKV` (`core.py`, PolarQuant) | Random rotation + per-vector norm + Lloyd-Max codebook, bit-packed; asymmetric key/value bits | `CompressedKV` (indices, norms (B,H,S), bits, shape); `nbytes`, `original_nbytes`, `compression_ratio(head_dim)` (RV); `compression_ratio(packed)`, `theoretical_compression_ratio()` (RV, formula); `estimate_memory(...)` (RV dict: original_gb, compressed_gb, ratio, saved_gb) | live (container) / estimated (formulas) | Per-token norm distribution (histogram per layer) for values; code-occupancy histogram per bit width (are all 2^b cells used?) | Models; IS-004 (codebook identity: bits, seed, rotation kind full-QR vs structured) | Norms and codes are on the container at spill; histogram in the sampled path. S | `estimate_memory` and the ratio methods are formulas (kind: estimated/derived), not measurements. |
| A10 | `TurboQuantKVCache` (+ `.robust`, `memory_stats`, `fused_decode`) (`core.py`) | numpy/CuPy streaming cache, flush-half-window policy, per-channel keys, fused compute-on-codes decode | `length`, `hot_length`, `cold_length` (RV); `memory_stats()` (SD: cold_bytes, hot_bytes, total_bytes, uncompressed_equivalent_bytes (fp32 baseline), effective_ratio); `_pck_fused_ok` (IN: whether fused form applies) | live | Same tier bars as A2/A3; a "decode path taken" badge per cache: fused M4 / fused M1-M2 / reconstruction fallback, with the reason (nuq grid, structured rotation, d%32, d/32>16) | Models, System; OV-006, QI-001 | Record which branch `fused_decode` took and why (one enum per call; the branch condition already exists). S | `memory_stats` ratio is vs fp32 (finding 8). This cache is not the HF path. |
| A11 | `PerChannelKV` options (`per_channel_kv.py`) | Keys: uniform, nuq quantile, NF4, asym-NF4, outlier_frac, zero_point calibrated/sparse/bias, `rope_averaged_bias`, `dc_channel_mask`, experimental `calibrate` | per chunk container fields (A8); zp_mode, rope_theta, position_start; bits (IN) | live | Codec card per tensor role (a small, fixed table: grid, zp mode, outlier_frac, stored metadata bytes per token); zero-point comparison chart (calibrated mean vs RoPE-averaged bias per channel) when k_bias exists | Models, ReadScope; RS-001, IS-004 | Attributes at attach. S | Measured LongBench numbers in docstrings are artifacts from `kvquant_matrix`, not live. |
| A12 | `RoPEFrequencyAnalyzer`, `RoPEAwareQuantizer` (`rope.py`) | Per-dimension bit allocation boosting long-wavelength pairs; splits dims into two PolarQuant groups | `bit_allocation()` (RV per-dim bits), `summary()` (SD: n_boosted_dims, avg_bits, frequency and wavelength ranges), `stats()` (SD) | artifact (config-only) | Per-dimension bar: rotary wavelength (log) with the context-window line and bits per dim as color | Models (config), ReadScope; RS-001 | Pure config; compute at attach. S | Uses PolarQuant per group; not what `TurboQuantCache` uses. Show as "available, not active" unless built. |
| A13 | `calibrate_key_quantizer` (`calibration.py`), `PerChannelKV.calibrate` | Experimental Lloyd-Max per (head, channel) key codebook fitted once | fitted levels (H,D,L) (IN on quantizer) | artifact | Per-channel level ladders for one head (small multiples); flag "experimental: does not beat asym-NF4 on softmax-KL" | ReadScope; RS-005 | Offline. S | Docstring states it lowers reconstruction error without improving the consumer metric. |
| A14 | `calibration_coverage`, `check_calibration_coverage` (`calibration_coverage.py`) | How far a calibration sample sits from serving activations | `CoverageReport` (SD: jeffreys, bhattacharyya, mean_shift, spectral_ratio, verdict, n_calibration, n_serving, dim) | live-capable (compare calibration keys vs a sampled window of serving keys) | Gauge per layer of Jeffreys divergence with WARN/FAIL bands, trended over time | ReadScope, Models; RS-005, VS-005 (drift) | Sample spilled keys (already on host) into a reservoir; compute every N seconds off the generation thread. M | Thresholds are conventions, not measurements (module says so). Only meaningful if a calibrated codebook is in use. |
| A15 | `AutoConfig` (`autoconfig.py`), `list_models`, `list_targets`, `TurboQuantKV.from_model`, `hardware_profile`, `with_hardware_tuning` | Reads HF config or registry, picks K/V bits, RoPE boost, zero-point mode per target preset | `summary()` (SD: head_dim, n_kv_heads, n_layers, max_seq_len, key_bits, value_bits, rope_aware, rope_theta, estimated_kv_cache_gb, compression_ratio, saved_gb); `estimate_memory(seq_len)` (SD) | artifact | Plan vs effective diff card (plan K4/V3 vs running K4/V4, zp auto vs calibrated); memory-vs-context curve (fp16 vs compressed estimate) with the current context marked | Models; RS-002, BM-005 | At attach. S | Estimated (formula, packed bits, no hot window, no per-channel metadata or outliers). Not wired to the HF cache (finding 6). |
| A16 | `tqp plan kv` (`cli.py`, schema `kv_plan.schema.json`) | Emits `turboquant-pro/kv-plan` v1 | request, policy (= `AutoConfig.summary()`), key_zero_point, risk_flags, reproduction (AR) | artifact | Plan card with risk flags as amber chips; "reproduce" CLI line | Models, Benchmarks; BM-005, UX-008 | None. S | Plan, not measurement. |

## 2. Kernels and backends

| # | Feature (files) | What it does | Data produced or exposable | Live / artifact | Best visualization | Workspace, req ids | Hook, effort | Honesty notes |
|---|---|---|---|---|---|---|---|---|
| B1 | Fused decode reference M0 (`kv_fused.py`: `fused_decode_attention`, `dequant_decode_attention`, `fused_decode`, `merge_partials`) | One decode step computed in code space; online-softmax partials (m, l, acc) merge hot and cold | output (H, d) (RV); partials per tier (RV) | live (when a numpy/CuPy cache is used) | Per-head hot vs cold attention mass: l_cold/(l_hot+l_cold) from the partials, as a layer x head heatmap ("how much attention lands in compressed memory") | Models; OV-003, RS-004 | Partials are already returned; record the mass split per call (sampled). S | Only on `TurboQuantKVCache.fused_decode`, not on the HF cache. Exact vs decompress-then-attend (tested). |
| B2 | M1/M2 CUDA kernels (`kv_kernel.py`: `fused_decode_cuda` method warp/block; M4 `pck_block_partials_cuda`, `fused_decode_pck_cuda`) | CuPy RawKernels, one block/warp per head, online softmax over ADC scores | latency (NO in library; benchmarks measure it); which kernel ran (NO) | live | Kernel-selection badge + per-step kernel time sparkline; bar of fused vs dequant ms at current S | System, Models; OV-006, section 10 Performance (kernel utilization) | CUDA events around the launch on sampled steps (event record is async; read later). M | Kernels compile lazily via NVRTC: first-call latency includes compile; mark warm-up. |
| B3 | M4 reference and prepared pages (`kv_fused_pck.py`: `PreparedPCKBlock`, `build_outlier_csr`, `pck_key_scores`) | Per-channel keys fused: bias + dense grid sum + CSR outlier deltas; one prepared block per cold page, cached | per page: S, nnz of outlier CSR (row_ptr), device (IN); backend chosen (cupy / torch-Triton / numpy) (IN) | live | Per-page table (S, outlier nnz, prepared yes/no, backend), with a bar of outlier nnz per page | Models, System; IS-001 analogue for pages | Read block attributes when prepared (once per page). S | `partials` raises for torch without Triton; record the failure reason. |
| B4 | Volta K2 kernels (`volta_kernels.py`: `k2_key_scores`, `k2_key_scores_packed`, `value_accum`, `apply_outlier_csr`) | Read 1-byte (or packed) codes directly; LUT-dequant + dot in registers; vec4 kernel when D%4==0 | variant chosen (vec4ns vs scalar fallback) (IN); latency (benchmark only) | live | Achieved-bandwidth gauge (bytes of codes read / kernel time) against HBM peak for the detected GPU | System; section 10 Performance (memory bandwidth) | CUDA events (sampled) plus bytes-read computed from shapes. M | Bandwidth is derived (bytes/time), label derived. Docstring figures (55-63% of HBM2 peak) are GV100 benchmark claims. |
| B5 | Triton port P5 (`kv_triton.py`, `_triton_kernels.py`: `has_triton`, `pck_block_partials_triton`, `pck_batched_partials_triton`, `polar_partials_triton`) | Portable M2/M4 kernels, split-K, batched per-page launch, BLOCK_D tiling | `has_triton()` (RV); nsplit, BLOCK_D, MAX_NNZ per launch (IN) | live | Capability row in System (Triton yes/no, device); per-launch nsplit/BLOCK_D in the kernel inspector | System; API-001, API-008 | Record launch params on sampled launches. S | Triton JIT compile on first call; exactness oracle is the CuPy kernel. |
| B6 | Packing kernels (`cuda_kernels.py`: `get_gpu_kernel`, `gpu_batch_quantize`, `gpu_batch_rotate_quantize`) | GPU bit-pack/unpack and fused rotate+quantize for `TurboQuantKV(use_gpu=True)` | compile cache contents (IN) | live (only when use_gpu=True) | Row in the kernel inventory (compiled / not compiled) | System; OV-006 | Read the kernel cache dict. S | Not used by the HF cache (use_gpu=False). |
| B7 | `backend.py` (`to_numpy`, `torch_decode`, `_TorchXP`) | Boundary conversion from torch/CuPy; torch reference decode after host reconstruction | none beyond outputs | live | None of its own; contributes the "backend" label (numpy / cupy / torch) to the mode badge | System; OV-006 | S | `torch_decode` is host reconstruction then torch attention, not device-native decode (its own docstring). |
| B8 | `hardware.py` (`detect_gpu`, `get_hardware_profile`, `profile_for_arch`) | Detects GPU via CuPy, maps compute capability to arch, recommends bits | `HardwareInfo` (name, arch, cc, memory_gb, device_id), `HardwareProfile` (recommended bits, fp8/fp4 support, use_fused_kernel, notes) (RV) | artifact (at start) | Runtime card: device, arch, cc, memory; recommended vs effective bits | System, Overview; API-001, section 7 Runtime | At attach. S | Needs CuPy; without CuPy it reports "CPU" even when torch sees a GPU. Show "detected via CuPy: no" rather than "no GPU". No utilization, power or temperature. |
| B9 | `cuda_search.py` (`gpu_adc_search`, `gpu_l2_search`, `gpu_hamming_search`) | GPU search over compressed embeddings | results only | n/a for models | Belongs to the embedding/Index workspaces | Index & Shards | n/a | Listed for completeness; not model-side. |

## 3. Weights

| # | Feature (files) | What it does | Data produced or exposable | Live / artifact | Best visualization | Workspace, req ids | Hook, effort | Honesty notes |
|---|---|---|---|---|---|---|---|---|
| C1 | `ModelCompressor.analyze` (`model_compress.py`), `turboquant-pro model` (`autotune.main_model`) | Weight-space SVD of FFN matrices | `CompressionReport` (n_layers, total_params, ffn_params, layers: `LayerAnalysis` per matrix with eigenvalues, variance_explained_50/75/90, effective_rank, condition_number; avg_effective_rank_ratio, recommended_ratio, estimated_speedup) (RV) | artifact | Layer x matrix heatmap of effective-rank ratio; per-matrix scree plot on click | Models (weights tab); VS-003 analogue | Offline; runs once at load (SVD of every FFN matrix is not cheap on a large model: sample_layers). S to wire | Variance is not downstream performance (module says so). `estimated_speedup` is estimated. |
| C2 | `analyze_activations`, `compress_activations`, `compress` | Activation-space PCA per layer and per attention head; in-place rank reduction | `HeadAnalysis` per (layer, head): effective_rank, variance_explained_90, compressible; n_compressible_heads (RV) | artifact | Layer x head grid colored by effective rank / head_dim, compressible heads outlined | Models; RS-004 | Offline, needs calibration data. M | Calibration-dependent; record the dataset (A14 applies). |
| C3 | `quantize_weights` with `rope_aware_k`, `rope_protected_rows`, `quantize_weight_rows` | Fake-quantize FFN+attention weights per output row; keep the long-wavelength k_proj rows full precision | returns the model only; counts (layers quantized, protected k rows) are only **logged** (NO as data) | artifact | Per-layer bar: rows protected vs quantized in k_proj; config card (bits, protect_frac, protect_bits) | Models; RS-001 | Return the counts (small change) or capture the log record. S | **Fake quantization**: dtype and memory are unchanged; no storage saving. Never label it as compressed bytes. |
| C4 | `ModelCompressor.sweep` | Ratio sweep with a user eval_fn | list of {ratio, quality, speedup} (RV) | artifact | Pareto line: quality vs ratio | Benchmarks; BM-004 | Offline. S | Quality is whatever eval_fn returns; name it. |
| C5 | `weight_plan.py` (`CostTable`, `solve`, `dual_bound`, `budget_for_rate`, `pin`, `check_matrices`) + `tqp plan weights` (schemas `weight_cost_table`, `weight_plan`) | Exact multiple-choice-knapsack bit allocation per matrix with a Lagrangian dual bound; refuses rather than approximates | `WeightPlan` (bits per matrix, cost, stored_bits, budget_bits, dual_bound, gap, lattice_unit, lattice_states, cost_table_hash, model, predictor) (RV, AR) | artifact | Layer x matrix-type heatmap of assigned bits (q,k,v,o,gate,up,down); budget bar (stored vs budget); cost vs dual-bound readout with the gap; bits histogram | Models (weights), Benchmarks; BM-004, BM-005, RS-002 (cost-table hash provenance) | None (offline). S | Exact given the cost table; any deficit belongs to the cost model. Show the predictor name and table hash next to the plan. |
| C6 | Weight observer campaign (`benchmarks/weight_observer/`: tables, atlas, sensitivity, variants, measure, plans, flatness, headroom, score) | Part III: which predictor (raw, act, fisher, outdiag, observer K-FAC) ranks weight-quantization damage (KL on WikiText) | `tables.json` cost per matrix per bits per predictor; `atlas.json` per matrix (sigma_x and p_y effective rank, participation, top share, massive channels, row hubness); `sensitivity.jsonl` single-matrix KL; `results_weights.json` rho-bar per predictor, verdicts W1-W3; `planned/*.json` plans per predictor and rate (AR) | artifact | (1) Predictor agreement: scatter of predicted cost vs measured single-matrix KL per predictor, log-log; (2) atlas small multiples per layer (effective rank of inputs vs read-outs); (3) rho-bar bar chart with CIs and the verdict chip | Benchmarks, ReadScope; BM-001, BM-002, RS-004 | Read committed JSON. S | Registered verdicts vs exploratory files must be labeled differently (the scripts say "EXPLORATORY; not a verdict"). |
| C7 | Weight plugins (`plugins/tqp-bnb`: bnb_nf4, bnb_llm_int8; `plugins/tqp-gptq-awq`: gptq, awq) | Out-of-tree codecs registered for the weight target | via registry: name, tier, targets; conformance report | artifact | Rows in the plugin table (G1) | System, Benchmarks | S | Availability depends on installed packages; show "not installed" distinctly. |

## 4. Tracing, (A2) instruments, consumers

| # | Feature (files) | What it does | Data produced or exposable | Live / artifact | Best visualization | Workspace, req ids | Hook, effort | Honesty notes |
|---|---|---|---|---|---|---|---|---|
| D1 | `trace_operators`, `recommend_quantization`, `OperatorPlan` (`operator_trace.py`) + `tqp trace` | Infers each parameter's operator regime (softmax_score, linear_residual, gate_selection, state_decay, norm, unknown) structurally and by torch.fx; maps regime x target (weight, kv_activation) to a discipline | per tensor: regime, method (structural/fx), confidence; discipline (family, protect_dc, sensitivity, rationale); `coverage()`, `by_regime()`, `families()`, `summary()` (RV) | artifact (at model load; meta device, no weights) | Model anatomy map: layers as rows, module slots as columns, colored by regime, with a second toggle for discipline family under the chosen target; confidence as opacity; fx-evidence marker | Models (anatomy), ReadScope; RS-001, RS-007 (which tensors depend on which discipline) | Run once at attach on the loaded model (structural is cheap; fx is best-effort). S | Structural is name-based; fx may fail on some models (`traced` flag). Show method per tensor. |
| D2 | `probe_quotient`, `recommend_key_quantizer`, `displacement_decomposition`, `tangential_fraction(s)` (`a2_probe.py`) + `tqp probe` + `agent_tools.recommend_kv_key_quantizer` | Calibration-time test: which quotient family (polar, per-channel, whitened polar) preserves the declared consumer's scores | `A2ProbeResult` (consumer, spearman_polar, spearman_per_channel, spearman_whitened, median_tangential_fraction, median_unit_displacement, recommendation, margin) (RV, JSON via CLI) | artifact, repeatable on live samples | Per-layer paired dot plot: Spearman polar vs per-channel (and whitened) with the recommendation; tangential-fraction gauge per layer | ReadScope, Models; RS-003 (compare families), RS-004 | Needs keys and ideally the real queries. Keys are on the host at spill; queries need the attention hook (A7). Run on a reservoir off-thread every N seconds. M | Probe quantizers are proxies (`_polar_proxy`, `_per_channel_proxy`), not the shipped codecs. `--demo` data is synthetic and labeled so; keep that label. |
| D3 | Routing sensitivity (`operator_sensitivity.py`: `routing_margins`, `differential_fraction`, `predict_routing_flips`, `routing_sensitivity`) | MoE router fragility: top-k margin distribution, predicted flip fraction | `RoutingSensitivity` (k, n_tokens, margin_p10, margin_p50, margin_mean, predicted_flip_fraction) (RV) | live-capable (MoE only) | Margin histogram per MoE layer with the policy floor line; per-token margin strip over the generated sequence | Models; RS-004 | Forward hook on router gate modules (logits are small: tokens x experts); sampled. S/M | Not applicable to dense models; hide the panel (capability absent) rather than show zeros. Flip fraction is a first-order prediction. |
| D4 | State-decay sensitivity (`decay_gain`, `decay_time_constant`, `decay_sensitivity`, `quantize_decay`, `state_decay_sensitivity`) | SSM decay fragility and the log-time-constant basis | `StateDecaySensitivity` (n_channels, seq_len, mean_gain, max_gain, slow_channel_fraction, recommended_basis) (RV) | artifact (from weights) | Per-channel scatter: time constant (log) vs gain, slow channels highlighted | Models; RS-004 | Read `A_log` parameters at attach. S | SSM models only. |
| D5 | `behavioral_agreement.py` (`correctness_agreement`, `flip_rate`, `behavioral_agreement`, `noise_floor`, `evaluate`) | Decision-level comparison of base vs quantized predictions with a noise floor | `FlipResult` (acc_base, acc_quant, CA, regressions, recoveries, churn, net_delta, McNemar stat and p), `NoiseFloor`, `BehavioralReport` (agreement, disagreement, excess_disagreement, excess_z) (RV) | live-capable (paired run), usually artifact | 2x2 flip matrix (base right/wrong x quant right/wrong) with counts; excess-over-floor bar with z; running token-agreement line during a paired generation | Models, ReadScope; RS-003, QI-003 | Live needs a paired fp16 run (teacher-forced or free-running) on the same prompts; token agreement per step is cheap once both logits exist. M/L | Needs a reference run; without a noise floor, disagreement cannot be attributed to quantization (module's own point). |
| D6 | Consumer metrics (`consumers.py`: `attention_softmax`, `attention_topk`, `topk_*`, `read_operator`, `declared`; `nominal_per_item`, `row_cosine`) + `tqp plan consumers` | Registered per-item metrics a plan is judged on; cosine only as a diagnostic | per item scores (RV); registry listing (RV) | live (the attention consumers are exactly A7's metric) | Registry table (name, exact/proxy, evidence kind, target); for a run, per-query score distribution (violin) | ReadScope, Models; RS-001, API-001 | Reuse in A7. S | `AttentionScoreConsumer` needs queries. |
| D7 | Read operators (`read_operators.py`: identity, declared, `attention_analytic`; `error_covariance`, `consumer_distortion`) + `read_operator_conformance.py` + `plugins/tqp-readscope` (readscope_blind, readscope_jacobian) | PSD `P_C` for a consumer; distortion `tr(P_C Sigma_delta)` | operator (D,D) (RV); distortion scalar (RV); conformance report checks (SD) | artifact, live-capable per head | Per-head eigen-spectrum of `P_C` (dB) against the codec's per-direction error, i.e. the existing spectrum-analyzer view applied to one attention head's keys | ReadScope; RS-004, RS-005 | `attention_analytic` needs the head's queries (A7 hook); compute off-thread on a reservoir. M | Two defensible operators differ materially (module: ~0.3 subspace overlap); always name the provider. |
| D8 | `QualityMonitor` (`monitor.py`) + `tqp monitor` | Rolling cosine and (A2) tangential fraction over (original, reconstructed) pairs; drift and radial-drift flags; Prometheus dict | `stats()`, `metrics_dict()` (mean/min/std/p95 cosine, alerts, is_healthy, drift_detected, median_tangential_fraction, radial_drift_detected, certificate validity fields) (SD) | live-capable (feed spilled K or V pairs from A6) | Tangential-fraction sparkline per layer with the floor; health chip | Models, Overview health; OV-004 | Feed the sampled spill pairs from A6. S once A6 exists | Its cosine fields are not acceptance for keys; show the tangential fraction and hide or grey cosine for keys. |
| D9 | `tqp plan run` (`planner.py`, schema `compression_plan`) with `--target kv_key/kv_value/weight`, `plan explain`, `plan replay` | Measures every registered codec for a target against the consumer on a calibration split, verifies on held-out data, emits a plan record with a fallback policy | `CompressionPlan` (candidate_results with quality intervals, selected codec/params, expected cost/quality, certificate_requirement, fallback_policy, evidence, environment, preflight) (AR); replay report (identity match, codec agreement, delta, reproduced) | artifact | Candidate forest plot (consumer metric interval per codec, floor line, winner marked); replay verdict chip | Benchmarks, ReadScope; BM-002, BM-005, RS-006, NFR-009 | Offline on a dumped key/value .npy (dump from the A6 reservoir). S | For KV targets the planner reshapes rows to (1, 1, S, D), i.e. one head, so per-head structure is flattened; say so. |
| D10 | Observer contracts for a model (`observer.py`, `.tqo`, `tqp observer`) and `tqp plan refine` / `plan compat` (`refinement.py`) | Contract naming target (kv_key possible), consumers, budget; refine/compat compare two or more observers' read operators | contract digest and reference; refinement tax, operator overlap, compatibility matrix (AR) | artifact | Compatibility matrix heatmap (built-for x read-by, loss ratio) | ReadScope; RS-001, RS-003, RS-005 | Offline. S | Built and tested around embeddings; applying to KV keys needs an attention_analytic operator with real queries. Mark as untested on KV. |
| D11 | `modality.py` presets | Embedding-model PCA/bit presets | `ModalityPreset` table (RV) | artifact | Plain table only | Benchmarks | S | Embedding presets, not LLM KV; "expected_cosine" is not an acceptance metric. |

## 5. Runtime policy

| # | Feature (files) | What it does | Data produced or exposable | Live / artifact | Best visualization | Workspace, req ids | Hook, effort | Honesty notes |
|---|---|---|---|---|---|---|---|---|
| E1 | `TQPRuntimePolicy` (`runtime_policy.py`): `evaluate_kv_keys`, `evaluate_routing`, `evaluate_decay`, `evaluate_a2`, `evaluate_retrieval`, `evaluate_certificate`, `evaluate_index_drift`, `evaluate_all` | Reads a fragility measurement, compares to a floor, returns a conservative action | `RuntimeDecision` (situation, action, conservative, reason, measured, params) (RV, `as_dict`); floors (routing_margin_floor 0.02, decay_slow_fraction_ceiling 0.02, radial_drift_floor 0.15, etc.) | live-capable, but **not called by any generation path** | Decision timeline: one lane per situation (kv_keys, routing, a2, ...), a mark per evaluation colored cheap/conservative, with reason on hover and the measured value vs floor | Overview events, Models; OV-005, OV-006, RS-008 (the `reason` string is computed, not generated) | Console calls `evaluate_*` on the sampled measurements it already has (D2, D3, D8) at a low rate. The decision is advisory unless someone acts on it. M | Label "advisory: the running cache does not change". Never imply automatic back-off (requirements section 13 forbids automatic production changes). |
| E2 | Planner fallback policy (`planner._fallback_policy`) | Writes the runtime action and triggers into the plan record | action, reason, thresholds, triggers (AR) | artifact | Trigger checklist next to the plan, each trigger lit if the live measurement crosses it | ReadScope; RS-005 | Compare live readings to the recorded thresholds. S | Triggers are prose; only some map to a live measurement. |

## 6. Serving integrations

| # | Feature (files) | What it does | Data produced or exposable | Live / artifact | Best visualization | Workspace, req ids | Hook, effort | Honesty notes |
|---|---|---|---|---|---|---|---|---|
| F1 | `TurboQuantKVConnector` + `TurboQuantBlockStore` (`connectors/vllm_v1.py`) | vLLM V1 KV connector: quantize finished layers into a host store (per-channel keys, polar values), restore matched prefixes; async saves with backpressure; persist to dir; evict | `stats()` (requests, records, key_plugin, value_plugin); `matched_tokens(request_id)`; records with num_tokens, shape, dtype (IN) | live | Tier view: requests x layers grid of stored records; matched-token bar per request; queue depth gauge | Models (serving tab), Index & Shards analogue; IS-001, IS-003 | Already has the metrics object (F2); poll `get_metrics()`. S | Real-engine lane runs only with vLLM installed; otherwise an importable shim. MVP load is synchronous. |
| F2 | `ConnectorMetrics` (`connectors/metrics.py`) | Thread-safe counters and 1024-op latency reservoirs; Prometheus exposition | saves, loads, hits, misses by cause (empty, corrupt, incompatible, timeout, declined), evictions, integrity_failures, backpressure_blocked/dropped, worker_fallbacks, bytes_logical/physical, records_persisted/restored, hit_rate, effective_expansion, save/load latency p50/p95/p99 (SD) | live | Hit/miss stacked bar by cause over time; save/load latency percentiles sparkline; logical vs physical bytes with expansion ratio | Overview, Models; OV-001, OV-002 (queue pressure), OV-004, section 10 Operational | Already live; register these as `kv_tier.*` MetricSpecs. S | Latency window is the last 1024 operations, not time-based; the spec must say "window: last 1024 ops". |
| F3 | `KVIdentityProfile`, `prefix_block_hashes` (`connectors/identity.py`) | Content-addressed identity of a KV tier (model, revision, weights, tokenizer, RoPE, backend, dtype, block size, quant); incomplete profile matches nothing | `digest()`, `unknown_fields()`, `is_complete()`, `compatible()` (RV) | artifact (per session) | Identity card listing each field with a missing-field warning; digest in monospace | ReadScope, Models; RS-002, IS-004, NFR-009 | At attach (from the model config). S | Rule: uncertain compatibility means miss and recompute; show unknown fields in amber. |
| F4 | `TurboQuantKVManager`, `TurboQuantKVBackend` (`vllm_plugin.py`) | Older multi-layer manager and a vLLM-style backend (not the V1 connector) | `layer_stats(i)` (`LayerStats`: length, cold/hot length, cold/hot/total bytes, uncompressed bytes, compression_ratio), `memory_stats()` (SD), `estimate_capacity(max_memory_gb)` (RV) | live | Same per-layer tier bars as A2/A3; capacity-vs-budget readout | Models | Poll. S | Uses the default non-robust key recipe and an fp32 baseline (findings 6, 8). |
| F5 | llama.cpp / GGUF (`examples/llama_integration.py`, `benchmarks/benchmark_llama.py`, `benchmark_e2e.py --backend llama-cpp`, `JETSON_NANO.md`) | Sketch with random K/V; simulated access-pattern benchmark; GGUF baseline runner | e2e row: decode_tok_s, j_per_token, peak_mem_mb, backend, quantizer provenance (AR) | artifact | Benchmarks row with provenance chip "GGUF baseline, not TurboQuant" | Benchmarks; BM-001 | n/a | No real llama.cpp KV hook exists. Do not present these as TurboQuant results. |
| F6 | NATS (`nats_codec.py`, `nats_transport.py`, `nats_worker.py`) | Embedding wire codec; request/reply transport and pod entrypoint for sharded vector search | `TurboQuantNATSCodec.stats()`, payload sizes | n/a for models | Belongs to Index & Shards | Index & Shards | n/a | Not a KV transport (finding 9). |

## 7. Plugins

| # | Feature (files) | What it does | Data produced or exposable | Live / artifact | Best visualization | Workspace, req ids | Hook, effort | Honesty notes |
|---|---|---|---|---|---|---|---|---|
| G1 | Registry (`plugins.py`: `register`, `get_plugin`, `create`, `available_plugins`, `load_entry_point_plugins`, `capabilities`, `affine_params`, `affine_codes`, `outlier_csr`, `native_dtype`, `resolve_plugins`) + `tqp plugin list` | Named quantizer factories (in-tree per_channel, polar, tq_embedding, faiss_*; entry points fp8_kv, nvfp4_kv, bnb_nf4, bnb_llm_int8, gptq, awq) with optional affine / native-dtype capabilities | per plugin: name, tier, targets, description (RV); capability flags: affine (fused M4 eligible), outlier CSR, native dtype, bit_widths, requires_calibration, hardware (RV) | artifact (discovery) | Capability matrix: plugins x {target, tier, affine/fused-eligible, native dtype, hardware req, installed}; `resolve_plugins(model)` output as the anatomy map (D1) recolored by chosen plugin | System, Models; API-001, API-008, IS-004 | At attach. S | "Registered" is not "installed and working"; run conformance (G2) before marking green. |
| G2 | Conformance kit (`plugin_conformance.py`: `run_conformance`, `assert_conformance`) + `tqp plugin conformance` | roundtrip, packed, affine==decompress, CSR validity, serialization | `ConformanceReport.results` per check: pass / skip: reason / FAIL: detail (SD) | artifact (seconds) | Plugin x check grid, green/grey/red with the reason on hover | System, ReadScope; RS-005 | On demand (a low-risk action). S | Default sample is synthetic (seeded Gaussian + per-head DC offset); say so. Validity, not accuracy. |
| G3 | Out-of-tree plugins (`plugins/tqp-trtllm` incl. native fp8 passthrough; `tqp-bnb`; `tqp-gptq-awq`; `tqp-readscope`) | Recipes and read operators registered via entry points | same fields as G1; trtllm `native.py` fp8 container with per-head scale | artifact | Rows in G1; for fp8_kv, a "native passthrough available on this GPU" chip | System | S | Hardware-dependent; `native_dtype` means storage passthrough, not fp8 compute (its docstring). |
| G4 | `benchmarks/fuzz_plugins.py` | Time-budgeted random combination sweep over contracts and backends | failures by combination | artifact | Failure list grouped by (plugin, backend, shape) | Benchmarks; BM-003 | Offline. S | Fuzz coverage, not proof. |

## 8. Benchmarks and campaign artifacts that produce model-side results

| # | Source | Model-side data | Best visualization | Workspace, req ids | Honesty notes |
|---|---|---|---|---|---|
| H1 | `benchmarks/kvquant_matrix/` (`results_matrix.json`, `results_rescue.json`, `results_longgen*.json`, `results_latency.json`, `gate_dispositions.json`; runners `tq_paper_lb_shard.py`, `wikitext_ppl.py`, `keys_*`, `score_keys.py`, `key_coding.py`) | LongBench task scores and WikiText-2 ppl per model x key codec (fp16, nf4, nf4a, pre/post RoPE); rescue sweep with verdicts; long-generation gaps; kernel latency sweep (fp16 / warp / dequant ms by H and S); gate dispositions with amendments | Model x codec heatmap of the gap to fp16 per task (diverging, zero centered); latency curves vs S per method; dispositions list | Benchmarks; BM-001..005 | The harness is "quantize prefill once", not the HF drop-in; say which method produced each cell. Part II (`score_keys`) results are sealed until records are committed (per project memory); show only committed files. |
| H2 | `bench_kv_serving.py` | Per sampled layer: attention KL vs fp16 per step; free-running token agreement; decode ms/token; cold-token footprint | The reference design for the live Models quality panel (A7) | Models, Benchmarks | No committed results. fp32 compute, eager attention: relative throughput meaningful, absolute not (its docstring). |
| H3 | `benchmark_longbench_parity.py` | Per layer: fp16 attention vs fused decode output on real activations (via `eager_attention_forward` recording hook) | Per-layer relative error bar | Benchmarks, ReadScope | Uses PolarQuant keys via `TurboQuantPGVector` codes; not the recommended key format. |
| H4 | Kernel latency: `bench_k2_volta.py`, `bench_outlier_latency.py` (+ `RESULTS_outlier_latency.md`), `h200_longcontext_bench.py`, `p5_triton_bench.py` (+ `RESULTS_p5_triton.md`), `benchmark_kv_kernel.py`, `benchmark_kv_decode.py`, `benchmark_kv_adc.py`, `benchmark_decode_overhead.py` (+ `RESULTS_decode_overhead.md`) | ms per decode step vs S per method; outlier_frac sweep; Triton vs RawKernel exactness and latency | Latency-vs-context curves, one line per method, with the running session's (S, ms) point overlaid | Benchmarks, System; BM-002, BM-004 | GPU-specific; always show the GPU name. |
| H5 | `benchmark_e2e.py`, `benchmark_edge.py` (`PowerSampler`: NVML or tegrastats) | decode tok/s, J/token, peak process memory; KV memory fp16 vs compressed | Energy-per-token bar per quantizer with provenance | Benchmarks, System (PowerSampler is reusable for live GPU power) | `PowerSampler` is the only NVML/power reader in the repo; it lives in `benchmarks/`, not the package. |
| H6 | Mechanism studies: `rope_offset_frequency.py`, `deterministic_zeropoint.py`, `lb_keys_4bit.py`, `validate_olmoe_routing.py`, `validate_mixtral_routing.py`, `validate_mamba_decay.py`, `experiments/k_wavelength_probe.py`, `experiments/matched_bit_projection_sensitivity.py`, `experiments/validate_rope_aware_k.py`, `experiments/behavioral_metric_demo.py` | DC-offset mass by rotary wavelength; zero-point variants' ppl; key 4-bit comparisons; routing flip vs bits; SSM drift by basis; K-row damage vs wavelength | Evidence cards linked from the live panels they justify (A8 links to rope_offset_frequency; D3 to the routing validations) | ReadScope, Benchmarks; RS-002 | Artifacts with their own model and date. |
| H7 | `benchmark_autoconfig.py`, `kv_quant_shootout.py/.ipynb`, `benchmarks/observer_advantage/` (embedding-side Part I) | Preset comparisons; shootout tables | Tables | Benchmarks | Mostly synthetic or embedding-side. |

---

## 9. What the telemetry contract needs for a live Models session

Nothing in the model path emits telemetry today (finding 7). A minimal instrumentation layer,
kept off by default and sampled (NFR-003):

1. **Per-token clock** (S): a `StoppingCriteria` or `LogitsProcessor` passed to `generate` is called
   once per step with the ids; record `perf_counter` per step. Gives prefill ms (first call), decode
   ms/token, tokens/s. No model changes.
2. **Instrumented layer** (S/M): subclass or wrap `TurboQuantLayer` (or set a module-level tap the
   layer checks with one truth test, like `telemetry.trace`) to record per spill: layer, tokens,
   compress ms, container `nbytes()` by component; per update: cold chunk count, decompress ms,
   materialized bytes. All values are already on the host.
3. **Sampled quality** (M): every Nth spill, decompress the new chunk and compare to the source
   block (key/value error per head, per channel; outlier counts). Diagnostic only.
4. **Attention consumer** (L): shadow fp16 K/V for 1-3 probe layers plus an attention-function
   wrapper on sampled steps, computing KL and top-k overlap per head (the `attention_softmax`
   consumer). Needs eager attention on the probe steps. Declares `kind: sampled`, reference
   "fp16 shadow cache of probe layers".
5. **Device resources** (S): `torch.cuda.memory_allocated` / `max_memory_allocated` per step (no
   sync); NVML utilization, power, temperature on a 1-10 Hz thread (reuse `PowerSampler` logic).
   Flip `capabilities().features.gpu_utilization` only when NVML is present.

Proposed metric specs (name, unit, kind): `model.decode.tok_s` (tokens/s, measured, rate over 10 s),
`model.decode.step_ms.p50/p95` (ms, measured), `model.prefill_ms` (ms, measured, last),
`kv.tokens.hot` / `kv.tokens.cold` (tokens, measured, last, per layer), `kv.chunks.cold` (count,
measured), `kv.bytes.host_compressed` (bytes, measured, split by component), `kv.bytes.fp16_equiv`
(bytes, derived), `kv.bytes.device_transient` (bytes, sampled), `kv.ratio_vs_fp16` (x, derived, the
baseline in the name), `kv.spill_ms` and `kv.materialize_ms` (ms, measured), `kv.key_err_db` /
`kv.value_err_db` (dB, sampled, reference "source block at spill", diagnostic),
`kv.attn_kl` (nats, sampled, reference "fp16 shadow of probe layers"), `kv.cold_attention_mass`
(fraction, derived, only on the fused cache), `gpu.util_pct`, `gpu.power_w`, `gpu.mem_mb`
(sampled, NVML), `policy.decision` (event), plus the existing `ConnectorMetrics` fields as
`kv_tier.*` with "window: last 1024 ops".

A new entity, `Model` (or `Runtime.model`), should carry: model id and revision, architecture,
n_layers, n_heads, n_kv_heads, head_dim, rope_theta, dtype, device, attention implementation,
effective cache configuration per tensor role, and the `KVIdentityProfile` digest (F3).

---

## 10. Top 10 visualizations for a live HF model session (by value)

1. **Per-layer KV tier and bytes bar** (A2+A3): hot vs cold tokens and bytes per layer, three
   baselines (fp16-equivalent, host compressed resident split by component, device transient).
   This one panel exposes findings 2, 3 and 4 at a glance. Cheap, exact.
2. **Decode throughput timeline with cost breakdown** (clock + A5): tokens/s and ms/step over
   generated tokens, stacked into model compute vs cold decompress vs spill compress, with the
   cold-chunk count as an overlay. Shows finding 5 directly.
3. **Layer x head attention-KL heatmap** (A7): the accepted consumer metric for keys, sampled on
   probe layers against an fp16 shadow. The quality truth of the session.
4. **Mode and effective-config header** (A1, A10, B2-B5, B8): codec per tensor role, host vs
   device, decompress-then-attend vs fused kernel (with the reason), plan-vs-effective diff.
   Stops the UI from implying fused kernels or a kv_plan that is not running.
5. **Head x channel DC-offset heatmap with RoPE long-wavelength channels marked** (A8, A11,
   `dc_channel_mask`): shows why asymmetric NF4 is needed on this model, from data the codec stores.
6. **Spill event waterfall** (A4): step x layer ticks colored by chunk size; makes the one-token
   decode chunks and their bytes visible.
7. **Operator-regime anatomy map** (D1, G1 `resolve_plugins`): layers x module slots colored by
   regime and recommended discipline, with fx confidence. One-shot at load, frames every other panel.
8. **(A2) probe per layer** (D2, D8): Spearman polar vs per-channel dot plot plus the tangential
   fraction sparkline, recomputed on a key reservoir; the early warning the project built.
9. **Runtime-policy decision timeline** (E1): lanes per situation, cheap vs conservative marks,
   measured value against the floor, reason on hover, labeled advisory.
10. **GPU resource strip** (resource hook, B8, H5 PowerSampler): device memory allocated and
    peak, utilization, power, J/token, with freshness indicators.

Close runners-up: key/value reconstruction error heatmap (A6, diagnostic only); cold attention mass
(B1, only on the fused cache); behavioral flip matrix from a paired run (D5); weight bit-plan
heatmap (C5) for a weight-quantized model.

## 11. Features with no sensible visualization (and why)

- `backend.to_numpy`, `is_torch_tensor`, `is_cupy_array`, `_TorchXP`: plumbing. They only
  contribute a backend label to the mode badge.
- `_pack_indices` / `_unpack_indices`, `packed_codes`, `cuda_kernels` pack/unpack: byte-layout
  utilities. Their only observable is bytes, already counted in A3.
- `_triton_kernels.py` kernel bodies, kernel caches (`_KERNEL`, `_kernels`): internals; the useful
  facts (compiled yes/no, launch params) belong in the kernel inventory row, not a chart.
- `TurboQuantLayer` contract methods (`get_mask_sizes`, `get_max_cache_shape`, `crop`,
  `reorder_cache`, `batch_repeat_interleave`, `batch_select_indices`, `offload`, `prefetch`,
  `reset`): correctness plumbing for `generate`. Only worth an event mark when `crop` or
  `reorder_cache` fires, because those decompress and re-ingest the whole cache (a cost spike).
- `KVIdentityProfile` internals and `prefix_block_hashes`: a card with fields and digest, not a
  chart. Hash chains have no meaningful geometry.
- `modality.py` presets: a static table of embedding presets, not model-side.
- NATS codec, transport and worker: not KV; belong to Index & Shards.
- `examples/llama_integration.py`, `benchmark_llama.py`: simulated K/V; visualizing them would
  present synthetic numbers as a model session.
- `CompressedKV.compression_ratio(packed)`, `theoretical_compression_ratio()`: formulas that
  duplicate the measured bytes of A3. Show the measured number and keep the formula as a tooltip.
- `plugins.native_dtype`, `capabilities` on their own: flags; they appear as matrix cells (G1).
- `agent_tools` wrappers: JSON-in/JSON-out duplicates of D2 and friends; nothing new to draw.

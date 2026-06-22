# Changelog

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

# Changelog

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

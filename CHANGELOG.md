# Changelog

## Unreleased

Theory-to-practice round: the companion paper's v0.8 results
([the-angular-observer](https://github.com/ahb-sjsu/the-angular-observer))
land as two instruments, following the house pattern of turning each incident
into an instrument.

### Added
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

# CLAIMS.md — what we claim, and exactly how to reproduce it

At-a-glance claim → reproduction table. Every row here is an entry in
[`claims.yaml`](claims.yaml) under the **Ledger id** shown, with the same status word;
[`tests/test_claims_ledger.py`](tests/test_claims_ledger.py) fails CI if the two files
disagree, and `tqp replay --list` prints the ledger. The **central result** is Track 1
(embedding compression + compressed-domain retrieval) — CPU/Colab-reproducible. Track 2
(KV-cache / operator-aware quantization) is the engineering-package extra and needs a GPU.
Full detail + scope caveats: [`docs/claims.md`](docs/claims.md).

**Status legend** (one word, identical in `claims.yaml`):
*executable* = `tqp replay <id>` runs it and gates on declared floors (hermetic subset in
CI; full run local) ·
*reproducible* = one *Run all* of the linked notebook on public data reproduces the claim ·
*needs-local-run* = reproducible on public data, but full scale / timing needs your
hardware and no result at that scale is committed ·
*partial* = notebook runs, one variant left to fill in ·
*experimental* = requires GPU + model weights, treat as engineering/preview ·
*reported* = measured once on data that is **not in the repo** (private sample or internal
deployment) — a real measurement, but one a third party cannot replay from here ·
*retracted* = published, then found irreproducible; kept visible on purpose.

> **What is actually CI-gated.** Exactly one retrieval number: `embedding_glove_recall`
> (~9.6× at recall@10 ≈ 0.999 on public GloVe-100, hermetic subset with loose floors on
> every push). The README's 32× / recall@10 ≈ 0.999 headline is a *reported* row: measured
> on a private LaBSE sample, corroborated at 1M on Gutenberg, not replayable from this repo.

## Track 1 — Embedding / vector-DB compression (central result)

| Claim | Ledger id | Public reproduction? | Dataset | Command / notebook | Hardware | Status |
|---|---|---|---|---|---|---|
| **~9.6× compression at recall@10 ≈ 0.999** (full-dim + 3-bit + 12× rerank) | `embedding_glove_recall` | Yes — **`tqp replay embedding_glove_recall`**, gated in CI | GloVe-100-angular (1.18M; hermetic subset bundled) | [`benchmarks/canonical_glove.py`](benchmarks/canonical_glove.py) (`--small` CI / `--full` local) | CPU | **executable** (reranked recall + compression floors) |
| **32× at recall@10 ≈ 0.999, ties OPQ, beats RaBitQ, builds ~20× faster than OPQ** — the README headline | `embedding_labse_32x_headline` | **No** — private 199k LaBSE sample; 1M Gutenberg corroboration regenerable via `gutenberg_embed.py` | LaBSE 768-d, 199k (private) · Gutenberg-1M LaBSE | [`benchmarks/RESULTS_labse_199k.md`](benchmarks/RESULTS_labse_199k.md) · [`RESULTS_gutenberg_1m.md`](benchmarks/RESULTS_gutenberg_1m.md) · [`RESULTS_rabitq_comparison.md`](benchmarks/RESULTS_rabitq_comparison.md) | CPU | **reported** (single measured run; RaBitQ builds ~100× faster than tq-pro, the build win is vs OPQ) |
| **Beats RaBitQ on recall / ties OPQ at matched bytes** on public data | `embedding_beats_rabitq_ties_opq` | Yes (notebook auto-downloads); no committed RaBitQ result on public data | GloVe-100 / NYTimes-256 / deep-image-96 | [`00_canonical_sota_embedding.ipynb`](notebooks/claims/00_canonical_sota_embedding.ipynb) · [`benchmarks/canonical_embedding.py`](benchmarks/canonical_embedding.py) | CPU (GPU optional) | **needs-local-run** (committed public rows are PQ/OPQ only: [`RESULTS_glove.md`](benchmarks/RESULTS_glove.md)) |
| **Up to 27.7× compression at high recall@10** (PCA-384 + 3-bit, 5× oversample + rerank) | `embedding_27x_high_recall` | **No** — only holds on high-dim concentrated-spectrum encoders; the public notebook datasets are compact and do not truncate | BGE-M3 3.3M (private) · LaBSE 199k (private) | [`docs/benchmarks/embeddings.md`](docs/benchmarks/embeddings.md) | CPU | **reported** |
| **PCA rotation makes non-Matryoshka models truncatable, no retraining** | `pca_truncation_no_retrain` | Yes | GloVe-100-angular | [`01_pca_truncation.ipynb`](notebooks/claims/01_pca_truncation.ipynb) | CPU | **reproducible** |
| **Learned codebooks reduce quantization error ~22%** | `learned_codebooks_error_reduction` | Partially | GloVe-100-angular | [`02_learned_codebooks.ipynb`](notebooks/claims/02_learned_codebooks.ipynb) | CPU | **partial** (learned-variant cell to fill) |
| **ADCIndex compressed-domain search throughput** | `adcindex_throughput` | Yes (recall); QPS local | GloVe-100-angular | [`03_adcindex_throughput.ipynb`](notebooks/claims/03_adcindex_throughput.ipynb) | CPU | **reproducible** (abs. QPS = local run) |
| **Up to 114× pipeline-storage compression** | `embedding_114x_pipeline_storage` | **No** — dataset-dependent; the figure comes from private high-dim corpora | dataset-dependent | operating point of [`00_...ipynb`](notebooks/claims/00_canonical_sota_embedding.ipynb) on the right data · [`docs/benchmarks/embeddings.md`](docs/benchmarks/embeddings.md) | CPU | **reported** (storage-only accounting, recall via rerank) |
| **Graceful behaviour on OOD anisotropic / heavy-tailed embeddings** | `ood_anisotropic_robustness` | Yes | synthetic anisotropic (power-law spectrum, Student-t tails) | [`04_ood_anisotropic.ipynb`](notebooks/claims/04_ood_anisotropic.ipynb) | CPU | **reproducible** (robustness envelope tracks spectral concentration) |
| **PCA+TQ preserves recall@10 at >10× in seconds** (mechanism smoke) | `track1_recall_smoke` | Yes — `tqp replay track1_recall_smoke` | synthetic, generated in-script | [`benchmarks/replay_smoke.py`](benchmarks/replay_smoke.py) | CPU | **executable** (not CI-gated) |

> **Honest scope.** PCA *truncation* wins only for high-dimensional / concentrated-spectrum
> embeddings (sentence, vision). On compact descriptor sets (GloVe-100, NYTimes-256) it loses to
> PQ/OPQ; at full dimension / matched bytes the TurboQuant scalar quantizer still **wins on GloVe and
> vision, ties on NYTimes** — see [`benchmarks/RESULTS_glove.md`](benchmarks/RESULTS_glove.md).
> Index-build cost: the 4–20× advantage is **over OPQ** (20× at 199k, 4× at 1M). RaBitQ builds
> in well under a second and is far cheaper to build than tq-pro; tq-pro's advantage over RaBitQ
> is recall at matched bytes, not build time.

## Track 2 — KV-cache / operator-aware quantization (engineering package; GPU)

| Claim | Ledger id | Public reproduction? | Dataset | Command / notebook | Hardware | Status |
|---|---|---|---|---|---|---|
| **KV *keys* need per-channel / asym treatment** (PolarQuant keys: 0.995 cosine, ppl ≈10⁴; incl. Qwen2.5 collapse & recovery) | `kv_keys_per_channel` | Yes (on GPU) | Llama / Mistral / Qwen + perplexity/LongBench | [`10_kv_keys_per_channel.ipynb`](notebooks/claims/10_kv_keys_per_channel.ipynb) · [`docs/KV_KEYS_FINDING.md`](docs/KV_KEYS_FINDING.md) | GPU | **experimental** (PolarQuant catastrophe is fake-quant on Qwen2.5 only; see model card scope) |
| **NF4 / asym-NF4 vs uniform K4** across model families | `kv_nf4_matrix` | Yes (on GPU) | Llama / Mistral / Qwen | [`11_kv_nf4_matrix.ipynb`](notebooks/claims/11_kv_nf4_matrix.ipynb) · [`benchmarks/RESULTS_longbench.md`](benchmarks/RESULTS_longbench.md) | GPU | **experimental** (short-gen matrix; long-gen rows retracted below) |
| **asym-NF4 long-generation degradation curve** (13.7 ROUGE-L at 512 tokens) | `kv_longgen_nf4a_degradation` | Irreproducible | Qwen2.5-7B / Llama-2 on LongBench gov_report, multi_news | [`benchmarks/RESULTS_longgen_revalidation.md`](benchmarks/RESULTS_longgen_revalidation.md) · [`REVAL-2026-08-08.md`](benchmarks/kvquant_matrix/REVAL-2026-08-08.md) | GPU | **retracted** (2026-08-15; re-validated gap −0.31; the real 26.64 collapse is symmetric NF4) |
| **Comparison vs KVQuant / KIVI** on LongBench/perplexity | `kv_vs_kvquant` | Partially | model + task | [`12_kv_vs_kvquant.ipynb`](notebooks/claims/12_kv_vs_kvquant.ipynb) | GPU | **experimental** (our KVQuant reimpl. not a faithful repro of their strongest number) |
| **Fused decode kernel speedup** | `kv_fused_decode_speedup` | Yes (on GPU) | microbenchmark | [`benchmarks/benchmark_kv_kernel.py`](benchmarks/benchmark_kv_kernel.py) | GPU + build toolchain | **experimental** |
| **MoE routing fragility is carried by the routing margin** (real OLMoE + Mixtral) | `moe_routing_margin` | Yes (on GPU) | OLMoE-1B-7B, Mixtral-8x7B on WikiText-2 | [`benchmarks/validate_olmoe_routing.py`](benchmarks/validate_olmoe_routing.py) · [model card](docs/model_cards/moe_routing.md) | GPU | **experimental** |
| **SSM decay must be quantized in the native A_log basis** (real Mamba-790m) | `ssm_decay_basis` | Yes (on GPU) | Mamba-790m on WikiText-2 | [`benchmarks/validate_mamba_decay.py`](benchmarks/validate_mamba_decay.py) · [model card](docs/model_cards/ssm_decay.md) | GPU | **experimental** |

> **Erratum (2026-08-15) — the long-generation degradation curve.** Track 2 materials
> (`KV_QUANT_GUIDE.md` §5, `results_longgen.json`, paper draft §5.5, CHANGELOG "Honest
> limitation") recorded asym-NF4 (`nf4a`) losing **13.7** ROUGE-L on LongBench gov_report
> (Qwen2.5-7B) at 512 generated tokens, with the gap growing 0.25 → 4.19 → 9.60 → 13.7 across
> 64/128/256/512. **That curve is irreproducible under `nf4a` as labeled**: re-validation with
> the same harness and LongBench's own metrics measures a gap of **−0.31** (n=40), and the
> multi_news / Llama-2 rows fail the same way. A **real and larger** long-generation collapse
> (**26.64** points on the same cell) exists under the **symmetric** `nf4` codebook. Root
> cause: most probably arm-labeling contamination between the back-to-back `nf4` / `nf4a`
> sweep arms during off-repo aggregation and hand transcription — implementation drift is
> ruled out by git history (`_quant_nf4a_group` is unchanged since it first appeared,
> `289bdfc`, before the results commit `4f7baab`) and every committed driver exports
> `CODEBOOK=nf4a` explicitly; definitive attribution is impossible because the raw run
> outputs were never committed. The recorded rows are left visible with this correction
> pointing at them, and the ledger carries it as `kv_longgen_nf4a_degradation` with status
> *retracted*. Evidence: [`benchmarks/kvquant_matrix/REVAL-2026-08-08.md`](benchmarks/kvquant_matrix/REVAL-2026-08-08.md)
> · analysis: [`benchmarks/RESULTS_longgen_revalidation.md`](benchmarks/RESULTS_longgen_revalidation.md).
> The Qwen **symmetric-NF4 collapse** claims above (`kv_keys_per_channel`) are unaffected —
> that collapse re-validates, larger than recorded.

---

*The bytes/vector figures are computed analytically (`out_dim × bits ÷ 8`) to keep the harness
library-agnostic. (`estimate_storage()` was dimension-agnostic before v1.4.1 and now tracks the real
pipeline config — see [`docs/claims.md`](docs/claims.md).) Notebooks ship with empty outputs; numbers
appear when you run them.*

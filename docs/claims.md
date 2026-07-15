# Claims & evidence ladder

> For the **at-a-glance claim → reproduction table** (Claim / Public reproduction? / Dataset /
> Command / Hardware / Status), see **[`CLAIMS.md`](../CLAIMS.md)** at the repo root. This page is the
> detailed version with evidence levels and scope caveats.

Every headline claim in this project sits at one of five evidence levels, and **each rung links to a
runnable notebook** under [`notebooks/claims/`](../notebooks/claims). The two tracks —
**embedding/vector-DB compression** and **KV-cache compression** — are evaluated with entirely
different metrics and are kept separate below.

**Levels**
- **L1 — Publicly reproducible in Colab.** Public data, no special hardware, one *Run all*.
- **L2 — Public data, local hardware required.** Reproducible but needs a GPU and/or large RAM/time.
- **L3 — Private or large-scale internal benchmark.** Data or scale not in the public repo.
- **L4 — Paper result, not yet independently reproduced.** Reported vs. published baselines.
- **L5 — Engineering claim, covered by unit tests only.** Correctness/perf asserted by the suite.

All embedding notebooks embed the **same** verified harness
([`benchmarks/canonical_embedding.py`](../benchmarks/canonical_embedding.py)) — the notebook code is
the CI-tested code — and score against a public dataset's **provided** ground-truth. Every ANN method
is measured single-stage *and* +rerank at an **identical** oversample factor, so the comparison is
apples-to-apples. Notebooks ship with empty outputs: numbers appear only when you run them.

---

## Track 1 — Embedding / vector-DB compression

| Claim | Level | Reproduce (click to run) |
|---|:--:|---|
| **Beats RaBitQ on recall / ties OPQ at scale, builds index 4–20× faster** — the headline result | L2 | **[`00_canonical_sota_embedding.ipynb`](../notebooks/claims/00_canonical_sota_embedding.ipynb)** — one table: flat / PQ / OPQ / IVFPQ / RaBitQ / PCA-only / TQ-only / PCA+TQ / ADCIndex, identical rerank, on public GloVe. |
| **27× storage compression @ high recall@10** (5× oversampling + reranking; all methods benchmarked identically) | L2 | Same notebook — read the compression×/recall columns at your operating point. |
| **PCA rotation makes non-Matryoshka models truncatable, no retraining** (Varici et al. 2025) | L1 | **[`01_pca_truncation.ipynb`](../notebooks/claims/01_pca_truncation.ipynb)** — naive vs PCA truncation + retained variance, CPU-only. |
| **Learned codebooks reduce quantization error ~22%** | L2 | **[`02_learned_codebooks.ipynb`](../notebooks/claims/02_learned_codebooks.ipynb)** — learned vs fixed reconstruction MSE at matched bits. |
| **ADCIndex compressed-domain search throughput** | L5 (+L2 at scale) | **[`03_adcindex_throughput.ipynb`](../notebooks/claims/03_adcindex_throughput.ipynb)** — QPS / RAM vs fp32-flat at matched recall. |
| **Graceful degradation on OOD anisotropic / heavy-tailed embeddings** | L1 | **[`04_ood_anisotropic.ipynb`](../notebooks/claims/04_ood_anisotropic.ipynb)** — synthetic pathological corpus (power-law spectrum, Student-t tails); robustness envelope tracks spectral concentration. |
| **Up to 114× storage compression (PCA-Matryoshka + TurboQuant)** | L2 / dataset-dependent | Storage-only, recall preserved via oversampling+reranking; *not* a single-vector reconstruction bound. An operating point of the canonical notebook (raise compression, read the recall). |
| **Distribution-free rank-agreement floors** — for any corpus, measured distortion κ and concentration μ̂(κ) give guaranteed floors Kendall τ ≥ 1−2μ̂, Spearman ≥ 1−3μ̂, with *no* distributional assumptions (a shipped theorem, not a benchmark: [Bond, *Keep the Angle* v0.8](https://github.com/ahb-sjsu/the-angular-observer) + Daniels 1950). A vacuous floor is the principled per-corpus "exact reranking required" signal; autotune reports κ / μ̂ / τ-floor per operating point. | L5 | `turboquant_pro/rank_certificate.py`, floors verified against brute-force Kendall τ in `tests/test_rank_certificate.py`; surfaced in `autotune` results. |
| **(A2) consumer-metric probe & tangential-fraction drift monitor** — calibration-time family selection against the *declared* consumer metric (reproduces the v1.2.0 KV-keys catastrophe on synthetic key statistics) and a streaming guardrail for norm-dominated drift that cosine cannot see | L5 | `turboquant_pro/a2_probe.py`, `QualityMonitor` tangential stream; `tests/test_a2_probe.py`, `tests/test_monitor.py`. Scope note: the streaming fraction guards radial drift; direction-concentration (the keys regime) needs the end-to-end probe — both documented in [`KV_KEYS_FINDING.md`](KV_KEYS_FINDING.md). |

> **Honest scope (`benchmarks/RESULTS_glove.md`, reproduced by notebook 00).** PCA *truncation* wins
> only for high-dimensional embeddings with a concentrated spectrum (sentence/vision encoders). On
> already-compact descriptor sets (GloVe-100, NYTimes-256) it discards real variance and loses to
> PQ/OPQ; at **full dimension / matched bytes** the TurboQuant scalar quantizer still **wins on
> GloVe and vision, ties on NYTimes**. The notebook reproduces that full picture, not a cherry-pick.

## Track 2 — KV-cache compression

> These need a **GPU runtime** + Hugging Face model weights (not free-CPU Colab). The notebooks wrap
> the same `benchmarks/` entry points that produced `RESULTS_longbench.md`.

| Claim | Level | Reproduce |
|---|:--:|---|
| **KV-cache *keys* need per-channel / asymmetric treatment; generic reconstruction metrics mislead** — the core scientific finding (incl. Qwen2.5 collapse) | L2 | **[`10_kv_keys_per_channel.ipynb`](../notebooks/claims/10_kv_keys_per_channel.ipynb)** + [`docs/KV_KEYS_FINDING.md`](KV_KEYS_FINDING.md). |
| **NF4 / asym-NF4 key quantization vs uniform K4** (Llama / Mistral / Qwen) | L2 | **[`11_kv_nf4_matrix.ipynb`](../notebooks/claims/11_kv_nf4_matrix.ipynb)** — fp16 / uniform-K4 / NF4 / asym-NF4 / +outliers matrix. |
| **Comparison vs KVQuant on LongBench/perplexity** | L4 | **[`12_kv_vs_kvquant.ipynb`](../notebooks/claims/12_kv_vs_kvquant.ipynb)** — our KVQuant reimplementation is **not** a faithful reproduction of their strongest number; treat the gap as indicative, not adjudicated. |
| **CUDA fused decode kernel speedups** | L5 | Kernel correctness + microbench in the test suite (`benchmark_kv_kernel.py`); requires GPU + build toolchain. |

---

## How to read this

- Each row's level says *what kind* of evidence backs the claim, so you can decide whether it meets
  your bar before depending on it. **L4/L5 is not a weakness** — it is a disclosure.
- The **strongest, most-scrutinized** claims (the RaBitQ/OPQ comparison; the KV-key finding) are
  wired to a single *Run all*.
- Stability of the *code* behind each claim is a separate axis — see
  [`docs/api-stability.md`](api-stability.md).

### Note on `estimate_storage()`
The bytes/vector figures in the notebooks are computed **analytically** (out_dim × bits ÷ 8) to keep
the harness self-contained and library-agnostic. `PCAMatryoshkaPipeline.estimate_storage()` was
dimension-agnostic (hard-coded 1024→384 @ 3-bit) before **v1.4.1**; as of v1.4.1 it defaults to the
pipeline's real `input_dim` / `output_dim` / `bits` (regression-tested in `tests/test_pca.py`).

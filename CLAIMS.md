# CLAIMS.md — what we claim, and exactly how to reproduce it

At-a-glance claim → reproduction table. Every claim links to a runnable notebook under
[`notebooks/claims/`](notebooks/claims). The **central result** is Track 1 (embedding compression +
compressed-domain retrieval) — CPU/Colab-reproducible. Track 2 (KV-cache / fused decode) is the
engineering-package extra and needs a GPU. Full detail + scope caveats: [`docs/claims.md`](docs/claims.md).

**Status legend:** *Reproducible* = one *Run all* on public data reproduces the claim ·
*Needs local run* = reproducible but full-scale/timing needs your hardware ·
*Partial* = notebook runs, one variant left to fill in ·
*Experimental* = requires GPU + model weights, treat as engineering/preview.

## Track 1 — Embedding / vector-DB compression (central result)

| Claim | Public reproduction? | Dataset | Command / notebook | Hardware | Status |
|---|---|---|---|---|---|
| **27× compression at high recall@10** (5× oversample + rerank) | Yes | GloVe-100-angular (public) | [`00_canonical_sota_embedding.ipynb`](notebooks/claims/00_canonical_sota_embedding.ipynb) | CPU | Reproducible |
| **Beats RaBitQ on recall / ties OPQ, builds 4–20× faster** | Yes (full GloVe); 1M-scale timing local | GloVe-100-angular (1.18M) | [`00_canonical_sota_embedding.ipynb`](notebooks/claims/00_canonical_sota_embedding.ipynb) · [`benchmarks/canonical_embedding.py`](benchmarks/canonical_embedding.py) | CPU (GPU optional) | Reproducible (abs. build-time = local run) |
| **PCA rotation makes non-Matryoshka models truncatable, no retraining** | Yes | GloVe-100-angular | [`01_pca_truncation.ipynb`](notebooks/claims/01_pca_truncation.ipynb) | CPU | Reproducible |
| **Learned codebooks reduce quantization error ~22%** | Partially | GloVe-100-angular | [`02_learned_codebooks.ipynb`](notebooks/claims/02_learned_codebooks.ipynb) | CPU | Partial (learned-variant cell to fill) |
| **ADCIndex compressed-domain search throughput** | Yes (recall); QPS local | GloVe-100-angular | [`03_adcindex_throughput.ipynb`](notebooks/claims/03_adcindex_throughput.ipynb) | CPU | Reproducible (abs. QPS = local run) |
| **Up to 114× storage compression** | Yes (scoped) | dataset-dependent | operating point of [`00_...ipynb`](notebooks/claims/00_canonical_sota_embedding.ipynb) | CPU | Reproducible (storage-only, recall via rerank) |
| **Graceful behaviour on OOD anisotropic / heavy-tailed embeddings** | Yes | synthetic anisotropic (power-law spectrum, Student-t tails) | [`04_ood_anisotropic.ipynb`](notebooks/claims/04_ood_anisotropic.ipynb) | CPU | Reproducible (robustness envelope tracks spectral concentration) |

> **Honest scope.** PCA *truncation* wins only for high-dimensional / concentrated-spectrum
> embeddings (sentence, vision). On compact descriptor sets (GloVe-100, NYTimes-256) it loses to
> PQ/OPQ; at full dimension / matched bytes the TurboQuant scalar quantizer still **wins on GloVe and
> vision, ties on NYTimes** — see [`benchmarks/RESULTS_glove.md`](benchmarks/RESULTS_glove.md).

## Track 2 — KV-cache compression (engineering package; GPU)

| Claim | Public reproduction? | Dataset | Command / notebook | Hardware | Status |
|---|---|---|---|---|---|
| **KV *keys* need per-channel / asym treatment** (incl. Qwen2.5 collapse & recovery) | Yes (on GPU) | Llama / Mistral / Qwen + perplexity/LongBench | [`10_kv_keys_per_channel.ipynb`](notebooks/claims/10_kv_keys_per_channel.ipynb) · [`docs/KV_KEYS_FINDING.md`](docs/KV_KEYS_FINDING.md) | GPU | Experimental |
| **NF4 / asym-NF4 vs uniform K4** across model families | Yes (on GPU) | Llama / Mistral / Qwen | [`11_kv_nf4_matrix.ipynb`](notebooks/claims/11_kv_nf4_matrix.ipynb) · [`benchmarks/RESULTS_longbench.md`](benchmarks/RESULTS_longbench.md) | GPU | Experimental |
| **Comparison vs KVQuant** on LongBench/perplexity | Partially | model + task | [`12_kv_vs_kvquant.ipynb`](notebooks/claims/12_kv_vs_kvquant.ipynb) | GPU | Experimental (our KVQuant reimpl. not a faithful repro of their strongest number) |
| **Fused decode kernel speedup** | Yes (on GPU) | microbenchmark | [`benchmarks/benchmark_kv_kernel.py`](benchmarks/benchmark_kv_kernel.py) | GPU + build toolchain | Experimental |

---

*The bytes/vector figures are computed analytically (`out_dim × bits ÷ 8`) to keep the harness
library-agnostic. (`estimate_storage()` was dimension-agnostic before v1.4.1 and now tracks the real
pipeline config — see [`docs/claims.md`](docs/claims.md).) Notebooks ship with empty outputs; numbers
appear when you run them.*

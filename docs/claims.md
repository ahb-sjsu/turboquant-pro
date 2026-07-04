# Claims & evidence ladder

Every headline claim in this project sits at one of five evidence levels. This page states which,
and points at where to reproduce it. The two tracks — **embedding/vector-DB compression** and
**KV-cache compression** — are evaluated with entirely different metrics and are kept separate below.

**Levels**
- **L1 — Publicly reproducible in Colab.** Public data, no special hardware, one notebook/command.
- **L2 — Public data, local hardware required.** Reproducible but needs a GPU and/or large RAM/time.
- **L3 — Private or large-scale internal benchmark.** Data or scale not in the public repo.
- **L4 — Paper result, not yet independently reproduced.** Reported vs. published baselines.
- **L5 — Engineering claim, covered by unit tests only.** Correctness/perf asserted by the suite.

Where a repro path is not yet wired to a single command, it is marked **[repro TODO]** — the
supporting benchmark exists in `benchmarks/`, but the one-liner is still being pinned down.

---

## Track 1 — Embedding / vector-DB compression

| Claim | Level | Reproduce |
|---|---|---|
| **27× storage compression @ 99.8% recall@10** (with 5× oversampling + reranking; all methods benchmarked identically) | L2 | `benchmarks/` embedding suite — public model + public corpus, GPU recommended. **[repro TODO: pin command]** |
| **Beats RaBitQ on recall / ties OPQ at 1M-vector scale, builds index 4–20× faster** — the headline reviewers will try first | L2 | Canonical embedding table (flat/PQ/OPQ/RaBitQ/PCA-only/TQ-only/PCA+TQ/ADCIndex) at identical rerank candidate counts. **[repro TODO: single command + fixed seed]** |
| **PCA rotation makes non-Matryoshka models truncatable with no retraining** (Varici et al. 2025 grounding) | L1 | [`notebooks/turboquant_benchmark.ipynb`](../notebooks/turboquant_benchmark.ipynb) — [open in Colab](https://colab.research.google.com/github/ahb-sjsu/turboquant-pro/blob/master/notebooks/turboquant_benchmark.ipynb). |
| **Learned codebooks reduce quantization error 22%** | L2 | Codebook benchmark in `benchmarks/`. **[repro TODO]** |
| **Up to 114× storage compression (PCA-Matryoshka + TurboQuant)** | L2 / dataset-dependent | Storage-only, recall preserved via oversampling+reranking; *not* a single-vector reconstruction bound. Operating points in the benchmark tables. |
| **ADCIndex compressed-domain search throughput** | L5 (+ L2 at scale) | Round-trip + throughput unit tests; scale numbers need local HW. |

## Track 2 — KV-cache compression

| Claim | Level | Reproduce |
|---|---|---|
| **KV-cache *keys* need per-channel / asymmetric treatment; generic vector-reconstruction metrics mislead** — the core scientific finding | L2 | `docs/KV_KEYS_FINDING.md` + KV benchmark scripts; perplexity/LongBench on public models, GPU required. |
| **NF4 / asymmetric-NF4 key quantization vs. uniform K4** (per-model: Llama / Mistral / Qwen) | L2 | KV-cache benchmark table (fp16 / uniform-K4 / NF4 / asym-NF4 / +outliers) with model versions + exact commands. **[repro TODO: publish the full matrix]** |
| **Qwen2.5 uniform-quant collapse case** | L2 | Single-model repro in the KV suite; documented in the README seed table. |
| **Comparison vs. KVQuant on LongBench/perplexity** | L4 | Reported against the published method; our KVQuant reimplementation is **not** a faithful reproduction of their strongest number — treat the gap as indicative, not adjudicated. |
| **CUDA fused decode kernel speedups** | L5 | Kernel correctness + microbench in the test suite; requires GPU + build toolchain. |

---

## How to read this

- A claim being **L4/L5** is not a weakness — it says *what kind* of evidence backs it, so you
  can decide whether it meets your bar before depending on it.
- The **strongest, most-scrutinized** claims (the RaBitQ/OPQ comparison; the KV-key finding) are
  L2 — public data, local hardware. Making these **one-command reproducible** is the top open item
  from Review 1 and is tracked in `REVIEW_RESPONSE_1.md`.
- Stability of the *code* behind each claim is a separate axis — see `docs/api-stability.md`.

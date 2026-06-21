# turboquant-pro — Comprehensive Feature Analysis

An honest, evidence-based evaluation of **every** turboquant-pro capability, toward
the goal of an industry-standard tool. Each feature is rated on **maturity**
(Production / Beta / Experimental) and **evidence** (Measured here / Shipped but
not independently benchmarked / Documented). Numbers are real runs on Atlas;
where we did not run a benchmark, we say so rather than imply one.

## Executive summary

turboquant-pro is a **broad, well-integrated compression system** whose *core*
(embedding + KV compression) is now **measured at scale and competitive with
2024 SOTA**, surrounded by an unusually deep set of deployment integrations. The
honest shape: a strong, validated core; a wide, mostly-shipped-but-not-all-
independently-benchmarked periphery; and two genuine differentiators (training-
free fast index build, SQL-native compressed search).

| Verdict axis | Assessment |
|---|---|
| Core compression quality | **Strong** — ties OPQ, beats RaBitQ single-stage (see §1) |
| Index build cost | **Best-in-class** — 4–20× faster than OPQ |
| Query throughput | **Weakness** — linear scan as benchmarked; GPU-ADC path unmeasured |
| Breadth of integrations | **Exceptional** — pgvector/FAISS/vLLM/NATS/4 vector DBs |
| Production tooling | **Good** — observability measured; AutoConfig shipped |
| Test/format rigor | **In progress** — 397 tests; format spec is the gap |

---

## 1. Core embedding compression (PCA-Matryoshka + TurboQuant) — **Measured**
**Maturity: Production.** The flagship. Training-free PCA rotation makes any
encoder truncatable, + scalar quantization for ~30× compression.

Real results (32× compression, two-stage protocol, recall@10):

| | single-stage | +rerank | build |
|---|---|---|---|
| 199k LaBSE | tq-pro **0.784** vs OPQ 0.780, RaBitQ 0.63 | 0.9993 vs 0.999 | **31 s** vs OPQ 632 s |
| 1M Gutenberg | tq-pro **0.857** vs OPQ 0.872 | 0.989 vs 0.989 | **131 s** vs OPQ 529 s |

**Assessment:** ties the strongest learned baseline (OPQ) and **beats RaBitQ on
the pure-compression metric**, at **4–20× lower build cost**. The truncatability
result (BGE-M3 256-d: cosine 0.467→0.974) is clean and real. *(RaBitQ +rerank
number pending the thermal-safe re-run.)*

## 2. KV-cache compression (RoPE-aware) — **Measured (analytical+sim)**
**Maturity: Production.** `TurboQuantKVCache`, RoPE-aware, hot-window in fp16.
Result: **5.3× KV memory reduction at 3-bit**; lifts a 7B model's weights+KV from
14 GB to 2.7 GB (fits a 4 GB edge device under TQ, not fp16). See
`benchmark_edge.py`. *Honest caveat:* memory is analytical + the library's
measured store; end-to-end energy needs on-device runs (`benchmark_e2e.py`).

## 3. Vector-DB integration — **Partially measured**
**Maturity: Production (pgvector, FAISS); Beta (exporters).**
- **pgvector-native (`tqvector`)** — *Measured.* Compressed vectors stored AND
  searched in PostgreSQL via `<=>`: **4.6× smaller** than fp32 at recall@10 0.90
  (single-stage). Caveats: no ANN index yet → slower per query; raw SQL path
  (no PCA/rerank). A genuine differentiator: no separate vector service.
- **FAISS** — used throughout as a baseline harness; `TurboQuantFAISS` shipped.
- **Exporters** (Milvus, Pinecone, Qdrant, Weaviate) — *Shipped, not benchmarked.*
  Breadth is real; round-trip fidelity not independently verified here.

## 4. Multi-modal presets — **Shipped, not independently benchmarked**
**Maturity: Beta.** Per-encoder presets for text (BGE-M3, E5, ada-002), vision
(CLIP, SigLIP), audio (Whisper), code (CodeBERT, CodeLlama). The presets exist
and are sensible (per-modality PCA rank + bit-width). We have **not** run a
cross-modal retrieval benchmark here — that is the honest gap (#12).

## 5. Adaptivity: AutoConfig + autotune + hardware detection — **Shipped (v1.0)**
**Maturity: Beta.** `AutoConfig.from_pretrained` (Llama-3-8B/70B presets),
`run_autotune`, `get_hardware_profile`. The zero-config story is the SQLite-like
selling point. *Not yet benchmarked* against a brute-force sweep (#21); the
mechanism is present and correct on inspection.

## 6. Observability: QualityMonitor — **Measured (v1.0)**
**Maturity: Production.** Rolling-window quality, **KS-test drift detection**,
alert callbacks, Prometheus metrics. Result: **no false alarm** on a clean 4-bit
stream; **drift detected within 50 samples** of a 4→2-bit regression. Genuine
production MLOps that most compression libraries lack.

## 7. Transport: NATS codec — **Shipped, not benchmarked**
**Maturity: Beta.** `TurboQuantNATSCodec` compresses embeddings/telemetry over a
message bus. Relevant for edge↔cloud. Not independently benchmarked here.

## 8. Advanced: learned codebooks, ANS, GPU ADC — **Mixed**
- **Learned codebooks** (`fit_codebook`, v1.0) — *Documented claim* (cosine
  0.978→0.99 at equal bits); not reproduced here (Atlas system pkg is v0.7.0).
- **ANS entropy coding** — *Shipped, not benchmarked.*
- **GPU ADC search** (`gpu_adc_search`, CuPy) — *Shipped;* this is the intended
  fix for the query-throughput weakness; benchmark pending (#25).

---

## Maturity / evidence matrix

| Feature | Maturity | Evidence |
|---|---|---|
| Embedding compression | Production | **Measured @ 1M** |
| KV-cache compression | Production | Measured (analytical) |
| pgvector-native | Production | **Measured** |
| QualityMonitor | Production | **Measured** |
| Edge/AutoConfig deploy | Production | Measured (budget) |
| FAISS integration | Production | Used |
| Multi-modal presets | Beta | Shipped |
| AutoConfig (model-aware) | Beta | Shipped |
| Learned codebooks | Beta (v1.0) | Documented |
| Vector-DB exporters | Beta | Shipped |
| NATS transport | Beta | Shipped |
| GPU ADC search | Beta | Shipped |
| ANS coding | Experimental | Shipped |

## Honest overall verdict
- **Strengths:** validated core (ties OPQ, beats RaBitQ single-stage, 4–20× faster
  build), real SQL-native search, real production observability, exceptional
  integration breadth, reproducible.
- **Weaknesses:** query throughput (linear scan as benchmarked; GPU-ADC unmeasured);
  several periphery features shipped but not independently benchmarked; no frozen
  compressed-format spec yet (the #1 standardization gap, see `CODE_QUALITY.md`).
- **Bottom line:** an **A− system today** — a strong, honest, validated core with
  unusual deployment breadth. The path to a clear **A+** is: prove competitive
  query speed (GPU ADC), benchmark the remaining periphery (multi-modal,
  AutoConfig), and ship the versioned format spec + conformance suite.

*This document is updated as each pending benchmark lands; see `FEATURE_COVERAGE.md`
for the per-feature benchmark map and `RESULTS_*.md` for raw numbers.*

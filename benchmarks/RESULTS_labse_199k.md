# Vector-DB retrieval benchmark — real 199k LaBSE embeddings

Atlas, `/archive/results_aesthetics/bip_sample_200k_labse.npy` (768-dim, 199,000
corpus / 1,000 held-out queries), exact cosine ground truth, CPU,
`OMP_NUM_THREADS=20`. Reproduce: `benchmarks/benchmark_vectordb.py`.

## Fair head-to-head at 32× compression (two-stage = oversample×5 + rerank)

Every method gets the same protocol: compressed first stage → rerank the top-50
candidates by exact fp32 on the retained originals.

| method | bytes/vec | build s | qps | recall@10 (1-stage) | recall@10 (+rerank) |
|---|---:|---:|---:|---:|---:|
| fp32-flat (exact) | 3072 | 7 | 212 | 0.9997 | — |
| faiss-PQ | 96 | 142 | 136 | 0.467 | 0.827 |
| faiss-IVFPQ | 96 | 355 | 9513 | 0.496 | 0.756 |
| **faiss-OPQ** | 96 | **632** | 915 | 0.780 | **0.999** |
| **turboquant-pro** TQ3 | 100 | **31** | 224 | 0.784 | **0.9993** |

## Honest takeaway
- **Accuracy: turboquant-pro ties the strongest baseline (OPQ).** With a fair
  rerank for all methods, tq-pro recall@10 = 0.9993 vs OPQ 0.999 — a statistical
  tie — and both clearly beat PQ (0.827) and IVF-PQ (0.756). tq-pro does **not**
  "dominate" OPQ on recall; the earlier impression was an artifact of giving
  rerank only to tq-pro.
- **Build cost is the real win: ~20× faster.** tq-pro builds in **31 s** vs OPQ's
  **632 s** at equal recall — because it is PCA-based (one eigen-decomposition),
  not OPQ's expensive iterative rotation+codebook optimization. At VLDB scale,
  index-construction time matters as much as query time.
- **Query caveat (honest):** as benchmarked, tq-pro searches a flat index over
  reconstructed vectors (224 qps) and is slower than OPQ's ADC (915 qps). tq-pro
  ships a native compressed/ADC + GPU search path (`gpu_adc_search`) that should
  close this — measuring it is future work, not claimed here.
- **System advantages (beyond this table):** SQL-native compressed search in
  pgvector, multi-modal presets, and zero-config `AutoConfig` — none of which the
  faiss baselines offer.

## Method contribution (the original novelty)
PCA-Matryoshka makes **non-Matryoshka embeddings truncatable without retraining**
(naïve truncation to 256-d: cosine 0.467 → 0.974), which is what lets a
training-free pipeline reach OPQ-class accuracy at a fraction of the build cost.

*Next:* 1M-scale (Gutenberg), a standard public dataset, multi-modal, and the
pgvector-native numbers — see the sprint.

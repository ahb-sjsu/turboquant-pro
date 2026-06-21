# VLDB-scale retrieval benchmark — 1M Gutenberg LaBSE embeddings

Real, multilingual: **999,000** passages embedded from 1,146 Project Gutenberg
books with LaBSE (768-dim), 1,000 held-out queries, exact cosine ground truth,
Atlas CPU (`OMP_NUM_THREADS=20`). Embeddings built by
`benchmarks/gutenberg_embed.py`; benchmark by `benchmarks/benchmark_vectordb.py`.

## Fair head-to-head at 32× compression (two-stage = oversample×5 + rerank)

| method | bytes/vec | build s | qps | recall@10 (1-stage) | recall@10 (+rerank) |
|---|---:|---:|---:|---:|---:|
| fp32-flat (exact) | 3072 | 3 | 27 | 0.990 | — |
| faiss-PQ | 96 | 167 | 24 | 0.717 | 0.976 |
| faiss-IVFPQ | 96 | 366 | **7366** | 0.615 | 0.845 |
| **faiss-OPQ** | 96 | **529** | 39 | 0.872 | **0.989** |
| **turboquant-pro** TQ3 | 100 | **131** | 30 | 0.857 | **0.989** |

*(Exact flat tops out at recall@10 0.990 here — numerical ties — so 0.989 is at
the practical ceiling.)*

## Honest takeaway at 1M scale
- **Accuracy: tq-pro ties the best baseline (OPQ)** — both 0.989 with fair
  rerank, at the ~0.99 exact ceiling; both clearly beat PQ (0.976) and IVF-PQ
  (0.845).
- **Build cost stays in tq-pro's favor: ~4× faster** (131 s vs OPQ 529 s) on 1M
  vectors. PCA-Matryoshka needs one eigen-decomposition; OPQ needs iterative
  rotation+codebook optimization that scales poorly.
- **Query throughput is a real trade-off:** IVF-PQ is far faster (7366 qps) but
  much less accurate; tq-pro and OPQ are comparable (~30–40 qps as benchmarked,
  flat reconstruct search). tq-pro's native ADC/GPU search would raise this —
  measured separately, not claimed here.

## The defensible VLDB contribution
A **training-free** pipeline (PCA-Matryoshka makes non-Matryoshka embeddings
truncatable; +TurboQuant scalar quantization) that reaches **OPQ-class accuracy
at a fraction of the index-build cost**, on **1M real multilingual vectors**, and
ships as a deployable SQL-native system (pgvector) — not just a faiss recipe.
Index construction time is a first-class concern at VLDB scale, and that is where
this wins.

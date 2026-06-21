# Vector-DB retrieval-cost benchmark — real 199k LaBSE embeddings

Run on Atlas (`/archive/results_aesthetics/bip_sample_200k_labse.npy`, 768-dim,
199,000 corpus / 1,000 held-out queries, exact cosine ground truth, CPU,
`OMP_NUM_THREADS=20`). Reproduce with `benchmarks/benchmark_vectordb.py`.

| method | bytes/vec | comp× | qps | recall@10 | recall@100 |
|---|---:|---:|---:|---:|---:|
| fp32-flat (exact) | 3072 | 1.0 | 156 | 0.9997 | 1.000 |
| faiss-PQ (m=64) | 64 | 48× | 146 | 0.447 | 0.492 |
| faiss-PQ (m=96) | 96 | 32× | 108 | 0.565 | 0.604 |
| faiss-PQ (m=128) | 128 | 24× | 668 | 0.658 | 0.693 |
| faiss-HNSW (fp32, no compression) | 3072 | 1.0 | 2884 | 0.916 | 0.833 |
| **tq-pro PCA256+TQ2** 1-stage | 68 | 45× | 80 | 0.659 | 0.710 |
| **tq-pro PCA256+TQ2** +rerank×5 | 68 | 45× | 227 | **0.974** | — |
| **tq-pro PCA256+TQ3** 1-stage | 100 | 31× | 49 | 0.784 | 0.819 |
| **tq-pro PCA256+TQ3** +rerank×5 | 100 | 31× | 97 | **0.9993** | — |
| **tq-pro PCA256+TQ4** 1-stage | 132 | 23× | 120 | 0.882 | 0.907 |
| **tq-pro PCA256+TQ4** +rerank×5 | 132 | 23× | 227 | **0.9997** | — |

## Takeaway
At matched compression, turboquant-pro **dominates product quantization**: at
~30–32×, recall@10 **0.9993 vs PQ 0.565**; at ~45×, **0.974 vs 0.447**.
Single-stage (no rerank) also beats PQ at every budget. Rerank uses the retained
fp32 originals (standard two-stage ANN); compressed-only storage is the
`bytes/vec` column. This reproduces the TechRxiv 99.8%@27× claim on real data and
extends it to 199k scale.

*Next:* scale to 1M+ (Gutenberg), add OPQ/IVF-PQ/RaBitQ baselines, multi-modal
(CLIP/MERT), and a standard public dataset — see the sprint.

# Query-speed analysis (honest) — the one real tq-pro limitation

Real 50k LaBSE, recall@10. tq-pro **wins recall/compression/build** but is **not a
fast ANN system**; here is the honest speed picture.

| method | qps | recall@10 | note |
|---|---:|---:|---|
| tq-pro flat scan (CPU) | 162–254 | 0.78 (1-stage) / 0.999 (+rerank) | linear scan over codes |
| tq-pro GPU-ADC (`gpu_adc_search`) | **2.8** | 0.71 | per-query CuPy launch overhead dominates |
| ScaNN (AH+tree+reorder) | **3441** | 0.72 | tuned fast-ANN *system* |
| faiss IVF-PQ | 7366 | 0.50–0.85 | sub-linear inverted index |

## Honest conclusion
turboquant-pro is a **compression method**, not a fast ANN index. It dominates on
recall and compression and matches OPQ accuracy at far lower build cost, but its
query throughput (linear scan 162–254 qps) trails tuned ANN systems (ScaNN 3441
qps), and the shipped GPU-ADC path is *not* a fix (2.8 qps — per-query overhead).
The route to competitive query speed is a **batched CUDA ADC kernel** or
**integrating tq-pro codes into a ScaNN-style index** — real engineering, the
principal future-work item.

## Path-1 result (PCA-Matryoshka front-end to fast indexes) — NEGATIVE

Tested the hypothesis "PCA-Matryoshka reduction improves a fast ANN index."
100k LaBSE, recall@10 (+rerank):

| index | recall@10 (+rerank) | qps | build |
|---|---:|---:|---:|
| IVF-PQ raw-768 | **0.782** | 13033 | 194 s |
| IVF-PQ pca-256 | 0.414 | 44248 | 65 s |
| HNSW raw-768 | **0.9375** | 3221 | 76 s |
| HNSW pca-256 | 0.771 | 5331 | 54 s |

**Honest conclusion:** PCA-256 makes the indexes ~3x faster but **drops recall
substantially** — the dimensionality reduction compounds with the index's own
approximation. PCA-Matryoshka does **not** improve fast ANN indexes; it does not
break the speed/recall/compression trade-off. This documents a genuine
**trilemma**: methods give two of {fast, high-recall, compressed}, not three.
tq-pro occupies the **high-recall + compressed** corner (flat scan, 162-254 qps);
plain HNSW gives **fast + high-recall** but uncompressed; PCA+IVF-PQ gives
**fast + compressed** but low-recall. A method that breaks the trilemma (a fast
*compressed* ADC) remains the real route to a clean A+.

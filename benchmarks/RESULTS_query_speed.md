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

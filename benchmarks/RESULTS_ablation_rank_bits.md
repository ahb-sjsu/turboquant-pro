# Ablation: PCA rank × quantization bits (199k LaBSE)

`benchmark_vectordb.py --out-dim {128,256,384} --bits {2,3,4} --methods flat tq`
on the 199k LaBSE corpus (768-d), recall@10 with oversample×5 + rerank.

| PCA rank | bits | bytes/vec | comp× | recall@10 (1-stage) | recall@10 (+rerank) |
|---:|---:|---:|---:|---:|---:|
| 128 | 2 | 36 | 85.3× | 0.516 | 0.889 |
| 128 | 3 | 52 | 59.1× | 0.630 | 0.971 |
| 128 | 4 | 68 | 45.2× | 0.708 | 0.992 |
| 256 | 2 | 68 | 45.2× | 0.659 | 0.974 |
| 256 | 3 | 100 | 30.7× | 0.784 | **0.9993** |
| 256 | 4 | 132 | 23.3× | 0.882 | 0.9997 |
| 384 | 2 | 100 | 30.7× | 0.711 | 0.992 |
| 384 | 3 | 148 | 20.8× | 0.821 | 0.9995 |
| 384 | 4 | 196 | 15.7× | 0.903 | 0.9997 |

## Findings
- **At a fixed byte budget, more bits / fewer dims tends to win.** At 68 bytes,
  PCA128+TQ4 (0.992) beats PCA256+TQ2 (0.974); at 100 bytes, PCA256+TQ3 (0.9993)
  beats PCA384+TQ2 (0.992). Spend the budget on bit-depth before extra dimensions.
- **The default PCA256+TQ3 (~30×) is near-optimal** — 0.9993 recall@10, matching
  the OPQ-class accuracy from the main comparison.
- **Aggressive compression still works:** even at **85×** (PCA128+TQ2, 36 bytes),
  rerank recovers recall@10 to 0.889 — useful for the most memory-starved tiers.
- Build time is uniformly low (12–27 s) across the grid — the training-free
  build-cost advantage holds regardless of configuration.

*Note:* the learned-codebook ablation (`fit_codebook`, the 0.978→0.99 claim) is a
turboquant-pro **v1.0** feature; Atlas runs v0.7.0, so it is reported separately,
not from this run.

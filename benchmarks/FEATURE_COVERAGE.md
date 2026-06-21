# Feature-coverage matrix

Every headline feature of turboquant-pro maps to a runnable benchmark and a
measured headline number. This is the reproducibility/trust contract for an
industry-standard tool: no feature is claimed without a benchmark behind it.

| Feature | Benchmark | Headline result | Data |
|---|---|---|---|
| **Compression vs. recall** (core) | `benchmark_vectordb.py` | recall@10 **0.9993 @ ~30×** (ties OPQ) | 199k LaBSE |
| **Beats classic PQ / IVF-PQ** | `benchmark_vectordb.py` | 0.999 vs PQ 0.83, IVF-PQ 0.76 (fair rerank) | 199k LaBSE |
| **VLDB scale** | `gutenberg_embed.py` + `benchmark_vectordb.py` | recall@10 **0.989 @ 1M**, ties OPQ | 1M Gutenberg |
| **Fast index build** (training-free) | `benchmark_vectordb.py` | **4–20× faster build** than OPQ | 199k / 1M |
| **Rank × bits trade-off** | `benchmark_vectordb.py` (sweep) | spend budget on bit-depth; 85× → 0.89 | 199k LaBSE |
| **pgvector-native compressed search** | `benchmark_pgvector_real.py` | compressed search in SQL via `tqvector`+`<=>`; **4.6× smaller** than fp32 @ recall@10 0.90 | 50k LaBSE in Postgres |
| **KV-cache compression** (RoPE-aware) | `benchmark_edge.py` | **5.3× @ 3-bit** KV memory | analytical + sim |
| **Edge memory/energy** | `benchmark_edge.py`, `benchmark_e2e.py` | 7B fits 4 GB under TQ (not fp16) | budget + device |
| **Reproducibility** (public, 1-click) | `notebooks/turboquant_benchmark.ipynb` | Colab reproduces the PQ gap on public data | AG-News |
| **Multi-modal presets** | _(pending #12)_ | text / vision / audio embeddings | LaBSE/CLIP/MERT |
| **Zero-config AutoConfig** | _(pending #21)_ | picks near-optimal bits/rank per model | — |
| **Observability (QualityMonitor)** | `benchmark_quality_monitor.py` | no false alarm clean; **drift caught ≤50 samples** after 4→2-bit regression; Prometheus metrics | 16k LaBSE (v1.0) |

## How to run the whole suite
```bash
# core retrieval (real embeddings)
python benchmarks/benchmark_vectordb.py --npy <embeddings.npy> --bits 2 3 4 \
    --methods flat pq opq ivfpq tq
# pgvector-native (in PostgreSQL, as the postgres user)
python benchmarks/benchmark_pgvector_real.py --npy <embeddings.npy>
# edge memory budget + end-to-end energy
python benchmarks/benchmark_edge.py --energy
python -m benchmarks.benchmark_e2e --model <gguf> --quantizer <prov> --energy
# one-click public reproduction
#   open notebooks/turboquant_benchmark.ipynb in Colab
```

Results are recorded in `benchmarks/RESULTS_*.md`. This table is updated as each
remaining benchmark lands.

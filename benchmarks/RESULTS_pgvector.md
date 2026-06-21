# pgvector (fp32) vs tqvector (compressed) — in real PostgreSQL

50,000 real LaBSE embeddings (768-d) loaded into PostgreSQL on Atlas, 500 queries,
exact cosine ground truth. `pgvector` (`vector` type) vs turboquant-pro's
`tqvector` extension type, both searched with the `<=>` operator (no ANN index;
sequential scan). Reproduce: `benchmarks/benchmark_pgvector_real.py`.

| storage | bytes/vec | vs fp32 | insert (s) | q_p50 (ms) | q_p95 (ms) | recall@10 |
|---|---:|---:|---:|---:|---:|---:|
| `vector` (fp32) | 4195 | 1× | 33.8 | 252 | 288 | 1.000 |
| `tqvector` 4-bit | 911 | **4.6×** | 34.2 | 611 | 654 | 0.896 |
| `tqvector` 3-bit | 631 | 6.6× | 33.3 | 539 | 574 | 0.813 |
| `tqvector` 2-bit | 456 | 9.2× | 33.9 | 527 | 592 | 0.664 |

## Takeaway
- **The differentiator works:** compressed vectors are **stored and searched
  inside standard PostgreSQL** via the `tqvector` type and the `<=>` distance
  operator — no decompression step, no separate vector-search service. At 4-bit,
  the column is **4.6× smaller** than fp32 `pgvector` while retaining recall@10
  0.90 on a single-stage scan.
- **Honest caveats.** (1) This is the *raw SQL path*: full-dimension scalar
  quantization, **no PCA-Matryoshka reduction and no rerank**, so recall is lower
  than the faiss pipeline (which adds both). (2) Query latency is *higher* than
  fp32 here because there is **no ANN index** and `tq_compress(query)` is computed
  per query; a `tqvector` ANN index is the obvious next step. The value
  demonstrated is *deployability* — compressed vector search in the database you
  already run — not raw query speed.

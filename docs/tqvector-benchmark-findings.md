# tqvector benchmark findings

Measurements of the `tqvector` pgrx extension against raw pgvector
`vector(fp32)` on Atlas PostgreSQL 16, synthetic unit-normalized
Gaussian corpora, brute-force sequential scan (no HNSW). Queries run
via persistent psycopg2 connection on Atlas (no SSH round-trip in
the measured latencies).

## Implementation fix shipped during this run

`compress()` and `decompress()` each called `generate_rotation(dim,
seed)` — a Gram-Schmidt QR of a dim×dim matrix. At dim=384 that is
~100 ms per call; at dim=768, ~300 ms. The `<=>` operator calls
decompress twice per comparison, so a brute-force ORDER BY query on
100 rows took **27.7 seconds** (EXPLAIN ANALYZE confirmed).

Fix: `OnceLock<RwLock<HashMap<(usize, u32), Vec<f32>>>>` computes each
rotation once and reuses the cached copy. Same query: **222 ms** —
a **~125× speedup**. Committed to `main` as `pgext: cache rotation
matrix per (dim, seed)`.

## Production-scale Pareto (n=100k, brute-force scan, parallel_safe)

| dim | storage | bytes/vec | compress | recall@10 | q_p50 (ms) | vs fp32 |
|---|---|---:|---:|---:|---:|---:|
| 384 | vector | 1639 | 1.0× | 1.00 | 40 | baseline |
| 384 | tqvector_4 | 456 | 3.6× | 0.71 | 12084 | 300× slower ⚠ |
| 384 | tqvector_3 | 357 | 4.6× | 0.50 | 12132 | 300× slower ⚠ |
| 384 | tqvector_2 | 265 | 6.2× | 0.31 | 12090 | 300× slower ⚠ |
| 768 | vector | 4195 | 1.0× | 1.00 | 501 | baseline |
| 768 | **tqvector_4** | **911** | **4.6×** | **0.78** | **391** | **1.28× faster** |
| 768 | tqvector_3 | 635 | 6.6× | 0.54 | 359 | 1.40× faster |
| 768 | tqvector_2 | 456 | 9.2× | 0.31 | 329 | 1.52× faster |
| 1536 | vector | 8337 | 1.0× | 1.00 | 794 | baseline |
| 1536 | **tqvector_4** | **1639** | **5.1×** | **0.78** | **731** | **1.09× faster** |
| 1536 | tqvector_3 | 1366 | 6.1× | 0.52 | 666 | 1.19× faster |
| 1536 | tqvector_2 | 889 | 9.4× | 0.32 | 635 | 1.25× faster |

**At dim 768 and 1536 (covering MiniLM and OpenAI `text-embedding-3`),
tqvector_4 is a clean Pareto win** against raw pgvector: 4.6–5.1×
smaller memory, faster queries (memory-bandwidth-bound scans benefit
from the smaller working set), and 0.78 recall@10.

## Known anomaly: dim=384 is 30× slower per comparison than dim=768

The dim=384 row is flagged. All three bit widths cluster at ~12,100 ms
and parallel Seq Scan with 2 workers is running (confirmed via EXPLAIN
ANALYZE). Per-comparison cost:

- dim=384 → ~240 µs
- dim=768 → ~7.8 µs
- dim=1536 → ~14.6 µs

Larger-dim should be slower, not faster. This is inverted, so there's
a genuine issue in our dim=384 code path. Candidates to investigate:

- Rotation cache miss behavior specific to dim=384 (cache is keyed by
  dim — should share across calls)
- Unrolled vs loop code in `apply_rotation` / `inverse_rotation` —
  LLVM might be vectorizing the dim=768/1536 paths and not dim=384
- Some threshold in pgrx / Rust that kicks in near this size

Workaround for now: use halfvec or raw fp32 at dim ≤ 512; the
compression savings at those dims don't justify the latency hit.

## Product gap: no HNSW operator class

All current numbers are brute-force sequential scan. pgvector ships
HNSW and IVFFlat operator classes for `vector` and `halfvec`;
tqvector has no access method.

- At **100k rows** brute-force with parallel scan is workable (391 ms
  at dim=768, 731 ms at dim=1536).
- At **1M rows** brute-force scales to ~4–7 seconds, which is too
  slow for interactive retrieval.
- Production vector search above 1M rows is universally HNSW-indexed.

Two paths to address:

1. **Implement tqvector HNSW operator class** via pgrx's index-AM
   hooks. Non-trivial; requires insert/search/delete, cost estimation,
   vacuum integration. Rough estimate: 1–2 sprints.
2. **Two-column pattern**: store `halfvec(dim)` with HNSW for
   candidate retrieval and `tqvector` for compressed persistent
   storage + reranking. Doubles memory unless halfvec is on slow
   disk and tqvector in RAM (which makes some deployments worthwhile).

## Open items

- [ ] Fix the dim=384 per-comparison anomaly (240 µs vs the expected
      <10 µs).
- [ ] Test with real embeddings (sentence-transformers/all-MiniLM-L6-v2
      on WikiText) — confirm synthetic-Gaussian recall carries over.
- [ ] Scale to 1M at dim=768 to see where brute-force breaks.
- [ ] Consider two-column pattern or HNSW opclass.
- [ ] Insert throughput is ~340 rows/sec for tqvector (vs ~1100 for
      vector). Acceptable for one-time ingest but an obvious target
      if we want bulk-import speed. The path `COPY ... → INSERT
      SELECT tq_compress()` gave 3500 r/s in v1 and can be re-adopted.

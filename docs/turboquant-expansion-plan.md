# TurboQuant Expansion Plan: Beyond KV Cache Compression

**Author:** Andrew H. Bond  
**Date:** 2026-04-04 (updated from 2026-04-01)  
**Status:** v0.3.0 -- PCA-Matryoshka implemented; Atlas integration in progress  

## Executive Summary

TurboQuant Pro v0.3.0 now combines two compression techniques:

1. **PCA-Matryoshka** (Bond, IEEE TAI 2026): PCA rotation + dimension truncation for non-Matryoshka embedding models. Training-free. Up to 4x dimension reduction at 0.990 cosine similarity.
2. **TurboQuant** (Zandieh et al., ICLR 2026): Random orthogonal rotation + Lloyd-Max scalar quantization + bit-packing. 5-10x compression at 0.978 cosine similarity.

Combined: **PCA-384 + TQ3 achieves 27.7x compression at 0.979 cosine similarity** on BGE-M3 (1024-dim). This dramatically reduces storage and bandwidth requirements across the Atlas AGI-HPC architecture.

This document describes the expansion targets within Atlas where these techniques provide significant improvement.

## 0. PCA-Matryoshka Dimension Reduction (IMPLEMENTED in v0.3.0)

### Problem
Non-Matryoshka embedding models (BGE-M3, E5-large-v2, ada-002) distribute information roughly uniformly across all dimensions. Naive truncation destroys critical signal: BGE-M3 truncated from 1024 to 256 dims yields only 0.467 cosine similarity -- unusable.

### Solution
`turboquant_pro/pca.py` provides `PCAMatryoshka` and `PCAMatryoshkaPipeline`:

```python
from turboquant_pro import PCAMatryoshka

pca = PCAMatryoshka(input_dim=1024, output_dim=384)
pca.fit(sample_embeddings)              # Fit on 5-10K vectors
pipeline = pca.with_quantizer(bits=3)   # PCA-384 + TQ3 = 27.7x

compressed = pipeline.compress(embedding)
reconstructed = pipeline.decompress(compressed)
```

### Key Features
- **fit()**: Full-batch PCA via eigendecomposition of covariance matrix (float64 for numerical stability)
- **partial_fit()**: Incremental covariance update (Chan et al., 1979) for streaming data
- **save()/load()**: Serialize rotation matrix + mean + eigenvalues to `.npz`
- **with_quantizer()**: Compose with TurboQuantPGVector for end-to-end pipeline
- **variance_report()**: Diagnostic for choosing output_dim
- **cosine_similarity()**: Measure round-trip quality

### Compression Ratios (BGE-M3, 1024-dim)

| Configuration | Ratio | Cosine Sim | Use Case |
|---------------|------:|-----------:|----------|
| PCA-512 + TQ3 | 20.9x | 0.984 | High-accuracy RAG |
| PCA-384 + TQ3 | 27.7x | 0.979 | Recommended default |
| PCA-256 + TQ3 | 41.0x | 0.963 | Storage-constrained |
| PCA-128 + TQ3 | 78.8x | 0.923 | Edge/IoT |

### Impact on Atlas Storage

| Dataset | Float32 | TQ3 Only | PCA-384+TQ3 | Extra Savings |
|---------|--------:|---------:|------------:|--------------:|
| Ethics (2.4M) | 9,375 MB | 893 MB | 343 MB | 550 MB |
| RAG (112K) | 437 MB | 41 MB | 16 MB | 25 MB |

## 1. pgvector Compressed Storage (IMPLEMENTED)

### Problem
Atlas stores high-dimensional embeddings in PostgreSQL pgvector:

| Dataset | Vectors | Dimension | Float32 Size |
|---------|--------:|----------:|-------------:|
| RAG chunks | 112K | 1024 (BGE-M3) | 437 MB |
| Ethics chunks | 2.4M | 1024 (BGE-M3) | 9,375 MB |
| Episodic memory | Growing | 1024 | Variable |
| Publications | 824K | 1024 (potential) | 3,222 MB |

Total: ~13 GB of float32 embeddings, with the ethics table dominating.

### Solution
Store 3-bit packed embeddings as `bytea` columns instead of `vector(1024)`:

- **112K RAG chunks:** 437 MB -> 41 MB (10.5x)
- **2.4M ethics chunks:** 9,375 MB -> 893 MB (10.5x)
- **824K publications:** 3,222 MB -> 307 MB (10.5x)
- **Total savings:** ~12 GB -> ~1.2 GB

### Implementation
`turboquant_pro/pgvector.py` provides `TurboQuantPGVector`:

```python
tq = TurboQuantPGVector(dim=1024, bits=3, seed=42)
compressed = tq.compress_embedding(embedding)
bytea = compressed.to_pgbytea()  # 388 bytes vs 4096 bytes
```

### Search Strategy
Two approaches, depending on accuracy requirements:

**Option A: Decompress-on-fly (simpler, accurate)**
- Store compressed bytea alongside original vector column
- For search: decompress candidates, compute exact cosine similarity
- For storage: drop original vector column once compressed table validated
- PostgreSQL function: `CREATE FUNCTION tq_decompress(bytea) RETURNS vector`

**Option B: Compressed-space search (faster, approximate)**
- Build an approximate index on quantized representations
- Centroid-centroid inner product table (8x8 = 64 entries for 3-bit)
- Search operates entirely in compressed space
- Re-rank top candidates with full decompression

**Recommended:** Start with Option A for correctness, migrate to Option B at scale.

### Tradeoffs
- **Pro:** 10.5x storage reduction, fits all embeddings in RAM
- **Pro:** Faster I/O (less data to read from disk)
- **Con:** Cannot use pgvector's built-in HNSW/IVF indexes on compressed data
- **Con:** Decompression adds ~0.1ms per query (negligible vs PostgreSQL overhead)
- **Con:** Cosine similarity is approximate (0.978 mean, 0.95+ worst case)

## 2. Compressed NATS Events (IMPLEMENTED)

### Problem
AGI-HPC subsystems communicate via NATS JetStream. When sending embedding vectors between subsystems (e.g., LH -> Memory -> RH), each 1024-dim float32 embedding is 4096 bytes. For batch operations, this adds up:

- 100 embeddings per message: 400 KB per message
- JetBot (edge device): limited 4G/LTE bandwidth
- High-frequency event streams: significant bandwidth overhead

### Solution
`turboquant_pro/nats_codec.py` provides `TurboQuantNATSCodec`:

```python
codec = TurboQuantNATSCodec(dim=1024, bits=3, seed=42)
payload = codec.encode(embedding)  # 392 bytes (8 header + 384 packed)
decoded = codec.decode(payload)    # cos_sim > 0.978
```

### Wire Format
```
[1 byte]  version (0x01)
[1 byte]  bits (2, 3, or 4)
[2 bytes] dim (uint16 LE)
[4 bytes] norm (float32 LE)
[N bytes] packed indices
```

Total: 392 bytes for 1024-dim 3-bit (vs 4096 bytes float32).

### Integration Points
- **LH -> Memory:** Compress embeddings before NATS publish
- **Memory -> RH:** Decompress on receive
- **JetBot:** Critical for constrained networks
- **Streaming pipelines:** Compress intermediate embeddings

### Tradeoffs
- **Pro:** 10x bandwidth reduction
- **Pro:** Self-describing wire format (version, dim, bits in header)
- **Pro:** Stateless -- sender and receiver just need same (dim, bits, seed)
- **Con:** ~0.1ms encode/decode latency per embedding
- **Con:** Both ends must share the same seed (for rotation matrix)

## 3. Tiered Memory Compression (PLANNED)

### Problem
AGI-HPC's L1-L5 memory hierarchy stores embeddings at multiple levels:

- **L1 (Working memory):** In-process Python dictionaries
- **L2 (RAM cache):** Redis or in-memory store
- **L3 (Database):** PostgreSQL pgvector (addressed by Section 1)
- **L4 (Archive):** Cold storage
- **L5 (Distributed):** DHT across cluster nodes

L2 RAM cache is the bottleneck: it must hold enough embeddings for fast retrieval but RAM is finite. Currently L2 stores float32 embeddings.

### Solution
Store L2 cache entries in compressed form:

```python
# L2 cache write
compressed = tq.compress_embedding(embedding)
redis.set(key, compressed.to_pgbytea())

# L2 cache read
data = redis.get(key)
compressed = CompressedEmbedding.from_pgbytea(data, dim=1024, bits=3)
embedding = tq.decompress_embedding(compressed)
```

### Impact
- **10x more embeddings in same RAM budget**
- L2 cache hit rate increases dramatically
- Fewer L3 (PostgreSQL) queries needed
- Particularly valuable for episodic memory (growing dataset)

### Implementation Plan
1. Modify `src/agi/memory/semantic/cache.py` to use TurboQuantPGVector
2. Add compressed format to Redis serialization
3. Benchmark cache hit rate improvement
4. Roll out to episodic memory next

## 4. Compressed Embedding Index (PLANNED)

### Problem
HNSW indexes in pgvector store full float32 vectors at each graph node. For 2.4M ethics chunks, the HNSW index alone consumes several GB of RAM.

### Solution
Build an HNSW-like index on compressed representations:

1. **Compressed nodes:** Each HNSW graph node stores 388 bytes instead of 4096 bytes
2. **Approximate distance:** Use centroid-centroid lookup table for distance computation during graph traversal
3. **Exact re-ranking:** Decompress top candidates for final ranking

### Expected Results
- **10x smaller index memory footprint**
- **Approximate recall@10 > 0.95** (based on our benchmarks)
- **Faster graph traversal** (less memory bandwidth per node)

### Implementation Plan
1. Implement compressed HNSW graph in Python (prototype)
2. Benchmark recall vs memory vs latency tradeoffs
3. If promising, write a PostgreSQL C extension for native support

## Architecture Diagram

```
                    AGI-HPC Subsystems
                    ==================

 [LH Subsystem]  ----NATS----> [Memory Subsystem] ----NATS----> [RH Subsystem]
      |                              |                                |
      |  TurboQuant                  |  TurboQuant                    |
      |  NATS Codec                  |  pgvector                      |
      |  (compress)                  |  (store compressed)            |
      |                              |                                |
      v                              v                                v
  Embedding                   PostgreSQL pgvector               Embedding
  Generator                   (bytea compressed)                Consumer
                                     |
                              L2 RAM Cache
                              (compressed)
                                     |
                              HNSW Index
                              (compressed nodes)
```

## Benchmarks

### Compression Quality (1024-dim BGE-M3 embeddings)

| Bits | Compressed Size | Ratio (vs float32) | Mean Cosine Sim | Min Cosine Sim |
|------|----------------:|--------------------:|----------------:|---------------:|
| 2    | 260 bytes       | 15.8x              | 0.926           | 0.87           |
| 3    | 388 bytes       | 10.5x              | 0.978           | 0.95           |
| 4    | 516 bytes       | 7.9x               | 0.995           | 0.98           |

**Recommendation:** 3-bit provides the best balance of compression and accuracy for vector database use cases. The 0.978 mean cosine similarity means search ranking is almost perfectly preserved.

### Search Accuracy (recall@10, 50K vectors)

| Method | Recall@10 | Latency |
|--------|----------:|--------:|
| Exact float32 | 1.000 | baseline |
| TurboQuant 3-bit | ~0.97 | +decompress overhead |
| TurboQuant 4-bit | ~0.99 | +decompress overhead |

### Throughput

| Operation | Rate |
|-----------|-----:|
| Compress (single, CPU) | ~50K emb/sec |
| Compress (batch, CPU) | ~100K emb/sec |
| Decompress (single, CPU) | ~80K emb/sec |
| NATS encode (single) | ~50K msg/sec |
| NATS decode (single) | ~80K msg/sec |

## Timeline

| Sprint | Deliverable | Status |
|--------|-------------|--------|
| 1 | KV cache compression module | Done |
| 1 | pgvector compression module | Done |
| 1 | NATS codec module | Done |
| 1 | Unit tests + benchmark suite + CI | Done |
| 2 | PCA-Matryoshka module (v0.3.0) | **Done** |
| 2 | PCAMatryoshkaPipeline + incremental PCA | **Done** |
| 2 | 47 new PCA tests (160 total) | **Done** |
| 2 | IEEE TAI paper experiments | **Done** |
| 3 | Atlas DB compression (RAG + ethics) | **Done** |
| 3 | Route RAG to PCA-384 column | **Done** (live, 91.4% recall) |
| 3 | tsvector FTS population | **Done** (112K rows) |
| 4 | KV cache q8_0 benchmark | **Done** (47% savings, 1.9x slower on Volta) |
| 4 | Adaptive LLM proxy (fast/long context) | **Done** |
| 4 | Shared BGE-M3 embedding service | **Done** |
| 4 | NATS embedding codec | **Done** (89.5x payload reduction) |
| 5 | CUDA Hamming search kernel | **Done** (1.5x faster than IVFFlat) |
| 5 | CUDA ADC kernel | **Done** (42-66x faster than Python, still slower than float) |
| 5 | GPU PCA projection benchmark | **Done** (42x speedup at 500K vec) |
| 6 | 3-tier search cascade benchmark | **Done** |
| 6 | Compiled wiki pipeline (Karpathy-style) | **In progress** |
| 6 | Hybrid RRF fusion | **Tested, NOT worth it** (hurts recall) |
| 7 | Publications embedding (824K) | **In progress** |

## Search Tier Benchmark Results (2026-04-04)

Tested on 112K chunks, 50 queries, vs exact full-dim ground truth:

| Tier | Method | Recall@10 | Latency | Verdict |
|------|--------|-----------|---------|---------|
| 0 | Full 1024-dim exact | 1.000 | 457.2 ms | Baseline (too slow) |
| **2a** | **PCA-384 IVFFlat** | **0.906** | **4.4 ms** | **BEST — 104x faster** |
| 2b | GPU Hamming funnel | 0.908 | 10.8 ms | Same recall, 2.5x slower |
| 2c | tsvector FTS | 0.102 | 2.9 ms | Keyword fallback only |
| 2d | Hybrid RRF (Hamming+FTS) | 0.852 | 14.9 ms | **NOT worth it** |
| 1 | Wiki article lookup | — | <1 ms | Compiling |

**Production cascade**: Wiki → PCA-384 IVFFlat → FTS fallback

**Key learning**: RRF fusion of vector + FTS HURTS recall because low-quality
FTS results (0.102 recall) dilute good vector results through rank fusion.
FTS is useful only as a standalone keyword fallback when vector search
returns nothing.

## Compression Benchmark Results (2026-04-04)

| Method | Cosine | Bytes/vec | Ratio | Latency |
|--------|--------|-----------|-------|---------|
| Full 1024 float32 | 1.000 | 4,096 B | 1.0x | 457 ms |
| PCA-384 float32 | 0.984 | 1,536 B | 2.7x | 4.4 ms |
| TQ3 1024-dim | 0.978 | 388 B | 10.6x | — |
| **PCA-384 + TQ3** | **0.971** | **148 B** | **27.7x** | — |
| Binary PCA-384 | — | 48 B | 85x | 4.9 ms (GPU) |
| NATS JSON payload | — | 254 B | 89.5x | — |

## References

1. Bond, A.H. "PCA-Matryoshka: Enabling Effective Dimension Reduction for Non-Matryoshka Embedding Models." IEEE TAI, 2026.
2. Zandieh, Han, Daliri, Karbasi. "Sub-linear Memory Inference via PolarQuant and QJL." ICLR 2026.
3. Karpathy, A. "LLM Knowledge Bases." X post, 2026. (Inspiration for wiki compilation tier.)
4. Cormack, Clarke, Butt. "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods." SIGIR 2009.
5. Johnson, Douze, Jegou. "Billion-scale similarity search with GPUs." IEEE TBBDATA, 2019.
6. pgvector: https://github.com/pgvector/pgvector
7. TurboQuant Pro: https://github.com/ahb-sjsu/turboquant-pro

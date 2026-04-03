# TurboQuant Expansion Plan: Beyond KV Cache Compression

**Author:** Andrew H. Bond  
**Date:** 2026-04-01  
**Status:** Implementation in progress  

## Executive Summary

TurboQuant Pro implements PolarQuant + QJL (Zandieh et al., ICLR 2026) for LLM KV cache compression, achieving 5.12x compression at 0.978 cosine similarity. The core algorithm -- random orthogonal rotation followed by Lloyd-Max scalar quantization and bit-packing -- is not specific to KV cache tensors. It works on **any high-dimensional vector** where coordinates become approximately i.i.d. Gaussian after rotation.

This document describes four expansion targets within the Atlas AGI-HPC architecture where TurboQuant provides significant improvement.

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
| 1 | pgvector compression module | Done |
| 1 | NATS codec module | Done |
| 1 | Unit tests (54 new) | Done |
| 1 | Benchmark suite | Done |
| 1 | CI integration | Done |
| 2 | Atlas real-data benchmarks | Planned |
| 2 | L2 cache integration | Planned |
| 3 | Compressed HNSW prototype | Planned |
| 4 | PostgreSQL C extension | Planned |
| 5 | CUDA-accelerated batch ops | Planned |

## References

1. Zandieh, Han, Daliri, Karbasi. "Sub-linear Memory Inference via PolarQuant and QJL." ICLR 2026.
2. Johnson, Douze, Jegou. "Billion-scale similarity search with GPUs." IEEE TBBDATA, 2019.
3. pgvector: https://github.com/pgvector/pgvector
4. TurboQuant Pro: https://github.com/ahb-sjsu/turboquant-pro

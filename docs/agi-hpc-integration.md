# AGI-HPC Integration Guide

How to integrate TurboQuant Pro's compressed embedding cache into the AGI-HPC memory subsystem.

## Overview

The AGI-HPC platform uses a tiered memory hierarchy (L1-L5). TurboQuant Pro's `CompressedEmbeddingCache` slots into the **L2 RAM cache** layer, storing compressed embeddings in Redis or in-process memory. This enables ~10x more embeddings at the same memory budget, significantly improving cache hit rates for episodic memory retrieval.

## Architecture

```
L1  │  Hot registers (per-request, ~100 embeddings)
L2  │  CompressedEmbeddingCache + Redis  ← THIS INTEGRATION
L3  │  PostgreSQL pgvector (persistent storage)
L4  │  Object storage archive
L5  │  Cold tape/glacier
```

## Quick Start

```python
import redis
from turboquant_pro.pgvector import TurboQuantPGVector
from turboquant_pro.cache_adapter import (
    CompressedEmbeddingCache,
    RedisCacheBackend,
)

# 1. Configure compression (must match the parameters used for L3 storage)
tq = TurboQuantPGVector(dim=1024, bits=3, seed=42)

# 2. Connect to Redis
redis_client = redis.Redis(host="redis.internal", port=6379, db=2)
backend = RedisCacheBackend(redis_client, prefix="agi:l2:")

# 3. Create the cache with 1-hour TTL
cache = CompressedEmbeddingCache(tq, backend, default_ttl=3600)

# 4. Use it
cache.put("episode:12345", embedding_float32)
result = cache.get("episode:12345")  # np.ndarray or None

# 5. Monitor
print(cache.stats())
# {'n_entries': 1, 'memory_bytes': ..., 'hit_rate': 0.85, ...}
```

## Wire Format Compatibility

The cache stores embeddings using the same `pgbytea` wire format as the PostgreSQL compressed tables and the NATS event codec:

```
[4 bytes float32 norm][packed_bytes]
```

This means:
- Data written by `CompressedEmbeddingCache` can be read by `TurboQuantPGVector.decompress_embedding(CompressedEmbedding.from_pgbytea(...))`
- Data from NATS events can be cached directly without re-compression
- The `dim` and `bits` parameters are **not** embedded in the wire format — all consumers must use the same `TurboQuantPGVector` configuration

## In-Memory Backend (Single Process)

For single-process workloads or testing, use `InMemoryCacheBackend` with LRU eviction:

```python
from turboquant_pro.cache_adapter import InMemoryCacheBackend

backend = InMemoryCacheBackend(max_entries=50_000)
cache = CompressedEmbeddingCache(tq, backend)
```

## Configuration Recommendations

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `dim` | Match L3 | Must be identical across all tiers |
| `bits` | 3 | Best quality/size tradeoff (0.978 cosine sim) |
| `seed` | 42 | Must be identical across all tiers |
| `default_ttl` | 3600 (1 hr) | Tune based on access pattern volatility |
| `max_entries` | RAM / 392 | ~392 bytes per 1024-dim 3-bit embedding |
| Redis prefix | `"agi:l2:"` | Namespace isolation from other Redis users |

## Batch Operations

For bulk cache warming (e.g., after a cold start or migration):

```python
import numpy as np

ids = [f"doc:{i}" for i in range(10_000)]
embeddings = np.load("embeddings.npy")  # (10000, 1024) float32

cache.put_batch(ids, embeddings)
```

## Monitoring

The `stats()` method returns:

```python
{
    "n_entries": 45_000,
    "memory_bytes": 17_640_000,      # ~17 MB compressed
    "equivalent_float32_bytes": 184_320_000,  # ~184 MB if uncompressed
    "effective_compression_ratio": 10.45,
    "hit_count": 123_456,
    "miss_count": 12_345,
    "hit_rate": 0.9091,
}
```

Expose `hit_rate` and `effective_compression_ratio` as Prometheus metrics for alerting on cache degradation.

## GPU-Accelerated Cache Population

When `use_gpu=True` is available (CuPy installed), batch compression can use the fused GPU kernels for faster cache warming:

```python
compressed_list = tq.compress_batch(embeddings, use_gpu=True)
```

This is particularly useful for initial cache population of large corpora.

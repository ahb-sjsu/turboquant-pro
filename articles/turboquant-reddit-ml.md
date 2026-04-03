# [P] TurboQuant Pro: Open-source vector compression toolkit — 5-42x smaller embeddings with 0.97+ recall

**TL;DR**: We built an open-source toolkit that compresses high-dimensional vectors (embeddings, KV cache, anything in pgvector/FAISS) by 5-42x while maintaining 0.95+ cosine similarity. Benchmarked 6 methods on 2.4M real embeddings. MIT licensed.

**GitHub**: https://github.com/ahb-sjsu/turboquant-pro  
**Install**: `pip install turboquant-pro`

## The Problem

Vector databases are eating RAM. If you're running RAG with BGE-M3 (1024-dim float32), each embedding is 4KB. At 1M vectors that's 4GB just for embeddings. At 10M you need 40GB. pgvector, FAISS, Pinecone — they all have this problem.

## What We Built

TurboQuant Pro implements and benchmarks 6 compression methods:

| Method | Ratio | Cosine Sim | Recall@10 | Complexity |
|--------|-------|-----------|-----------|------------|
| Scalar int8 | 4x | 0.999 | 0.99 | Trivial |
| Matryoshka truncation | 4x | 0.97 | 0.96 | Trivial |
| TurboQuant 3-bit | 5.1x | 0.978 | 0.97 | Medium |
| pgvector bytea (TQ) | 10.5x | 0.978 | 0.95 | Medium |
| Matryoshka + int8 | 16x | 0.97 | 0.94 | Low |
| Matryoshka + TQ 3-bit | 42x | 0.93 | 0.90 | Medium |

The core algorithm is PolarQuant + QJL from Zandieh et al. (ICLR 2026) — random rotation maps vectors onto a hypersphere, then Lloyd-Max scalar quantization compresses each coordinate to b bits. We added bit-packing, CUDA kernels, and a streaming KV cache manager on top.

## What's Novel

1. **First open-source implementation** of the Zandieh et al. TurboQuant algorithm
2. **Multi-method benchmarking** on real data (2.4M embeddings from a cross-civilizational ethics corpus spanning 5,000 years — long story)
3. **Practical recommendations** — we found that for most RAG use cases, Matryoshka truncation + scalar int8 (16x, zero training, 3 lines of code) beats fancy methods. TurboQuant's rotation trick only wins for KV cache where you need quality at high compression.
4. **pgvector integration** — store compressed embeddings as bytea, search in compressed space
5. **Streaming KV cache** with L1 (hot, uncompressed) / L2 (cold, compressed) tiering

## Origin Story

This started as a beam search optimization in a symbolic AI system (Theory Radar — formula search engine). The beam candidates were high-dimensional vectors that we compressed to fit wider beams in GPU memory. Then we realized the same trick works for LLM KV cache, then for RAG embeddings, then for... everything.

Adapted from a production system running on 2x Quadro GV100 32GB. Benchmarked against a real pgvector database with 2.4M vectors. This isn't synthetic data — it's actual BGE-M3 embeddings from texts spanning Ancient Greek philosophy to Buddhist suttas to Reddit advice columns.

## Key Finding

**Simple beats clever for most use cases.** Scalar int8 gives you 4x compression at 0.999 cosine similarity with literally 3 lines of NumPy. Matryoshka truncation (just slicing the vector) gives another free 4x if your embedding model supports it (BGE-M3 does). Combined that's 16x with zero moving parts, zero training, zero codebooks that can go stale.

TurboQuant's rotation trick is worth the complexity only when you need the last bit of quality at high compression — specifically for KV cache in long-context inference where the quality/compression tradeoff directly affects output quality.

## Technical Details

The PolarQuant step:
```python
# Random rotation maps any distribution onto unit hypersphere
Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
rotated = x @ Q  # Now each coordinate ~ N(0, 1/sqrt(dim))

# Lloyd-Max scalar quantizer (optimal for Gaussian)
indices = np.digitize(rotated, codebook_boundaries)  # b-bit per coordinate

# Bit-pack: 8 x 3-bit values → 3 bytes
packed = pack_3bit(indices)  # 5.12x compression
```

Decompression:
```python
reconstructed = codebook_centroids[unpack_3bit(packed)]
original_approx = reconstructed @ Q.T  # Inverse rotation
# cosine_similarity(original, original_approx) ≈ 0.978
```

CuPy CUDA kernels for GPU: ~25 GB/s throughput on Volta.

## Usage

```python
from turboquant_pro import TurboQuantKV

tq = TurboQuantKV(head_dim=1024, bits=3)
compressed = tq.compress(embeddings, packed=True)  # 5.12x smaller
recovered = tq.decompress(compressed)               # 0.978 cosine sim
```

For pgvector:
```python
from turboquant_pro.pgvector import TurboQuantPGVector

tqpg = TurboQuantPGVector(dim=1024, bits=3)
bytea_data = tqpg.to_pgbytea(embedding)  # 4096 bytes → 388 bytes
```

## What's Next

- Autotune CLI: `turboquant-pro autotune --source postgres://... --min-recall 0.95`
- Native pgvector extension (C, not Python wrapper)
- FAISS integration
- vLLM KV cache plugin
- Proper paper (arXiv draft is in the repo)

## Links

- **Code**: https://github.com/ahb-sjsu/turboquant-pro
- **Install**: `pip install turboquant-pro`
- **Paper**: Zandieh et al., "Sub-linear Memory Inference via PolarQuant and QJL", ICLR 2026
- **License**: MIT

Feedback welcome. We're particularly interested in benchmarks on other embedding models (OpenAI ada-002, Cohere, etc.) and at larger scale (100M+ vectors).

---

*Built as part of the Atlas AI cognitive architecture project. The 2.4M ethics embeddings come from texts spanning Ancient Greek (Homer, Plato, Aristotle), Hebrew (Talmud, Mishnah), Buddhist (Pali Canon), Sanskrit (Vedas, Upanishads), Old Norse (Eddas), and modern advice columns. Because why not.*

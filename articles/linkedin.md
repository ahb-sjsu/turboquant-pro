# TurboQuant Pro v0.8.0: HNSW on Compressed Embeddings, Fused CUDA Kernels, and a 10x Cache

Excited to share v0.8.0 of TurboQuant Pro, our open-source embedding compression library built on the PolarQuant + QJL algorithm (Zandieh et al., ICLR 2026).

This release adds three major features:

## Compressed HNSW Index

What if your vector index stored 3-bit packed embeddings instead of float32? That's CompressedHNSW -- a pure-Python HNSW graph where every node is ~388 bytes instead of ~4,096 bytes (at dim=1024). Distance computation during graph traversal uses a precomputed centroid-centroid lookup table, so there's no decompression during search. Top candidates are optionally reranked with exact cosine similarity after decompression.

Early benchmarks show 0.85+ recall@10 at ~4x less memory than float32 HNSW.

## Fused CUDA Compression Kernels

The rotation + quantization step in TurboQuant has always been two passes: matrix multiply, then searchsorted. v0.8.0 fuses them into a single tiled GEMM kernel that quantizes inline -- eliminating one full (N, dim) float32 intermediate from global memory. For 100K vectors at dim=1024, that's 400MB of memory traffic saved per batch.

The standalone quantize kernels use unrolled binary search (3 comparisons for 3-bit) instead of generic searchsorted.

## L2 Compressed Embedding Cache

CompressedEmbeddingCache is a pluggable cache (in-memory LRU or Redis) that stores embeddings in TurboQuant's compressed wire format. At 3-bit with dim=1024, each cached embedding is ~392 bytes vs ~4,096 bytes in float32 -- so you fit ~10x more vectors in the same memory budget.

Under Zipf-distributed access patterns (which model real workloads), a compressed cache at 100MB jumps from ~60% to ~95% hit rate compared to float32 caching at the same budget.

---

244 tests. MIT licensed. pip install turboquant-pro.

GitHub: https://github.com/ahb-sjsu/turboquant-pro
PyPI: https://pypi.org/project/turboquant-pro/0.8.0/

#MachineLearning #VectorDatabase #Embeddings #CUDA #OpenSource #Python #LLM #KVCache

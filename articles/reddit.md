# What if your HNSW index stored 3-bit embeddings instead of float32?

I've been experimenting with an approach to vector indexing where the HNSW graph nodes store quantized embeddings (~388 bytes each at dim=1024) instead of float32 vectors (~4,096 bytes).

The key insight: if you quantize embeddings using Lloyd-Max scalar quantization after a random orthogonal rotation (the PolarQuant approach from Zandieh et al., ICLR 2026), you can precompute a centroid-centroid inner product table (8x8 = 64 floats for 3-bit). During graph traversal, distance computation becomes 1024 table lookups instead of 1024 float multiplies + accumulate. No decompression needed during search.

Top-k candidates are then decompressed and reranked with exact cosine similarity for the final result.

**Early results on random embeddings (dim=128, 2K vectors, M=16):**
- recall@10 > 0.85 with reranking (vs brute force)
- Memory per node: ~4x less than float32 HNSW
- Build is slower (Python prototype, not optimized)

**What doesn't work well yet:**
- The quantization noise in distances can cause the greedy search to take suboptimal paths, requiring higher ef values to compensate
- At very small dimensions (dim=64), the neighbor list and cached index overhead dominates, so you don't actually save memory
- Build time is slow -- the Python HNSW implementation isn't competitive with FAISS for construction speed

**Interesting side finding:** When you also build a compressed embedding cache (storing vectors in their packed wire format), you can fit ~10x more embeddings in the same RAM budget. Under Zipf-distributed access patterns, this jumps cache hit rates from ~60% to ~95% at 100MB.

The fused CUDA kernel for rotation + quantization (tiled GEMM that quantizes inline instead of writing the float32 intermediate) also turned out nicely -- eliminates one full N*dim float32 round-trip to global memory.

Has anyone else explored operating ANN indices directly on quantized representations without decompression? Curious about approaches beyond PQ (product quantization), which is the standard but operates on subvectors rather than scalar-quantized full vectors.

The code is at https://github.com/ahb-sjsu/turboquant-pro if anyone wants to poke at it.

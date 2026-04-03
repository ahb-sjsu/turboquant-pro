# TurboQuant: 5x Memory Compression for AI Vector Databases -- An Open-Source Implementation

*Andrew H. Bond | San Jose State University*

---

Every AI system that uses embeddings faces the same problem: vectors are expensive to store.

A single 1024-dimensional embedding in float32 takes 4,096 bytes. That sounds small until you scale. A RAG system with 2.4 million document chunks -- the kind you need for comprehensive retrieval -- requires 9.4 GB just for the embeddings. Add HNSW indexes and you are looking at 15-20 GB of RAM for a single pgvector table. On a system with multiple embedding stores (semantic, episodic, ethics), you quickly exhaust available memory.

This is the problem we hit while building a distributed cognitive architecture that manages millions of embeddings across PostgreSQL, NATS message buses, and multi-tiered memory caches. We needed compression that was fast, accurate, and simple to integrate.

## The Algorithm: PolarQuant + Lloyd-Max

TurboQuant is our open-source implementation of the PolarQuant + QJL algorithm described by Zandieh, Han, Daliri, and Karbasi (ICLR 2026). The key insight is elegant: if you apply a random orthogonal rotation to a high-dimensional vector, the resulting coordinates become approximately i.i.d. Gaussian -- regardless of the original distribution. This is a consequence of the concentration of measure on high-dimensional hyperspheres.

Once coordinates are Gaussian, you can quantize each one independently using a Lloyd-Max scalar quantizer. For 3-bit quantization, each coordinate maps to one of 8 centroids optimized for the Gaussian distribution. Eight 3-bit indices pack neatly into 3 bytes (24 bits).

The full pipeline:

1. **Extract** the L2 norm of the vector (stored separately as float32)
2. **Normalize** to a unit vector
3. **Rotate** with a random orthogonal matrix (QR factorization of a Gaussian matrix)
4. **Quantize** each coordinate to a 3-bit index using precomputed centroids
5. **Bit-pack** the indices (8 values into 3 bytes)

To decompress: unpack, look up centroids, inverse-rotate, scale by stored norm. The entire round-trip preserves 0.978 mean cosine similarity -- more than sufficient for retrieval tasks.

## Real Numbers

We applied TurboQuant to our production embedding workloads:

| Dataset | Vectors | Float32 | Compressed (3-bit) | Ratio |
|---------|--------:|--------:|-------------------:|------:|
| RAG chunks | 112K | 437 MB | 41 MB | 10.5x |
| Ethics corpus | 2.4M | 9,375 MB | 893 MB | 10.5x |
| Publications | 824K | 3,222 MB | 307 MB | 10.5x |

That is 13 GB reduced to 1.2 GB -- all of it now fits comfortably in RAM on a single machine.

For NATS message bus transport, each 1024-dim embedding shrinks from 4,096 bytes to 392 bytes. In batch operations moving hundreds of embeddings between subsystems, the bandwidth savings are substantial.

The compression runs at approximately 100,000 embeddings per second on CPU (NumPy), with optional CuPy GPU acceleration for even higher throughput.

Search accuracy (recall@10) stays above 0.97 -- meaning that 97% of the time, the true top-10 nearest neighbors are correctly identified even when searching over compressed representations.

## How to Use It

TurboQuant Pro is pip-installable and dependency-light (only NumPy):

```bash
pip install turboquant-pro
```

For pgvector embedding compression:

```python
from turboquant_pro import TurboQuantPGVector

tq = TurboQuantPGVector(dim=1024, bits=3, seed=42)

# Compress an embedding: 4,096 bytes -> 388 bytes
compressed = tq.compress_embedding(embedding)
bytea_data = compressed.to_pgbytea()

# Decompress
reconstructed = tq.decompress_embedding(compressed)
```

For message bus transport:

```python
from turboquant_pro import TurboQuantNATSCodec

codec = TurboQuantNATSCodec(dim=1024, bits=3, seed=42)
payload = codec.encode(embedding)   # 392 bytes
decoded = codec.decode(payload)     # cos_sim > 0.978
```

PostgreSQL integration is built in -- create compressed tables, bulk insert, and search directly from Python.

## Why This Matters

Vector databases are a core infrastructure component for modern AI systems, yet their memory requirements scale linearly with the number of embeddings. Current solutions (dimensionality reduction, product quantization) either lose too much information or are complex to implement correctly.

TurboQuant offers a different tradeoff: keep the full dimensionality, quantize the coordinates. The mathematical guarantee (concentration of measure after rotation) means you know in advance what quality to expect. 3-bit gives you 0.978 cosine similarity. 4-bit gives you 0.995. You choose based on your accuracy requirements.

This approach is orthogonal to indexing strategies. You can use TurboQuant with flat scan, HNSW, IVF, or any other index. The compressed vectors are just smaller, so everything runs faster and fits in less memory.

## What Is Next

We are working on several extensions:

- **CUDA kernels** for GPU-accelerated batch compression (targeting 1M+ embeddings/sec)
- **PostgreSQL C extension** for native compressed vector operations (`CREATE TYPE tqvector`)
- **Streaming compression** for real-time embedding pipelines
- **Compressed HNSW** that operates entirely in quantized space

## Get Involved

TurboQuant Pro is MIT-licensed and open source:

- **GitHub:** https://github.com/ahb-sjsu/turboquant-pro
- **PyPI:** https://pypi.org/project/turboquant-pro/
- **Paper:** Zandieh et al., "Sub-linear Memory Inference via PolarQuant and QJL" (ICLR 2026)

If you are working with large embedding stores and memory is a constraint, give TurboQuant a try. We would love to hear how it works for your use case.

---

*Andrew H. Bond is a consultant and researcher working on distributed AI architectures and geometric reasoning at San Jose State University. Connect at linkedin.com/in/andrew-bond or reach out at andrew.bond@sjsu.edu.*

#MachineLearning #VectorDatabases #Compression #OpenSource #pgvector #AI #Embeddings

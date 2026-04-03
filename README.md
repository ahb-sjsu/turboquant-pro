# TurboQuant-KV

[![PyPI version](https://img.shields.io/pypi/v/turboquant-kv)](https://pypi.org/project/turboquant-kv/)
[![Tests](https://img.shields.io/github/actions/workflow/status/ahb-sjsu/turboquant-kv/ci.yml?label=tests)](https://github.com/ahb-sjsu/turboquant-kv/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)

**First open-source implementation of TurboQuant (Zandieh et al., ICLR 2026) for LLM KV cache compression, pgvector embedding compression, and NATS transport.**

5-10x memory reduction with 0.978 cosine similarity. Works on consumer GPUs (Volta+) and CPU.

## Installation

```bash
pip install turboquant-kv

# With GPU support (CUDA 12.x)
pip install turboquant-kv[gpu]

# With pgvector support (PostgreSQL)
pip install turboquant-kv[pgvector]

# With NATS transport support
pip install turboquant-kv[nats]

# Everything
pip install turboquant-kv[all]
```

## Quick Start

```python
import numpy as np
from turboquant_kv import TurboQuantKV

tq = TurboQuantKV(head_dim=256, n_heads=16, bits=3, use_gpu=False)
compressed = tq.compress(kv_tensor, packed=True)   # 5.1x smaller
reconstructed = tq.decompress(compressed)           # cos_sim > 0.978
```

## How It Works

TurboQuant-KV implements the PolarQuant + QJL algorithm from Zandieh et al. (ICLR 2026) for compressing the key-value cache in transformer inference:

```
                    KV Tensor (B, H, S, D)
                           |
                    [L2 Norm Extract]
                           |
                    [Unit Normalize]
                           |
                   [Random Rotation Pi]        <-- QR of Gaussian matrix
                           |
                [Lloyd-Max Scalar Quantize]    <-- b-bit per coordinate
                           |
                     [Bit-Pack Indices]        <-- 8x3-bit = 3 bytes
                           |
              CompressedKV {indices, norms, bits}
                           |
                     [Unpack + Lookup]
                           |
                   [Inverse Rotation]
                           |
                    [Scale by Norms]
                           |
                Reconstructed KV Tensor
```

**Key idea**: A random orthogonal rotation maps head-dimension vectors onto the unit hypersphere, making coordinates approximately i.i.d. Gaussian. This enables efficient scalar quantization with precomputed Lloyd-Max codebooks.

## Benchmark Results

Compression quality and ratios on random Gaussian KV tensors (head_dim=256, n_heads=16, fp16 baseline):

| Bits | Compression Ratio | Cosine Similarity | MSE      |
|------|------------------:|------------------:|---------:|
| 2    |             7.5x  |            0.926  | 0.001178 |
| 3    |             5.1x  |            0.978  | 0.000349 |
| 4    |             3.9x  |            0.995  | 0.000082 |

Memory estimates for popular models at 8K context (3-bit, packed):

| Model           | Original | Compressed | Saved   | Ratio |
|-----------------|----------|------------|---------|-------|
| Llama 3.1 8B   | 0.500 GB | 0.098 GB   | 0.402 GB| 5.1x  |
| Llama 3.1 70B  | 1.250 GB | 0.244 GB   | 1.006 GB| 5.1x  |
| Gemma 4 27B    | 1.125 GB | 0.220 GB   | 0.905 GB| 5.1x  |
| Mistral 7B     | 2.000 GB | 0.391 GB   | 1.609 GB| 5.1x  |

## Streaming Cache

TurboQuant-KV includes a streaming tiered cache for autoregressive generation:

- **L1 (hot window)**: Recent tokens stored uncompressed for zero-latency attention
- **L2 (cold storage)**: Older tokens bit-packed at b-bit precision (~5x compression)

```python
from turboquant_kv import TurboQuantKVCache

cache = TurboQuantKVCache(head_dim=256, n_heads=16, bits=3, hot_window=512)

for token in tokens:
    k, v = model.forward_one(token)
    cache.append(k, v)                          # auto-compresses old entries
    keys = cache.get_keys(0, cache.length)       # seamless hot+cold retrieval
    values = cache.get_values(0, cache.length)
```

## pgvector Embedding Compression

TurboQuant-KV can compress high-dimensional embeddings stored in PostgreSQL pgvector, reducing storage by 10x (from float32) or 5x (from float16):

```python
from turboquant_kv import TurboQuantPGVector

tq = TurboQuantPGVector(dim=1024, bits=3, seed=42)

# Compress a single embedding (4096 bytes -> 388 bytes)
compressed = tq.compress_embedding(embedding_float32)

# Store as bytea in PostgreSQL
bytea_data = compressed.to_pgbytea()

# Batch compress for bulk operations
compressed_batch = tq.compress_batch(embeddings_array)

# Search compressed embeddings
scores = tq.compressed_cosine_similarity(query, compressed_batch)

# PostgreSQL integration
tq.create_compressed_table(conn, "embeddings_compressed")
tq.insert_compressed(conn, "embeddings_compressed", ids, embeddings)
results = tq.search_compressed(conn, "embeddings_compressed", query, top_k=10)
```

**Storage savings for real workloads (1024-dim BGE-M3, 3-bit):**

| Dataset | Vectors | Float32 | Compressed | Ratio | Saved |
|---------|--------:|--------:|-----------:|------:|------:|
| RAG chunks | 112K | 437 MB | 41 MB | 10.5x | 396 MB |
| Ethics chunks | 2.4M | 9,375 MB | 893 MB | 10.5x | 8,482 MB |
| Publications | 824K | 3,222 MB | 307 MB | 10.5x | 2,915 MB |

## NATS Transport Codec

Compress embeddings for transmission over NATS JetStream or any message bus:

```python
from turboquant_kv import TurboQuantNATSCodec

codec = TurboQuantNATSCodec(dim=1024, bits=3, seed=42)

# Encode for transport (4096 bytes -> 392 bytes)
payload = codec.encode(embedding_float32)

# Decode on the receiving end
embedding_approx = codec.decode(payload)

# Batch operations
payloads = codec.encode_batch(embeddings_2d)
embeddings = codec.decode_batch(payloads)

# Check compression stats
print(codec.stats())
# {'dim': 1024, 'bits': 3, 'payload_bytes': 392,
#  'float32_bytes': 4096, 'compression_ratio': 10.45, ...}
```

## Components

| Class | Purpose |
|-------|---------|
| `TurboQuantKV` | Stateless compress/decompress with optional bit-packing |
| `TurboQuantKVCache` | Streaming L1/L2 tiered cache for autoregressive inference |
| `CompressedKV` | Container dataclass for compressed tensors |
| `TurboQuantPGVector` | Compress pgvector embeddings for PostgreSQL storage |
| `CompressedEmbedding` | Container for a single compressed embedding |
| `TurboQuantNATSCodec` | Encode/decode embeddings for NATS transport |

## Integration Options

### llama.cpp / llama-cpp-python

See `examples/llama_integration.py` for a wrapper pattern that intercepts KV tensors and stores them in a `TurboQuantKVCache`.

### vLLM

TurboQuant-KV can be integrated into vLLM's PagedAttention by compressing cold KV pages:

```python
# Conceptual: compress a page of KV cache
tq = TurboQuantKV(head_dim=128, n_heads=8, bits=3)
compressed_page = tq.compress(kv_page, packed=True)
# Store compressed_page instead of raw fp16
```

### HuggingFace Transformers

Wrap the KV cache in `generate()` by subclassing the model's attention:

```python
# Override the cache update in the attention layer
compressed_k = tq.compress(key_states, packed=True)
compressed_v = tq.compress(value_states, packed=True)
# Decompress when computing attention scores
```

## GPU Acceleration

When CuPy is available, TurboQuant-KV uses CUDA RawKernels for bit-packing operations. All kernels are Volta-compatible (compute capability 7.0+).

```python
tq = TurboQuantKV(head_dim=256, n_heads=16, bits=3, use_gpu=True)
# Automatically uses CuPy for rotation, quantization, and bit-packing
```

Falls back to NumPy automatically when CuPy is not installed.

## Citation

If you use TurboQuant-KV in your research, please cite both this implementation and the original algorithm:

```bibtex
@software{bond2025turboquantkv,
  title={TurboQuant-KV: Open-Source PolarQuant+QJL Implementation for LLM KV Cache Compression},
  author={Bond, Andrew H.},
  year={2025},
  url={https://github.com/ahb-sjsu/turboquant-kv},
  license={MIT}
}

@inproceedings{zandieh2026sublinear,
  title={Sub-linear Memory Inference via PolarQuant and QJL},
  author={Zandieh, Amir and Han, Insu and Daliri, Majid and Karbasi, Amin},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## Acknowledgments

- **Algorithm**: Zandieh, Han, Daliri, and Karbasi -- "Sub-linear Memory Inference via PolarQuant and QJL" (ICLR 2026)
- **Origin**: Adapted from the Theory Radar project's TurboBeam beam-search compression, which first implemented PolarQuant+QJL in Python
- **Author**: Andrew H. Bond, San Jose State University

## License

MIT License. See [LICENSE](LICENSE) for details.

# TurboQuant Pro

[![PyPI version](https://img.shields.io/pypi/v/turboquant-pro?v=0.7.0)](https://pypi.org/project/turboquant-pro/)
[![Tests](https://img.shields.io/github/actions/workflow/status/ahb-sjsu/turboquant-pro/ci.yml?label=tests)](https://github.com/ahb-sjsu/turboquant-pro/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)

**PCA-Matryoshka dimension reduction + TurboQuant scalar quantization for embedding compression, LLM KV caches, model weight pruning, pgvector, FAISS, and NATS transport.**

Up to 27x embedding compression at 0.979 cosine similarity. Activation-space PCA for model weight compression. 175 tests. Works on consumer GPUs (Volta+) and CPU.

## What's New in v0.7.0

- **Activation-space PCA** ([FLAT-LLM](https://arxiv.org/abs/2505.23966) inspired): Compress model weights based on which dimensions matter during actual inference, not just weight structure. Requires calibration data.
- **Head-wise granularity**: Each attention head analyzed separately — identifies which heads are compressible and which aren't.
- **Differential compression**: `sweep(mode="activation")` compares weight-space vs activation-space at each ratio.

### Previous releases

- **v0.6.0**: Model weight compression (`ModelCompressor`), weight-space SVD, [MatFormer](https://arxiv.org/abs/2310.07707) inspired.
- **v0.5.0**: Autotune CLI, FAISS integration, vLLM KV cache plugin, Rust pgext.
- **v0.4.0**: Autotune CLI.
- **v0.3.0**: PCA-Matryoshka (`PCAMatryoshka`, `PCAMatryoshkaPipeline`).

## Installation

```bash
pip install turboquant-pro

# With pgvector + autotune
pip install turboquant-pro[pgvector]

# With FAISS
pip install turboquant-pro[faiss]

# With GPU support (CUDA 12.x)
pip install turboquant-pro[gpu]

# Everything
pip install turboquant-pro[all]
```

## Quick Start

```python
import numpy as np
from turboquant_pro import TurboQuantKV

tq = TurboQuantKV(head_dim=256, n_heads=16, bits=3, use_gpu=False)
compressed = tq.compress(kv_tensor, packed=True)   # 5.1x smaller
reconstructed = tq.decompress(compressed)           # cos_sim > 0.978
```

## PCA-Matryoshka Compression

PCA-Matryoshka applies a PCA rotation to any non-Matryoshka embedding model's output, reordering dimensions by explained variance so that truncation becomes effective without retraining. Combined with TurboQuant quantization, this achieves up to 114x compression.

```python
from turboquant_pro import PCAMatryoshka

# Fit PCA on a sample of embeddings (5-10K vectors is sufficient)
pca = PCAMatryoshka(input_dim=1024, output_dim=384)
result = pca.fit(sample_embeddings)
print(f"Variance explained: {result.total_variance_explained:.1%}")

# Create the full pipeline: PCA-384 + TurboQuant 3-bit
pipeline = pca.with_quantizer(bits=3)  # ~27x compression

# Compress/decompress
compressed = pipeline.compress(embedding)      # 4096 bytes -> ~148 bytes
reconstructed = pipeline.decompress(compressed)  # cosine ~0.979
```

**15-method compression comparison on BGE-M3 (1024-dim, 2.4M vectors):**

| Method | Compression | Cosine Sim | Recall@10 |
|--------|----------:|----------:|----------:|
| Scalar int8 | 4x | 0.9999 | 97.2% |
| TurboQuant 4-bit | 7.9x | 0.995 | 90.4% |
| TurboQuant 3-bit | 10.6x | 0.978 | 83.8% |
| **PCA-384 + TQ3** | **27.7x** | **0.979** | **76.4%** |
| PCA-256 + TQ3 | 41.0x | 0.963 | 78.2% |
| Binary quantization | 32.0x | 0.758 | 66.6% |
| PCA-128 + TQ2 | 113.8x | 0.924 | 78.7% |
| PQ M=16 K=256 | 256.0x | 0.810 | 41.4% |

**Production deployment (PCA-384 + TQ3, BGE-M3):**

Deployed on 3.3M production vectors (BGE-M3, 1024-dim). PCA-384 + TQ3 compresses every vector from 4,096 bytes to ~148 bytes (27.7x) regardless of content — the ratio is a property of the config, not the data:

| Corpus | Vectors | Original | Compressed |
|--------|--------:|---------:|-----------:|
| Ethics (37 langs) | 2.4M | 9.4 GB | 338 MB |
| Publications | 824K | 3.2 GB | 116 MB |
| Code repos | 112K | 437 MB | 16 MB |
| **Total** | **3.3M** | **13 GB** | **470 MB** |

## Autotune CLI

Find the optimal compression for your data in ~10 seconds:

```bash
turboquant-pro autotune \
  --source "dbname=mydb user=me" \
  --table chunks --column embedding \
  --min-recall 0.95
```

Real output on 194K production embeddings:

```
              Config   Ratio   Cosine   Recall   Var%   Time
--------------------------------------------------------------
       PCA-128 + TQ2  113.8x   0.9237   78.7%  79.9%   2.2s
       PCA-256 + TQ3   41.0x   0.9700   92.0%  92.3%   0.7s
       PCA-384 + TQ4   20.9x   0.9906   96.0%  97.3%   0.6s
       PCA-512 + TQ4   15.8x   0.9949   96.3%  99.0%   0.6s

Recommendation (min recall >= 95%):
  PCA-384 + TQ4: 20.9x compression, 96.0% recall@10
```

## FAISS Integration

Wrap FAISS indices with automatic PCA compression:

```python
from turboquant_pro import PCAMatryoshka
from turboquant_pro.faiss_index import TurboQuantFAISS

pca = PCAMatryoshka(input_dim=1024, output_dim=384)
pca.fit(sample_embeddings)

index = TurboQuantFAISS(pca, index_type="ivf", n_lists=100)
index.add(corpus)  # Auto PCA-compressed
distances, ids = index.search(query, k=10)  # Auto PCA-rotated
print(index.stats())  # 2.7x smaller index
```

Supports Flat, IVF, and HNSW. Save/load indices to disk.

## How It Works

TurboQuant Pro implements the PolarQuant + QJL algorithm from Zandieh et al. (ICLR 2026) for compressing the key-value cache in transformer inference:

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

## Native PostgreSQL Extension (Rust + CUDA)

The `pgext/` directory contains a native PostgreSQL extension written in Rust (pgrx) that adds the `tqvector` data type directly to PostgreSQL — no Python needed.

```sql
-- Compress your entire table in one command
CREATE TABLE embeddings_tq AS
SELECT id, tq_compress(embedding::float4[], 3) AS tqv
FROM embeddings;

-- Search with cosine distance operator
SELECT id, tqv <=> tq_compress(query::float4[], 3) AS dist
FROM embeddings_tq ORDER BY dist LIMIT 10;

-- Check compression
SELECT tq_dim(tqv), tq_bits(tqv), tq_ratio(tqv) FROM embeddings_tq LIMIT 1;
-- 1024, 3, 10.6
```

**Production benchmark (194K BGE-M3 1024-dim vectors on Atlas):**

| Metric | Result |
|--------|--------|
| Compression speed | 23,969 vec/sec |
| Storage (original) | 5,237 MB |
| Storage (compressed) | 169 MB |
| Compression ratio | 31x (including table overhead) |
| Rust unit tests | 12 passing |

Build and install:
```bash
cd pgext
cargo install cargo-pgrx && cargo pgrx init --pg16 $(which pg_config)
cargo pgrx install --release
psql -c "CREATE EXTENSION tqvector;"
```

Optional GPU acceleration: `cargo build --features gpu` (requires CUDA 12.0+, cudarc).

See [`pgext/README.md`](pgext/README.md) for full API documentation.

## Model Weight Compression (v0.6-0.7)

PCA-Matryoshka applied to model parameters. Two modes:

**Weight-space SVD** (v0.6, fast, no data needed): SVD on weight matrices directly.
**Activation-space PCA** (v0.7, FLAT-LLM inspired): Run calibration data, PCA the activations, compress in the directions that matter least for inference. More accurate.

Head-wise granularity: Each attention head is analyzed separately — some heads are highly compressible, others aren't.

Inspired by [MatFormer](https://arxiv.org/abs/2310.07707) and [FLAT-LLM](https://arxiv.org/abs/2505.23966).

**Important caveat:** Eigenspectrum analysis is *diagnostic*, not a performance guarantee. Keeping 95% of SVD variance does NOT mean keeping 95% of downstream accuracy. Always validate with `sweep()` + `eval_fn`.

```bash
# Weight-space analysis (fast, no calibration data)
turboquant-pro model --model "meta-llama/Llama-3.2-1B"

# Activation-space analysis (accurate, needs calibration data)
turboquant-pro model --model "meta-llama/Llama-3.2-1B" \
  --mode activation --calibration cal_data.txt --n-samples 64
```

```python
from turboquant_pro.model_compress import ModelCompressor

compressor = ModelCompressor(model)

# Weight-space (fast)
report = compressor.analyze()
compressed = compressor.compress(0.5)

# Activation-space (accurate, needs calibration data)
report = compressor.analyze_activations(
    calibration_data=texts,  # list of strings
    tokenizer=tokenizer,
    n_samples=64,
)
# Per-head analysis
for head in report.heads:
    if head.compressible:
        print(f"{head.layer_name} head {head.head_idx}: "
              f"rank {head.effective_rank}/{head.head_dim} — COMPRESS")

# Compress using activation-space PCA basis
compressed = compressor.compress_activations(target_ratio=0.5)

# ALWAYS validate on downstream tasks
results = compressor.sweep(
    ratios=[0.3, 0.5, 0.7],
    eval_fn=lambda m: evaluate_perplexity(m, test_set),
    mode="activation",
)
```

## Benchmark Results

Compression quality and ratios on random Gaussian KV tensors (head_dim=256, n_heads=16, fp16 baseline):

| Bits | Compression Ratio | Cosine Similarity | MSE      |
|------|------------------:|------------------:|---------:|
| 2    |             7.5x  |            0.926  | 0.001178 |
| 3    |             5.1x  |            0.978  | 0.000349 |
| 4    |             3.9x  |            0.995  | 0.000082 |

KV cache memory estimates at 8K context (3-bit packed, ~5.1x compression for all models — ratio depends on bit width, not model):

| Model | KV Cache (fp16) | Compressed (3-bit) | Saved |
|-------|----------------:|-------------------:|------:|
| Llama 3.1 8B | 0.50 GB | 0.10 GB | 0.40 GB |
| Llama 3.1 70B | 1.25 GB | 0.24 GB | 1.01 GB |
| Gemma 4 27B | 1.13 GB | 0.22 GB | 0.91 GB |
| Mistral 7B | 2.00 GB | 0.39 GB | 1.61 GB |

## Streaming Cache

TurboQuant Pro includes a streaming tiered cache for autoregressive generation:

- **L1 (hot window)**: Recent tokens stored uncompressed for zero-latency attention
- **L2 (cold storage)**: Older tokens bit-packed at b-bit precision (~5x compression)

```python
from turboquant_pro import TurboQuantKVCache

cache = TurboQuantKVCache(head_dim=256, n_heads=16, bits=3, hot_window=512)

for token in tokens:
    k, v = model.forward_one(token)
    cache.append(k, v)                          # auto-compresses old entries
    keys = cache.get_keys(0, cache.length)       # seamless hot+cold retrieval
    values = cache.get_values(0, cache.length)
```

## pgvector Embedding Compression

TurboQuant Pro can compress high-dimensional embeddings stored in PostgreSQL pgvector, reducing storage by 10x (from float32) or 5x (from float16):

```python
from turboquant_pro import TurboQuantPGVector

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

**Storage savings (1024-dim BGE-M3, 3-bit, no PCA truncation):**

TurboQuant 3-bit alone compresses each vector from 4,096 to ~388 bytes (10.5x):

| Corpus | Vectors | Original | Compressed |
|--------|--------:|---------:|-----------:|
| RAG chunks | 112K | 437 MB | 41 MB |
| Ethics | 2.4M | 9,375 MB | 893 MB |
| Publications | 824K | 3,222 MB | 307 MB |

## NATS Transport Codec

Compress embeddings for transmission over NATS JetStream or any message bus:

```python
from turboquant_pro import TurboQuantNATSCodec

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
| `PCAMatryoshka` | PCA rotation + truncation for dimension reduction |
| `PCAMatryoshkaPipeline` | Combined PCA + TurboQuant end-to-end pipeline |
| `TurboQuantKV` | Stateless compress/decompress with optional bit-packing |
| `TurboQuantKVCache` | Streaming L1/L2 tiered cache for autoregressive inference |
| `TurboQuantKVManager` | Multi-layer KV cache manager (vLLM plugin) |
| `TurboQuantFAISS` | FAISS index wrapper with auto PCA compression |
| `TurboQuantPGVector` | Compress pgvector embeddings for PostgreSQL storage |
| `TurboQuantNATSCodec` | Encode/decode embeddings for NATS transport |
| `run_autotune` | Sweep configs and recommend optimal compression |
| `ModelCompressor` | SVD analysis + low-rank compression of model FFN weights |

## Integration Options

### llama.cpp / llama-cpp-python

See `examples/llama_integration.py` for a wrapper pattern that intercepts KV tensors and stores them in a `TurboQuantKVCache`.

### vLLM KV Cache Plugin

Multi-layer KV cache manager with hot/cold tiering:

```python
from turboquant_pro.vllm_plugin import TurboQuantKVManager

mgr = TurboQuantKVManager(
    n_layers=32, n_kv_heads=8, head_dim=128,
    bits=3, hot_window=512
)

# Store tokens as they're generated
mgr.store(layer_id=0, keys=k_tensor, values=v_tensor)

# Load back (transparently decompresses cold storage)
keys, values = mgr.load(layer_id=0, start=0, end=1024)

# Estimate max context for a memory budget
max_ctx = mgr.estimate_capacity(max_memory_gb=4.0)  # ~32K instead of ~8K

print(mgr.memory_stats())  # compression_ratio, saved_mb, etc.
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

When CuPy is available, TurboQuant Pro uses CUDA RawKernels for bit-packing operations. All kernels are Volta-compatible (compute capability 7.0+).

```python
tq = TurboQuantKV(head_dim=256, n_heads=16, bits=3, use_gpu=True)
# Automatically uses CuPy for rotation, quantization, and bit-packing
```

Falls back to NumPy automatically when CuPy is not installed.

## Citation

If you use TurboQuant Pro in your research, please cite both this implementation and the original algorithm:

```bibtex
@software{bond2026turboquantpro,
  title={TurboQuant Pro: PCA-Matryoshka + TurboQuant Compression for Embeddings and LLM KV Caches},
  author={Bond, Andrew H.},
  year={2026},
  url={https://github.com/ahb-sjsu/turboquant-pro},
  license={MIT}
}

@article{bond2026pcamatryoshka,
  title={PCA-Matryoshka: Enabling Effective Dimension Reduction for Non-Matryoshka Embedding Models with Applications to Vector Database Compression},
  author={Bond, Andrew H.},
  journal={IEEE Transactions on Artificial Intelligence},
  year={2026}
}

@inproceedings{zandieh2026sublinear,
  title={Sub-linear Memory Inference via PolarQuant and QJL},
  author={Zandieh, Amir and Han, Insu and Daliri, Majid and Karbasi, Amin},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}

@article{devvrit2023matformer,
  title={MatFormer: Nested Transformer for Elastic Inference},
  author={Devvrit and Kudugunta, Sneha and Kusupati, Aditya and others},
  journal={arXiv:2310.07707},
  year={2023}
}

@article{flatllm2025,
  title={FLAT-LLM: Fine-grained Low-rank Activation Space Transformation for Large Language Model Compression},
  journal={arXiv:2505.23966},
  year={2025}
}
```

## Acknowledgments

- **PolarQuant algorithm**: Zandieh, Han, Daliri, and Karbasi — "Sub-linear Memory Inference via PolarQuant and QJL" (ICLR 2026)
- **MatFormer**: Devvrit et al. — "Nested Transformer for Elastic Inference" (2023). Inspired the model weight compression module.
- **FLAT-LLM**: "Fine-grained Low-rank Activation Space Transformation for LLM Compression" (2025). Inspired activation-space PCA and head-wise analysis.
- **Matryoshka Representation Learning**: Kusupati et al. (2022). PCA-Matryoshka extends this concept to non-Matryoshka models via training-free PCA rotation.
- **Origin**: Adapted from the Theory Radar project's TurboBeam beam-search compression, which first implemented PolarQuant+QJL in Python.
- **Community**: Thanks to DigThatData and others on r/machinelearning for feedback on evaluation methodology, the varimax connection, and the FLAT-LLM pointer.
- **Author**: Andrew H. Bond, San Jose State University

## License

MIT License. See [LICENSE](LICENSE) for details.

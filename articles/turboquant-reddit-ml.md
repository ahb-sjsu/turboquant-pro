# [P] [R] PCA-Matryoshka: 27x embedding compression at 0.979 cosine sim — now with autotune, FAISS, and vLLM KV cache

**TL;DR:** Most embedding models can't be truncated — naive dimension reduction destroys them. We show that fitting PCA once on a sample and rotating before truncation makes it work. BGE-M3 truncated to 256d: naive = 0.467 cosine (useless), PCA first = 0.974 cosine (+109%). Combined with 3-bit quantization: 27x compression at 0.979 cosine sim. Deployed on 3.3M vectors in production. v0.5 adds autotune CLI, FAISS integration, and vLLM KV cache compression. Open source.

**GitHub**: https://github.com/ahb-sjsu/turboquant-pro
**Install**: `pip install turboquant-pro[all]`

---

## The Problem

If you're running a RAG system with millions of embeddings, memory is your bottleneck. A 2.4M-vector corpus in float32 at 1024 dimensions costs 9.4 GB just for embeddings. Add indexes and you're at 15-20 GB for one table.

Matryoshka-trained models (OpenAI text-embedding-3, etc.) let you truncate dimensions cheaply. But **most deployed models weren't trained that way** — BGE-M3, Cohere Embed, ada-002, E5-large. For these models, information is distributed roughly uniformly across dimensions, and naive truncation is catastrophic.

## The Fix: PCA Rotation

The insight is embarrassingly simple: **PCA reorders the dimensions by importance, then truncation works.**

1. Fit PCA on a sample of your embeddings (5K-10K vectors is enough)
2. Rotate all vectors into the PCA basis
3. Now truncation works — trailing dimensions are the least important

Results on BGE-M3 (1024-dim, 10K vectors):

| Dims | Naive Truncation | PCA First | Improvement |
|------|-----------------|-----------|-------------|
| 512 | 0.707 | 0.996 | +41% |
| 384 | 0.609 | 0.990 | +63% |
| **256** | **0.467** | **0.974** | **+109%** |
| 128 | 0.333 | 0.933 | +180% |

**Why it works:** Learned embeddings have rapidly decaying eigenvalues. The effective dimensionality is ~400 despite nominal 1024. PCA concentrates signal into the leading components — Eckart-Young theorem guarantees this is optimal among linear projections.

## Full Compression Pipeline: 15-Method Comparison

We benchmarked 15 compression methods on the same corpus (2.4M BGE-M3 embeddings from a cross-civilizational ethics dataset spanning 37 languages):

| Method | Compression | Cosine Sim | Recall@10 |
|--------|------------|-----------|-----------|
| Scalar int8 | 4x | 0.9999 | 97.2% |
| TurboQuant 4-bit | 7.9x | 0.995 | 90.4% |
| TurboQuant 3-bit | 10.6x | 0.978 | 83.8% |
| **PCA-384 + TQ3** | **27.7x** | **0.979** | **76.4%** |
| PCA-256 + TQ3 | 41x | 0.963 | 78.2% |
| Binary quantization | 32x | 0.758 | 66.6% |
| PQ M=16, K=256 | 256x | 0.810 | 41.4% |
| Matryoshka 512d | 2x | 0.736 | 69.6% |
| Matryoshka 256d | 4x | 0.466 | 57.4% |

**Key finding:** PCA-384 + TQ3 *matches* standalone TurboQuant's cosine similarity (0.979 vs 0.978) at **2.6x higher compression**. It fills the previously empty gap in the Pareto frontier between scalar quantization (<10x) and binary/PQ (>32x).

PCA-Matryoshka + TQ **strictly dominates** both binary quantization and product quantization across the practical range.

## Production Deployment

Running on 3.3M vectors across 6 corpora (pgvector + IVFFlat):

| Corpus | Vectors | Float32 | Compressed | Ratio |
|--------|---------|---------|------------|-------|
| Ethics (37 languages) | 2.4M | 9.4 GB | 338 MB | 27x |
| Academic papers | 824K | 3.2 GB | 116 MB | 27x |
| Code repos | 112K | 437 MB | 16 MB | 27x |
| **Total** | **3.3M** | **13 GB** | **470 MB** | **27x** |

Search: 1,840 QPS. Compression throughput: 100K/sec CPU (NumPy), 2.1M/sec GPU (CuPy Volta kernels).

## New in v0.5: Autotune, FAISS, vLLM

### Autotune CLI

Stop guessing your compression config. One command sweeps 12 configurations on your actual data:

```bash
turboquant-pro autotune \
  --source "dbname=mydb user=me" \
  --table chunks --column embedding \
  --min-recall 0.95
```

On our 194K production corpus (10.8 seconds, no GPU):

```
       PCA-128 + TQ2  113.8x   0.9237   78.7%
       PCA-384 + TQ3   27.7x   0.9823   93.7%
       PCA-384 + TQ4   20.9x   0.9906   96.0%  << RECOMMENDED
       PCA-512 + TQ4   15.8x   0.9949   96.3%
```

### FAISS Integration

Wraps FAISS with auto PCA rotation. Index stores compressed vectors, queries auto-rotated:

```python
from turboquant_pro.faiss_index import TurboQuantFAISS

index = TurboQuantFAISS(pca, index_type="ivf", n_lists=100)
index.add(corpus)  # 1024-dim -> 384-dim automatically
distances, ids = index.search(query, k=10)
```

Supports Flat, IVF, HNSW. 2.7x smaller index, same search API.

### vLLM KV Cache Compression

Same principle for transformer inference. Hot/cold tiering — recent tokens uncompressed, older tokens 3-bit compressed:

```python
from turboquant_pro.vllm_plugin import TurboQuantKVManager

mgr = TurboQuantKVManager(n_layers=32, n_kv_heads=8, head_dim=128, bits=3)
max_ctx = mgr.estimate_capacity(max_memory_gb=4.0)  # ~32K instead of ~8K
```

Gemma 4 31B KV cache: 2 GB -> 340 MB. Same memory, 4x longer context.

## Limitations (Being Honest)

- **Recall@10 degrades faster than cosine.** 27x compression gives 0.979 cosine but only 76.4% recall@10. If you need >95% recall, use PCA-384+TQ4 (21x, 96% recall).
- **PCA needs fitting once.** ~30 seconds on 10K vectors. 5K samples converge to within 0.002 cosine of the full-corpus basis.
- **KV cache quality depends on model.** Tested on Gemma 4; your mileage may vary on different architectures.

## Code

```python
from turboquant_pro import PCAMatryoshka, PCAMatryoshkaPipeline, TurboQuantPGVector

pca = PCAMatryoshka(input_dim=1024, output_dim=384)
pca.fit(sample_embeddings)
tq = TurboQuantPGVector(dim=384, bits=3)
pipeline = PCAMatryoshkaPipeline(pca, tq)

compressed = pipeline.compress(embedding)  # 4096 bytes -> 150 bytes
recovered = pipeline.decompress(compressed)  # cos_sim > 0.979
```

175 tests passing. MIT licensed. Core dependency: just NumPy.

## What's Next

- Native pgvector C extension (`CREATE TYPE tqvector` — search in compressed space without Python)
- Async vLLM backend for non-blocking KV offload
- Compressed HNSW that operates entirely in quantized space

---

**GitHub:** https://github.com/ahb-sjsu/turboquant-pro
**PyPI:** `pip install turboquant-pro[all]` (v0.5.0)
**Paper:** IEEE TAI submission (15-method comparison, eigenspectrum analysis, cross-lingual evaluation on 2.4M vectors across 37 languages)

*The 2.4M ethics embeddings span Homer to the Talmud to Reddit advice columns, across 37 languages and 5,000 years. The PCA doesn't care — eigenvalues decay the same way regardless of whether the text is the Bhagavad Gita or r/AmItheAsshole.*

# [R] PCA recovers the same ordered features as Matryoshka — and we used it to build eigenvalue-weighted quantization

**TL;DR**: A colleague pointed us to a new paper proving PCA (Rayleigh-Ritz) recovers the same ordered features as Matryoshka representation learning. We used this insight to build eigenvalue-weighted quantization that allocates more bits to important dimensions. Result: **+0.35% cosine similarity at the same compression ratio** (closing 85% of the gap to 4-bit quality at 3-bit cost). Open source in `turboquant-pro` v0.8.0.

---

## The story

We published TurboQuant Pro a few weeks ago — an embedding compression toolkit that uses PCA rotation to convert any embedding model into a Matryoshka-like representation, then applies scalar quantization. Our [PCA-Matryoshka pipeline](https://github.com/ahb-sjsu/turboquant-pro) gets 27x compression on BGE-M3 at 0.979 cosine similarity, which was already pretty good.

Then a colleague on Reddit dropped a link to [Varici et al., "Eigenfunction Extraction for Ordered Representation Learning"](https://arxiv.org/abs/2510.24672) from Carnegie Mellon, saying:

> *"One paper actually proves that PCA (or Rayleigh-Ritz in the paper) is actually recovering the same ordered features as Matryoshka from spectral perspective."*

We read the paper and realized: **if eigenvalues are theoretically grounded importance scores for each PCA dimension, we shouldn't be wasting the same number of bits on every dimension.** The top PCA components carry 49% of the variance in the first 25% of dimensions — they deserve more bits. The bottom dimensions carry <7% of the variance — they can get by with fewer.

## What we built

**Eigenvalue-weighted quantization**: instead of uniform 3-bit across all PCA dimensions, we allocate bits proportional to eigenvalue importance:

| Dimension range | Eigenvalue importance | Bits allocated |
|---|---|---|
| Top 25% (dims 1-64) | ~49% variance | 4 bits |
| Middle 50% (dims 65-192) | ~44% variance | 3 bits |
| Bottom 25% (dims 193-256) | ~7% variance | 2 bits |

Average: still 3.0 bits/dim. Same compression ratio. Better quality.

## Real benchmarks on BGE-M3

10,000 text embeddings (1024-dim BGE-M3), PCA to various target dimensions:

### At 256 dimensions (41x compression)

| Method | Cosine | Compression |
|---|---|---|
| Naive truncation (no PCA) | 0.4750 | — |
| PCA + uniform 3-bit | 0.9934 | 41.0x |
| **PCA + weighted 4+3+2** | **0.9969** | **41.0x** |
| PCA + uniform 4-bit | 0.9970 | 31.0x |

The weighted 3-bit essentially matches uniform 4-bit quality but at **32% more compression** (41x vs 31x).

### At 128 dimensions (78.8x compression)

| Method | Cosine | Compression |
|---|---|---|
| Naive truncation (no PCA) | 0.3489 | — |
| PCA + uniform 3-bit | 0.9914 | 78.8x |
| **PCA + weighted 4+3+3+2** | **0.9947** | **78.8x** |
| PCA + uniform 4-bit | 0.9949 | 60.2x |

At extreme compression (78.8x), eigenvalue-weighted 3-bit closes **85% of the gap to 4-bit quality** at zero extra storage cost.

### At 384 dimensions (27.7x compression)

| Method | Cosine | Compression |
|---|---|---|
| PCA + uniform 3-bit | 0.9943 | 27.7x |
| **PCA + weighted 4+3+2** | **0.9978** | **27.7x** |
| PCA + uniform 4-bit | 0.9980 | 20.9x |

Same pattern. The weighted approach consistently matches 4-bit quality at 3-bit cost.

## Why this works (the theory)

Varici et al. prove that contrastive and non-contrastive learning implicitly perform spectral decomposition of a "contextual kernel." PCA extracts the **ordered eigenfunctions** from this kernel, with eigenvalues serving as importance scores. The first PCA component captures the most variance, the second captures the next most, etc.

When you apply uniform quantization, you're treating dimension 1 (49% of variance) the same as dimension 256 (0.03% of variance). That's wasteful. Eigenvalue-weighted quantization allocates precision proportional to information content — **more bits where it matters, fewer where it doesn't**.

Mathematically, this is the Eckart-Young-Mirsky theorem meets Shannon rate-distortion theory: optimal bit allocation is proportional to log of the signal variance, which is exactly the eigenvalue.

## What's new in v0.8.0

```python
from turboquant_pro.pca import PCAMatryoshka

pca = PCAMatryoshka(input_dim=1024, output_dim=256)
pca.fit(embeddings)

# Old way: uniform 3-bit
pipe_uniform = pca.with_quantizer(bits=3)

# New way: eigenvalue-weighted (same avg bits, better quality)
pipe_weighted = pca.with_weighted_quantizer(
    bit_schedule=[(64, 4), (128, 3), (64, 2)]  # 4+3+2 bits
)

# Or auto-compute from eigenvalue spectrum
pipe_auto = pca.with_weighted_quantizer(avg_bits=3.0)
```

`pip install turboquant-pro>=0.8.0`

**GitHub**: [github.com/ahb-sjsu/turboquant-pro](https://github.com/ahb-sjsu/turboquant-pro)

## Acknowledgments

Thanks to the Reddit commenter who pointed us to Varici et al.'s paper — sometimes the best improvements come from a single link in a comment thread. And thanks to Varici, Tsai, Ray, Boffi, and Ravikumar at CMU for the spectral-theoretic foundation that made this click.

The earlier r/machinelearning discussion that flagged the SVD variance vs. downstream accuracy distinction (u/DigThatData) also influenced our thinking — eigenvalue-weighted quantization is precisely how you bridge that gap.

---

*Disclosure: I'm the author of turboquant-pro. This is an open-source project, no commercial affiliation. The benchmark code and all results are reproducible from the repo.*

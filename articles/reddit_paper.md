# [R] PCA rotation makes non-Matryoshka embeddings truncatable — 27x compression at 99% recall with reranking

Most embedding models (BGE-M3, E5, ada-002, Cohere) weren't trained with Matryoshka losses, so you can't just drop trailing dimensions. We tried: truncating BGE-M3 from 1024 to 256 dims gives 0.467 cosine similarity. Unusable.

**The fix is embarrassingly simple.** Fit PCA on a sample of your embeddings (~5K vectors is enough), then rotate all vectors into the principal component basis before truncating. The eigenvalues reorder dimensions by importance, so truncation now discards the least important ones instead of arbitrary ones.

Result: PCA truncation to 256 dims gives 0.974 cosine similarity. That's a 109% improvement from a one-line linear transformation with no retraining.

## The compression pipeline

Stack PCA dimension reduction with scalar quantization (3-bit per coordinate, using the PolarQuant rotation trick from Zandieh et al. ICLR 2026):

1. PCA rotate + truncate to 384 dims (from 1024)
2. Random orthogonal rotation (makes coordinates ~Gaussian)
3. Lloyd-Max 3-bit quantization + bit-packing

Result: **27x compression** (4096 bytes → 148 bytes per embedding).

## The recall numbers (this is the part that matters)

We benchmarked on a 2.4M-vector cross-civilizational ethics corpus (BGE-M3 embeddings). Here's what we found:

| Method | Compression | Recall@10 |
|--------|------------|-----------|
| Scalar int8 | 4x | 97.2% |
| TurboQuant 3-bit | 10.6x | 83.8% |
| PCA-384 + TQ3 | 27.7x | 77.0% |
| PCA-256 + TQ3 | 41.0x | 78.2% |
| Binary quantization | 32x | 66.6% |
| Product quantization (M=16) | 256x | 41.4% |

79% recall single-stage isn't great. But with standard 5x oversampling + exact reranking (fetch 50 candidates, rescore with original vectors), it jumps to **99.8% recall@10**. Critically, we benchmarked ALL methods with the same reranking:

| Method | Compression | No rerank | 5x rerank |
|--------|------------|-----------|-----------|
| Scalar int8 | 4x | 99.0% | 100% |
| TQ3 | 10.5x | 83.4% | 100% |
| **PCA-384 + TQ3** | **27.7x** | 79.2% | **99.8%** |
| PCA-256 + TQ3 | 41x | 75.4% | 98.6% |
| Binary | 32x | 54.4% | 85.6% |
| PQ (M=16) | 256x | 38.4% | 73.6% |

Binary at 32x only reaches 85.6% with the same reranking. PQ only 73.6%. The dominance holds under reranking, not just single-stage. Verified on 50K production embeddings.

## The surprising finding: cosine similarity lies to you

This was the most interesting part of the paper. Look at these two rows:

- PCA-384 + TQ3: **0.979** cosine similarity, **76.4%** recall@10
- PCA-256 + TQ3: **0.963** cosine similarity, **78.2%** recall@10

PCA-256 has *lower* cosine similarity but *higher* recall. The per-vector reconstruction fidelity metric diverges from the ranking quality metric at high compression. Small perturbations distributed across many vectors can swap the order of closely-ranked items even when each individual vector looks good.

**Takeaway:** If you're evaluating embedding compression for retrieval, report recall@k, not just cosine similarity. We almost made this mistake ourselves — the cosine numbers made PCA-384 look better than PCA-256, but recall tells the opposite story.

## What doesn't work

- **Naive truncation of non-Matryoshka models.** Just dropping dims is catastrophic (0.467 cosine at 50% dims, 0.333 at 25% dims). The information is distributed roughly uniformly — you need PCA to concentrate it.
- **Product quantization at the same compression range.** PQ (M=16 K=256) gets 256x compression but only 41% recall. PCA-128 + TQ3 gets 79x compression at 79% recall — strictly dominates PQ in the 30-80x range.
- **Relying on cosine similarity to evaluate compression quality.** We keep repeating this because it's the easiest trap to fall into.

## Two bonus findings from the implementation work

**Learned codebooks:** The standard Lloyd-Max quantization assumes rotated coordinates are Gaussian. They're not — the tails are heavier. Training a codebook on your actual rotated data (just 1D k-means, 50 iterations) reduces quantization error by 22% at the same 3 bits. Works consistently across models.

**Asymmetric K/V allocation for KV caches:** Keys are more sensitive to quantization than values because softmax amplifies errors in K. Using 4-bit keys / 2-bit values gives 0.995 key cosine similarity at the same storage as uniform 3-bit. Free quality win on the dimension that matters.

The paper is under review at IEEE TAI. Code: https://github.com/ahb-sjsu/turboquant-pro (`pip install turboquant-pro`)

Happy to discuss the methodology or the cosine-vs-recall finding — that's the part I think has the broadest implications beyond our specific use case.

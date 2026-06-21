# Is PolarQuant's polar transform worth adopting? — NO (for our use case)

PolarQuant (Han et al., arXiv:2502.02617) recursively converts randomly-rotated
coordinates to polar (radius + angles); after preconditioning the angles concentrate
to a known analytic distribution, so they quantize tightly and need no per-block
normalization. We implemented the recursive transform (their Def. 1) and compared
reconstruction + retrieval against our rotate + per-dim Lloyd-Max scalar quant, on
the **same** randomly-rotated PCA-256 LaBSE unit vectors (50k corpus, matched bits;
angles quantized with a per-level adaptive codebook — the fair, paper-faithful setup).

| method | bits/coord | recon cosine | recall@10 |
|---|---:|---:|---:|
| scalar-TQ3 (ours) | 3 | **0.9779** | **0.3998** |
| polar (3b angles + 8b radius) | 3 | 0.9703 | 0.3972 |
| scalar-TQ4 (ours) | 4 | **0.9947** | **0.4175** |
| polar (4b angles + 8b radius) | 4 | 0.9922 | 0.4158 |

**Conclusion.** Polar **ties, slightly below** our scalar quant at matched bits. Both
methods exploit the same random-rotation→i.i.d.-Gaussian insight, and both already
avoid per-block normalization (we store one L2 norm per vector, a fixed Gaussian
codebook otherwise). So the polar transform adds a recursive transform + per-level
codebooks for **no reconstruction-quality gain** on embeddings — not worth adopting
for the vector-search path.

**Where polar *does* win (KV cache, not embeddings):**
1. **RoPE-radius-invariance** — under RoPE each 2D key pair is rotated by a
   position-dependent angle, leaving the *radius* invariant; polar coordinates
   isolate that invariant, which Cartesian scalar quant cannot exploit. Relevant only
   to KV *keys*, not embeddings.
2. **Fused decode kernel** — PolarQuant ships Triton kernels that fold dequant into
   attention (its "decoding acceleration"), a systems win addressing the KV-decode
   latency critique. This is orthogonal to quantization quality.

**Recommendation:** do not adopt polar for embedding compression. Consider the polar
*key* representation only if/when we build fused KV-decode kernels (see the decode-
overhead work item); the value there is RoPE-invariance + kernel fusion, not bits.

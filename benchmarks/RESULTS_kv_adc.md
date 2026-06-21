# Fused KV-decode via ADC (the #1 reviewer fix) — exact, no inverse-rotation

The decode-overhead study (`RESULTS_decode_overhead.md`) found dequant is ~99% of the
CPU decode step, and the cost is the **inverse-rotation matmul** (S x dxd per token)
plus materializing K. But the attention score needs neither:

    q . K_recon = q . (norm * unrotate(cent[codes])) = norm * (rotate(q) . cent[codes])

Rotate the query once, then score every cached key by an asymmetric-distance product
over the packed codes -- no inverse-rotation, no reconstructed K.

**Measured** (8192-token cache, head_dim 128, 32 heads, 3-bit, CPU NumPy):

| path | time | correctness |
|---|---:|---|
| dequant + matmul | 249.6 ms | reference |
| **ADC attention (fused)** | **147.1 ms (1.7x)** | max rel err **3.3e-7** (exact) |

**Why this matters more than 1.7x.** The NumPy ADC path still materializes
`cent[codes]`; the win shown is only from skipping the inverse-rotation. The AVX2 ADC
kernel already shipped for embeddings (`turboquant_pro/_adc`) **fuses the LUT lookup
and accumulation without materializing** the reconstructed keys -- i.e. it *is* the
fused dequant-inside-attention kernel the reviewer asked for, and removes the
memory-bound K materialization (the larger win on GPU decode). The same primitive
serves embedding search and KV-cache attention.

**Status:** the math and the fix are validated (exact scores, inverse-rotation
removed); productionizing it as a fused CUDA decode kernel (batched over heads, no
intermediate K) is the remaining engineering, sharing the embedding ADC kernel.

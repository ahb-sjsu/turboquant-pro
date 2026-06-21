# ADC distance during HNSW traversal (#3) — design-space map

Reviewer question: routing packed codes into HNSW, is distance computed by unpacking
to fp32, or by ADC directly on the compressed stream? Measured on PCA-256 LaBSE (50k):

| index | distance in traversal | bytes/vec | qps | recall@10 (single) |
|---|---|---:|---:|---:|
| HNSW-flat | fp32 dot, uncompressed | 1024 | 8190 | 0.419 |
| HNSW-PQ (m=64) | **ADC LUT, C++, no unpack** | 64 | 17615 | 0.338 |
| HNSW-PQ (m=128) | **ADC LUT, C++, no unpack** | 128 | 10246 | 0.410 |
| ADCIndex (ours) | **asym. ADC SIMD, no unpack** | 96 | 6748 | **0.795** |

## Findings
1. **We never unpack to fp32 to compute distances.** Two existing mechanisms:
   `CompressedHNSW` scores graph hops via a centroid-pair inner-product table
   (symmetric ADC, O(dim) lookups); `ADCIndex` uses asymmetric ADC (query fp32,
   per-query LUT) over the packed codes via an AVX2 kernel.
2. **ADC-during-traversal is real and fast** — faiss `IndexHNSWPQ` does it in C++ at
   10k--17k qps with no unpacking, confirming the pattern scales sub-linearly.
3. **But PQ codes in the graph lose recall** (0.34--0.41), while our asymmetric ADC
   over scalar codes reaches **0.795** at the same byte budget -- because it operates
   at the mean-corrected 768-d cosine, which PQ-in-256-d does not capture.

## The open integration (honest gap)
Our codes give the better recall but `ADCIndex` is a *flat* scan (linear in N). The
ideal -- our asymmetric ADC as the **HNSW graph-hop distance** -- needs a C++ HNSW
whose per-hop distance calls our LUT (faiss HNSW-PQ is hardwired to PQ codes). That is
the same kernel-integration family as the flat ADC kernel, and the route to
sub-linear *and* high-recall *and* compressed. Until then: `ADCIndex` flat ADC
(high recall, linear) or `CompressedHNSW` cc-table traversal (sub-linear, symmetric).

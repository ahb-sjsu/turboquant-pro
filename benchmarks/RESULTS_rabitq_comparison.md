# RaBitQ / ScaNN comparison — official libraries, estimator-isolated

Corroborates [`RESULTS_labse_199k.md`](RESULTS_labse_199k.md) (which uses faiss's
RaBitQ) with the **official RaBitQ library** (`rabitqlib`) and **ScaNN**, and
isolates the *estimator* rather than the index: RaBitQ is run exhaustively
(`nprobe = nlist`) and ScaNN AH-only (no reorder), so recall reflects
quantized-scoring quality, not partition pruning.

Atlas, real LaBSE `bip_sample_200k_labse.npy` (768-dim, 198,000 corpus / 2,000
held-out queries), exact cosine ground truth, CPU. Reproduce:
`benchmarks/benchmark_rabitq_comparison.py --npy <labse.npy>`.

## Single-pass recall@10 at matched bytes

| method | bytes/vec | recall@10 (single-pass) |
|---|---:|---:|
| **turboquant-pro** PCA256 · 3-bit | 102 | **0.785** |
| **turboquant-pro** PCA384 · 3-bit | 150 | **0.819** |
| **turboquant-pro** PCA512 · 3-bit | 198 | **0.837** |
| RaBitQ `rabitqlib` 1-bit | 104 | 0.543 |
| RaBitQ `rabitqlib` 2-bit | 200 | 0.752 |
| RaBitQ `rabitqlib` 3-bit | 296 | 0.859 |

At matched bytes, tq-pro leads single-pass by **+24 pts @ ~100 B** and **+8.5 pts
@ ~200 B**; RaBitQ only pulls ahead at 296 B (its 3-bit, ~1.5× tq-pro's bytes).
The tq-pro numbers are verified bit-faithful to the shipped `ADCIndex` (a
standalone re-implementation of the ADC math matched it to ±0.005 recall).

## The single-pass-vs-rerank apples-to-oranges

RaBitQ's headline "90–99%" is a **with-rerank** number. Matched on the rerank
axis, tq-pro's own rerank saturates almost immediately (PCA384 · 3-bit):

| stage | recall@10 |
|---|---:|
| single-pass | 0.8190 |
| + 5× rerank | **0.9997** |
| + 10× rerank | 0.9998 |

So the fair readings are: *single-pass* — tq-pro ahead at matched bytes;
*with-rerank* — both saturate near 1.0 (and tq-pro edges RaBitQ here too; see
`RESULTS_labse_199k.md`, tq-pro 0.9993 vs RaBitQ 0.962). Comparing RaBitQ-reranked
to tq-pro-single-pass is the apples-to-oranges to avoid.

## ScaNN (AH-only) — anisotropic vs reconstruction loss

On LaBSE (998k gutenberg subset), ScaNN's anisotropic (score-aware) loss gives
**no advantage** over plain reconstruction: 0.7727 vs 0.7710 (AH-only), 0.972 vs
0.972 with reorder. The anisotropic advantage is real on other data (GloVe:
+6.4 pts AH-only) but is data-dependent and does not transfer to LaBSE — a caution
against assuming score-aware quantization always helps.

## Why tq-pro wins here

It is a *system*, not one quantizer: PCA-Matryoshka spends bytes on high-SNR
components while RaBitQ quantizes all 768 raw dims; the ADC estimator reconstructs
cosine exactly in the compressed domain; the norm is preserved; and rerank is
one call away. RaBitQ is a strong single quantizer, but on real embeddings the
PCA front-end plus tq-pro's own estimator is already ahead at matched bytes —
before eigenvalue bit-allocation, ANS coding, or AutoConfig are even switched on.

## Honest caveats

- One dataset (LaBSE), one RaBitQ implementation (`rabitqlib`), one operating
  regime. The picture can differ on data where PCA truncation is less effective
  (a flat spectrum) — the [GloVe benchmark](RESULTS_glove.md) is the low-rank-free
  control.
- RaBitQ is run exhaustively to isolate its estimator; its production use adds an
  IVF/graph index and a rerank stage (which, matched, tq-pro also uses).
- Numbers are single-run at the stated seed; treat sub-0.01 differences as noise.

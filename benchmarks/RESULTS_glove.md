# Standard public benchmark: GloVe-100-angular (#10)

To remove the "all results on private LaBSE" weakness, we ran tq-pro against PQ/OPQ
on the canonical **GloVe-100-angular** dataset (ann-benchmarks; 1.18M train, scored
against its provided ground-truth neighbours), 2000 queries, ~16x compression.

| method | bytes/vec | comp x | recall@10 (1-stage) | recall@10 (+rerank) |
|---|---:|---:|---:|---:|
| PQ (m=25) | 25 | 16x | 0.520 | 0.862 |
| OPQ (m=25) | 25 | 16x | 0.523 | 0.866 |
| tq-pro PCA**64**+TQ3 (explains 73% var) | 24 | 17x | 0.380 | **0.685** |
| tq-pro PCA**100**+TQ3 (100% var) | 37 | 11x | 0.735 | 0.989 |
| **tq-pro PCA100+TQ2 (100% var, matched bytes)** | 25 | 16x | 0.574 | **0.906** |

## Honest finding: PCA-Matryoshka truncation is dataset-dependent
With our *default* PCA-256-style truncation (here PCA-64), tq-pro **loses to PQ/OPQ**
(0.685 vs 0.862). The diagnostic shows why: **GloVe-100 has no Matryoshka structure
-- PCA-64 retains only 73% of the variance**, so truncation discards real signal.
This is the opposite of high-dimensional sentence embeddings (LaBSE 768->256 keeps
~99%, where tq-pro reaches 0.999 and beats RaBitQ).

## ...but the core quantizer still wins when configured correctly
With **no truncation** (full 100-d) at the *same byte budget* (PCA100+TQ2 = 25 B),
tq-pro reaches **0.906 +rerank -- beating PQ (0.862) and OPQ (0.866)**. So the
TurboQuant scalar quantizer is competitive/better even on GloVe; the failure was
*truncating data that isn't truncatable*.

## Actionable lesson (-> AutoConfig)
Truncation depth must track **spectral concentration**: pick the PCA rank from the
cumulative explained-variance curve (e.g. >=95%), not a fixed dimension. On GloVe
that means little/no truncation; on LaBSE it means aggressive truncation. This is a
model/data-aware decision AutoConfig should make automatically -- the honest
limitation and its fix in one. **Recommendation:** tq-pro's truncation is for
high-dimensional embeddings with concentrated spectra; for low-dim descriptor sets,
keep full dimension (still beats PQ) or use PQ.

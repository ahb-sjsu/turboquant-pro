# Hubness quotient in ADCIndex: the density dissociation, in retrieval

**Script:** [`hubness_local_scaling.py`](hubness_local_scaling.py) · deterministic, CPU-only, ~1 min.

**Motivation** (companion theory paper, "Radius is density, angle is geometry" + the α=1
dissociation): quotienting a density/hubness coordinate is the right move **iff** that
coordinate is nuisance for the task truth — on sampled manifolds density is sampling noise
(quotient helps); on emergent substrates density is structure (quotient hurts). The retrieval
analogue: a per-vector density proxy (mean/std similarity to a fixed 256-sample of the
*reconstructed* corpus, computable at index build) correcting single-stage candidate scores.

**Protocol:** corpus = power-law-spectrum signal + a common mean offset (embedding-anisotropy
nuisance), L2-normalized; PCA-128 + 2-bit TurboQuant (aggressive operating point) in ADCIndex;
identical top-50 candidate fetch for all single-stage variants; **two ground truths = the two
regimes of the dissociation** (top-10 cosine in *signal* space vs on the *observed* vectors).

## Results (n=5000 corpus, 500 queries, 1024-d)

**Regime 1 — the mean is nuisance** (ground truth: signal-space cosine — the
"all-but-the-top" situation of real embedding anisotropy):

| variant | recall@10 | max occupancy | skewness |
|---|---:|---:|---:|
| baseline (raw ADC) | 0.2204 | 13 | 2.40 |
| **centered (score − μᵢ)** | **0.3770** | 10 | 1.84 |
| zscored ((score−μ)/s) | 0.3598 | **9** | **1.45** |
| +rerank ×5 (exact, observed space) | 0.2456 | 14 | 2.48 |

The quotient gains **+15.7 recall points** single-stage and **beats exact reranking** —
reranking is exact *in observed space*, so it cannot remove the nuisance the quotient targets.
Hub occupancy and skew drop as predicted.

**Regime 2 — the mean is truth** (ground truth: observed-space cosine — hubs are genuine
neighbors):

| variant | recall@10 |
|---|---:|
| baseline (raw ADC) | 0.6556 |
| centered | 0.3032 |
| zscored | 0.3738 |
| +rerank ×5 (exact) | **0.9882** |

The same correction **costs 28 points**, and exact reranking is near-perfect (protocol sanity
check).

## Verdict

The principled fix is also the practical one — **conditionally, and the condition is the
paper's own dissociation**: quotient density only when density is nuisance for the metric
being preserved. Since the two regimes are indistinguishable from the observed vectors alone,
the quotient should ship as an *option with a diagnostic*, not a default: the sign of a
held-out response to the correction (or an external relevance signal) decides the regime —
exactly the paper's "the sign of the α=1 response is itself a diagnostic."

**Caveats:** synthetic single-run corpus; the nuisance here is a rank-1 common mean (real
anisotropy is messier); production integration (an optional `local_scaling` argument on
`ADCIndex.search`) is future work pending a public-data replication.

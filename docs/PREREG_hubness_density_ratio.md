# Pilot preregistration: hubness as a query/corpus density ratio at the k-NN scale

**Status:** exploratory pilot. Predictions were written and committed **before** any measurement. Not a frozen
openvector-bench registration. Author of the predictions: Claude (assistant), on the owner's request,
2026-10-28.

## Hypothesis

The capture-basin identity of `BOND_METRIC.md` is N_k(x_j) ≈ |Q| · μ_Q(A_k(x_j)). To first order the basin
A_k(x_j) is a ball whose radius is the local k-NN radius, r_k ∝ (k / (n f_C))^{1/d}. So the Poisson-deconvolved
attractiveness is

  w_j ∝ (f_Q / f_C)(x_j),

smoothed at bandwidth r_k, where f_Q is the query density and f_C the corpus density. Three consequences:

1. **First order.** With real queries (f_Q ≠ f_C), the density ratio predicts who is a hub better than query mass
   alone.
2. **Scale collapse.** n and k enter only through the bandwidth, so attractiveness skew is a function of n/k
   alone. This is predicted at fixed corpus and query measures, with skew deconvolved for budget.
3. **Second order.** With queries drawn from the corpus distribution (f_Q = f_C) the first-order term
   vanishes. The ratio carries no signal, and residual hubness is smaller and comes from geometry: centrality
   and local density, the Radovanović mechanism.

## Data and protocol

- **Corpus.** Real Cohere Embed-V3 wiki1024 (`/archive/tqp_real/wiki1024/part_000.npy`), cosine on
  L2-normalized rows.
  - A uniform random sample of rows avoids the topical clustering of Wikipedia row order.
  - Corpus pool: 250,000 rows.
  - Held-out query pool QB: disjoint rows, split into QB1 (counts) and QB2 (predictors).
- **Real queries.** QA (`queries.npy`), split in half: QA1 (counts) and QA2 (predictors).
- **Anti-circularity.** Predictors use only the predictor half (QB2/QA2). Counts use only the counting half
  (QB1/QA1).
- **Counts.** Exact top-k by brute force, N_k = reverse-neighbour counts over the counting queries.
- **Statistic.** Hubness = `attractiveness_skew` (Poisson moment deconvolution, as in openvector-bench
  `hubness.py`). Budget ρ = nq·k/n is reported on every cell.
- **Predictors** for corpus point x_j, all from the predictor half, at a global similarity threshold t equal to
  the median similarity of corpus points to their k-th corpus neighbour (the basin scale):
  - qmass_j = number of predictor queries with cos ≥ t;
  - cmass_j = number of other corpus points with cos ≥ t;
  - ratio_j = (qmass_j + ½)/(cmass_j + ½);
  - centrality_j = cos(x_j, corpus mean);
  - density_j = similarity to the k-th corpus neighbour.

  Agreement is Spearman correlation with N_k.

## Registered predictions

**P1 (first order, real queries).** In each QA predictor cell (n = 12,500 with k ∈ {20, 40}):

  Spearman(N_k, ratio) − Spearman(N_k, qmass) ≥ 0.05.

Fails if any cell misses.

**P2 (scale collapse).** Grid: n ∈ {12.5k, 25k, 50k, 100k, 200k} and k ∈ {5, 10, 20, 40, 80}, QB queries,
ρ = 2.5, 3 seeds (seed = corpus subsample + query subsample). For every pair of cells sharing the same n/k, the
attractiveness skews must agree within max(2·SE_pair, 10% of their mean), where SE_pair = √(SE₁² + SE₂²) over
seeds.

P2 holds if at least 80% of same-n/k pairs agree **and** the skew varies across n/k values. Specifically, the
range of the n/k-averaged skews must exceed 3 × the median SE_pair; otherwise collapse is vacuous. As a
discriminating control, report the same agreement rate for pairs that share n but differ in k. Collapse is only
informative if that control rate is clearly lower.

**P3 (second order, same-distribution queries).** In each QB predictor cell (n ∈ {25k, 50k}, k = 10):
- |Spearman(N_k, ratio)| < 0.2;
- max(|Spearman(N_k, centrality)|, |Spearman(N_k, density)|) > |Spearman(N_k, ratio)| + 0.1.

In addition, at the matched cell n = 12,500, k = 40: attractiveness_skew(QA) > attractiveness_skew(QB).

## What would count against the hypothesis

- **P1 failing:** query mass alone is as good as the ratio, and the corpus-density division adds nothing.
- **P2 failing:** n and k act separately, so the bandwidth is not the only channel.
- **P3 failing:** same-distribution queries still carry a ratio signal, or real queries are not hubbier.

Each failure is reported as a failure, with the numbers.

## Limits stated in advance

- One corpus, one embedding model, one metric.
- Only 1,000 real queries, so the QA cells are restricted to small n.
- A global threshold t is a crude bandwidth. A wrong bandwidth biases P1 against the hypothesis, not for it.
- Three seeds.

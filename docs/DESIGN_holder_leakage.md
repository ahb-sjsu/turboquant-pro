# Design: leakage to a holder, strata under one level, codes along a sequence

`turboquant_pro.leakage`, `tqp plan leakage`, `tqp plan strata`,
`tqp plan sequence`. Source: the rate-leakage-distortion series in
geometric-observation:

| paper | folder | status |
|---|---|---|
| I | `paper/tit-rate-leakage` | in review, IEEE T-IT |
| II | `paper/tit-stationary` | submitted to IEEE T-IT 2026-11-18 (IT-26-1961) |
| III | `paper/tit-beyond-stationary` | draft, not submitted |

Statement numbers below are the papers'.

## 0. Thesis

A stored code is read by its consumer, and it is also seen by parties that
never decode it but already hold something correlated with the data:

- an index host with public-model embeddings of the same documents;
- a tenant with the metadata;
- a worker holding part of a context.

What the code adds to what such a **holder** already knows is its leakage.
Every allocation in this package so far minimizes the consumer's distortion
at a rate and is blind to leakage. Paper I gives the whole rate-leakage
frontier at a distortion, with a structure simple enough to compute and to
ship.

## 1. The model and the frontier

The source is `T ~ N(0, Sigma)`, reproduced under the weight `Q` (a read
operator). The holder observes `S = H T + U`, `U ~ N(0, Sigma_U)`. A
description costs its rate `I(T; T_hat)` and its leakage `I(T; T_hat | S)`
per vector. The achievable pairs at distortion `D` are the quadrants above
`(r(Delta), l(Delta))`, over error covariances `Delta` with
`tr(Q Delta) <= D` (Theorem 3). In the whitened coordinate
`X = Sigma^-1/2 Delta Sigma^-1/2` (paper I, Section IV):

    r(X) = -1/2 log2 det X
    l(X) = r(X) - 1/2 log2 det(Ht Htᵀ + Sigma_U) + 1/2 log2 det(Ht X Htᵀ + Sigma_U)
    d(X) = tr(Qt X).

**Theorem 7 (water-filling on a tilted weight).** The point of weight
`alpha` is the unique `X` with `d(X) = D` and

    X = Phi(2 nu Qt + (1 - alpha) Htᵀ (Ht X Htᵀ + Sigma_U)^-1 Ht),

with `Phi` reverse water-filling at level one. At `alpha = 1` this is the
consumer-only design. For `alpha < 1` the holder tilts the weight:
Corollary 8 reads it as the rate-distortion design for a decoder that also
reproduces what the holder sees.

**The solver** is Proposition 11's majorize-minimize iteration. At each step
it freezes the tilt, finds the level that meets `D`, and water-fills. The
weighted cost is nonincreasing and the iterates converge. The level search is
bracketed Illinois regula falsi, because the distortion is continuous and
nonincreasing in the level. It needs a few eigendecompositions per step where
bisection needs dozens. Every returned point records its fixed-point
residual, whether the cost was monotone, and whether it converged.

**Blocks with one level.** The program separates over independent blocks that
share one level, and the majorize-minimize argument is separable over them.
One solver therefore serves three settings:

- paper I, one block;
- paper II, frequencies of a stationary process (Theorem 4; Hermitian
  blocks, Lemma 1);
- paper III, regimes or strata (Proposition 23).

## 2. What the numbers are, and are not

They are the Gaussian-model frontier of test channels. Paper III,
**Theorem 2**, needs only second moments: for a source that is *not*
Gaussian, the Gaussian region with the same second moments is still
achievable. So a frontier computed from a sample covariance *estimates* an
achievable target for real embeddings. **Theorem 3**, that every code is
bounded below by the Gaussian frontier less two relative entropies, needs
more: a holder `S = H T + U` with `U` *independent* of `T`, a source with a
density, and `I(T; S)` finite. A least-squares fit only makes the residual
uncorrelated with `T`, so the lower bound holds only when the holder really
is such a channel.

A practical scalar quantizer in the frontier basis approaches the frontier
but is not certified by it. `described_components` turns a design into
directions and bits (component `v_iᵀ Sigma^-1/2 T` at `-1/2 log2 x_i` bits).
`linear_leakage` measures a real code's leakage as a labelled proxy: exact for
jointly Gaussian variables, a linear holder's gain otherwise. Paper I's
Appendix C is the evidence that the effect survives quantization: at equal
rate and budget, a Lloyd-Max quantizer on the leakage-aware projection leaks
13 to 72% less than on the principal axis, on one two-dimensional instance.

## 3. The commands

**`tqp plan leakage`.** Fits the holder from row-paired samples by least
squares: the linear model with the same second moments, which is the Gaussian
problem Theorem 2 needs. It then reports:

- the frontier from `alpha = 1` to `alpha = 0`;
- the consumer-only design's **excess leakage**;
- the least-leaking design's **rate premium**;
- the Euclidean angle, at this one budget, between the two designs' leading
  directions (the top eigenvectors of `Sigma - Delta`). Theorem 22 says the
  two designs differ when the rate-optimal one is one-dimensional along
  `b_R` and `J Sigma b_R` is not parallel to `b_R`. The angle is a summary
  of that difference, not the paper's angle measure, and not Theorem 20's
  turning across budgets;
- a verdict read against a noise floor: the same fit with the holder's rows shuffled, which knows nothing, so whatever it shows is estimation noise. `TRADEOFF` needs both the excess and the premium above `--noise-factor` (default 10) times the floor, otherwise `WITHIN_NOISE`. (The library's `tradeoff_report` without a floor keeps the exact-model rule, `NO_TRADEOFF` below 1e-6 bits.) The first version used that exact rule on estimated holders and called a shuffled holder a trade-off.

`--pca K` is rigorous rather than approximate. The top `K` principal
components are uncorrelated with the rest. The `K`-dimensional problem with
the holder fitted to `(T_K, S)` absorbs the rest into the holder's noise, so
it gives such designs' exact rate and leakage, because given `T_K` the
description is independent of `(T_rest, S)`. The undescribed rest adds its own
distortion. So the result is an achievable point of the full problem, not the
full frontier.

**`tqp plan strata`.** Paper III, **Corollary 6**: for a common weight and
holder,

    V(alpha, D; sum_c lambda_c K_c) >= min { sum_c lambda_c V(alpha, D_c; K_c) : sum_c lambda_c D_c <= D },

and the right side is the block program with one level. So per-stratum codes
under a shared water level never cost more than one pooled code. The command
reports:

- the saving, and the rate each stratum gets under the shared level (not
  equal budgets per stratum);
- the label entropy, which is the most a decoder that must be told the labels
  pays; a shard or a partition carries its label for free;
- the between-means term, which a pooled code that does not subtract stratum
  means also faces. It is kept separate because the corollary is about
  centred strata.

This is the model-based counterpart of `strata_ops.allocate_by_fragility`,
which spends on measured fragility.

**`tqp plan sequence`.** A memoryless code sees only the lag-0 covariance, the
mean of the spectrum. A memoryless channel's block information is at most the
sum of its per-letter informations, so coding along the sequence never costs
more. The command estimates the spectral density (Welch, Hann, half overlap)
and compares the two rates at one distortion (paper II, Theorem 4 at
`alpha = 1`, which is Kolmogorov's formula for rate alone). It reports where
the bits sit in frequency.

This is the target issue #22 (differential KV compression) aims at. A delta
coder is a fixed first-order whitening filter; the spectral allocation is the
optimum it approximates, and the gain here is what any sequence codec can win
at most, in the Gaussian model. The figure is for rate only. The leakage
counterpart with a holder that has memory is paper II's whole
`spectral_blocks(..., holder_sequence=...)` model, which fits the holder's
filter frequency by frequency. It is exposed for the sequence code but not
compared, because the memoryless side would need its own holder model.

## 4. Tests

`tests/test_leakage.py`. Paper I's printed numbers (Section X):

- the scalar instance's excesses at `(rho², tau², D) = (0.75, 0.5, 0.3)`:
  0.0349 bits of leakage and 0.0400 bits of rate;
- the least leakages of two other scalar sources, `(rho², tau²) = (3/5, 1/5)`
  and `(3/4, 1/2)`, at `D = 0.1`: 1.1880 and 1.2105 bits;
- the coupled instance's leakage-optimal directions: 47.48, 46.76, 45.07 and
  41.15 degrees at `D = 1.38, 1.3, 1.2, 1.1`, with the rate-optimal one on the
  first axis;
- the independent-components split: `R = L = 0.1610` on component 1 against
  `L = 0.0197`, `R = 0.3272` on component 2. This checks the shared level
  across blocks, and blocks are compared against the block-diagonal problem.

Then:

- an optimality check that only evaluates costs, by feasible perturbations,
  with an **I0 self-test**: the consumer-only design is caught as improvable
  for leakage;
- `alpha = 1` against classical reverse water-filling;
- a blind holder gives `NO_TRADEOFF`;
- invariance under rotation and holder scaling;
- the costs against a Monte Carlo of the Gaussian test channel;
- the holder fit;
- pooling never cheaper, and identical strata gain nothing;
- an AR(1) sequence against Kolmogorov's closed form, and white noise gains
  nothing;
- the per-frequency holder fit recovers a known filter;
- the CLI.

## 5. What a claim would need

Every number here is a model quantity. A product claim is a measurement,
registered before it is run. For example: on real embeddings with a
public-model holder, a scalar quantizer in the leakage-aware basis has a
lower measured linear leakage than the consumer-only allocation at matched
bytes and matched recall@k. The consumer's metric is the acceptance gate, as
everywhere in this package, and the leakage proxy is a second, labelled
measurement. The same holds for a sequence codec on KV caches (#22): it would
be measured on perplexity in the `kvquant_matrix` harness, with
`tqp plan sequence` as its stated target.

## 6. Not yet

- An observer-contract clause naming holders (`holders:`). It would change the
  contract schema, so it waits for a decision.
- Decoder side information (paper I, Theorem 31). The region is known in the
  degraded cases, when one party's information dominates the other's: a
  decoder that knows more than the holder leaves no trade-off, and one that
  knows less reduces exactly to the problem without decoder side information.
  The general case is only bracketed (Proposition 32).
- An integer-width allocation in the frontier basis from the Lloyd-Max table,
  so that `plan leakage` can emit a buildable code next to the test channel.

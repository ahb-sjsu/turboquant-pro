# Results — consumer-read bases for retrieval truncation

Registered in `docs/PREREG_consumer_basis.md` before any arm ran; the verdict rules, arms, dimensions
and endpoint are that file's, not this one's. Scored by `benchmarks/consumer_basis/score.py` from the
per-arm records on the campaign volume, 2026-09-16.

Data: `sentence-transformers/msmarco-*` style embedding sets staged at revision
`f018922a46a51388348a307ec2fef28019a33026`, 1M corpus rows per large arm, payload hashes in each
arm's `hashes.json`. Bases: **P** corpus PCA, **Q** query PCA, **S** the symmetrised operator, **O**
the read-operator construction (a query map and a document map). Endpoint: single-pass recall@10 at
the kept dimension, paired percentile bootstrap over the arm's evaluation queries (10,000 resamples,
seed 0), BETTER/WORSE at ±0.01.

## Registered verdicts

| hypothesis | verdict | cells |
|---|---|---|
| **H1** O beats P on asymmetric arms | **HOLDS** | BETTER in 6 of 6 |
| **H2** O ties P on symmetric arms | **HOLDS** | TIE in 3, BETTER in 3, WORSE in none |
| **H3** S beats P on asymmetric arms | **REFUTED** | BETTER in 3, TIE in 1, WORSE in 2 |

H1, paired difference in recall@10 (O − P), 95% interval:

| arm | k=64 | k=128 | k=256 |
|---|---|---|---|
| msmarco | +0.0147 [+0.0121, +0.0172] | +0.0125 [+0.0104, +0.0146] | +0.0132 [+0.0114, +0.0150] |
| hotpotqa | +0.0455 [+0.0426, +0.0483] | +0.0302 [+0.0277, +0.0328] | +0.0256 [+0.0233, +0.0278] |

H3 fails where it matters most: S is BETTER at k=64 on both arms (+0.019, +0.057) and WORSE at
k=256 on both (−0.030, −0.036). A basis that wins at the smallest budget and loses at the largest is
not the free lunch the hypothesis proposed.

## What the registration licenses

The outcome table in section 5 anticipated "H1 and H3 HOLD, H2 HOLDS" and two ways of failing H1. It
did not anticipate this combination, so the reading is stated rather than claimed: H1 and H2 together
license an **opt-in consumer basis built as O, with its two maps**, and H3's refutation removes S,
the cheaper single-map variant, from that offer. Nothing here licenses changing the default, which
stays corpus PCA.

## Everything the registration left unscored

Reported, not scored: the Q basis, k = 32 and 512, the secondary arms, and the rerank endpoint.

- **Query PCA is most of the gain.** On msmarco at k=256, P 0.764, Q 0.776, O 0.777; on hotpotqa,
  P 0.671, Q 0.693, O 0.697. O's margin over plain query PCA is a few thousandths, well inside what
  this design can separate. The registered contrast was O against P, and that is what holds; the
  practical question of whether the read-operator construction earns its second map over simply
  fitting PCA to a query sample is **not answered here**, and on these numbers it looks close.
- **On one secondary arm P wins.** nq at k=256: P 0.807, O 0.802, Q 0.779. nq was registered as
  secondary and is unscored, but it is the one arm where the corpus basis is ahead at a large budget.
- **The gain tracks the query/document mismatch.** Mismatch index 0.518 (hotpotqa) gives the largest
  gains, 0.440 (msmarco) smaller ones, 0.0048 (msmarco-sym) a tie. hotpotqa-sym, built as a symmetric
  control, still carries mismatch 0.095 and still shows O ahead, which is why H2 counts three BETTER
  cells: the control is less symmetric than intended, and that weakens it as a control.
- **Fit size barely matters.** O@2000 and O@10000 on msmarco differ from full-sample O by ≤0.004 at
  every k, so a few thousand queries are enough to fit the basis.

## All arms, recall@10 (single-pass / after reranking the top 100)

| arm | kind | mismatch | basis | k=32 | k=64 | k=128 | k=256 | k=512 |
|---|---|---:|---|---|---|---|---|---|
| fiqa | asymmetric-secondary | 0.3109 | P | 0.250 / 0.642 | 0.457 / 0.867 | 0.653 / 0.963 | 0.827 / 0.995 | 0.956 / 1.000 |
| fiqa | asymmetric-secondary | 0.3109 | Q | 0.303 / 0.715 | 0.502 / 0.881 | 0.683 / 0.960 | 0.834 / 0.992 | 0.955 / 1.000 |
| fiqa | asymmetric-secondary | 0.3109 | S | 0.291 / 0.703 | 0.490 / 0.880 | 0.660 / 0.957 | 0.788 / 0.993 | 0.859 / 1.000 |
| fiqa | asymmetric-secondary | 0.3109 | O | 0.304 / 0.719 | 0.505 / 0.890 | 0.690 / 0.966 | 0.842 / 0.995 | 0.965 / 1.000 |
| hotpotqa | asymmetric | 0.5178 | P | 0.049 / 0.182 | 0.178 / 0.455 | 0.408 / 0.779 | 0.671 / 0.968 | 0.922 / 1.000 |
| hotpotqa | asymmetric | 0.5178 | Q | 0.093 / 0.301 | 0.222 / 0.540 | 0.435 / 0.806 | 0.693 / 0.969 | 0.932 / 1.000 |
| hotpotqa | asymmetric | 0.5178 | S | 0.095 / 0.308 | 0.234 / 0.564 | 0.429 / 0.810 | 0.635 / 0.963 | 0.748 / 0.997 |
| hotpotqa | asymmetric | 0.5178 | O | 0.086 / 0.283 | 0.223 / 0.541 | 0.439 / 0.809 | 0.697 / 0.971 | 0.934 / 1.000 |
| hotpotqa-sym | symmetric | 0.0952 | P | 0.067 / 0.240 | 0.251 / 0.596 | 0.499 / 0.881 | 0.740 / 0.987 | 0.942 / 1.000 |
| hotpotqa-sym | symmetric | 0.0952 | Q | 0.100 / 0.338 | 0.287 / 0.662 | 0.524 / 0.901 | 0.755 / 0.988 | 0.946 / 1.000 |
| hotpotqa-sym | symmetric | 0.0952 | S | 0.092 / 0.310 | 0.280 / 0.645 | 0.507 / 0.890 | 0.736 / 0.987 | 0.905 / 1.000 |
| hotpotqa-sym | symmetric | 0.0952 | O | 0.098 / 0.327 | 0.290 / 0.662 | 0.521 / 0.897 | 0.754 / 0.988 | 0.946 / 1.000 |
| msmarco | asymmetric | 0.4395 | P | 0.087 / 0.295 | 0.302 / 0.661 | 0.564 / 0.900 | 0.764 / 0.982 | 0.945 / 1.000 |
| msmarco | asymmetric | 0.4395 | Q | 0.091 / 0.306 | 0.313 / 0.678 | 0.573 / 0.906 | 0.776 / 0.988 | 0.950 / 1.000 |
| msmarco | asymmetric | 0.4395 | S | 0.104 / 0.331 | 0.321 / 0.677 | 0.557 / 0.893 | 0.734 / 0.979 | 0.852 / 1.000 |
| msmarco | asymmetric | 0.4395 | O | 0.097 / 0.318 | 0.317 / 0.682 | 0.576 / 0.907 | 0.777 / 0.986 | 0.953 / 1.000 |
| msmarco | asymmetric | 0.4395 | O@10000 | 0.096 / 0.318 | 0.314 / 0.679 | 0.575 / 0.907 | 0.775 / 0.985 | 0.952 / 1.000 |
| msmarco | asymmetric | 0.4395 | O@2000 | 0.091 / 0.308 | 0.306 / 0.669 | 0.566 / 0.904 | 0.766 / 0.985 | 0.946 / 1.000 |
| msmarco-sym | symmetric | 0.0048 | P | 0.111 / 0.361 | 0.348 / 0.745 | 0.607 / 0.950 | 0.796 / 0.996 | 0.955 / 1.000 |
| msmarco-sym | symmetric | 0.0048 | Q | 0.113 / 0.362 | 0.344 / 0.736 | 0.605 / 0.947 | 0.792 / 0.996 | 0.954 / 1.000 |
| msmarco-sym | symmetric | 0.0048 | S | 0.113 / 0.363 | 0.348 / 0.742 | 0.607 / 0.949 | 0.796 / 0.996 | 0.953 / 1.000 |
| msmarco-sym | symmetric | 0.0048 | O | 0.113 / 0.362 | 0.347 / 0.742 | 0.607 / 0.949 | 0.795 / 0.997 | 0.954 / 1.000 |
| nq | asymmetric-secondary | 0.5074 | P | 0.092 / 0.309 | 0.319 / 0.688 | 0.611 / 0.918 | 0.807 / 0.990 | 0.956 / 1.000 |
| nq | asymmetric-secondary | 0.5074 | Q | 0.087 / 0.282 | 0.280 / 0.629 | 0.562 / 0.894 | 0.779 / 0.984 | 0.940 / 1.000 |
| nq | asymmetric-secondary | 0.5074 | S | 0.109 / 0.335 | 0.332 / 0.693 | 0.604 / 0.912 | 0.776 / 0.986 | 0.872 / 1.000 |
| nq | asymmetric-secondary | 0.5074 | O | 0.096 / 0.312 | 0.311 / 0.672 | 0.597 / 0.910 | 0.802 / 0.987 | 0.956 / 1.000 |
| quora | symmetric-secondary | 0.0140 | P | 0.209 / 0.525 | 0.467 / 0.818 | 0.687 / 0.959 | 0.846 / 0.993 | 0.967 / 1.000 |
| quora | symmetric-secondary | 0.0140 | Q | 0.195 / 0.503 | 0.447 / 0.800 | 0.672 / 0.952 | 0.833 / 0.992 | 0.960 / 1.000 |
| quora | symmetric-secondary | 0.0140 | S | 0.206 / 0.520 | 0.460 / 0.812 | 0.681 / 0.956 | 0.841 / 0.993 | 0.959 / 1.000 |
| quora | symmetric-secondary | 0.0140 | O | 0.201 / 0.512 | 0.456 / 0.809 | 0.680 / 0.956 | 0.841 / 0.992 | 0.965 / 1.000 |

## Limits

One embedding family and one revision; 1M corpus rows per arm, not the full sets; exact search, so
nothing here measures interaction with quantization or with an index; queries come from each set's
own query split, which is the distribution the basis is fitted to, so this is the favourable case for
any consumer-aware method. Reranking the top 100 closes most of the gap between all four bases at
k ≥ 256, which bounds how much the choice of basis can matter in a pipeline that reranks.

## Prior art

As registered in section 6: consumer-aware and score-aware compression is not new (ScaNN's
anisotropic quantization, ADSampling and DADE/DDCpca for early termination, query-aware subspace
methods such as TaCo). The narrow question here was whether choosing the kept subspace from the query
distribution changes exact-search recall at fixed dimension, with a symmetric control. It does, on
the registered arms. No novelty is claimed, and the Q comparison above is the reason to be careful
about claiming one.

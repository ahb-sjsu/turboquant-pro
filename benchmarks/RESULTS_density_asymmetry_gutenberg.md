# Density sets the direction of retrieval flow between language strata: Gutenberg result

**Registered before computation:** [`docs/PREREG_density_asymmetry_gutenberg.md`](../docs/PREREG_density_asymmetry_gutenberg.md),
frozen in commit `427f3c2`. It has no amendments.
**Script:** [`strata_density_gutenberg.py`](strata_density_gutenberg.py), sha256 `eceef429…`.
**Raw output:** `results/gut_gate.{json,log}` and `results/gut_score.{json,log}`.
**Setup:** Gutenberg pair corpus on Atlas, 150,067 passages from 3,861 books in 20 languages, 13 of them
eligible. Stored exact 10-NN lists from G1 BGE-M3 and G2 LaBSE. Corpus rows serve as queries. Per-row
density files are `sdens_gut_G{1,2}[_xbook].npy` in `/archive/experiments/hubness-density-ratio/`. Run
2026-10-28 on Atlas cores 12–19.

## Verdicts

Both registered predictions hold.

| test | verdict | G1 BGE-M3 | G2 LaBSE | family p (larger arm), Holm |
|---|---|---|---|---|
| **Gate.** Density test on row-shuffled densities, fire if > 17 of 200 rejections | quiet | 2 | 5 | — |
| **P1.** Net flow runs into denser language strata | **HOLDS** | Λ_ρ = +0.86, p = 0.0062 | Λ_ρ = +0.98, p = 0.00005 | 0.0062 → 0.0100 |
| **P2.** That alignment exceeds random partitions of the same sizes | **HOLDS** | above all 200 (mean 0.36, sd 0.25) | above all 200 (mean 0.36, sd 0.25) | 0.0050 → 0.0100 |

P1 uses 20,000 sampled orderings of the 13 strata. P2's p-value of 1/201 is the floor of its reference
distribution, so it says only that no random partition reached the observed value.

## What this shows

**Between language strata in this corpus, net retrieval flow runs into the denser stratum, on both
encoders.** It does so more strongly than random groupings of rows of the same sizes would give. So
density acts as a property of the language areas here, not only as a row-level effect that any
partition would inherit.

**The density differences are small, and the flow still follows them.** Per-stratum densities span
0.609–0.644 on G1 and 0.621–0.665 on G2. The two encoders rank the languages differently. Spanish and
Italian are among the sparsest on G1 and among the densest on G2, for example. Each encoder's flow
follows its own density ordering. That is what a geometric mechanism predicts, and what a property of
the texts alone would not.

**How much of the asymmetry density accounts for.** In the weighted fit of the net-flow share on the
density difference, `R²` is 0.56 on G1 and 0.97 on G2.

## Robustness readouts (descriptive)

| readout | G1 | G2 |
|---|---|---|
| Λ_ρ, border-row density | 0.84 (p 0.008) | 0.89 (p 0.0004) |
| Λ_ρ, density excluding same-book neighbours | 0.84 (p 0.009) | 0.98 (p 0.00005) |
| Λ_n, stratum size in place of density | 0.76 (p 0.022) | 0.12 (p 0.39) |
| Spearman(ρ, size) | 0.04 | 0.12 |

- **Same-book clustering does not drive the effect.** Same-book neighbour shares vary a lot by stratum,
  from 0.17 to 0.46 on G1. Latin, Esperanto, Greek and Portuguese have the highest. Recomputing density
  from neighbours outside the row's own book leaves Λ_ρ almost unchanged. 5,772 rows on G1 and 1,705 on
  G2 had no such neighbour and were dropped from that readout.
- **Size is a partial rival on G1 only.** On G1 flow also tends toward larger strata, but density and
  size are nearly uncorrelated (0.04), so these are two different orderings, and density aligns more
  strongly. On G2 size carries nothing.
- **Disclosed transit mass does not track density.** Dutch and French carry the most transit on G1 and
  sit among the sparser strata.

## The date map (descriptive only)

Rows binned by the first author's birth year (50-year bins, 7 eligible bins from −450 to 1900).

| | G1 | G2 |
|---|---|---|
| Λ_chron, later-born → earlier-born | 0.99 (p 0.001) | 0.97 (p 0.005) |
| Λ_ρ | 0.49 (p 0.12) | 0.95 (p 0.012) |

On this map flow runs from later-born authors' strata to earlier-born ones on both encoders. That is the
chronology prereg's direction. It was registered as descriptive only, with no correction. The prereg
names two reasons it is weak evidence:
- Gutenberg is not a derivational corpus.
- 2 of the 7 bins, 1800 and 1850, hold 81% of the dated rows. Bins also differ in language mix, and the
  language result above shows that language strata differ in density.

It is a lead for the chronology hypothesis, not support.

## Relation to the Ethics result

The Ethics test ([`RESULTS_density_asymmetry.md`](RESULTS_density_asymmetry.md)) failed every registered
prediction, with density pointing the predicted way on every arm. This result does not change that
verdict. It shows the effect in a second corpus, with almost twice as many strata (13 against 7) and
six times as many pairs (78 against 21). On Ethics, density did not stand out from other orderings of
seven strata. Here it stands out from other orderings and from random partitions on both encoders.

## Limits

- One corpus, two encoders, corpus→corpus queries only. Real queries were shown by the density-ratio
  pilot to behave differently.
- English is 40% of rows, so its pairs carry much of the flow mass.
- P2's reference has 200 partitions, so its p-value cannot go below 1/201.
- The analyst knew each language's transit mass before registration. It does not order the strata the
  way density does (see above).

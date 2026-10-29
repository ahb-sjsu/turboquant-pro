# Density versus chronology as the direction of STRATA retrieval flow: result

**Registered before computation:** [`docs/PREREG_density_asymmetry.md`](../docs/PREREG_density_asymmetry.md),
frozen in commit `cb057fa`. It includes a joint amendment to the chronology prereg
[`docs/PREREG_directed_asymmetry.md`](../docs/PREREG_directed_asymmetry.md) (blob `0b4de32`).
Amendment 1 (commit `c3e3282`) replaced the density-test gate after the first gate fired.
**Scripts:** [`strata_density_asym.py`](strata_density_asym.py), which uses the chronology functions of
[`strata_dirasym.py`](strata_dirasym.py) unchanged.
**Raw output:** `results/dasym_gate.{json,log}` (first gate), `results/dasym_gate_a1.{json,log}`
(Amendment-1 gate), `results/dasym_score.{json,log}`.
**Setup:** Ethics corpus on Atlas, 350,000 rows, stored exact 10-NN lists from two encoders, D1 BGE-M3
and D2 LaBSE. Corpus rows serve as queries. Per-row density files `sdens_D1.npy` (sha256 dbaeaab1…) and
`sdens_D2.npy` (1f8aacb2…) are in `/archive/experiments/hubness-density-ratio/`. Run 2026-10-28 on
Atlas cores 12–19.

## Verdicts

No registered prediction passed. The chronology tests were not scored.

| test | verdict | numbers |
|---|---|---|
| **T1, T2** (chronology prereg) | **VOID** | its own control T3 fired on D1, see below |
| **R1.** Net flow runs into denser language strata (BGE-M3) | **FAILS** | Λ_ρ = +0.66, exact p = 0.33, Holm 0.35 |
| **R2.** Same (LaBSE) | **FAILS** | Λ_ρ = +0.91, exact p = 0.18, Holm 0.35 |
| **J-chron.** Later → earlier flow with density held fixed, both arms | **FAILS** | β = +0.052 (p 0.028) and +0.033 (p 0.25); registered p 0.25 |
| **J-ρ.** Flow into denser strata with chronology held fixed, both arms | **FAILS** | β = +0.084 (p 0.022) and +0.093 (p 0.083); registered p 0.083, Holm 0.17 |

## The gates

**G1, the chronology prereg's T3, fired on D1.** Under the alphabetical ordering of the seven language
strata, Λ was 0.933 with two-sided p = 0.031. On D2 it was 0.559, p = 0.364. By that prereg's rule, T1
and T2 are void. The amendment had recorded in advance that T3 cannot detect a partition artifact. It
fires about 5% of the time whenever the flow follows a single strong ordering, and that appears to be
what happened. The language flow on D1 is very close to a total order, and alphabetical order happens
to lie near it. The rule was applied anyway.

**The first G2 fired, and the fault was in the gate.** Permuting area codes over rows should have made
every new test reject about 10 times in 200.

| arm | R (Λ_ρ) | J β_ρ | J β_chron |
|---|---|---|---|
| D1, first gate | 37 | 46 | 4 |
| D2, first gate | 27 | 53 | 10 |
| D1, Amendment-1 gate | 15 | 8 | 11 |
| D2, Amendment-1 gate | 7 | 12 | 7 |

The fire limit is 17. Permuting labels keeps each row's density and in-degree. A random stratum that
collects denser rows also collects more hubs, so the density tests found real flow into it. The density
mechanism therefore works at the level of single rows, inside random partitions. That is a finding in
itself. It means the first gate was the wrong null, not that the instrument was faulty. Amendment 1
replaced it with a null that permutes the densities over rows and keeps the true flow, and the new gate
stayed quiet on both arms.

## What the scores show

**Density points the predicted way everywhere, and no registered test can separate it from other
orderings.** All four registered density statistics are positive. On the language map the seven strata
give 5,040 orderings, and the flow is so close to a single total order that many orderings fit it
nearly as well as density does. So the exact test cannot single density out. In the joint model, D1
supports both terms at p < 0.03 and D2 supports neither at 0.05. The registered rule needs both arms.

**The language map adds nothing beyond row-level density on D1.** Over 200 random partitions with the
true stratum sizes, Λ_ρ averaged 0.39 on D1 and 0.29 on D2. The observed value was matched or exceeded
by 30.5% of partitions on D1 and 4.5% on D2. So on D1 the true language strata carry no more
density-aligned flow than any random grouping of rows of the same sizes.

**The two encoders disagree on how much density explains.** In the weighted fit of the net-flow share on
the density difference, `R²` was 0.03 on D1, with a negative slope, and 0.61 on D2.

## Descriptive results (outside every family, uncorrected)

These were registered as descriptive. None of them can stand in for a confirmatory p-value.

- **Density alignment on `tradition × century_bin` is strong on both arms.** Λ_ρ = 0.96 (p = 0.003) on
  D1 and 0.92 (p = 0.012) on D2, each from 5,000 sampled orderings. This is the clearest pro-density
  signal in the run. It was not a registered test, so it is a lead for a future registration, not a
  result. Spearman(ρ, century bin) is −0.25 on D1 and −0.27 on D2, so density and chronology are only
  weakly confounded on this map.
- **On D1, language flow runs from large strata into small ones.** Λ_n = −0.95. Only about 5 of the
  5,040 orderings give a value that low. Pali and Sanskrit are the two smallest strata and the two
  densest (ρ = 0.79 and 0.81 against 0.68–0.72 for the others). On D2, Λ_n = +0.69, so this does not
  replicate across encoders.
- **Chronology on raw counts:** Λ_chron = 0.19 on D1 and 0.20 on D2, both in H-D's direction. It has no
  p-value, and the frozen T1 that would have tested it is void.
- **Border-row density.** On D2, Λ with border-row density was 0.88 (p = 0.20). On D1 it is undefined,
  because no Pali row has an out-edge into another eligible stratum. The script reports p = 0.0 there.
  That number is an artifact of comparing with an undefined value and should be ignored.

Per-stratum density (median cosine to the 10th stored neighbour):

| stratum | n | ρ D1 | ρ D2 |
|---|---|---|---|
| aramaic | 123,820 | 0.720 | 0.789 |
| english | 44,929 | 0.692 | 0.744 |
| greek | 32,289 | 0.685 | 0.762 |
| hebrew | 134,524 | 0.719 | 0.798 |
| latin | 7,038 | 0.683 | 0.755 |
| pali | 2,132 | 0.791 | 0.858 |
| sanskrit | 4,780 | 0.815 | 0.873 |

## What this leaves

The directed asymmetry is still real and still unexplained. The registered tests cannot tell whether it
follows density, size or chronology at seven or ten strata. The run did establish two things. Density
already drives flow at the level of single rows, which the first gate showed by firing. And on the
century map, density alignment is strong on both encoders but was not registered. A follow-up worth
registering would test Λ_ρ on `tradition × century_bin` directly, with chronology scored jointly under
a control that can detect a partition artifact. T3 cannot.

## Limits

- Seven and ten strata. With few strata and a flow close to a total order, an ordering test has little
  power to single out one ordering.
- Corpus→corpus only. Ethics has no real query set.
- One run per arm. The kNN lists are fixed, so the only randomness is in the permutations.
- One deviation from the frozen plan: Amendment 1, made after the first gate fired and before any
  scoring. It is documented in the prereg with the gate counts that prompted it.

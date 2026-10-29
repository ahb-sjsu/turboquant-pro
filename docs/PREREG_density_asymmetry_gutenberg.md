# Pre-registration — does density set the direction of retrieval flow between language strata? Gutenberg replication

**Status: REGISTERED ⚪, FROZEN on filing 2026-10-28** at the author's instruction. Nothing in §2–§5
has been computed. Harness: [`benchmarks/strata_density_gutenberg.py`](../benchmarks/strata_density_gutenberg.py),
committed with this document, smoke-tested on synthetic data and planted flows only.

**Why this exists.** The Ethics test ([`PREREG_density_asymmetry.md`](PREREG_density_asymmetry.md),
result in [`benchmarks/RESULTS_density_asymmetry.md`](../benchmarks/RESULTS_density_asymmetry.md)) failed
every registered prediction. Two things from it motivate a new test on new data:
- **Density pointed the predicted way on every arm.** With 7 strata the ordering test could not single
  density out.
- **The first gate showed the mechanism at row level.** On random partitions of the rows, flow ran into
  whichever stratum happened to collect denser rows.

The Ethics graphs are spent for this question. This document moves the test to a corpus whose pairwise
flow and per-stratum density have never been computed, with almost twice as many strata (13 against 7)
and six times as many pairs (78 against 21).

## 0. What is known, and what is blind

**Known to the analyst before registration:**
- The full Ethics result, including its descriptive signals.
- From the July STRATA runs on this corpus, which are committed on Atlas:
  - `p2_score.json`, including per-language **transit mass**, which the analyst has seen. For BGE-M3
    the largest are Dutch 20,097, French 16,283, German 10,283, Hungarian 8,136, Italian 8,090,
    Finnish 7,091, Spanish 6,510, English 6,056.
  - `strata_report_anatomy_*.json`, giving per-language hub skew and class. The analyst has seen
    only the first entries (Dutch).
- The stratum sizes (Appendix).

Transit mass counts cross-stratum traffic through a language. It does not give the direction of flow
between pairs, and it is not density. It is disclosed because a busy transit stratum could plausibly
be a dense one.

**Never computed:** per-row or per-stratum density, and the pairwise counts `C_ab`. Every prediction
below is about those.

## 1. Materials (committed artifacts, no new kNN)

On Atlas `/archive/tqp_strata/gutenberg_pair/` (built 2026-07-23 by `build_gutenberg_pair.py`; text
fingerprint `f3e19747…`):

| arm | vectors | neighbour ids |
|---|---|---|
| G1 BGE-M3 | `gut_bge_m3.npy` (150,067 × 1024, normalized) | `exact_ids_bge.npy` |
| G2 LaBSE | `gut_labse.npy` (150,067 × 768, normalized) | `exact_ids_labse.npy` |

`labels.txt` gives the language per row. `passages.jsonl` gives the source book per row. Sha256 of every
input is recorded in the gate output. `k = 10`, `n_min = 2000`. Thirteen strata are eligible. Edges into
or out of ineligible strata are dropped.

**Date map (descriptive only).** Each book's first author with life dates in
`/archive/gutenberg/pg_catalog.csv` gives a birth year. That is the first `year[?][ BCE]-` match in the
`Authors` field, with BCE years negative. Rows are binned by birth year in 50-year bins.
Bins with at least 2,000 rows are eligible, and books without dates are dropped. 3,286 of 3,861 books
have dates.

## 2. Statistics (defined before computation)

Unchanged from the Ethics prereg except where marked.

- **Per-row density.** `s(x)` = cosine of row `x` to its last stored neighbour (column 9).
- **Per-stratum density.** `ρ_a` = median `s(x)` over the stratum.
- **Flow.** Raw directed edge counts `C_ab`.
- **Λ_ρ** = Σ_{a<b} (C_ab − C_ba)·sgn(ρ_b − ρ_a) / Σ|C_ab − C_ba|.
- **Ordering null (changed).** 13! orderings cannot be enumerated, so the null uses 20,000 orderings
  drawn with seed 20261101. One-sided p = (1 + #{Λ* ≥ Λ}) / 20,001.
- **Row-level reference (new as a test).** Λ_ρ on 200 random partitions of the rows with the true
  stratum sizes, seed 20261102. This gives the alignment that row-level density alone produces. The
  test's p-value is (1 + #{Λ_rand ≥ Λ_obs}) / 201.

## 3. Registered predictions

One family of two tests, Holm–Bonferroni at family-wise α = 0.05. Each test must hold on **both** arms,
and its family p-value is the larger of the two arms' p-values.

- **P1 — density sets direction.** `Λ_ρ > 0` on both arms, ordering-null p < 0.05 after Holm.
- **P2 — area density matters beyond row-level density.** `Λ_ρ` on the true strata exceeds the
  row-level reference on both arms, p < 0.05 after Holm.

**How the outcomes read.**
- P1 and P2 hold: the true language strata carry more density-aligned flow than random groupings of
  rows. Density is a property of the areas, not only of the rows.
- P1 holds and P2 fails: direction follows density, but no more than row-level density gives. The
  strata are passive containers for a row-level effect.
- P1 fails with `Λ_ρ > 0`: the Ethics pattern again, direction leaning toward density but not singled
  out. Two corpora agreeing in sign is still not a result.
- P1 fails with `Λ_ρ` significantly below 0 (two-sided, reported): flow runs into sparser strata. That
  inverts the mechanism and is the headline.

## 4. Gate (computed first, separate invocation)

The gate is the Ethics Amendment-1 calibration, with nothing borrowed from T3.
- `s(x)` is permuted over rows 200 times per arm, with the true labels and flow kept. Seed 20261103.
- In each replicate P1's test is recomputed with 2,000 orderings. The gate's ordering count is smaller
  than the 20,000 used for scoring, for cost.
- Under this null P1 should reject about 10 times in 200. The gate fires if it rejects more than 17
  times on either arm.
- If it fires, scoring is blocked.

P2 has no separate gate. Its reference distribution is itself a null built from the same data.

The gate writes rejection counts only. The scoring script refuses to run without a quiet gate file.

## 5. Descriptive readouts (outside the family, no confirmatory use)

- `Λ_n`, with stratum size in place of density, and Spearman(`ρ_a`, `n_a`).
- `R²` of net-flow share on the density difference.
- `Λ` with border-row density. A stratum with no border rows is dropped from this statistic and named.
  This fixes the Ethics script, which reported p = 0.0 for an undefined value.
- **Same-book share.** The fraction of each stratum's stored neighbours that come from the same book,
  and Λ_ρ recomputed with `s(x)` taken from the last stored neighbour outside the row's own book (rows
  with none are dropped).
- **Date map:** Λ_ρ and a later → earlier Λ_chron, each with its own 20,000-ordering null and no
  correction.
- Per-stratum `ρ` and transit mass side by side, so the reader can judge the disclosed transit numbers.

## 6. Limitations

- **Books are clusters.** Up to 40 passages come from one book, and same-book neighbours can inflate
  `s(x)`. If a language's passages come from fewer, more uniform books, it looks denser for that
  reason. The same-book readout in §5 shows how much this matters. It is not a correction.
- **English is 40% of rows.** Its pairs carry much of the flow mass, so Λ_ρ is weighted toward them.
- **Corpus→corpus only,** as in Ethics.
- **The date map is weak.** Most authors are 19th-century, and Gutenberg is not a derivational
  corpus, so it says little about the chronology hypothesis.
- **This is a different corpus, not a replication of the same graph.** A pass here does not
  retroactively rescue the Ethics result. It shows the effect in a second corpus.

## 7. Cost

Minutes of CPU on Atlas, cores 12–23. 150k dot products, 400 recounts in the gate, 400 in the P2
reference, and 20,000 orderings of a 13 × 13 matrix per arm.

## 8. Before freezing

The scoring script will be written, black/ruff clean, and smoke-tested on synthetic data and on planted
flows before this document is frozen, so a crash cannot force an amendment. Both go into the freeze
commit.

## Appendix — stratum sizes (from labels only)

Eligible (13): english 60,000 · french 15,000 · finnish 15,000 · german 15,000 · dutch 10,000 ·
spanish 10,000 · italian 6,615 · greek 4,000 · hungarian 3,985 · esperanto 2,332 · latin 2,189 ·
portuguese 2,103 · swedish 2,000.
Ineligible: chinese 984 · polish 312 · japanese 291 · hebrew 120 · czech 80 · russian 46 · persian 10.

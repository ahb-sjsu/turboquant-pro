# Pre-registration — density as a rival to chronology for the direction of retrieval flow, with a joint amendment to `PREREG_directed_asymmetry.md`

**Status: REGISTERED ⚪, FROZEN on filing 2026-10-28.** Nothing in §3 has been computed on any arm,
and neither has any statistic of the chronology prereg. The author asked for a joint amendment
(2026-10-28) so that both hypotheses are registered before either is computed.

**Companion documents.**
- [`PREREG_directed_asymmetry.md`](PREREG_directed_asymmetry.md), the chronology prereg. Frozen
  2026-07-29 in commit `0f84916`, blob `0b4de32`, unchanged here. Its T1–T3 stand as written, except
  for the two clarifications in §2.
- Harness [`benchmarks/strata_dirasym.py`](../benchmarks/strata_dirasym.py), the author's chronology
  harness from Atlas `/archive/tqp_strata/dirasym.py` (sha256 `1a1600f2…`, dated 2026-07-29, never
  run). It was copied and reformatted with `black` only.
- Harness [`benchmarks/strata_density_asym.py`](../benchmarks/strata_density_asym.py), which runs
  every test below. It imports the chronology functions from `strata_dirasym.py` unchanged.

> **The question.** Directed asymmetry of the stratified kNN flow, `A = ‖F − Fᵀ‖_F / ‖F‖_F`, is
> significant in 4 of 4 arm×map cells and tracks neither compression nor training objective. The
> chronology prereg says it encodes textual derivation, with flow running later → earlier. This
> document registers a rival with a known mechanism: **net flow runs from sparser strata into
> denser ones**, because in corpus→corpus kNN a row's in-degree follows its local density. It also
> registers a joint test of which ordering the flow follows once the other is held fixed.

## 0. What is known, and what is blind

**Mechanism.** Hubs form where density is higher (Radovanović et al., JMLR 2010; Low et al. 2013).
The density-ratio pilot in this repo
([`benchmarks/RESULTS_hubness_density_ratio.md`](../benchmarks/RESULTS_hubness_density_ratio.md))
found count tracking local density at Spearman 0.37–0.38 with held-out corpus rows as queries. A row
in a sparse stratum `a` has distant neighbours, some of them in a denser stratum `b`. A row in `b` has
close neighbours, mostly inside `b`. So `a → b` edges should outnumber `b → a` edges.

**Known to the analyst:** the existence and magnitude of `A` in all four cells, and the stratum sizes
(Appendix). The analyst also checked metadata that carries no flow or density information: about 5%
of rows list themselves among their stored neighbours, and both vector files are unit-normalized.
**Never computed and never opened by the analyst:** per-stratum density, and any pairwise structure
of the flow. The Amendment-1 score files on Atlas (`score_D*_*.json`) may contain flow matrices.
The analyst has not opened them.

**Honest prior.** If density wins, the asymmetry is a known geometric effect seen between strata, not
a new phenomenon. Any chronology signal would then have to be shown net of density, which is what J
does.

## 1. Materials (committed artifacts, no new kNN)

On Atlas `/archive/tqp_strata/`:

| arm | vectors | neighbour ids | map files |
|---|---|---|---|
| D1 BGE-M3 | `sample_emb.npy` (350,000 × 1024) | `exact_ids.npy` (sha256 5a99aae6…) | `labels.txt` (85e10dba…), `labels_tradcent.txt` (ce1be324…) |
| D2 LaBSE @ 512 | `ethics_labse.npy` (350,000 × 768) | `exact_ids_labse_ethics.npy` (60286a94…) | same |

Both vector files share row order (`ethics_labse.npy.meta.json`). `k = 10` and `n_min = 2000` are
inherited. Edges into or out of ineligible strata are dropped, as in the harness. The `language` map
has 7 eligible strata (21 pairs). The `tradition × century_bin` map has 10 (45 pairs).

## 2. Clarifications to the chronology prereg (made before any computation)

These settle two places where the frozen text and the frozen harness disagree, or the text is wrong
about the data. They do not change any prediction, threshold or null.

1. **Pair set for Λ.** §3 of that document defines Λ over all pairs with differing bins, and the
   harness computes exactly that. Its §6 says cross-tradition pairs "are not part of T1". The
   definition and the code are binding. T1 uses all pairs with differing bins. The Λ restricted to
   within-tradition pairs is reported beside it as a descriptive number, as §6 asks.
2. **A factual error.** §6 says seven of the ten strata are `jewish`. The labels give five
   (`jewish|−2, 0, 1, 2, 3`). The other five are `greco_roman|−1`, `greco_roman|0`,
   `american_advice|4`, `hindu|−1` and `buddhist|−1`. Within-tradition pairs with differing bins
   number 11.
3. **Unstated details, fixed here.** T3 is computed on both arms and fires if either arm gives
   two-sided `p < 0.05`. Its p-value in the Holm family is the smaller of the two. T1's p-value in the
   family is the larger of the two arms' one-sided p-values, since T1 requires both. D3 (T2) is the
   Pearson correlation over the same pair set as T1. Its row-permutation null applies one shared
   permutation of the area code to both arms in each replicate, so encoder agreement is measured
   against what two kNN graphs on the same rows share by chance.

**A note on T3, recorded without changing it.** T3 compares one fixed ordering (alphabetical) with all
orderings of the same flow matrix. A partition artifact moves every ordering together, so T3 cannot
see one. If the language flow has a real total order, alphabetical order is just one of 5,040
orderings, and T3 fires with probability about 5% whatever the instrument does. It is kept as frozen.
It governs T1 and T2 only. The calibration gate G2 below is the instrument check for the new tests.

## 3. Statistics (defined before computation)

**Per-row density.** `s(x)` = cosine similarity of row `x` to its last stored neighbour (column 9 of
the stored ids), computed from the arm's own vectors. Higher means denser. For the ~5% of rows that
list themselves, this is their 9th distinct neighbour. That is fixed here and not tuned.

**Per-stratum density.** `ρ_a` = median `s(x)` over rows in stratum `a`. Secondary `ρ_a^∂` uses
border rows only, meaning rows with at least one out-edge into another eligible stratum.

**Flow for the new tests uses raw edge counts.** `C_ab` = number of directed edges from rows in `a` to
rows in `b`. Under random assignment of rows to strata, `C_ab` and `C_ba` have the same expectation,
so raw counts carry no size term. The row-normalized `F` of the chronology prereg carries one
(`F_ab − F_ba → (n_b − n_a)/N`). T1–T3 keep `F` as frozen.

**Λ_ρ — density alignment.** Over unordered pairs `a < b`:

    Λ_ρ = Σ (C_ab − C_ba) · sgn(ρ_b − ρ_a)  /  Σ |C_ab − C_ba|

`Λ_ρ > 0` means net flow runs into the denser stratum. **Null:** hold `C` fixed and permute the
density values among strata. On `language` this is exact over all 5,040 orderings, and the one-sided
p-value is the share of orderings with `Λ ≥` the observed value.

**D-ρ2 (descriptive).** Weighted least squares through the origin of `m_ab = (C_ab − C_ba)/(C_ab + C_ba)`
on `ρ_b − ρ_a`, weights `C_ab + C_ba`. Report `R²`.

**Rival and companions (descriptive, with p-values but outside every family).** `Λ_n` = Λ with stratum
size in place of density. `Λ` with `ρ^∂`. Spearman(`ρ_a`, `n_a`).

**J — the joint model, on `tradition × century_bin`, raw counts.** Over all 45 pairs `a < b`, fit by
weighted least squares through the origin, weights `C_ab + C_ba`:

    m_ab = β_chron · sgn(c_a − c_b) + β_ρ · sgn(ρ_b − ρ_a)

`β_chron > 0` is flow running later → earlier with density held fixed. `β_ρ > 0` is flow running into
denser strata with chronology held fixed. Pairs with equal bins inform only `β_ρ`. **Nulls:** for
`β_chron`, permute the century bins among the ten strata with `ρ` fixed. For `β_ρ`, permute the
densities with bins fixed. 5,000 draws each. One-sided p = (1 + #{β* ≥ β}) / 5001.

**Descriptive on `tradition × century_bin`:** Λ_ρ and Λ_n (5,000 sampled orderings), Λ_chron on raw
counts, Λ within tradition, and Spearman(`ρ_a`, bin).

## 4. Registered predictions

Three families, each Holm–Bonferroni at family-wise α = 0.05. The T family is the chronology prereg's,
unchanged. No family is corrected against another. Nothing outside the families may be substituted for
a confirmatory p-value.

**T family (frozen, `PREREG_directed_asymmetry.md` §4).** T1, T2, T3 as written there, with §2's
clarifications.

**R family — density sets direction on the `language` map.**
- **R1 (BGE-M3).** `Λ_ρ > 0` with exact one-sided `p < 0.05` after Holm.
- **R2 (LaBSE).** Same.

**J family — which ordering survives the other, on `tradition × century_bin`.**
- **J-chron.** `β_chron > 0` on both arms, with the larger of the two p-values below 0.05 after Holm.
- **J-ρ.** `β_ρ > 0` on both arms, the same way.

**How the outcomes read.**
- J-ρ holds and J-chron fails: density explains the direction, and chronology adds nothing beyond it.
  This holds even if T1 passes, because T1 does not control for density.
- J-chron holds and J-ρ fails: chronology survives density. H-D is strengthened beyond T1.
- Both hold: two separate orderings contribute.
- Neither holds: either the 45 pairs cannot separate the two orderings or neither drives direction.
  Spearman(`ρ_a`, bin) says which. A high value means the orderings are confounded at this resolution.
- R1 or R2 significant with `Λ_ρ < 0`: flow runs into sparser strata. That inverts the mechanism and is
  the headline.
- R1 and R2 hold but `Λ_n` is as large: direction follows size as well as density. Report as
  unresolved, not as support.
- R1 and R2 hold with small `R²`: density sets the sign, not the bulk of the asymmetry.

## 5. Order of operations

Two separate invocations, with the guard written in code.

1. **Gate** (`strata_density_asym.py gate`).
   - **G1** is the frozen T3, computed with the harness functions and the harness null (2,000 draws,
     seed 106). If it fires, T1 and T2 are void and are not computed.
   - **G2** is a calibration of every new test. For each arm, the area code is permuted over rows 200
     times on both maps. In each replicate the gate recomputes `ρ`, the counts, the exact R test and
     the J tests (1,000 draws each). Under this null each test should reject 10 times in 200. G2 fires
     if any test rejects more than 17 times on any arm (`P(X ≥ 18) ≈ 0.01` per count, about 5% across
     the six counts). If G2 fires, R and J are blocked.
   - The gate writes no statistic computed on the true labels except T3's own.
2. **Score** (`strata_density_asym.py score`). Refuses to run unless the gate file exists and G2 is
   quiet. Skips T1 and T2 if G1 fired. Computes everything in §3 and §4 in one pass.

## 6. Limitations

- Seven language strata give 21 pairs. Ten tradcent strata give 45, and the five `jewish` strata carry
  most of the edge mass.
- Corpus→corpus only. Ethics has no real query set, and the pilot showed that real queries produce a
  different and much stronger hub structure.
- J uses signs, not magnitudes, for both predictors, so the two compete on equal terms. It cannot say
  how much of `A` either explains.
- Permuting one predictor while holding the other fixed is valid when the two are not strongly
  correlated across strata. Spearman(`ρ_a`, bin) is reported so the reader can judge.
- LaBSE's residual truncation (≤ 11.7% of rows per language) can change density. It is not proven
  neutral.
- `century` is corpus metadata of unaudited provenance, as the chronology prereg already notes.

## 7. Cost

Minutes of CPU on Atlas, cores 12–23. Per-row density is one dot product per row. Everything else is
arithmetic on 7 × 7 and 10 × 10 count matrices, plus 400 recounts of 3.5 million edges in the gate and
200 in the T2 null.

## Amendment 1 (2026-10-28, after the gate, before any scoring)

**What happened.** The gate of §5 ran once (record: `benchmarks/results/dasym_gate.json` and
`dasym_gate.log`).
- **G1 (the frozen T3) fired on D1.** Λ under the alphabetical ordering was 0.933, two-sided
  p = 0.031. On D2 it was 0.559, p = 0.364. By the chronology prereg's own rule, T1 and T2 are void
  and are not computed. This amendment does not change that.
- **G2 fired on both arms.** The rejection counts out of 200 were as follows.

  | arm | R (Λ_ρ) | J β_ρ | J β_chron |
  |---|---|---|---|
  | D1 | 37 | 46 | 4 |
  | D1 limit | 17 | 17 | 17 |
  | D2 | 27 | 53 | 10 |

**Why G2 was the wrong null for the density tests.** Permuting area codes over rows breaks the link
between rows and areas, but each row keeps its own density and its own in-degree. A random area that
happens to collect denser rows also collects more hubs, so flow into it really is larger. The density
mechanism therefore operates inside random partitions, and the density tests correctly detect it
there. The excess rejections show the mechanism at row level. They do not show a defect in the
ordering test. The chronology term stayed calibrated, as it should, since a random partition carries
no chronology.

**What this changes.**
1. **The gate for the density tests (R and J β_ρ) is replaced.** The new null permutes `s(x)` over
   rows and keeps the true labels and the true flow. Densities then carry no information about the
   flow, and each test should reject about 10 times in 200. The fire rule is unchanged (more than 17
   on any count, any arm). The J β_chron calibration keeps the row-permuted area codes. The new gate
   is `strata_density_asym.py gate-a1`, seed `20261030`. It recomputes G1 identically, so G1 still
   fires and T1 and T2 stay void.
2. **`score` refuses any gate file that is not an Amendment-1 gate.**
3. **A descriptive comparison is added to `score`.** The language-map Λ_ρ is also compared with its
   distribution over 200 random partitions with the true stratum sizes. This shows whether the true
   language areas carry more density-aligned flow than row-level density alone would give. It is
   reported beside R1 and R2 and is not a confirmatory test.
4. **The reading of R1 and R2 is narrowed.** A pass now shows that the row-level mechanism appears
   between the true areas. It does not by itself show that area-level density matters beyond that.
   The comparison in item 3 speaks to that question.

**Unchanged:** every statistic, prediction, threshold and family in §3 and §4, apart from T1 and T2,
which G1 voided.

**What is known at the time of this amendment.** The G1 values above and the G2 counts above. The G2
counts say that the row-level mechanism exists in these graphs, which raises the prior that R1 and R2
pass. No statistic of the R or J families has been computed on the true labels with the true
densities.

**A note on G1, for the record.** Λ = 0.93 on D1 under an arbitrary ordering means the language flow
is very close to a single total order, and alphabetical order happens to lie near it. §2 predicted
that this case makes T3 fire by coincidence. The prereg's rule is applied anyway.

## Appendix — stratum sizes (known before registration, from labels only)

`language` (7 eligible): hebrew 134,524 · aramaic 123,820 · english 44,929 · greek 32,289 ·
latin 7,038 · sanskrit 4,780 · pali 2,132.

`tradition × century_bin` (10 eligible): jewish|1 140,689 · jewish|2 74,995 · jewish|−2 32,462 ·
greco_roman|−1 32,289 · jewish|0 25,695 · jewish|3 25,151 · greco_roman|0 7,038 ·
american_advice|4 3,330 · hindu|−1 3,022 · buddhist|−1 2,132. Spearman(size, bin) = 0.09
(p = 0.80).

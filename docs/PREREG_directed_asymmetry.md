# Pre-registration — what directed retrieval asymmetry is a property of

**Status: REGISTERED ⚪, FROZEN on filing 2026-07-29.** The statistics of §3 have
never been computed on any arm. Successor to
[`PREREG_role_differentiation.md`](PREREG_role_differentiation.md) Amendment 1,
whose P5 was refuted and whose P6 passed only marginally. This document exists
because that round produced exactly one robust, replicated signal — and it was
**unregistered**, so it is currently an observation and not a result.

> **The question.** Directed asymmetry of the stratified kNN flow matrix,
> `A = ‖F − Fᵀ‖_F / ‖F‖_F`, is significant at the permutation floor
> (`p = 0.0050`, observed above all 200 replicates) in **4 of 4** arm×map cells,
> at 0.0265–0.0483 against control values of 0.002–0.013. It does **not** track
> the observer's coarse-graining (P3: flat across 27.7× compression) and does
> **not** track the training objective (P5: refuted, both terms, with the
> residual confound favouring the prediction). So what is it a property of?

## 0. What is already known, and what is therefore not blind

**Known to the analyst:** that `A` is large and significant in all four cells,
its magnitudes, and that it tracks neither compression nor training objective.
No prediction below may rest on the *existence* or *magnitude* of `A`.

**Not computed on any arm, and therefore blind:** the *internal structure* of the
antisymmetric part of `F` — specifically whether its direction aligns with the
corpus's chronology. Every prediction in §4 is about that structure.

## 1. Hypothesis

**H-D: directed retrieval asymmetry encodes derivational direction in the
corpus, not a property of the encoder or the quantizer.**

The Ethics corpus is derivational: later texts comment on, cite, and elaborate
earlier ones (Torah → Talmud → medieval commentary; and the `tradition ×
century_bin` map spans bins −2 … 4). If a commentary passage is semantically
near its source, then when the commentary row is a *query* its neighbours include
source rows, producing an edge commentary → source. The reverse is weaker: a
source passage's nearest neighbours are other sources and its own restatements,
not any particular later commentary. **Net flow should therefore run from later
strata to earlier strata.**

This would explain both prior nulls in one stroke: derivational structure is a
property of *the texts*, so it is invariant to the encoder that embeds them (P5's
null) and to the quantizer that compresses them (P3's flatness). The asymmetry
was never going to track either.

## 2. Materials

Committed artifacts only, no new compute beyond flow-matrix arithmetic:
`exact_ids.npy` (D1, BGE-M3), `exact_ids_labse_ethics.npy` (D2, LaBSE @ 512),
`labels_tradcent.txt` (the pre-declared secondary map, 10 eligible strata at
`n_min = 2000`), and `labels.txt` (language; used only for the negative control
in §4, since language carries no chronology).

All ten eligible `tradition × century_bin` strata have numeric bins
(−2, −1, 0, 1, 2, 3, 4), so the ordering is total and no stratum is dropped for
a missing century. Bins are `century // 5`.

## 3. Statistics (defined before computation)

For eligible strata `a, b` with century bins `c_a, c_b`, from the row-normalized
directed flow matrix `F`:

**Net flow** `N_ab = F_ab − F_ba` (antisymmetric, `N_ab = −N_ba`).

**D1 — chronological alignment (PRIMARY).** Over all ordered pairs with
`c_a ≠ c_b`, the mass-weighted directional score

    Λ = Σ_{a,b : c_a > c_b} N_ab / Σ_{a,b : c_a ≠ c_b} |N_ab|

`Λ ∈ [−1, 1]`. `Λ > 0` means net flow runs later → earlier (H-D's direction);
`Λ ≈ 0` means the asymmetry is chronologically unstructured; `Λ < 0` means it
runs earlier → later.

**D2 — rank agreement (SECONDARY).** Kendall τ between `sign(N_ab)` and
`sign(c_a − c_b)` over the same pairs.

**D3 — cross-encoder stability (SECONDARY).** Pearson correlation between the
`N_ab` matrices of D1 and D2, over the same pair set.

**Nulls.** Two, and they test different things:
- **Row-permutation null** (as in Amendment 1): permute the area code over rows,
  `R = 200`. Preserves stratum sizes; destroys content–stratum pairing.
- **Chronology-permutation null (new, and the one that matters):** hold `F`
  fixed and permute the *century bins among the ten strata*, all `10!`-many
  approximated by `R = 2000` draws. This isolates the *ordering* from the
  partition — it asks whether the observed alignment is special to the true
  chronology, which the row-permutation null cannot ask.

## 4. Registered predictions, with the multiplicity rule fixed

**Multiplicity (correcting Amendment 1's P6 defect).** Exactly **three**
confirmatory tests are registered: T1, T2, T3 below. Holm–Bonferroni across
those three at family-wise α = 0.05. No fourth test may be added, and a p-value
from any exploratory quantity may not be substituted. Amendment 1's P6 was
written as "at least one of four cells, uncorrected" and passed at p = 0.0498;
that will not recur.

**T1 (primary) — chronological alignment exists and runs later → earlier.**
`Λ > 0` on **both** D1 and D2 under the `tradition × century_bin` map, with
chronology-permutation `p < 0.05` on both after Holm correction. *Direction is
predicted, so a significant `Λ < 0` refutes H-D rather than confirming a "some
structure" result.*

**T2 — the asymmetry is encoder-invariant.** D3 (cross-encoder correlation of
`N_ab`) is ≥ 0.5 with row-permutation `p < 0.05`. H-D says the structure lives
in the texts; if the two encoders disagree about direction, it does not.

**T3 — negative control: language carries no chronology.** Under the `language`
map, which has no derivational ordering, `|Λ|` computed against an arbitrary
fixed label ordering must be indistinguishable from its chronology-permutation
null (`p ≥ 0.05`). A fire here means `Λ` detects partition artifacts rather than
chronology, and **T1/T2 are void regardless of their own p-values.**

**Failure clauses.**
- T1 fails with `Λ ≈ 0`: the asymmetry is real but chronologically unstructured.
  H-D is wrong and the next candidate is stratum generality (semantic breadth),
  not derivation — a different round, not a re-analysis of this one.
- T1 fails with `Λ < 0` significantly: net flow runs earlier → later, which
  inverts the retrieval-direction argument in §1 and is a more interesting
  finding than a confirmation; report it as the headline.
- T2 fails while T1 holds: the alignment is encoder-specific, which contradicts
  the explanation offered for P5's null and reopens it.
- T3 fires: instrument defect, everything blocked, exactly as Amendment 1's
  control gate blocked scoring.

## 5. Order of operations (the D1-blindness lesson, encoded)

T3's control is computed and read **first**, in its own invocation. Only if it
stays quiet are T1/T2 computed, in one pass. The Amendment-1 scoring lost arm
D1's blindness to a single unattended script that ran past a stop; the gate and
the scoring are separate invocations here for that reason and not as ceremony.

## 6. Declared limitations

Century bins are coarse (`// 5`) and were chosen for stratum sizes, not for
chronological resolution · seven of ten eligible strata are `jewish`, so most of
the ordering signal is within one tradition and the test is largely about
internal Jewish textual chronology; the cross-tradition pairs are reported
separately and are **not** part of T1 · `century` is corpus metadata of unaudited
provenance and an error there propagates directly into `Λ` · the D2 arm retains
its ≤ 11.7% residual truncation, which has no obvious directional relationship
to chronology but is not proven neutral · `A`'s magnitude is known to the analyst
and is not evidence for anything here.

## 7. Cost

Minutes of CPU. Every input is committed; the work is flow-matrix arithmetic plus
2000 cheap re-orderings of a 10×10 matrix. This round is deliberately small
because the previous one spent ~3 GPU-hours to refute a prediction that a
`sentence-transformers` default had nearly invalidated in advance.

*The signal that replicated is the one worth registering.*

# Pre-registration — role differentiation as constraint-induced symmetry breaking (STRATA Phase-2)

**Status: §0–§4 SCORED (see [`RESULTS_role_differentiation.md`](RESULTS_role_differentiation.md)
— P4 fired, P3 not resolved). [Amendment 1](#amendment-1--corrected-instrument-frozen-2026-07-29)
is FROZEN 2026-07-29 and carries the live predictions (P5, P6).**
Originally filed 2026-07-28 as REGISTERED ⚪ for P1–P3 on the blind arms (§1),
with the statistics of §3 never yet computed on any arm. Successor hypothesis to
the falsified P2 of [`PREREG_multilingual_strata.md`](PREREG_multilingual_strata.md)
— **not a re-read of it**. Results land in `RESULTS_role_differentiation.md`;
this file is never edited to fit them.

> **The question.** The multilingual run found that the *trained*-transfer encoder
> produced heterogeneous per-language roles (7/13 backbone-class areas) while the
> *emergent* one produced none, inverting a registered prediction built on
> concentration. Is that inversion the signature of a different mechanism —
> **role differentiation as symmetry breaking forced by a constraint**, in the
> sense of Yamaguti & Tsuda (*Neural Networks* 62:3–10, 2015) and Tsuda,
> Yamaguti & Watanabe (*Entropy* 18:74, 2016) — and if so, does tightening the
> observer's coarse-graining *amplify* it?

---

## 0. Theoretical claim being tested (stated before the operationalization)

Two apparently opposite information objectives both produce functional
differentiation in the neural-modeling literature: maximizing transfer entropy
**between** subnetworks (Yamaguti & Tsuda 2015) and minimizing mutual
information **between** subgroups (Cognitive Neurodynamics, 2025). These are not
contradictory because **the homogeneous state is simultaneously the MI-maximum
and the TE-minimum** — two fully synchronized units share maximal mutual
information and near-zero transfer entropy, since each already predicts itself.
Both objectives are therefore repelled by the same configuration.

The load-bearing consequence for *this* project: an information objective alone
cannot select differentiation, because mutual information is invariant under
invertible reparameterization. Differentiation requires the map to be
**non-invertible** — finite dimension, finite units, quantization. That is
precisely the "with constraints" in Tsuda's framing, and it is the same
observation as this project's own thesis that the observer's coarse-graining
basis, not the ambient space, fixes which geometry survives.

**Therefore:** role differentiation should track the *strength of the
constraint*, not merely the training objective. P3 is the test that separates
this account from "LaBSE simply has different training data."

**Instrument explicitly refused.** Mutual information estimated by a learned
critic (MINE, Belghazi et al. 2018, and the InfoNCE/NWJ family) is **not** an
acceptance instrument here, and no claim in §4 may be scored with one. Reasons,
both dispositive: any distribution-free high-confidence lower bound on MI from
`N` samples cannot exceed `O(ln N)` and MINE's variance can grow exponentially
in the true MI (McAllester & Stratos, PMLR 108, 2020) — so a low reading is
non-identifiable between "low MI" and "underfit critic"; and estimated MI does
not track downstream behaviour, with downstream performance improving under
invertible models while true MI is constant (Tschannen et al., ICLR 2020). A
critic-optimized bound is the same class of defect as accepting on
reconstruction cosine: a number that moves for reasons other than the thing
being claimed. Every statistic in §3 is instead discrete, plug-in, and
distribution-free.

## 1. Arms, and what each arm can license

The Gutenberg pair has **already been scored** for P2 of the prior
registration. Its qualitative outcome is known to the analyst. Therefore:

| arm | corpus × encoder | status | licenses |
|---|---|---|---|
| **D1** | Ethics × BGE-M3 (exists) | §3 statistics never computed | **blind** |
| **D2** | Ethics × LaBSE (**one re-encode required**) | does not exist yet | **blind** |
| **C** | Gutenberg/Ethics × `xbse` language-invariant instance on the BGE-M3 base | does not exist; gated | **blind, and the only arm that isolates the objective** |
| **G** | paired Gutenberg × {BGE-M3, LaBSE} | P2 already scored | **discovery only — consistency, never confirmation** |

**Pre-committed reporting rule for arm G.** The §3 statistics will be computed
on it and reported, and agreement with §4 there will be labelled *consistent
with* the hypothesis and **explicitly denied the word "confirmed"**, because the
analyst knew the qualitative answer before choosing the statistics. Arm G cannot
resolve P1–P3 in either direction. If a reader can only be persuaded by arm G,
the honest response is that this registration has not yet been tested.

**Arm C is the decisive arm** because it varies the training objective while
holding architecture, pooling, dimension, and base weights fixed — the
LaBSE-vs-BGE-M3 contrast confounds objective with training data, tokenizer, and
dimension (768 vs 1024), all of which were already declared in the prior
registration and none of which are fixed by adding new statistics.

## 2. Area map and protocol

Unchanged from `PREREG_multilingual_strata.md` §2–3 and inherited wholesale:
`tqp-area-map/1`, language assignment with the labeler in the profile, the
high-confidence threshold with a reported `und` stratum, `k = 10`, seed 7,
`n_min = 2000`, `q_min = 500`, ABSTAIN recorded as a result. The langid-noise
hazard (label error masquerades as transit) applies identically and the stricter
threshold sensitivity rerun accompanies any P1–P3 claim.

New requirement: the **directed** kNN edge list must be retained per arm (source
row, neighbour row, both area labels), since S2 is defined on edge direction.
`hub_census` already treats `N_k` as in-degree, so the directedness is present in
the relational surface; this registration only requires that it not be
symmetrized before scoring.

## 3. Statistics (defined here, before measurement)

**S1 — role bimodality (primary).** Per-area transit fraction `τ_a` over
non-ABSTAIN areas, tested against unimodality by **Hartigan's dip test**
(reported statistic and p-value), with a **2-means silhouette** on the
multivariate role vector `(τ_a, transit-row centrality percentile_a, in-degree
Gini_a)` as the corroborating form. Bounded, scale-free coordinates only, so
values are comparable across encoders of differing dimension.

*Rationale, stated to prevent a post-hoc slide:* symmetry breaking predicts
**two qualitatively distinct role classes**, which is a distributional
*bifurcation*, not merely inequality of a scalar.

**S1 is explicitly NOT the Gini of transit mass, and the two are predicted to
move in opposite directions** — see P2 below. This is the sharpest falsifiable
content in this document.

**S2 — directed role asymmetry (secondary).** Build the area→area flow matrix
`F[a,b]` = share of directed kNN edges originating in area `a` and landing in
area `b`. Score `A = ‖F − Fᵀ‖_F / ‖F‖_F`, the antisymmetric share. `A → 0` is
the exchangeable/homogeneous configuration.

**S3 — declared-dependent, reported not scored.** Discrete plug-in
`I(area_query ; area_neighbour)` on the edge distribution (≈150k edges over
13–20 areas: plug-in estimation is sound at this alphabet size, no critic
needed). **Declared now:** S3 is close to monotone with mean τ, so it is *not
independent evidence* for any claim S1/S2 support, and it is reported as
descriptive context only.

## 4. Registered predictions (frozen 2026-07-28)

**P1 — differentiation is bimodal, and tracks the objective.** On the blind arms,
per-area `τ_a` is multimodal under an explicit transfer/invariance objective and
unimodal under emergent multilinguality. *Operationalized:* Hartigan dip rejects
unimodality at p < 0.05 for D2 (and for arm C when it exists) and fails to reject
for D1; corroborated by 2-means silhouette ≥ 0.50 versus < 0.35 respectively.
*If it fails:* the backbone/non-backbone split seen on Gutenberg was a
property of that corpus's language composition rather than of role
differentiation, and the Tsuda reading is wrong for embeddings — reported with
equal prominence, as the prior registration's inversion was.

**P2 — bimodality and mass inequality dissociate.** The more differentiated arm
has *higher* S1 and *lower* Gini of transit mass. *Operationalized:* the sign of
(S1 difference) is opposite to the sign of (transit-mass Gini difference) across
the D1/D2 contrast. *Stakes:* this is the prediction that distinguishes this
account from the falsified P2 of the prior registration, which assumed
differentiation would appear **as** concentration. A confirmation says the
earlier prediction was not merely wrong in magnitude but **wrong in the
statistic** — that concentration was the wrong observable for a bifurcation.
*If it fails* in the direction of both statistics moving together, then
concentration and role differentiation are not separable observables here and
this framework adds nothing over the original.

**P3 — differentiation is constraint-induced (the decisive prediction).** Within
a single arm, S1 increases monotonically as the observer's coarse-graining
tightens. *Operationalized:* dip statistic and silhouette computed at
{uncompressed, PCA-384, PCA-384+TQ3 at the deployed 27.7×} on the same rows and
the same area map, requiring a strictly monotone non-decreasing dip statistic
across the three, with the endpoints separated at p < 0.05.
*Why it matters more than P1:* P1 can be explained away by training data; P3
cannot. If role differentiation is amplified by the *quantizer* while the corpus,
encoder, and labels are held fixed, then differentiation is a property of the
observer's non-invertible coarse-graining — which is this project's thesis stated
as a measurement rather than an interpretation. *If it fails* (differentiation
flat or decreasing under tighter compression), the "constraint induces the
symmetry breaking" claim is dead as stated, and P1 survives only as a fact about
training objectives.

**P4 — negative controls (registered expectation, not a prediction).** A random
permutation of area labels, and an i.i.d. Gaussian corpus at matched `n`/dim,
both yield dip failing to reject, silhouette < 0.2, and `A` within noise of 0.
A control that fires invalidates the instrument, not the hypothesis, and blocks
scoring of P1–P3 until resolved.

## 5. Analysis plan

One report JSON per (arm, operating point) carrying the area-map digest, the
directed-edge fingerprint, seeds, thresholds, and the dip/silhouette/`A` values.
Comparisons only where the replication predicate holds (identical text
fingerprints, differing encoder digests) — the predicate refused the orphaned 1M
rung before and refuses ad-hoc pairs here. The statistics named in §3 are the
statistics scored; **the dip test is chosen now precisely so that a later switch
to a different unimodality test would be visible as an amendment.**

Arm G is computed first for engineering shakeout (it exercises the directed-edge
path on data whose answer is known) and is reported under the §1 labelling rule.
Arms D1/D2 are scored only after the code is frozen against arm G.

## 6. Declared confounds and limitations

Encoder contrast confounds objective with training data, tokenizer, and
dimension — **only arm C removes this**, and until arm C exists P1 is a
correlational claim about encoder families, stated as such · language↔topic
correlation inside Ethics (the Gutenberg rung was the check, and it is now
discovery-only for these statistics, so this weakens rather than strengthens the
Ethics arm — declared, not corrected) · dip test power at 13–20 areas is modest,
so a failure to reject is weak evidence for unimodality and will not be
reported as evidence *of* unimodality · corpus→corpus query distribution, as
before · `A` is sensitive to `k` (fixed at 10 by inheritance, not tuned).

## 7. Deliverables

`RESULTS_role_differentiation.md` with the §4 scoreboard, predictions typeset
beside measurements and confirmations beside embarrassments in identical
formatting · report JSONs (publishable without text or vectors) · the directed
flow matrices per arm · a `tqp` subcommand or documented script for S1/S2 so
the statistics are reproducible rather than notebook-resident · if P3 resolves
either way, it belongs in the Quantizer's Blindfold line of argument, since it is
that paper's claim made measurable.

## 8. Cost and blockers

S1/S2/S3 on existing embeddings: an afternoon of CPU. **P1 is blocked on one GPU
re-encode** (Ethics texts × LaBSE) to create arm D2 under the replication
predicate. **Arm C is blocked** on the `xbse` language-invariance instance
existing and clearing its own gate first — unchanged from the prior
registration's §8, and still the highest-value unblock in this line of work. P3
requires no new encode and is therefore the cheapest decisive test in this
document: it can be scored on arm D1 alone.

## 9. Amendment rule

§0, §3, and §4 are frozen. Protocol clarifications append to the changelog
below; operationalizations may be **tightened** with a dated entry, never
loosened, and never after seeing results for the statistic in question. Arm G's
labelling rule in §1 may not be amended at all — it is the provision that keeps
this document from becoming a post-hoc narrative about data already seen. The
results file cites this document by content hash.

*Symmetry is the redundant state; both signs of the information objective flee it,
and only the constraint decides that they must.*

---

---

# Amendment 1 — corrected instrument, FROZEN 2026-07-29

**Status: FROZEN.** §A1 below is registered as of this date and is scored only
after the control gate in §A1.4 passes. Nothing in §0–§4 above is altered; the
original P1–P4 were scored in
[`RESULTS_role_differentiation.md`](RESULTS_role_differentiation.md) and their
verdicts stand as recorded (P4 fired; P3 not resolved). This amendment exists
because **P4 did its job**: the 2-means silhouette fired on all three controls,
which the pre-registration defines as invalidating the instrument and blocking
scoring until resolved. This is that resolution.

## A1.1 What was wrong, in one line each

1. **τ_a is size-confounded.** Under label permutation a row's transit fraction
   tends to `1 − n_a/N`; measured, C3's thirteen near-equal areas all read
   0.918–0.925 against a predicted 0.923. Raw per-area τ therefore carries a
   size signal with no role content.
2. **A fixed-k=2 silhouette cannot decline to split.** C1's 0.808 was a 12-vs-1
   outlier split; C3's 0.461 split thirteen identical values.

## A1.2 Corrected statistics (frozen)

Let `R = 200` label permutations of the *same* directed kNN graph (permutation
preserves every area's size, so the null absorbs the size confound exactly).

**E — excess transit.** `e_a = τ_a − mean_r τ_a^(r)`. Under any null this is ≈ 0
for every area regardless of size, which is the property a role statistic must
have and raw τ_a lacks.

**S1′ (PRIMARY) — role multiplicity by permutation test.** Separation
`T = 1 − SS_within(2-means on {e_a}) / SS_total({e_a})`, with
`p = (1 + #{r : T^(r) ≥ T}) / (R + 1)` computed by running the identical
2-means/`T` pipeline on each permutation replicate's `{e_a^(r)}`.
**Registered because its validity does not depend on the number of areas** — it
is exact under the permutation null at any `n_areas`, unlike the dip test, whose
failure to resolve P3 at 7 areas was a power failure. It also fixes defect (2) by
construction: a fixed-k split of noise is exactly what the null produces, so
noise cannot be significant.

**S1′b (SECONDARY, descriptive) — Hartigan dip on `{e_a}`**, reported with its
power caveat. Not a gate.

**S2′ — directed asymmetry with a permutation p-value.** `A` as in §3, plus
`p_A` from the same `R` replicates. `A` passed its controls unchanged (0.010 /
0.002 / 0.013 against 0.048–0.097 observed); the p-value is a tightening, not a
replacement.

**S3 — unchanged and still context-only** (declared near-monotone with τ̄).

## A1.3 Area maps (declared now, before any D2 scoring)

No area key on the Ethics corpus yields 12 eligible strata at the inherited
`n_min = 2000` — measured on the committed sample: `language` 7,
`tradition` 5, `corpus` 5, `language×tradition` 8,
`tradition×century_bin` 10. A ≥12 floor would therefore be an unsatisfiable
precondition, the exact failure mode this project has hit twice, so it is not
registered.

- **Primary map: `language`** (inherited; Ethics 7 eligible, Gutenberg 13).
- **Secondary map, pre-declared: `tradition × century_bin`** (Ethics 10
  eligible), scored and reported alongside the primary. Declared here so it is
  not a post-hoc second look.
- **Floor: ≥ 5 eligible areas**, below which the whole test ABSTAINs with
  registered cause `too_few_areas`. `n_min = 2000` / `q_min = 500` are inherited
  and **not** loosened.

## A1.4 Control gate (must pass BEFORE D2 is scored)

The corrected statistics are run on the three P4 controls first. Required:
`|mean e_a| < 0.01` and `p(S1′) ≥ 0.05` and `p_A ≥ 0.05` on **all three**. Any
control firing blocks D2 scoring again, and the instrument is wrong again.
**D2 is scored in the same run only after that gate passes**, in one pass, with
no intermediate look — the D1 loss came from an unattended script running past
a stop, so the control gate and the scoring are separate invocations and the
gate's output is read before the second is launched.

## A1.5 Registered predictions (frozen 2026-07-29)

Arm D2 (`ethics_labse.npy`, built and predicate-verified, `"scored": false`) is
blind. **Arm D1's τ_a and `A` are already known to the analyst** (0.0483 at rung
1) — so every prediction below is carried by the D2 term alone, and that is
stated rather than glossed.

**P5 — the arm-G direction replicates out of sample.** On identical Ethics
texts, the trained-transfer encoder shows greater role differentiation than the
emergent one: `T(D2) > T(D1)` **and** `A(D2) > A(D1)`. This is a genuine
out-of-sample test of the only directional signal arm G produced
(`A`: LaBSE 0.0974 > BGE-M3 0.0848), on a different corpus, with D1 fixed and
D2 unseen. *If it fails:* that signal does not replicate, and "trained transfer
differentiates more" is unsupported on this corpus — reported, not retuned.

**P6 — differentiation is present at all.** On at least one arm and at least one
declared map, `p(S1′) < 0.05` — i.e. per-area roles split beyond what size and
draw noise produce. *If it fails on both arms and both maps:* there is no
detectable multi-class role structure in this corpus at these instruments, which
retires the bimodality reading for Ethics rather than leaving it unresolved.

**Declared confound that biases toward P5, and therefore weakens a
confirmation.** LaBSE truncates 0–11.7% of rows per scored language even at 512
(`d2_truncation_audit_256_512.json`). Truncation shortens text, which plausibly
makes it more generic, more central, and more retrieved — the same direction P5
predicts. A P5 confirmation is therefore **weak** evidence and must be reported
as such; a P5 refutation is strong, since the residual confound worked in the
prediction's favour and it still failed. Dimension (768 vs 1024) and training
data remain confounded as previously declared; only arm C removes them.

**P3 is not re-opened.** It was scored and recorded as not resolved. The
corrected instrument may be applied to the three compression rungs in a later
round, but D1's rungs are no longer blind and no P3 verdict may be revised from
them.

## A1.6 Amendment discipline

§A1.2, §A1.3 and §A1.5 are frozen as of 2026-07-29 and were written before any
corrected statistic was computed on any arm. Every change relative to §3/§4 is a
*tightening or a repair of a control failure*: the primary statistic gains an
exact null, the invalid statistic is demoted to descriptive, `A` gains a
p-value, an area-count floor and a second declared map are added, and `n_min`
is untouched. No band, no threshold, and no inherited protocol parameter is
loosened.

---

## Changelog (protocol clarifications; §0/§3/§4 untouched)

- **2026-07-29 — Amendment 1 filed and frozen (above).** Triggered by P4 firing
  on all three controls. §0/§3/§4 are unaltered and their scored verdicts stand.
  Also recorded: arm D2 was built at LaBSE's architectural limit (512) rather
  than the `sentence-transformers` default (256) after a per-language truncation
  audit showed the default's truncation spread (0.766) would have manufactured
  the predicted signal; the 512 spread is 0.117 with every scored language
  keeping its median row whole. The D1/D2 replication predicate was verified
  empirically (200 rows re-encoded with BGE-M3 against the stored embeddings:
  cosine min = median = 1.000000), not assumed from row order.

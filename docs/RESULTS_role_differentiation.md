# Results — role differentiation as constraint-induced symmetry breaking

Scored against [`PREREG_role_differentiation.md`](PREREG_role_differentiation.md),
blob `fc91233e3bdba70f42e4d9a98ce1f8b500ccfe0d`, frozen and pushed
(`7744046`) before any statistic in §3 was computed. Run 2026-07-28 on Atlas
under `batch_probe.ThermalController`. Harness `role_diff.py`, artifacts in
[`role_diff_out/`](role_diff_out/).

## Headline

**P4 FIRED. The silhouette component of S1 is invalid, on all three controls.**
Per the pre-registration ("a control that fires invalidates the instrument, not
the hypothesis, and blocks scoring of P1–P3 until resolved"), P3 cannot be
scored on that statistic.

**P3 is NOT RESOLVED** on its registered primary either: the dip statistic is
flat and non-monotone across a 27.7× compression range, with no endpoint
separation, and at 7 scored areas the test has almost no power. This is not a
refutation — the pre-registration states in advance that a failure to reject
unimodality will not be reported as evidence *of* unimodality.

**Instrument validity was established first, and that is what saved the
round.** Two control-valid statistics (S2 directed asymmetry, S3 plug-in MI)
did behave correctly and do carry a reading, reported below as supporting
context rather than as the registered test.

## P4 controls — the blocking result

| control | dip p | silhouette | S2 `A` | S3 MI (bits) | verdict |
|---|---|---|---|---|---|
| C1 permuted labels, Gutenberg/BGE-M3 | 0.172 (no reject ✓) | **0.808** | 0.010 ✓ | 0.0001 ✓ | **silhouette FIRED** (bound: < 0.2) |
| C2 permuted labels, Ethics | 0.373 ✓ | **0.815** | 0.002 ✓ | 0.0001 ✓ | **silhouette FIRED** |
| C3 i.i.d. Gaussian, n=50k (reduced n, declared) | 0.782 ✓ | **0.461** | 0.013 ✓ | 0.0003 ✓ | **silhouette FIRED** |

### Diagnosis (two independent defects, both mine, neither about the hypothesis)

**(1) τ_a is confounded with area size.** Under a label permutation a row's
retrievers are label-random, so its transit fraction tends to
`1 − n_a/N`. Measured, exactly: C1's 60,000-row area gives τ_a = 0.564 against
a predicted `1 − 60000/150067 = 0.60`, while its 2,000-row areas give
0.93. C3 is the cleanest case — 13 near-equal areas, predicted
`1 − 1/13 = 0.923`, measured τ_a ∈ [0.918, 0.925] for all thirteen. So raw
per-area τ carries a size signal that has nothing to do with role, and the
prior registration had already flagged exactly this class of confound for `S_k`
("languages differ wildly in n… use the Robin Hood index as primary"). I
reintroduced it in a new statistic.

**(2) 2-means silhouette cannot decline to split.** With k fixed at 2 it always
returns a partition, and the silhouette does not penalize partitioning noise.
C1's 0.808 comes from a **12-vs-1** split — one size-outlier area against the
rest. C3's 0.461 comes from splitting thirteen essentially identical values
(`[9, 4]`). A statistic that scores 0.81 on a tight blob with one outlier
cannot certify bimodality.

The dip test did *not* fire on any control, which is the correct behaviour: a
tight blob is genuinely unimodal. So the primary statistic is sound in kind;
the corroborating one is not.

## P3 — the blind arm (Ethics/BGE-M3, coarse-graining tightened)

| rung | τ̄ | dip | dip p | S2 `A` | S3 MI |
|---|---|---|---|---|---|
| 1 — uncompressed 1024-d | 0.2242 | 0.1405 | 0.150 | 0.0483 | 1.0935 |
| 2 — PCA-384 | 0.2289 | 0.1428 | 0.120 | 0.0568 | 1.0749 |
| 3 — PCA-384 + TQ3, deployed 27.7× | 0.2276 | 0.1412 | 0.141 | 0.0484 | 1.0858 |

Registered requirement: "strictly monotone non-decreasing dip statistic across
the three, with the endpoints separated at p < 0.05."

Measured: 0.1405 → 0.1428 → 0.1412 — **not monotone** (rises then falls), and
the endpoints differ by 0.0007 (0.5%) with no separation. **P3 fails its
registered test.** But the honest verdict is *not resolved*, not *refuted*,
because with 7 non-ABSTAIN areas the dip test cannot detect bimodality that is
present, and the pre-registration says so in advance.

**Supporting context from the control-valid instruments.** τ̄ moves 0.2242 →
0.2289 → 0.2276 and `A` moves 0.0483 → 0.0568 → 0.0484 across a 27.7×
tightening of the observer's coarse-graining — both flat and both
non-monotone. These are not P3's registered statistic, but they passed their
controls, and they show **no amplification of role differentiation under tighter
compression.** That is a real if weakly-powered signal against P3's mechanism:
if the observer's non-invertible coarse-graining were driving the
differentiation, 27.7× should have moved something, and nothing moved.

## Arm G — discovery only, and denied the word "confirmed" by §1

| arm | τ̄ | dip | dip p | S2 `A` | S3 MI |
|---|---|---|---|---|---|
| BGE-M3 (emergent) | 0.2907 | 0.0733 | 0.821 | 0.0848 | 0.9847 |
| LaBSE (trained transfer) | 0.3892 | 0.0992 | 0.292 | 0.0974 | 0.6776 |

τ̄ reproduces the published 0.291 / 0.389 exactly, and the eligible-area count
(13, with 7 ABSTAIN) matches — the harness agrees with the existing instrument
before it was pointed at anything new.

Directionally, LaBSE shows **higher** directed asymmetry (0.0974 vs 0.0848) and
**lower** inter-area redundancy (0.678 vs 0.985 bits), which is what the
hypothesis predicts for the more differentiated encoder. Under the §1 rule this
is *consistent with* and nothing more: the qualitative answer on this pair was
known before these statistics were chosen. S3's direction is additionally
non-independent, as declared — it tracks τ̄ nearly monotonically.

Neither arm shows bimodality (dip p = 0.82, 0.29). At 13 areas that is
uninformative in both directions.

## Protocol deviation, disclosed

The pre-registration orders controls before scoring precisely so an invalid
instrument is caught before it touches a blind arm. The controls did fire
first — but the run was a single unattended script, and it completed the D1
rungs before I could halt it. The blind arm's numbers were therefore produced
by an instrument already known to be partly invalid.

Consequences, stated plainly:

- The **dip-based P3 verdict above stands**, because the dip passed all three
  controls. It is reported as measured.
- The **silhouette values on D1 are discarded**, not interpreted.
- **Arm D1's blindness is spent for any revised statistic.** A size-corrected
  role statistic cannot be validated on it — that would be fitting the
  instrument to data whose answer I have now seen. It needs a fresh blind arm:
  D2 (Ethics × LaBSE, one GPU re-encode) or arm C (`xbse`).

## What a corrected instrument would look like (for a future registration)

1. **Size-correct τ.** Score `τ_a − (1 − n_a/N)`, or a per-area z-score against
   the label-permutation null the controls already compute. Under a null this is
   ≈ 0 for every area regardless of size, which is what a role statistic must do.
2. **Let the clustering decline to split.** Replace the fixed-k silhouette with a
   test that can return "one cluster" — a gap statistic against a unimodal null,
   or a dip on the size-corrected coordinate.
3. **Require enough areas.** Seven is not enough for any unimodality test.
   Either pool to a coarser area map with more balanced strata, or accept that
   the bimodality question needs a corpus with more eligible languages.

## Arm D2 (Ethics × LaBSE) — fatal at the library default, RESCUED at LaBSE's real limit

> **Resolution (added after the first audit, see §"Rescued at 512" below): arm D2
> IS being built.** The blocking confound below is an artifact of the
> `sentence-transformers` default `max_seq_length = 256`, not of LaBSE's
> architecture, which supports 512. At 512 the confound drops to second order.
> The analysis at 256 is retained because it is why the arm nearly got thrown
> away, and because it is the configuration the Gutenberg LaBSE arm used.

### At the sentence-transformers default (256): fatal

Before spending the GPU pass, the per-language truncation rate was measured
(`trunc_audit.py`, 30,000-row uniform subsample of the committed seed-7 sample,
LaBSE's own tokenizer, CPU). LaBSE's `max_seq_length` is 256 tokens; Ethics
content averages 878 characters (median 934, max 1208), and tokens-per-character
varies by script by 3–4×. Result
([`role_diff_out/d2_truncation_audit.json`](role_diff_out/d2_truncation_audit.json)):

| language | n sampled | median tokens | % truncated | median content kept |
|---|---|---|---|---|
| english | 3738 | 232 | 20.8% | ~100% |
| greek | 2823 | 240 | 41.3% | ~100% |
| latin | 624 | 256 | 48.4% | ~100% |
| aramaic | 10643 | 331 | 70.8% | 77% |
| pali | 178 | 302 | 80.3% | 85% |
| hebrew | 11536 | 356 | 86.5% | 72% |
| sanskrit | 419 | 469 | 97.4% | **55%** |

Across the seven languages with ≥ 150 sampled rows the truncation fraction
spans **0.208 → 0.974**, a spread of 0.766.

**Why this disqualifies the arm rather than merely complicating it.** D1 uses
BGE-M3 (8192-token context, no truncation); D2 would use LaBSE at 256. So the
two encoders would not consume the same text, and the amount withheld would vary
from ~0% of English to 45% of Sanskrit. The object of study is *per-language
role differentiation*. Truncation is itself a role-relevant transformation —
shortening a passage makes it more generic, plausibly more central and more
retrieved — so a per-language comparison across D1/D2 would be confounded with
per-language text loss in a way that could **manufacture** the predicted
backbone structure. The replication predicate is satisfied on the input strings
and violated in effect.

This also explains, retrospectively, why the Gutenberg pair worked: it was built
as a deliberate paragraph-pack of 400–900 characters, short enough that both
encoders see whole passages. Ethics chunks are simply too long for LaBSE.

### Rescued at 512: the cap was a library default, not the architecture

The tokenizer warns at **512**, not 256. Confirmed on the model:
`max_position_embeddings = 512`, `tokenizer.model_max_length = 512`,
`position_embedding_type = absolute` — while `SentenceTransformer`'s default
`max_seq_length` for LaBSE is **256**. Half the model's context was being
discarded by a library default. Re-measured on the same 30k subsample
([`role_diff_out/d2_truncation_audit_256_512.json`](role_diff_out/d2_truncation_audit_256_512.json)):

| language | median tokens | % trunc @256 | % trunc @512 | median kept @512 |
|---|---|---|---|---|
| english | 232 | 20.8% | **0.0%** | 100% |
| greek | 240 | 41.3% | **2.3%** | 100% |
| latin | 256 | 48.4% | **0.0%** | 100% |
| aramaic | 331 | 70.8% | **5.4%** | 100% |
| pali | 302 | 80.3% | **0.0%** | 100% |
| hebrew | 356 | 86.5% | **11.7%** | 100% |
| sanskrit | 469 | 97.4% | **9.5%** | 100% |

Over the seven languages with ≥ 150 sampled rows: truncation fraction spread
collapses **0.766 → 0.117**, and the median-kept-fraction spread collapses
**0.45 → 0.00** (every scored language keeps its median row whole).
`classical_chinese` remains fully truncated (median 926 tokens) but has 3 rows in
the sample and ABSTAINs on `n_min` regardless.

That converts a signal-manufacturing confound into a second-order one, with **no
training and no architectural change**. Arm D2 is therefore being built at
`max_seq_length = 512`.

**Declared configuration deviation.** 512 departs from the ST default, and the
Gutenberg LaBSE arm (`gut_labse.npy`) was encoded at 256. The two LaBSE arms
therefore differ in configuration. The effect there is small — that rung is a
deliberate 400–900-character paragraph pack, mostly under 256 tokens — but it is
recorded, not assumed away.

**Why not extend LaBSE beyond 512.** Its position embeddings are *absolute*, so
they do not extrapolate; going further needs interpolation or a RoPE/ALiBi swap
plus continued pretraining. The objection is not cost but validity: D2's purpose
is to contrast a *published* trained-transfer encoder with an emergent one, and
retraining LaBSE makes the training a confound, collapsing the contrast into
"BGE-M3 vs a locally modified LaBSE".

**Consequences for the blind-arm problem.** Arm D1's blindness is spent
(protocol deviation above). Two usable arms remain, and they are complementary:

1. **Arm D2 (Ethics × LaBSE @ 512)** — being built. Contrasts encoder
   *families*; confounded by training data and dimension (768 vs 1024), as the
   prior registration already declared, and now by a small residual
   truncation asymmetry (≤ 11.7% of rows in the worst language).
2. **Arm C (`xbse` language-invariance objective on the BGE-M3 base)** — still
   the only arm that isolates the *objective*, holding architecture, tokenizer,
   context length and base weights fixed. It also has zero truncation asymmetry
   by construction, which is what the 256-cap episode shows is worth having.
   Blocked on the instance existing and clearing its own gate.

Arm D2 is built **but not scored**: the corrected role statistic (§"What a
corrected instrument would look like") is not yet registered, and scoring before
that amendment is frozen would spend this arm's blindness for nothing. The
artifact records `"scored": false` for that reason.

One GPU pass spent, after two CPU tokenization audits established it would not
be wasted.

## Standing conclusions

- The hypothesis is untouched: nothing here bears on whether role
  differentiation is constraint-induced. P1/P2 were never scored (they need the
  D2 re-encode and arm C).
- The one substantive empirical signal is **negative and control-valid**: two
  validated statistics are flat across 27.7× compression on the blind arm.
- The pre-registration did its job twice — the arm-G labelling rule stopped a
  known-answer arm from being read as confirmation, and the controls-first
  ordering caught an invalid statistic. The value was in the ordering, not in
  the result.

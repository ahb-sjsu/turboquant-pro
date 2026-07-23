# Pre-registration — multilingual stratified hubness (STRATA Phase-1, first run)

**Status: REGISTERED ⚪ (no stratified measurement exists yet).** Filed
2026-07-23. Predictions below are frozen as of this date; protocol amendments
are logged and may only *tighten* (bar discipline, applied to the reviewer's
own claims). Companion to [`HUBNESS_PRIMER.md`](HUBNESS_PRIMER.md) (phenomenon),
[`STRATA_RFC.md`](STRATA_RFC.md) §1–2 (instruments, ABSTAIN, area-map identity),
and, for arm C, the `xbse` language-invariant instance. Results land in
`RESULTS_multilingual_strata.md`; this file is never edited to fit them.

> **The questions.**
> **Q1** Does per-language hubness vary across a multilingual embedding
> space — and by which mechanism (density vs centrality)?
> **Q2** Is there a cross-lingual transit backbone (a de facto Area 0), and
> does its strength depend on whether multilinguality was *trained*
> (translation-ranking objective) or *emergent*?
> **Q3** Under the deployed compression operating point, does retrieval
> damage concentrate in particular languages while aggregate recall stays
> green? ("Trust the tail," spatial edition, on production settings.)

---

## 1. Materials

| corpus | encoder | dim | n | languages | public? |
|---|---|---|---|---|---|
| Ethics | BGE-M3 | 1024 | 2.4M | 37 | vectors/reports yes; text no |
| multilingual Gutenberg | LaBSE | 768 | 1M | long-tail (heavily EN-skewed) | yes — companion-notebook rung |
| LaBSE-199k | LaBSE | 768 | 199k | multilingual | yes |

Both encoders emit L2-normalized vectors scored by cosine ⇒ the
normalization×metric confound is controlled by construction. Dimensions
differ (768 vs 1024) ⇒ **reported, not hidden**; an optional controlled pair
compares both at matched PCA-384 via `PCAMatryoshka`.

**Replication-predicate gap (protocol requirement).** The encoder contrast
in P2 requires *equal text fingerprints with differing encoder digests*
(STRATA/xbse replication predicate). The existing corpora do **not** share
texts across encoders. Therefore one re-encode is REQUIRED before P2 can be
scored: encode the Gutenberg text set under BGE-M3 (preferred: public rung)
or the Ethics text set under LaBSE. Without it, P2 comparisons are
refused — the predicate exists precisely to refuse them.

**Arms.**
- **A** base BGE-M3 (Ethics; Gutenberg after re-encode)
- **B** LaBSE (Gutenberg, LaBSE-199k)
- **C** *(optional, unlocks the objective-isolated contrast)* xbse
  language-invariant instance on the same BGE-M3 base — same architecture,
  pooling, dim; only the training objective varies.

## 2. Area map

`tqp-area-map/1`, `assignment_rule: metadata-key language` where labels
exist; otherwise a langid model, whose **id, revision, and confidence
threshold enter the area-map profile** — the labeler is configuration.
Query assignment: corpus→corpus for the primary run, declared as such
(primer: hubness depends on who is asking); any real-query battery is a
separately-labeled secondary run, never pooled.

**Declared hazard — langid errors masquerade as transit.** A mislabeled row
is, by construction, a "cross-lingual" neighbor: label noise inflates τ and
manufactures fake backbone. Mitigation, fixed now: high-confidence labeling
threshold (rows below it are assigned to a reported `und` stratum, not
guessed); the labeler-confidence distribution ships in the report; a
sensitivity rerun at a stricter threshold accompanies any P2 claim.

## 3. Protocol (exact)

```bash
# anatomy, per arm
tqp anatomy --npy <corpus>.npy --by language --k 10 --seed 7
# hubdiff at the DEPLOYED operating point (PCA-384 + TQ3, 27.7×)
tqp hubdiff --exact exact_ids.npy --approx adc_ids.npy \
    --strata language --min-anti-recall 0.90 --min-stratum-n 2000
```

`k = 10` (matches recall@10 throughout the project). Seeds fixed and listed.
Estimator declared per report (`exact@sample` for anatomy; the ADC-bias rule
from the primer applies and is restated in every artifact). `n_min = 2000`,
`q_min = 500` (STRATA defaults). **ABSTAIN is expected** for tail languages
— on the Gutenberg set especially — and is recorded as a *result*
(§1.5 exercised on real data), never silently dropped.

**Cross-size comparability rule, pre-committed:** languages differ wildly in
n, and `S_k` grows with n; therefore per-language comparisons use the
**Robin Hood index and `frac_above_2k` as primary**, `S_k` secondary with
per-stratum n printed beside it. (Disposition item #3's companions, doing
the job they were added for.)

## 4. Registered predictions (frozen 2026-07-23)

**P1 — heterogeneity.** Under base encoders, per-language hubness varies
widely. *Operationalized:* among non-ABSTAIN languages, max/min per-language
Robin Hood ratio ≥ 1.5 **and** max/min `S_k` ratio ≥ 3. *Mechanism claimed:*
training-data volume and subspace density differ by orders of magnitude
across languages. *If it fails:* the encoder homogenizes geometry across
languages far more than expected — itself a publishable finding about
BGE-M3, and the reviewer eats it publicly.

**P2 — trained interlingua > emergent interlingua.** Cross-lingual transit
concentrates in a semantically central (translationese) region, and the
concentration is **stronger under LaBSE / arm C than under base BGE-M3**,
because an explicit translation-ranking (resp. language-invariance)
objective pulls languages onto a shared backbone that emergent
multilinguality only approximates. *Operationalized:* (i) top-decile-τ rows
have mean centrality above the population at declared significance;
(ii) the concentration statistic (share of top-decile-τ mass in the top 3
source languages, and its Gini) is higher for LaBSE/arm C than for BGE-M3
**on the same texts** (re-encode required; comparison refused otherwise).
*Scoring rule:* both (i) and (ii) must hold for P2 to count as confirmed;
(i) alone = partial.

**P3 — compression damage is linguistically concentrated (directional
hypothesis, weaker prior).** Per-language anti-hub recall at the deployed
27.7× operating point varies by language, with the worst strata among
low-resource languages, while aggregate recall stays within its historical
green band. *Operationalized:* min-over-strata anti-hub recall at least
0.10 below the aggregate, with the bottom-3 strata drawn from the bottom
half of languages by n. *Stakes:* a confirmation directly motivates STRATA
Phase-3 fragility-driven allocation; a refutation is genuinely good news
about the deployed system and gets reported with equal prominence.

**P4 — registered expectation, not a prediction:** the Gutenberg run
produces ABSTAIN strata, exercising the minimum-evidence path on real data.

## 5. Analysis plan

One strata-report JSON per (arm, corpus), with full provenance blocks
(area-map digest incl. labeler, estimator, seeds, thresholds). Comparisons
occur **only** where the replication predicate holds. Correlation fields are
Spearman (as shipped). No post-hoc metric substitutions: the statistics
named in §4 are the statistics scored; anything else is labeled exploratory.

## 6. Declared confounds & limitations

Corpus→corpus query distribution (primary run) · dimension differences
across encoders (reported; matched-PCA pair as control) · tokenizer and
truncation differences at long rows · langid noise (§2 hazard + mitigation)
· possible language↔topic correlation inside the Ethics corpus (languages
may not be topically exchangeable; noted, not correctable here — the
Gutenberg replication is the check).

## 7. Deliverables

Strata-report JSONs (publishable even where text is not: fingerprints,
skews, τ, recall-by-decile — no vectors, no text) · a public companion
notebook on the Gutenberg rung · `RESULTS_multilingual_strata.md` with the
§4 scoreboard, predictions beside measurements, confirmations and
embarrassments typeset identically · a workshop-paper skeleton if ≥2 of
P1–P3 resolve either way. Stated plainly: predictions published next to
their measurements are this project family's working recruitment mechanism.

## 8. Cost

An afternoon of CPU for anatomy/hubdiff over existing embeddings; one GPU
re-encode (Gutenberg × BGE-M3) to unlock P2; arm C additionally requires the
xbse language-invariance instance to exist and clear its gate first.

## 9. Amendment rule

§4 is frozen. Protocol clarifications append to a changelog at the foot of
this file; operationalizations may be *tightened* (thresholds made stricter)
with a dated entry, never loosened, never after seeing results for the
statistic in question. The results file cites this document by content
hash.

*Trust the tail, not the mean — and hold the reviewer to it too.*

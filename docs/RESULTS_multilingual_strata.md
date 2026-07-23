# Results — multilingual stratified hubness (STRATA Phase-1, first measured run)

**Scored 2026-07-23** against the frozen pre-registration
[`PREREG_multilingual_strata.md`](PREREG_multilingual_strata.md)
(sha256
`5c6f74f7cfa3cdf6539ed1fb10a938a37a6ebd2f0def46b835d4d668022d4b55` — the
file as committed at `3b02ae6`, i.e. *before* the dated 2026-07-23
materials-changelog entry, which postdates this scoring; predictions were
frozen before any stratified measurement existed).
Raw artifacts, committed beside this file:
[`results/strata-multilingual-2026-07-23/`](results/strata-multilingual-2026-07-23/)
(anatomy report, hubdiff report, aggregate hubdiff — `tqp-strata-report/1`,
no vectors, no text). Predictions and measurements appear side by side;
confirmations and embarrassments are typeset identically.

## Provenance (the meter meters itself)

| field | value |
|---|---|
| corpus | Ethics (BGE-M3, 1024-d), arm A |
| universe | 2,391,361 embedded rows, **40** language labels (prereg materials said 37 — the measured count governs) |
| sample | uniform N=350,000, seed 7 (uniform sampling preserves the corpus measure; per-language n falls where it falls) |
| battery | corpus→corpus (declared primary; primer: hubness depends on who is asking) |
| estimator | exact kNN on given vectors, k=10; vectors unit-normalized (Euclidean rank ≡ cosine rank) |
| area map | `tqp-area-map/1` digest `f72eededc967…`, assignment `metadata-key language` |
| dataset fingerprint | `350000x1024:float32:s1367:0a7477c4756b05dc` (stride declared: sampled hash) |
| thresholds | n_min=2000, q_min=500, τ=0.5, N=50 (fields of the report) |
| operating point (P3) | tqp `TQEIndex` PCA-384 + TQ3, measured **27.676×**, **single-pass ADC, no rerank**, build seed 42 |
| software | turboquant-pro 2.0.0a2 + STRATA Phase-1 (`209902f`); run log on Atlas `/archive/tqp_strata/` |

Eligible strata (7 of 14): hebrew 134,524 · aramaic 123,820 · english
44,929 · greek 32,289 · latin 7,038 · sanskrit 4,780 · pali 2,132.
ABSTAIN (7): old_norse_english 336 · arabic 112 · classical_chinese 35 ·
tamil 2 · french/georgian/indonesian 1 (cause `stratum_insufficient_n`).

## Scoreboard

### P1 — heterogeneity: **NOT CONFIRMED** ✗

Registered: among non-ABSTAIN languages, max/min Robin Hood ratio ≥ 1.5
**and** max/min S_k ratio ≥ 3.

Measured: Robin Hood 0.3447 (english) … 0.4469 (sanskrit) → ratio
**1.30 < 1.5**. S_k 2.68 (english) … 4.40 (greek) → ratio **1.64 < 3**.
Both legs fail; the cross-size rule was honored (RH primary, S_k printed
beside per-stratum n).

The registered fallback therefore stands as the finding: **BGE-M3
homogenizes per-language hubness geometry far more than predicted.** Seven
languages spanning 63× in sample size, three scripts, and ~3 millennia of
text land in a Robin Hood band of width 0.10 and all classify NSHA. The
reviewer eats it publicly, as promised.

### P2 — trained > emergent interlingua: **NOT SCORED** (refused by design)

The replication predicate (equal text fingerprints, differing encoder
digests) is unmet — no re-encode exists yet. The comparison is refused,
which is the predicate doing its job. Unlocks after the Gutenberg ×
BGE-M3 re-encode.

### P3 — linguistically concentrated compression damage: **NOT CONFIRMED** at the registered threshold ✗

Registered: min-over-strata anti-hub recall ≥ 0.10 below the aggregate,
worst strata among low-resource languages, aggregate in its green band.

Measured at the deployed operating point: aggregate recall@10 **0.8233**
(p05 0.60), aggregate anti-hub recall **0.7933**; min-over-strata anti-hub
recall **0.7355** (sanskrit). Gap vs aggregate recall@10 = **0.088**; vs
aggregate anti-hub = **0.058**. Under either reading of "the aggregate,"
the gap is **below the registered 0.10** — no post-hoc reading selection;
both are reported, both fail.

The *directional* component did land, and is recorded as exploratory
(sub-threshold, not a confirmation): the bottom-3 anti-hub strata —
sanskrit 0.7355, latin 0.7733, pali 0.7758 — are exactly the three
lowest-resource eligible languages, and per-stratum anti-hub recall orders
near-monotonically with resource level (english 0.8110 > aramaic 0.8041 >
hebrew 0.7874 > greek 0.7793 > pali 0.7758 > latin 0.7733 > sanskrit
0.7355). A magnitude-honest summary: the damage concentration is real but
roughly half the registered effect size at this operating point.

Green-band caveat, stated plainly: this run used the single-pass ADC path
(no rerank), declared in provenance. Historical single-pass readings in
this project (e.g. 0.592 ADC-only at the 1B fleet run, different corpus)
are not a calibrated band for this corpus; the per-stratum *contrast* is
the pre-registered object, and it is measured under one fixed operating
point for all strata.

CI-gate demonstration (tooling, not a prereg claim): at
`--min-anti-recall 0.90` all 7 eligible strata fail ⇒ exit 1 — the
min-over-strata gate fires on real production-shaped data.

### P4 — ABSTAIN strata occur on real data: **CONFIRMED** ✓

7 of 14 strata ABSTAIN with registered cause `stratum_insufficient_n`,
excluded from gating, reported and counted. The minimum-evidence path is
exercised exactly as designed (§1.5).

## Exploratory observations (not pre-registered; labeled as such)

- **All seven eligible strata classify NSHA** — local hubs, low global
  transit. No backbone-class area exists at this (corpus, query, k):
  under corpus→corpus querying, each language is its own hub economy.
- **Transit τ is where the languages differ**: aramaic 0.302 > hebrew
  0.214 > english 0.189 > latin 0.185 > greek 0.074 ≫ sanskrit 0.0024 >
  pali 0.0004. The hebrew↔aramaic pair (Sefaria mixed-language corpora)
  shows an order of magnitude more cross-language neighbor traffic than
  the Indic strata, which are hermetically self-contained. This is the
  natural target for the P2 backbone analysis once the predicate unlocks.
- Per-area mechanism evidence (Spearman, per doctrine non-gating):
  `corr(count, −d_k)` 0.48–0.76 with `corr(count, centrality)` 0.14–0.35
  across strata — density-flavored throughout, no centrality-artifact
  signature in any stratum.

## Amendments

None. §4 was not edited after registration; this file cites the prereg by
content hash above.

*Trust the tail, not the mean — and hold the reviewer to it too.*

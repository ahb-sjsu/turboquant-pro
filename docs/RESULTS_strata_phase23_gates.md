# Results — STRATA Phase-2 & Phase-3 measured gates

**Measured 2026-07-23/24.** Artifacts:
[`results/strata-phase23-gates-2026-07-24/`](results/strata-phase23-gates-2026-07-24/).
Corpora: the Ethics 350k seed-7 sample (Gate A/A′; deployed operating
point PCA-384+TQ3, 27.676×, single-pass ADC) and the public paired
Gutenberg rung, BGE-M3 arm (Gates B/B2; 150,067 rows, 20 languages).
Approved claim shape throughout: "Δ measured in this artifact on this
corpus." Failures are typeset like passes; three of the four gates below
exist *because* an earlier gate failed.

## Gate A — CSLS at the compressed layer, scored against cosine truth: **FAILED, by design error**

As pre-declared, CSLS-rescored ADC rankings were scored against
UNCORRECTED exact-cosine ground truth. Anti-hub recall "dropped" ~0.32 in
every stratum (min-over-strata 0.437 at w=1.0, 0.571 at w=0.5). Diagnosis:
the gate conflated two different questions — CSLS is a different ranking
objective, so scoring it against cosine truth measures disagreement, not
damage. The gate design, not the remedy, failed first. Kept on the
scoreboard as the reason A′ exists.

## Gate A′ — the two questions separated (declared before running)

**A′1 — does CSLS reduce hubness on the exact graph? YES.**
Ethics 350k, exact layer, w=1.0:

| statistic | cosine graph | CSLS graph | Δ |
|---|---|---|---|
| count skew | 3.970 | 3.177 | −20% |
| count max | 287 | 213 | −26% |
| Robin Hood index | 0.372 | 0.261 | −30% |
| frac(N_k > 2k) | 0.117 | 0.079 | −33% |

The r_k scalar earns its trailer: measured hub-tail reduction on this
corpus at every statistic.

**A′2 — does COMPRESSED CSLS reproduce EXACT CSLS? NO — gate failed.**
Min-over-strata anti-hub recall 0.663 vs the 0.90 bar; all 7 strata fail;
overall per-stratum recalls 0.62–0.69 — substantially WORSE fidelity than
the uncorrected path (0.76–0.84). The reading: CSLS promotes exactly the
isolated, fine-distinction rows that 27.7× quantization damages first, so
the corrected ranking is *harder* to reproduce from compressed codes than
the uncorrected one. **Phase-2 consequence (normative for the efficacy
claim):** the `hubness-scalar/1` trailer semantics stay PROVISIONAL;
CSLS belongs at the exact/rerank layer (rescoring reconstructed or
original candidates), not in single-pass compressed-domain scoring at
this operating point. No unconditional improvement claim is made.

## Gate A″ — CSLS at the rerank layer: **FAILED — and the failure is diagnostic**

Declared: ADC candidates k=51 (deployed single-pass), rerank = cosine on
RECONSTRUCTED candidates − 1.0·r_k, scored against exact-CSLS truth, same
0.90 bar. Measured: min-over-strata anti-hub recall **0.6627 — identical
to A′2 to four decimals**, per-stratum values matching within ±0.0001.

The identity is the finding: moving CSLS from compressed-domain scoring to
reconstruction rerank changed *nothing*, so the bottleneck is not the
scoring layer — it is **candidate coverage**. Exact-CSLS's true neighbours
(isolated, low-r_k rows) frequently do not appear in the plain ADC top-51
at all; no rerank scheme can recover a candidate that was never fetched.
Fidelity to a hubness-corrected ranking at this operating point is
candidate-limited, not scoring-limited.

**Phase-2 deployment guidance, updated:** CSLS-corrected retrieval under
27.7× compression requires either substantially deeper candidate lists
(a measured cost/fidelity trade, unexplored) or correction applied where
candidates are generated — not just where they are ranked. The
`hubness-scalar/1` trailer semantics remain PROVISIONAL; the trailer's
measured value today is A′1's exact-layer hub-tail reduction and the
poisoning-monitoring signal, not compressed-path retrieval correction.

## Gate B — fragile-first greedy allocation: **FAILED, instructively**

Uniform 3-bit baseline: min-over-strata anti-hub 0.7251 (spanish). The
fragile-first allocator upgraded all eight of its targets (+0.004 to
+0.034) — and cut its unprotected donors below the old floor (greek
−0.062, latin −0.053 → 0.7118 < 0.7251). It flattened the distribution
and LOWERED the minimum. Min-over-strata is not improved by spending on
the weak; it is improved by not digging under the floor.

## Gate B2 — max-min allocation with protected donors: **PASS**

Redesigned allocator (declared rules: recipients = fragile half; donors
need headroom ≥ 0.06 over the baseline minimum — the one-bit penalty Gate
B measured; one donation step max; unfundable upgrades revert their
donations). Same 3.0 bits/row budget (achieved 2.995):

- **min-over-strata anti-hub: 0.7251 → 0.7697** (+0.045), no stratum
  below the old minimum, verdict PASS under the pre-declared rule.
- Every recipient-half stratum improved (spanish +0.047, swedish +0.054,
  italian — the one funded 4-bit upgrade — +0.050); protected latin
  IMPROVED (+0.035) instead of being cut; donors paid at most −0.014
  (greek), and two donors (chinese, esperanto) improved despite
  downgrades.
- The 3-bit-unchanged strata improving (+0.03–0.05) reveals a real
  mechanism: **cross-area score interference**. The global top-k merges
  ADC scores across areas; downgrading high-headroom areas reduces their
  rows' ability to steal candidate slots from other areas' true
  neighbours. Allocation is not per-area-local — a finding the uniform
  baseline could not have shown.
- Structural limitation, stated plainly: with this corpus's donor
  capacity (~8k rows eligible), the largest fragile strata (spanish 10k,
  finnish/french/german 15k) were individually unfundable; the allocator
  funded what it could and the floor still rose. On donor-scarce corpora
  the safe answer converges toward uniform — that convergence is a
  feature.

## Allocator defects found by the gates (both fixed + regression-tested)

1. Fragile-first greedy lowers the floor (Gate B) → replaced by
   `allocate_max_min`.
2. First `allocate_max_min` leaked donations when a recipient was
   unfundable (donors cut, nobody upgraded — strictly worse than uniform)
   and gave up instead of trying cheaper recipients (first B2 attempt) →
   donations now revert, iteration continues.
3. `StratifiedIndex` crashed on areas thinner than the PCA dim → per-area
   dim now clamps and declares itself in `area_codec_params`.

## Phase-gate status after measurement

- **Phase 2:** r_k trailer ratified (§9a) · hubness reduction MEASURED
  (A′1) · compressed-domain application REFUTED (A′2) · rerank-layer
  application REFUTED at candidate depth 51 (A″ — candidate-limited, not
  scoring-limited); remedy scope narrowed to exact-layer application and
  monitoring; open knob: candidate depth vs fidelity trade.
- **Phase 3:** identity fields + refuse-on-mismatch shipped and tested ·
  the measured capacity/recall trade on a public corpus EXISTS (Gate B2:
  floor +0.045 at equal budget) — the §4 gate is met, with the
  cross-area-interference caveat recorded.
- **Phase 4:** normative core (extended contract key, §5 stale sets)
  implemented + tested; catalog wiring remains 2.2-class.

*Trust the tail, not the mean — and when you spend bits, don't dig under
the floor.*

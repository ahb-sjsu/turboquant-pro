# Pre-registration: do the two Gutenberg encoders disagree about which language is denser, beyond book-sampling noise?

**Status: REGISTERED ⚪, FROZEN on filing 2026-10-29** at the author's instruction. Nothing in §3 or
§4 has been computed. Harness: [`benchmarks/strata_density_reversal.py`](../benchmarks/strata_density_reversal.py)
(sha256 `fc5ab622…b46ab`), committed with this document. Smoke-tested on synthetic data only:
the weighted median matches `np.median` of the expanded multiset exactly, a shared-order planted
corpus gives 0 clear reversals and a quiet gate, and a fully reversed one gives 57 of 78.

**Why this exists.** The Gutenberg density test ([`PREREG_density_asymmetry_gutenberg.md`](PREREG_density_asymmetry_gutenberg.md),
results in [`benchmarks/RESULTS_density_asymmetry_gutenberg.md`](../benchmarks/RESULTS_density_asymmetry_gutenberg.md))
found flow running toward the denser language on both encoders. An unregistered look at the stored
results afterwards (§0) found that the two encoders rank the 13 languages by density in nearly
unrelated orders. If that holds, "which language is denser" is a property of the encoder and corpus
together, not of the language. That is a two-consumer verdict inversion, in the sense of the
observation-theory campaigns' CSI and QoT flip registrations. It is not the compression flip of
`observation-theory-campaigns/encyclopedia/entries/flip.md`: there are no matched bits, no
reconstruction comparison and no anti arm.

The look had no noise floor. Each language's density is a median over 2,000 to 60,000 passages,
drawn about 40 at a time from 50 to 1,545 books. Some reversals are differences in the third decimal
place and may be sampling noise. This document fixes, before any interval is computed, how a
reversal must clear book-level sampling noise to count.

## 0. What is known, and what is blind

**Known to the analyst** from the exploratory look of 2026-10-29 (`rev.py` and `rev2.py` in the Atlas
run directory, which read `gut_score.json` only):
- The point estimates of all 13 per-language densities on both encoders, and so the full list of
  reversed pairs. **44 of 78 pairs reverse.** Kendall τ between the encoders' orders is −0.128.
  Every pair-level difference in density is known.
- 16 of the 44 reversed pairs have |difference| > 0.01 on both encoders. The smallest include
  Dutch–Swedish (−0.0012 / +0.0013), French–Swedish (+0.0007 / −0.0116) and Greek–Latin (+0.0005 / −0.0202).
- The same-book-excluded variant reverses 44 of 78 pairs. The border-row variant reverses 51 of 78
  (τ −0.308).
- The net flow on every pair, on both encoders. On the 44 reversed pairs, the net flow flips
  direction between the encoders on 34. Flow toward the denser side is Λ 0.71 for G1 and 1.00 for G2.
- Books per language and rows per book (Appendix).

**Never computed:** any resampled density, any standard error, any interval. Every registered
prediction in §3 depends on those.

The predictions are therefore not blind to the point estimates. They are blind to the noise. N1 and N2
ask whether point estimates already seen survive a noise floor fixed here. They do not test whether
reversals exist at all. N2's threshold was chosen with the point estimates in view, and is disclosed
as such.

## 1. Materials

There is no new embedding and no new kNN. Everything comes from artifacts the frozen Gutenberg test
used, on Atlas:
- **Per-row density.** `s(x)` is the cosine to the 10th stored neighbour. It is cached by the frozen
  harness as `/archive/experiments/hubness-density-ratio/sdens_gut_G1.npy` (BGE-M3) and `sdens_gut_G2.npy`
  (LaBSE). The same-book-excluded variants are `sdens_gut_G{1,2}_xbook.npy`.
- **Language and book per row.** These come from `/archive/tqp_strata/gutenberg_pair/labels.txt` and
  `passages.jsonl`, loaded with the frozen harness's `load_language` and `load_books`.
- **Net flow counts `C`.** These are in `gut_score.json`, for descriptives only.

The two encoders read the same 148,224 passages, so every resample is paired: one draw of books
serves both encoders.

## 2. Statistics (defined before computation)

- **Density of language a on encoder e.** `ρ_e(a)` is the median of `s_e(x)` over rows of `a`. This is
  the frozen statistic, by `nan_median`.
- **Pair difference.** `d_e(a,b) = ρ_e(a) − ρ_e(b)` for each of the 78 unordered pairs (a before b in
  the frozen language order).
- **Resampling unit: the book.** Passages from one book are correlated, and the book is the unit the
  corpus was sampled on. One bootstrap draw resamples, independently within each language, as many
  books as that language has, with replacement. All rows of each drawn book come along, with
  multiplicity. Both encoders' medians are computed on the same draw.
- **Draws.** B = 2,000, seed 20261104.
- **Standard error.** `SE_e(a,b)` is the standard deviation of `d_e(a,b)` over the B draws.
- **Clear sign.** `d_e(a,b)` has a clear sign when `|d_e(a,b)| > z* · SE_e(a,b)`, where
  `z* = Φ⁻¹(1 − 0.05/156) = 3.42`. That is a two-sided Bonferroni bound over 78 pairs at family
  level 0.05. The normal interval is used because the Bonferroni tail is too far out for 2,000
  percentile draws to resolve.
- **Clear reversal.** Pair (a,b) is a clear reversal when both `d_1(a,b)` and `d_2(a,b)` have a clear
  sign and the signs differ.
- **Why this controls error.** Suppose the two encoders shared one true order. A clear reversal
  would then need at least one encoder's difference to clear zero on the wrong side. At each
  encoder and pair, that has probability at most 0.05/156 under the normal approximation. Over 78
  pairs and 2 encoders, the chance of any false clear reversal is at most 0.05. The approximation
  is checked by the gate (§4), not assumed.
- **Rank agreement.** Kendall τ between the encoders' 13-language density orders. It is computed on
  each draw, and its 95% interval is the 2.5th to 97.5th percentiles over the B draws.

## 3. Registered predictions

All three are scored only if the gate (§4) is quiet. Each passes or fails on its own. There is no
combined family verdict beyond N1, which carries the error control of §2.

| ID | Prediction | Passes when |
|---|---|---|
| **N1** | The two encoders disagree on which language is denser beyond book-sampling noise | ≥ 1 clear reversal |
| **N2** | The disagreement is widespread, not one or two pairs | ≥ 10 clear reversals |
| **N3** | The encoders' density orders are far from agreement | Upper end of τ's 95% interval < +0.5 |

**N2's threshold is informed** by the known point estimates: 16 reversed pairs differ by more than
0.01 on both encoders. Ten is a guess at how many of those survive. It is not derived from any
computed noise.

**N3** is stated as an upper bound because τ's point estimate (−0.128) is known. The test is whether
book noise could put the orders in substantial agreement.

**Reading.** If N1 passes: on these data, the density order of the 13 languages belongs to the
encoder, not to the language, beyond book-sampling noise. If N1 fails, the exploratory reversals
are not distinguishable from sampling noise at this level, and no claim of encoder-relative density
is made. Neither outcome says which encoder is right. Neither bears on the registered Gutenberg
result (flow toward the denser language, within each encoder), which stands as scored.

## 4. Gate (computed first, separate invocation)

The gate checks that the clear-reversal rule does not produce reversals when there is only one
channel. This is a same-channel null.

- **Procedure.** For each encoder separately, split each language's books at random into two halves,
  A and B. Treat half A and half B as if they were two encoders: compute `ρ` on each half, bootstrap
  each half's books independently (B = 2,000), and count clear reversals between half A and half B by the §2 rule.
- **Replicates.** 20 random half-splits per encoder, seed 20261105.
- **Fires** if, for either encoder, 3 or more of the 20 splits show at least one clear reversal.
  Under the rule's nominal ceiling (5% per split), 3 or more of 20 happens with probability 0.075.
  The Bonferroni rule should sit well under its ceiling, so firing means the noise floor is too
  narrow.
- **If the gate fires:** nothing in §3 is scored. Any repair is a dated amendment made before
  scoring.

Halves have half the books, so their standard errors are larger than the full data's. The gate
tests the rule's calibration, not its power at full size.

## 5. Descriptive readouts (outside the registered predictions)

- The same rule applied to the same-book-excluded densities (`_xbook`) and to border rows (rows with
  at least one out-edge to another language). Count of clear reversals and τ interval for each.
- For each clear reversal: the net flow `C_ab − C_ba` on each encoder, and whether each runs toward
  that encoder's denser language.
- **The G1 anomaly.** G1's Λ toward-denser is 0.71 on reversed pairs against 0.94 on agreeing ones.
  List the reversed pairs where G1's net flow runs against G1's density, with their SE-scaled
  density differences, to see whether they are the noisy ones.
- Per-language bootstrap SE of `ρ_e(a)` on both encoders.

## 6. Limitations

- **Not blind to point estimates (§0).** This test asks only whether known estimates survive noise.
- **Only book-sampling noise is covered.** `s(x)` is treated as fixed per row. Rebuilding the kNN
  index on a resampled corpus would move every row's 10th neighbour. That noise is not in the floor.
  The floor is therefore a lower bound on the true uncertainty, and N1 and N2 could pass too easily.
  The gate checks calibration under the same simplification, so it does not cure this.
- **Two encoders.** A clear reversal shows the order depends on the encoder. Two encoders cannot
  show how many orders there are, or whether one is closer to some ground truth. No ground truth for
  "density of a language" is defined.
- **Same corpus.** Gutenberg's language strata differ in genre and era (the date map in the
  Gutenberg results). An encoder that is sensitive to genre and one that is not could produce
  reversals for that reason alone. That is still encoder-relative density, but its source is not
  identified here.

## 7. Cost

Atlas, cores 12–23, the agi-hpc venv, CPU only, under 20 threads. Each draw needs 13 medians over
about 148,000 values per encoder. The gate (2 encoders × 20 splits × 2 halves × 2,000 draws)
dominates. Estimated under an hour in total. This is not measured.

## 8. Decisions at freezing

The author approved the draft as written ("write the script, freeze it, then run", 2026-10-29), so the
draft's values stand:
- N2's threshold is 10 and N3's bound is +0.5.
- B = 2,000 draws with the normal-theory interval of §2.
- This test runs now. The kNN-rebuild noise of §6 stays an open limitation, not measured here.

## Appendix: books per language (from labels and `passages.jsonl` only)

| Language | Rows | Books | Largest book's share | Median rows per book |
|---|---|---|---|---|
| dutch | 10,000 | 253 | 0.004 | 40 |
| english | 60,000 | 1,545 | 0.001 | 40 |
| esperanto | 2,332 | 59 | 0.017 | 40 |
| finnish | 15,000 | 377 | 0.003 | 40 |
| french | 15,000 | 380 | 0.003 | 40 |
| german | 15,000 | 386 | 0.003 | 40 |
| greek | 4,000 | 100 | 0.010 | 40 |
| hungarian | 3,985 | 100 | 0.010 | 40 |
| italian | 6,615 | 167 | 0.006 | 40 |
| latin | 2,189 | 63 | 0.018 | 40 |
| portuguese | 2,103 | 62 | 0.019 | 40 |
| spanish | 10,000 | 252 | 0.004 | 40 |
| swedish | 2,000 | 50 | 0.020 | 40 |

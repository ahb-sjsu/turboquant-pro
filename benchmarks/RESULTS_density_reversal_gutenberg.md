# The two Gutenberg encoders disagree about which language is denser: noise-floor result

**Registered before computation:** [`docs/PREREG_density_reversal_gutenberg.md`](../docs/PREREG_density_reversal_gutenberg.md),
frozen in commit `609ce48`. It has no amendments.
**Script:** [`strata_density_reversal.py`](strata_density_reversal.py), sha256 `fc5ab622…b46ab`.
**Raw output:** `results/density_rev_gate.{json,log}` (sha256 `7f379481…`) and
`results/density_rev_score.{json,log}` (sha256 `442786eb…`).
**Setup:** the same 148,224 passages in 13 languages that the Gutenberg density test used, with the
per-row densities that test cached (`sdens_gut_G{1,2}[_xbook].npy`, cosine to the 10th stored
neighbour). G1 is BGE-M3 and G2 is LaBSE. Books were resampled within each language, 2,000 draws, one
draw shared by both encoders. Run 2026-10-29 on Atlas cores 12–23.

## Verdicts

The gate stayed quiet, and all three registered predictions pass.

| test | verdict | result |
|---|---|---|
| **Gate.** Same-encoder half-splits of each language's books, fire if 3 or more of 20 splits show a clear reversal | quiet | 0 of 20 splits on G1, 0 of 20 on G2; no split had even one |
| **N1.** At least 1 clear reversal | **PASSES** | 17 |
| **N2.** At least 10 clear reversals | **PASSES** | 17 |
| **N3.** Upper end of Kendall τ's 95% interval below +0.5 | **PASSES** | τ = −0.13, interval [−0.26, +0.03] |

A pair is a clear reversal when each encoder's density difference exceeds 3.41 bootstrap standard
errors and the two differences have opposite signs. 3.41 (3.414; the prereg rounds it to 3.42) is the two-sided Bonferroni bound over 78
pairs at 0.05, so under a shared true order the chance of any false clear reversal is at most 5%.

## What this shows

**The density order of the languages depends on the encoder.** 25 of the 78 pairs have a clear
sign on both encoders. They disagree on 17 and agree on 8. τ's interval excludes any substantial
agreement, and its point estimate is slightly negative.

| language | G1 density (SE) | G1 rank | G2 density (SE) | G2 rank |
|---|---|---|---|---|
| dutch | 0.6188 (0.0016) | 10 | 0.6460 (0.0013) | 6 |
| english | 0.6275 (0.0005) | 4 | 0.6613 (0.0008) | 3 |
| esperanto | 0.6321 (0.0029) | 3 | 0.6281 (0.0026) | 11 |
| finnish | 0.6246 (0.0008) | 6 | 0.6240 (0.0010) | 12 |
| french | 0.6208 (0.0013) | 8 | 0.6331 (0.0011) | 10 |
| german | 0.6247 (0.0013) | 5 | 0.6526 (0.0015) | 5 |
| greek | 0.6435 (0.0029) | 1 | 0.6427 (0.0023) | 9 |
| hungarian | 0.6241 (0.0020) | 7 | 0.6211 (0.0025) | 13 |
| italian | 0.6087 (0.0020) | 13 | 0.6606 (0.0015) | 4 |
| latin | 0.6431 (0.0032) | 2 | 0.6629 (0.0052) | 2 |
| portuguese | 0.6181 (0.0034) | 11 | 0.6439 (0.0035) | 8 |
| spanish | 0.6128 (0.0011) | 12 | 0.6650 (0.0011) | 1 |
| swedish | 0.6201 (0.0026) | 9 | 0.6447 (0.0024) | 7 |

Spanish and Italian drive most of the reversals: G1 places them near the bottom and G2 near the top.
Greek, Esperanto, Finnish and Hungarian move the other way. The clear reversals, strongest first by
the smaller of the two z-scores:

| pair (a, b) | ρ(a) − ρ(b), G1 / G2 | z, G1 / G2 |
|---|---|---|
| finnish, spanish | +0.0118 / −0.0410 | +9.0 / −27.6 |
| greek, spanish | +0.0308 / −0.0222 | +10.0 / −8.5 |
| finnish, italian | +0.0159 / −0.0366 | +7.4 / −20.3 |
| german, spanish | +0.0119 / −0.0124 | +7.1 / −6.8 |
| esperanto, italian | +0.0234 / −0.0325 | +6.6 / −10.8 |
| greek, italian | +0.0348 / −0.0179 | +9.9 / −6.5 |
| esperanto, spanish | +0.0193 / −0.0369 | +6.3 / −13.3 |
| english, greek | −0.0160 / +0.0186 | −5.5 / +7.5 |
| hungarian, italian | +0.0154 / −0.0395 | +5.3 / −13.4 |
| french, italian | +0.0121 / −0.0275 | +5.1 / −15.0 |
| hungarian, spanish | +0.0113 / −0.0439 | +4.9 / −16.1 |
| french, spanish | +0.0080 / −0.0318 | +4.8 / −20.1 |
| dutch, esperanto | −0.0132 / +0.0179 | −4.1 / +6.1 |
| dutch, italian | +0.0101 / −0.0146 | +4.0 / −7.2 |
| german, italian | +0.0160 / −0.0080 | +6.7 / −3.8 |
| german, greek | −0.0188 / +0.0099 | −5.9 / +3.6 |
| italian, swedish | −0.0114 / +0.0159 | −3.5 / +5.7 |

Of the 44 pairs that reversed in the unregistered look, 17 survive the floor. The other 27 are not
distinguishable from book-sampling noise at this level.

## Descriptive readouts (no confirmatory use)

- **Flow follows each encoder's own density on the clear reversals.** Net flow runs toward the
  encoder's denser language on 16 of 17 pairs for G1 and 17 of 17 for G2. The net flow direction
  itself flips between the encoders on 16 of 17.
- **The G1 anomaly from the exploratory look is mostly noise.** G1's net flow ran against G1's own
  density on 9 of the 44 point-reversed pairs. On 7 of the 9, G1's density difference is below the
  floor (|z| ≤ 3.3). Two are not:
  - **English vs Greek** is a clear reversal (z −5.5 / +7.5). G1 rates Greek denser, yet G1's net
    flow runs 885 edges from Greek into English.
  - **Esperanto vs French** is clear on G1 (z +3.6) but not on G2 (z −1.8). G1's net flow there is
    small, 94 edges from Esperanto into French.
- **Same-book-excluded densities:** 19 clear reversals of 44 point reversals, τ −0.13, interval
  [−0.26, +0.03].
- **Border rows only** (rows with an out-edge to another language): 19 clear reversals of 51 point
  reversals, τ −0.31, interval [−0.44, −0.13].

## Limitations

- **Not blind to the point estimates.** The 44 point reversals were seen before registration. This
  test establishes only that 17 of them survive book-sampling noise. N2's threshold was chosen with
  the point estimates in view.
- **The floor covers book sampling only.** Each passage's density is held fixed. Rebuilding the kNN
  index on resampled data would move every passage's 10th neighbour, and that noise is not included.
  17 is therefore an upper bound on what a full floor would keep. The clear reversals sit at 3.5 to 28
  SE on both encoders, so the headline is unlikely to fall, but this is not measured.
- **Two encoders cannot say which order is right.** No ground truth for the density of a language
  is defined. The result shows the order depends on the encoder, nothing more.
- **The source of the disagreement is not identified.** The language strata differ in genre and era.
  An encoder sensitive to genre and one that is not could produce these reversals.

## Reading for observation theory

This is a two-consumer verdict inversion. The same passages under the same language labels give
opposite answers to "which language is denser", and so to "which way neighbours flow between them",
depending on the encoder that reads them. It is the same kind of outcome as the CSI and QoT flip
registrations in observation-theory-campaigns. It is not the compression flip of that repo's
encyclopedia entry: there are no matched bits, no reconstruction comparison and no anti arm.

It does not touch the registered Gutenberg result
([`RESULTS_density_asymmetry_gutenberg.md`](RESULTS_density_asymmetry_gutenberg.md)). Within each
encoder, flow runs toward the denser language, and that stands as scored. This result adds that
"denser" is the encoder's judgement, not the language's.

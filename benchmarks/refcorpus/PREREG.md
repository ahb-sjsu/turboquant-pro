# Pre-registration — procedural reference corpus (RC-1)

**Registered before fitting or generating anything.** Discipline follows
*Geometric Methods* (Book 1): posited/measured separation, admission filters
fixed in advance, frozen nulls, misses reported as misses.

## The claim to be tested

**RC-1.** A corpus defined by a deterministic generator, fitted to a real
embedding corpus, can be made *geometrically indistinguishable from real
embeddings on the properties that govern nearest-neighbour search*, at any
scale, without materializing or distributing the vectors.

If RC-1 passes, a procedural 1T reference corpus (+ published exact ground
truth) is scientifically legitimate. **If it fails, it is a random-number
benchmark and must not be published as a reference corpus.** No reframing
after the fact.

## Reference (target) corpus

`/archive/tqp_real/wiki1024` — Cohere Wikipedia Embed-V3, 1024-d fp32, real
passages, real held-out queries. Second target once acquired: BIGANN/SIFT1B
(128-d uint8), which additionally has published ground truth.

## Frozen nulls (controls that must FAIL the gates)

1. **iid Gaussian**, matched dimension and global scale.
2. **Per-feature shuffle** of the real corpus — preserves every marginal,
   destroys the joint structure.
3. **Low-rank + noise** at matched effective rank only (our current seeded
   synthetic corpus) — expected to pass some gates and fail others; this is
   the honest baseline the new generator must beat.

A gate that the nulls also pass is not measuring anything and is struck from
the battery (reported, not silently dropped).

## Admission filters (thresholds fixed NOW, before any fit)

Measured on n=200k subsamples, k=100 neighbours, 5 seeds; report median and
IQR. Generated corpus must match the real target within:

| # | diagnostic | statistic | tolerance vs real |
|---|---|---|---|
| G1 | intrinsic dimension | two-NN MLE (Facco/Levina-Bickel) | ±15% |
| G2 | intrinsic dimension | ball-growth slope | ±20% |
| G3 | spectral profile | effective rank (participation ratio) | ±15% |
| G4 | spectral profile | dims for 90% energy | ±20% |
| G5 | distance concentration | relative contrast at k=100 | ±10% |
| G6 | hubness | skewness of k-occurrence N_k | ±25% |
| G7 | local ID spread | IQR of per-point two-NN LID | ±25% |
| G8 | angle/radius split | ρ(angular, full) and norm CV | ±15% |
| G9 | neighbourhood overlap | mean Jaccard of k-NN under PCA-256 vs full | ±10% |

**Pass = ≥7 of 9 gates within tolerance, AND G1/G5/G6 (the three that govern
ANN hardness) all within tolerance.** Nulls are expected to fail ≥4.

## Secondary, non-gating (reported for the record)

Two-sample energy distance on subsamples; δ-hyperbolicity growth-law R²
(the growth-law form, not the Gromov-δ implementation — the latter was
found buggy in prior work); persistent-homology H0/H1 summaries at n=5k.

## What a pass licenses / does not license

Licenses: using the generator as a *systems* benchmark corpus — scale, I/O,
recall mechanics, storage economics.
Does **not** license: claims about semantic retrieval quality, downstream
task accuracy, or that the corpus carries meaning. Vectors have no
semantics; only geometry is claimed.

## Analysis plan (fixed)

`corpus_geometry.py` computes the battery; run on target, nulls, and each
candidate generator. Fit uses a held-out split: fit on half the real corpus,
gate against the other half, so the generator cannot memorize the sample it
is scored on.

## Deviations

Any deviation from this document is recorded here with a date and reason,
before results are read.

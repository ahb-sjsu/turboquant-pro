# What the fleet corpus actually is, measured

Recall has been flat across the fleet runs at 0.999, 0.9988 and 0.9986 from one
billion rows to one hundred billion. The 1T run adds a fourth point at a cost of
27.3 TiB held for about six weeks. This note measures the corpus those numbers
were taken on, because the value of the fourth point depends entirely on whether
the corpus gets harder as it grows.

Measured 2026-08-03 on Atlas with openvector-bench's own geometry battery, so
the numbers are directly comparable with the real-embedding targets that battery
was built against. Raw data in
[`benchmarks/fleet/results/fleet_corpus_geometry.json`](../../benchmarks/fleet/results/fleet_corpus_geometry.json).
Generator copied verbatim from `benchmarks/fleet/fleet_common.py`.

## The corpus

`gen_block(g)` draws a fresh rank-16 Gaussian basis per shard, projects Gaussian
coefficients through it with a linear decay from 1.0 to 0.3, and adds 5% noise.
Ambient dimension is 32. It is the low-rank recipe, and it is the same recipe
openvector-bench's RC-1 battery already rejects as a stand-in for real
embeddings.

## Measurements

Growing at a fixed shard count, which is how a real corpus grows:

| rows | intrinsic dim | eff rank | dims90 | hubness k10 | k30 | k100 |
|---|---|---|---|---|---|---|
| 25,000 | 15.42 | 20.47 | 21 | 1.34 | 0.99 | 0.75 |
| 100,000 | 16.98 | 20.36 | 21 | 2.40 | 1.51 | 1.00 |
| 200,000 | 16.76 | 20.40 | 21 | 3.24 | 2.01 | 1.28 |

Growing the way the fleet actually grows, adding rows as new shards so every
added row lands on a new random basis:

| rows | shards | intrinsic dim | eff rank | dims90 | hubness k10 | k30 | k100 |
|---|---|---|---|---|---|---|---|
| 25,000 | 5 | 15.77 | 22.14 | 22 | 1.36 | 0.99 | 0.72 |
| 100,000 | 20 | 16.23 | 28.61 | 27 | 2.33 | 1.51 | 0.97 |
| 200,000 | 40 | 15.81 | 30.39 | 28 | 3.26 | 1.97 | 1.19 |

Real Cohere Embed-V3 wiki1024 for comparison: intrinsic dimension 52 to 61,
effective rank about 190 in 1024 ambient dimensions.

## What this says

**The corpus is a 16-dimensional object and stays one at every scale.**
Intrinsic dimension reads 15.4 to 17.0 everywhere and does not move with n. That
is exactly the recipe's nominal rank, so the estimator is reading the truth
rather than missing structure. Real embeddings sit three and a half times
higher. Intrinsic dimension is the main thing that makes nearest-neighbour
search hard, so the flat 0.999 is in large part a statement about a corpus where
the problem is easy.

**Growing by adding shards adds the kind of complexity that does not matter.**
This is the sharpest result and it is visible only because both growth modes
were measured. At a fixed shard count the effective rank is pinned at 20.4. When
the fleet grows, effective rank climbs 22.1 to 30.4 as new random bases fill out
the 32 ambient dimensions. Intrinsic dimension does not move in either case.
Approximate search is decided by local neighbourhood structure, so the fleet's
growth inflates a global statistic while leaving the local geometry that
determines recall untouched.

**One thing I expected and got wrong.** I predicted difficulty would be flat in
n because each shard carries its own basis, so a query's neighbours would stay
inside its own shard. Hubness rises with n instead, 1.34 to 3.24 at k=10, about
+0.42 per decade against roughly +0.25 per decade for real. The shards overlap
rather than separating, since two random 16-dimensional subspaces of a
32-dimensional space cannot avoid each other. So the corpus does get harder in
one respect, faster than real data does, and the flat recall holds despite that
rather than because nothing changes.

## What follows for scale

Per-shard work is already constant across runs. 1T over 500 servers is two
billion rows per server, exactly the ratio the 100B run used over 50 servers, so
nothing about the build gets harder per node and the scale-out is clean. What
grows is only the number of chances to draw a slow node.

The limit worth respecting is therefore not storage or compute. It is that
another order of magnitude on this corpus measures the same 16-dimensional
problem again. The fourth point is worth having as a scaling-mechanics result,
which is a real result, but it should not be reported as evidence that 4-bit
compression preserves recall on realistic data. It is not measured on realistic
data.

The change that would make a further order of magnitude informative is a corpus
with real intrinsic dimension, which is what openvector-bench's RC-1 generator
programme exists to produce and has not yet achieved.

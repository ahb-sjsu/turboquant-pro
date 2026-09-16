# Results — the pruned two-pass ADC scan

Registered in `docs/PREREG_pruned_scan.md` before the kernel ran on real data; the configurations,
grid, seeds, endpoint and the K1/K2 rules are that file's. Scored by `benchmarks/score_pruned_scan.py`
(committed before any evaluation number existed) from the records on the campaign volume, 2026-09-16.

Calibration picked one (prefix, z) per configuration and k on 500 held-out queries, taking the
largest speedup with recall ≥ 0.999 against the unpruned v2 kernel. Evaluation then ran 1,000
different held-out queries on three index seeds. Recall is measured against v2's own top-k, so it
asks whether pruning changes the answer, not whether the answer is right.

## Verdicts

| arm | d′ | bits | k | prefix | z | recall mean | recall min | survivors | speedup | K1 | K2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| wiki1024-1m | 1024 | 2 | 10 | 1/4 | 2 | 0.9999 | 0.9999 | 1.79% | 1.86× | HOLDS | HOLDS |
| wiki1024-1m | 1024 | 2 | 50 | 1/2 | 2 | 1.0000 | 1.0000 | 0.49% | 1.55× | HOLDS | HOLDS |
| wiki1024-1m | 1024 | 4 | 10 | 1/4 | 2 | 0.9999 | 0.9999 | 1.42% | 1.63× | HOLDS | HOLDS |
| wiki1024-1m | 1024 | 4 | 50 | 1/2 | 2 | 1.0000 | 1.0000 | 0.41% | 1.42× | HOLDS | HOLDS |
| wiki1024-1m | 256 | 4 | 10 | 1/2 | 2 | 0.9999 | 0.9999 | 1.19% | **0.82×** | HOLDS | FAILS |
| wiki1024-1m | 256 | 4 | 50 | 1/2 | 2 | 1.0000 | 0.9999 | 2.65% | **0.76×** | HOLDS | FAILS |
| dbpedia-3large-1536 | 1536 | 2 | 10 | 1/4 | 2 | 1.0000 | 1.0000 | 0.21% | 2.25× | HOLDS | HOLDS |
| dbpedia-3large-1536 | 1536 | 2 | 50 | 1/4 | 2 | 1.0000 | 1.0000 | 0.86% | 1.84× | HOLDS | HOLDS |
| dbpedia-3large-1536 | 1536 | 4 | 10 | 1/4 | 2 | 1.0000 | 1.0000 | 0.16% | 2.29× | HOLDS | HOLDS |
| dbpedia-3large-1536 | 1536 | 4 | 50 | 1/4 | 2 | 1.0000 | 1.0000 | 0.66% | 1.96× | HOLDS | HOLDS |
| dbpedia-3large-1536 | 384 | 4 | 10 | 1/2 | 2 | 1.0000 | 1.0000 | 0.25% | **0.99×** | HOLDS | FAILS |
| dbpedia-3large-1536 | 384 | 4 | 50 | 1/2 | 2 | 1.0000 | 1.0000 | 0.86% | **0.91×** | HOLDS | FAILS |
| wiki1024-5m | 1024 | 4 | 10 | 1/4 | 2 | 0.9998 | 0.9997 | 0.56% | 2.18× | HOLDS | HOLDS |
| wiki1024-5m | 1024 | 4 | 50 | 1/4 | 2 | 0.9999 | 0.9999 | 1.40% | 1.89× | HOLDS | HOLDS |

**K1 (recall) holds in all fourteen cells.** The lowest single-seed recall anywhere is 0.9997, and
eight of the fourteen cells are exactly 1.0000: the extrapolating bound at z = 2 essentially does not
drop a true top-k vector at these scales.

**K2 (speed) holds on every d′ ≥ 1024 configuration and fails on both narrow ones.** The registered
shipping condition is "K1 everywhere and K2 on every d′ ≥ 1024 configuration", so it is met, and
`score_pruned_scan.py` reports SHIP AS OPT-IN. K2 as written also asks for no slowdown below 1024
dimensions, and that half of it fails: 0.76× to 0.99×.

## What the numbers say about the mechanism

The prefix pass is a fixed tax and the savings scale with the scan it replaces, so the two-pass scan
wins exactly where the unpruned scan is long. At 1536 dimensions it is 1.84–2.29×; at 1024 it is
1.42–1.89×; at 384 and 256 it loses. The 5M arm behaves like the 1M arm of the same width (2.18×
against 1.63×, the difference being the larger absolute scan), which says the effect follows d′ and
the survivor fraction rather than N.

Survivors fall between 0.16% and 2.65%. Reading across the table, the configurations that lose are
not the ones that prune badly: dbpedia at 384 dimensions keeps only 0.25% of vectors and is still
0.99×, because 1/2 of 384 dimensions is already most of the work. Pruning cannot help when the
prefix needed to decide is a large fraction of the vector.

The chosen z was 2 in every cell, and the chosen prefix was 1/4 wherever the configuration won and
1/2 wherever it lost. Calibration selected the grid's most aggressive admissible point at width and
was pushed to the conservative one when the arithmetic left nothing to gain.

## Amendments, and one thing not accounted for

Both amendments in the registration were recorded before any evaluation number was read.

- **Amendment 1**: the pods ran 4 CPUs and 4 OpenMP threads, not the registered 8. Every row above
  is from a 4-thread pod, which the `env` record confirms for each of the four result files.
- **Amendment 2**: the run was split across pods sized to their own configurations, and the
  benchmark was changed to hold one index at a time. Configurations are disjoint across the files
  and the scorer asserts that rather than assuming it.

One pod, carrying configurations 4 and 5, was killed at about 10:05 on 2026-09-16 by something the
utilization guard's log does not account for and that none of the campaign's scripts target. Its
finished configurations were already on the volume; the two it was running were rerun in a separate
pod (`rbq-pruned-1m-c`), which is the fourth `env` entry. The cause is unknown and is recorded here
rather than guessed at.

## Limits

One CPU family, whatever node each pod landed on, and 4 threads throughout. Timings include the
per-query repack of the codes, which both kernels do. The bound's parameters come from one
calibration split per configuration. Queries are held-out corpus rows, not user queries. Recall is
against v2's ranking, so it bounds the damage pruning does and says nothing about either kernel's
agreement with exact search.

## What this licenses

The pruned scan ships as an **opt-in**, not a default, which is what the registration allowed on this
outcome. It belongs behind a flag with a documented range: worth using at 1024 dimensions and above,
and worth avoiding below that, where it costs 1–24% rather than saving anything. A planner choosing
search operators has a measured rule here rather than a guess, which is the form `docs/DESIGN_planner.md`
section 2.4 asks for.

Nothing here is claimed as new. The idea is ADSampling's (Gao & Long, SIGMOD 2023) and the
variance-ordered early termination of DADE / DDCpca; what is measured is whether it pays inside
tq-pro's quantized, randomly rotated codes, and at which widths.

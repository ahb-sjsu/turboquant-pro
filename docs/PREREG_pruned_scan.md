# Pre-registration — a pruned two-pass ADC scan for tq-pro

**Status: REGISTERED 2026-09-15, before the pruned kernel was run on real data.** Results land in
`benchmarks/RESULTS_pruned_scan.md`; this file is never edited to fit them.

> **The question.** tq-pro's compiled search is a full SIMD scan whose cost is linear in N·d'
> (the v2 kernel benchmark, 2026-09-15, found it compute-bound: nibble packing and streaming
> top-k changed wall time by −6% to +10%). Can a two-pass scan that stops most vectors after a
> prefix of dims keep top-k results and cut wall time?

This borrows the idea of ADSampling (Gao & Long, SIGMOD 2023) and the PCA-ordered early
termination of DADE / DDCpca: extrapolate a vector's distance from a prefix and stop when it
cannot win. Nothing here is claimed as new; the question is whether it pays inside tq-pro's
quantized, randomly rotated codes.

## 0. What is already known

A simulation on 200k real rows (`benchmarks/prune_sim.py`, commit `ef964ea`) with the oracle
threshold: the extrapolating bound at z = 3 pruned 90–99% of vectors after d'/8 to d'/2 dims at
recall 0.999–1.000 against the unpruned ranking; an exact bound pruned almost nothing. A cost model
(scalar work for sparse survivors, SIMD for the prefix) predicts a wall-time gain of about 1.5–2.5×
only where survivors fall to about 1–2%. The simulation used the oracle threshold and in-sample
parameters; this registration replaces both.

## 1. The kernel

`search_pruned` in `turboquant_pro/_adc/adc_scan.cpp` (the function is defined in the commit that
registers this file; v2 `search` is unchanged). Pass 1 sums the first m dims (SIMD) for every vector
and forms `est = acc_m + μ_rest + (acc_m − μ_pre)(d − m)/m` from per-query means and variances of the
uint8 table under the index's code frequencies, with `sd = sqrt(var_rest + var_pre((d − m)/m)²)`.
The threshold is the k-th largest score at `est − z·sd`. Pass 2 finishes, with the exact uint8
score, every vector whose score at `est + z·sd` reaches it: blocks with ≥ 6 survivors in SIMD, the
rest scalar. Returned scores equal the unpruned kernel's; a true top-k vector can be pruned.
Verified before registration: with z = 1e9 results equal v2's; returned scores equal v2's for the
same ids; recall ≥ 0.95 at z = 3 on structured synthetic data (tests/test_adc_kernel.py).

## 2. Configurations and data

Real rows staged for the RaBitQ public campaign (hashes in
`benchmarks/rabitq_public/DATA_MANIFEST.json`), normalized; tq-pro PCA to d' then TurboQuant at the
given bits (`benchmarks/bench_pruned_scan.py`):

| arm | corpus | d' | bits |
|---|---|---:|---:|
| wiki1024-1m | Cohere Wikipedia en part_000 (1M) | 1024 | 2 |
| wiki1024-1m | same | 1024 | 4 |
| wiki1024-1m | same | 256 | 4 |
| dbpedia-3large-1536 | DBpedia text-embedding-3-large, 989,500 rows | 1536 | 2 |
| dbpedia-3large-1536 | same | 1536 | 4 |
| dbpedia-3large-1536 | same | 384 | 4 |
| wiki1024-5m | Wikipedia parts 000–004 (5M) | 1024 | 4 |

Queries: 500 calibration rows and 1,000 evaluation rows, disjoint from each other and from the
corpus (wiki: the first 1,500 rows of the next part; DBpedia: the last 500 rows of part_000 for
calibration, the held-out `queries.npy` for evaluation).

## 3. Procedure

1. **Calibration** (index seed 0, calibration queries): for k ∈ {10, 50}, every grid point
   prefix ∈ {1/8, 1/4, 1/2} × z ∈ {2, 3, 4}; choose the largest speedup with recall@k ≥ 0.999
   against v2 (ties: larger z, then larger prefix). A configuration with no admissible point is
   reported as such.
2. **Evaluation** (index seeds 0, 1, 2; evaluation queries; the chosen point only): recall@k of the
   pruned top-k against v2's top-k, survivor fraction, and paired wall time (minimum of 3
   repetitions of each, same process and thread count) as speedup = t_v2 / t_pruned.

One 8-CPU pod runs everything, 8 OpenMP threads.

## 4. Verdicts

Per configuration and k, over the 3 evaluation seeds:

- **K1 (recall)** HOLDS if the mean recall@k against v2 is ≥ 0.995 and no seed is below 0.99.
- **K2 (speed)** HOLDS if the mean speedup is ≥ 1.3× on every d' ≥ 1024 configuration and ≥ 1.0×
  (no slowdown) on every other.

The pruned scan is worth shipping as an opt-in only if K1 holds everywhere and K2 holds on every
d' ≥ 1024 configuration. Otherwise it stays an experiment and the reason is reported.

## 5. Limits

One CPU family per run, whatever node the pod lands on; timings include the per-query repack of
the codes, which both kernels do; the bound's parameters come from one calibration split; queries
are held-out corpus rows, not user queries.

## 6. Amendment log

**Amendment 1 (2026-09-15, disclosed before any result was read).** The first run of
`bench_pruned_scan.py` on NRP used a pod with 4 CPUs and 4 OpenMP threads, not the 8 registered in
section 3. It was stopped during the calibration of the first configuration, because the campaign's
pods were flagged by the cluster for using too little of their requested CPU and memory, so no
evaluation number from it exists or is used. The rerun keeps the registered procedure; whatever
thread count the rerun pod gets is reported with the results, and section 5 already limits timings
to one CPU family per run. The only figure produced before the stop, for the record: wiki1024-1m at
d' = 1024, 2 bits, k = 10, calibration picked prefix 1/4 and z = 2 at recall 1.000 against v2, 2.0%
survivors, 1.45x. Calibration numbers do not enter K1 or K2.

**Amendment 2 (2026-09-16, before any evaluation number was read).** The rerun is split across two
pods instead of the single pod of section 3: one for the six 1M-row configurations and one for
wiki1024-5m. The reason is the cluster's utilization rule, not the experiment: a single pod must
request the memory the 5M configuration peaks at, and would then sit far below the memory floor for
the hours it spends on the 1M ones. Each pod requests what its own configurations use, and the thread
count each receives is reported with the results. Nothing about the calibration grid, the seeds, the
queries or the verdict rules changes, and the two pods run disjoint configurations, so no
configuration is measured twice.

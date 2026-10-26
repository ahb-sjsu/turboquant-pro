# Preregistration: routed recall at 10^12 rows predicted from 10^8 to 10^9

Registered 2026-10-01, before any measurement of the registered query set on the calibration
corpora was made and before the registered run's value at 10^12 was read. Procedure:
`benchmarks/fleet/scale_transfer_small.py` (sha256 `e434fc552fdea37da554abc86f9b3fd86480b2292a4a3c73b43bb79db611bc01`).
Development record: `benchmarks/fleet/record/1t/post/scale_transfer_small_dev.json`
(sha256 `2e50c2c1a4321c8a4f8d2b3a807f5e78c0c5817ec22fc6fa87038ca02f6d4268`).

This is the second registration of the scale-transfer question. The first
(`docs/PREREG_scale_transfer.md`) calibrates on subsets of 2×10^9 to 4×10^10 rows inside the
trillion-row index and lists a calibration at 10^8 as left to a later registration. This is that
registration. Both are graded on the same run.

## 1. Question

Can routed recall of a trillion-row index be predicted from corpora a thousand to ten thousand
times smaller, built and searched on one machine? If so, the routing quality of an index at
10^12 can be checked before the index is built.

## 2. Object

- **Index at 10^12**: the 500-server index of the 1T run, unchanged.
- **Registered run**: tag `1tnm`, the same run as the first registration. 100 queries, 25 from
  each of the seeded shards 200000, 250000, 300000 and 350000, which no corpus shard uses.
- **Recall**: recall at ten of the routed top ten against the exact top ten of the same corpus.

## 3. Calibration corpora

- **Shards**: 600 shards of the trillion-row index (5×10^6 rows each), rebuilt from their seeds
  on one workstation with the fleet's own bootstrap. The bootstrap is the basis file, the global coarse
  quantizer and the radius file that the fleet copied to every server. A calibration corpus is
  therefore the fleet's index restricted to those shards and routed by the fleet's router.
- **Design**: three permutations of 200 shards drawn without replacement from the 200000 global
  shards (seed 20261001). For each, the first 20, 40, 100 and 200 shards, that is 10^8, 2×10^8,
  5×10^8 and 10^9 rows. Each corpus is searched as one sharded index, exact and routed.
- **Curve**: per-query recall averaged over the three permutations, then over queries, at each
  size and each width of 16, 32, 64, 128 and 256 probes.
- **Target**: N* = 10^12 rows, all 500 servers.

## 4. Models and how the registered ones were chosen

The models are those of the first registration, fitted on the four calibration sizes.

| model | form |
|---|---|
| M0 | no change: recall at 10^9 carried to 10^12 |
| M1 | power law in the deficit: log(1 − R) linear in log N, least squares |
| M2 | recall linear in log N, least squares, capped at 1 |

The models were compared on the **development set**, the member queries of run `1t`, whose
values at 10^12 were already known. The rule was fixed before the comparison and is the same as
in the first registration. Per width, the model with the smallest absolute error at 10^12 wins,
and ties go to the simpler model (M0, then M1, then M2). Development result (prediction, error):

| width | M0 | M1 | M2 | measured |
|---|---|---|---|---|
| 16 | 0.755 (−0.197) | 0.900 (−0.052) | **1.000 (+0.048)** | 0.952 |
| 32 | 0.868 (−0.121) | 0.966 (−0.023) | **1.000 (+0.011)** | 0.989 |
| 64 | 0.943 (−0.054) | 0.991 (−0.006) | **1.000 (+0.003)** | 0.997 |
| 128 | 0.986 (−0.013) | **0.999 (+0.0001)** | 1.000 (+0.001) | 0.999 |
| 256 | 0.998 (−0.002) | 1.000 (−0.00001) | **1.000 (0.000)** | 1.000 |

The run `1tnm` measures 32 and 128 probes at 10^12.
**Registered models for the `1tnm` run: M2 at 32 probes (primary), M1 at 128 probes.**

M2 wins at 32 probes because its fit passes 1 before 10^12 and the cap holds it at 1. Its
prediction there is "routed recall reaches 1", not a fitted value below 1. The rule was applied
as written and the registration stands on it. Section 5 reports every model, so a reader can
judge M1 at 32 probes as well.

## 5. Predictions and grading

Grading follows the first registration. For each width the error is the registered model's
prediction minus the measured recall at 10^12. Its standard error comes from a **paired
bootstrap over queries** (2000 resamples, seed 7). The calibration corpora and the run share the
100 queries in the same order, so each resample recomputes the calibration curve, every model's
prediction and the measured value, and takes their difference.

- **H1, primary** (32 probes, M2): |error| ≤ 2 × SE of the difference.
- **H2** (32 probes): the registered model's |error| is smaller than M0's.
- **H3** (128 probes, M1): |error| ≤ 2 × SE of the difference.

Every model's prediction and error are reported at both widths. Results of this registration
and the first are reported side by side, whichever passes.

**What counts against the claim.** H1 failing means a law fitted at 10^8 to 10^9 rows does not
carry to 10^12 for these queries. H2 failing means the fitted law adds nothing to "no change".
Either result is reported at equal prominence with a pass.

## 6. Order and blinding

1. This registration is committed before the registered queries are measured on the calibration
   corpora.
2. `scale_transfer_small.py measure --qname q1tnm` runs for every permutation and size, then
   `scale_transfer_small.py curve --qname q1tnm --registered '{"32":"M2","128":"M1"}'` writes
   `scale_transfer_small_q1tnm.json` with every model's prediction and the script's sha256.
3. That JSON is committed to the repository **before** the run's value at 10^12 is read by the
   person running this test. The fleet session has been asked to hold the run's score back from that
   person until it is told the prediction is committed.
4. `scale_transfer_small.py grade --qname q1tnm --prediction <committed JSON> --results <the
   run's 500 partials>` then writes the grade, which is committed beside the prediction.

The grade requires the query file used here (`queries1tnm.npy`, sha256
`7d9e4a48152da1ee397af5e5e27ed1cfc38da1d7a82f42e6c05ffdf55a19d452`, regenerated from the
run's seeds) to match the run's own query file byte for byte. If it does not, the grade is not
computed and the mismatch is reported. If the run's value at 10^12 is read before the prediction
is committed, the test is void and reported as void, not graded.

## 7. Procedure history

- The development record was written by the script at sha256 `1cc7945e…` (in the development
  JSON). The registered script differs from it only by the `grade` phase and the `--registered`
  option, added after the development result. The build, measure and curve code is unchanged.
- The first measurement attempt failed before producing any result because the corpus assembly
  linked shards by relative paths that did not resolve. The links now use absolute paths. No
  measurement from the failed attempt exists.

## 8. What this does not test

- **One corpus.** The generator has intrinsic dimension about 16, as in the first registration.
- **Four calibration sizes over one decade.** The fit extrapolates three decades. The first
  registration covers 1.3 decades and extrapolates 1.4.
- **The models were selected on the development set.** Their development errors are therefore
  optimistic. The registered run, with queries of a different kind, is the real test.

# Preregistration: routed recall at 10^12 rows predicted from at most 4×10^10

Registered 2026-09-29, before any partial of the registered run was read. Procedure:
`benchmarks/fleet/scale_transfer.py` (sha256 `81ec880ef800719b32e6cded72c8d1b5c84828817d7b2d9e11932859ddff4478`).
Development record: `benchmarks/fleet/record/1t/post/scale_transfer_dev.json`
(sha256 `3eba0f0911dbad5119fbd57f91a7f17558f8d51b5621f9581be13661c6d0f0e3`).

## 1. Question

Inside the trillion-row index, routed recall against the exact scan of a subset of servers is
flat at wide probe widths and rises with the number of rows at narrow ones
(`docs/notes/1T_RECALL_RESULT_2026-09-27.md`, nested scale). If that rise follows a law that can
be fitted on small subsets, the routing quality of a trillion-row index can be predicted without
building or scanning it. This registration asks whether a model fitted on subsets of at most 20
servers (4×10^10 rows) predicts recall on all 500 (10^12 rows), 25 times beyond the largest
calibration point, for a query set the model has never seen.

## 2. Object

- **Index**: the 500-server, 10^12-row index of the 1T run, unchanged.
- **Registered run**: tag `1tnm`, launched by the fleet session on 2026-09-29. 100 queries, 25
  from each of the seeded shards 200000, 250000, 300000 and 350000, which no corpus shard uses,
  so no query is a corpus row and none has a home server. Reference exact scan and routed passes
  at 32 and 128 probes on all 500 servers.
- **Recall**: recall at ten of the routed top ten against the exact top ten of the same subset of
  servers, from the per-server partials, as in `fleet_nested.py`.

## 3. Calibration

- **Calibration servers**: 20 servers drawn once with seed 20260929:
  0, 43, 103, 109, 136, 138, 143, 168, 179, 249, 256, 265, 325, 341, 346, 370, 379, 421, 483, 494.
- **Subsets**: five orderings of the 20 (seed 20260930); for each, the first 1, 2, 3, 5, 10 and
  20 servers. Recall at each size is the mean over the five orderings, giving a curve at
  2×10^9, 4×10^9, 6×10^9, 10^10, 2×10^10 and 4×10^10 rows.
- **Target**: N* = 10^12 rows, all 500 servers.

## 4. Models and how the registered one was chosen

| model | form |
|---|---|
| M0 | no change: recall at 4×10^10 carried to 10^12 |
| M1 | power law in the deficit: log(1 − R) linear in log N, least squares on the six points |
| M2 | recall linear in log N, least squares, capped at 1 |

The models were compared on the **development set**, the member-query run (tag `1t`), whose
nested scale result had already been seen. The rule, fixed before the comparison, is the model
with the smallest absolute error at 10^12 per width, with ties going to the simpler model (M0,
then M1, then M2). Development result:

| width | M0 | M1 | M2 | measured |
|---|---|---|---|---|
| 16 | 0.891 (−0.061) | **0.939 (−0.013)** | 0.983 (+0.031) | 0.952 |
| 32 | 0.957 (−0.032) | **0.982 (−0.007)** | 1.000 (+0.011) | 0.989 |
| 64 | 0.989 (−0.008) | **0.998 (+0.0005)** | 1.000 (+0.003) | 0.997 |
| 128 | **0.999 (0.000)** | 1.000 (+0.001) | 1.000 (+0.001) | 0.999 |
| 256 | **1.000 (0.000)** | 1.000 (0.000) | 1.000 (0.000) | 1.000 |

**Registered models for the `1tnm` run: M1 at 32 probes (primary), M0 at 128 probes.**

## 5. Predictions and grading

For each width, the error is the registered model's prediction minus the measured recall at
10^12. Its standard error comes from a **paired bootstrap over queries** (2000 resamples, seed 7).
The calibration curve and the measurement share the 100 queries, so each resample recomputes
both and takes their difference. A threshold in units of that noise is the only fair bar, because
100 queries set the resolution.

- **H1, primary** (32 probes, M1): |error| ≤ 2 × SE of the difference.
- **H2** (32 probes): the registered model's |error| is smaller than M0's, that is, the fitted
  law beats "recall does not change" at predicting the trillion-row value.
- **H3** (128 probes, M0): |error| ≤ 2 × SE of the difference.

Every model's prediction and error are reported at both widths, not only the registered ones.

**What counts against the claim.** H1 failing means a law fitted below 4×10^10 rows does not
carry to 10^12 for these queries. H2 failing means the fitted law adds nothing to "no change".
Either result is reported at equal prominence with a pass.

## 6. Blinding

The test is blind only if the prediction exists before the trillion-row value is known.

1. `scale_transfer.py --predict --tag 1tnm --registered '{"32":"M1","128":"M0"}'` is run as soon
   as the reference and routed partials of the 20 calibration servers exist. It reads those 60
   files and nothing else, and writes `scale_transfer_predict_1tnm.json` with the sha256 of every
   file it read, of the script, and a UTC timestamp.
2. That JSON is committed to the repository **before** `score_1Tnm.log`, the run's own score, or
   any merge over more than the 20 calibration servers is read by the person running the test.
3. `scale_transfer.py --grade` is then run on all 500 servers and its JSON committed beside the
   prediction.

If the run's score is read before the prediction is committed, the test is void and reported as
void, not graded.

## 7. What this does not test

- **Calibration is at 2×10^9 to 4×10^10 rows, not 10^8.** The smallest unit available inside this
  index is one server of two billion rows. A calibration at 10^8 needs a separately built index
  and is left to a later registration.
- **One corpus.** The generator has intrinsic dimension about 16. Whether the exponent of M1
  depends on intrinsic dimension is the next question, not this one.
- **The model was selected on the development set.** Its development error is therefore
  optimistic. The registered run, with queries of a different kind, is the honest test.

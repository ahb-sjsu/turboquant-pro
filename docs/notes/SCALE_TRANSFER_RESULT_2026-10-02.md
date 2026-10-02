# Scale transfer to 10^12 rows: results of both registrations (2026-10-02)

Two registrations predicted routed recall of the 500-server, 10^12-row index on the non-member
queries of run `1tnm`, before that run's score existed.

- First registration (`docs/PREREG_scale_transfer.md`) calibrates on subsets of 2×10^9 to 4×10^10
  rows inside the index. Prediction committed at `b859fba` (03:24 UTC).
- Second registration (`docs/PREREG_scale_transfer_small.md`) calibrates on separately built
  corpora of 10^8 to 10^9 rows. Prediction committed at `108ed8e`.

The run was scored at 04:31 UTC (`record/1t/post/score_1Tnm.log`, committed at `e1839e1`). Grades:
`record/1t/post/scale_transfer_grade_1tnm.json` and `scale_transfer_small_grade_1tnm.json`.
Before grading, all 1997 files of the run were checked against their SHA256SUMS (no failure),
and both grader scripts matched their registered hashes (`81ec880e`, `e434fc55`). The measured
values the graders computed from the partials equal the run's own score.

## Measured

Routed recall at ten against the exact scan of all 500 servers, 100 non-member queries.

| probes | measured |
|---|---|
| 32 | 0.969 |
| 128 | 0.999 |

## Grades

Error is the registered model's prediction minus the measured value. SE is the standard error of
the difference from the paired bootstrap over queries (2000 resamples, seed 7). A prediction passes
when its error is within 2 SE.

| registration | probes | model | prediction | error | SE | result |
|---|---|---|---|---|---|---|
| first, H1 | 32 | M1 | 0.953 | −0.016 | 0.010 | pass (1.6 SE) |
| first, H2 | 32 | M1 against M0 (0.914, error −0.055) | | | | pass |
| first, H3 | 128 | M0 | 0.994 | −0.005 | 0.003 | pass (1.7 SE) |
| second, H1 | 32 | M2 | 1.000 | +0.031 | 0.007 | **fail (4.4 SE)** |
| second, H2 | 32 | M2 against M0 (0.819, error −0.150) | | | | pass |
| second, H3 | 128 | M1 | 0.998 | −0.001 | 0.001 | pass (0.5 SE) |

## What this says

The first registration's law, fitted on at most 4×10^10 rows, carried to 10^12 on queries it had
never seen. At 32 probes it predicted 0.953 against 0.969, and it beat the assumption that recall
does not change (0.914) by a wide margin.

The second registration's primary test failed. Its development comparison chose M2 at 32 probes
because the cap at 1 happened to sit close to the member-query value (0.989), and the registration
said so when it was made. On the non-member queries the true value is 0.969, so a prediction of 1
is 4.4 standard errors too high. The power law in the deficit (M1), fitted on the same 10^8 to
10^9 corpora but not registered at 32 probes, predicted 0.953, the same as the first registration.
That observation is reported as context only. It was not the registered model, and selecting it
now would be selecting on the test set.

At 128 probes both registrations passed, with M1 from 10^8 to 10^9 rows within 0.001.

## Limits

- One index and one synthetic corpus of intrinsic dimension about 16.
- 100 queries set the resolution. The standard errors above are of that size.
- The 10^8 to 10^9 corpora rebuild the index's own shards with its own router. They test transfer
  of a routing law across corpus size, not across corpora or encoders.

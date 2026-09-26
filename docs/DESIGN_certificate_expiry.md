# Design — certificates expire

Issue #177, the fifth of the Observation Theory family (#183). Phase 1 on
`feat/certificate-validity`.

## 0. Thesis

A rank certificate says that a statement was true for an observer `O` under an
environment `E`: these inputs, this read operator, this calibration sample.
When `O` becomes `O'`, or the data drifts out of the calibration's coverage,
the certificate is not false. It is no longer applicable. Monitoring reports
drift; this is different: it is invalidation, a status with a reason and an
action, computed from what the certificate itself recorded at issue.

The certificate already binds its reference operator by hash, so a changed
operator is detectable; but a hash says only that something changed, not how
much or in which direction. readscope's C-11c measured a consumer's read
operator drifting along a sequence, and the certificate's own `reference`
section exists because two defensible operators for one head differ by 0.3 in
overlap. The certificate should therefore carry enough of the operator to
measure that overlap later, and enough of the certified sample to measure
coverage later, and `tqp verify` should read both.

## 1. What the certificate records at issue (additive `validity` section)

Emitted by `tqp certify --validity`, and by default when `--observer` or
`--reference` is given, since both name an observer the certificate depends
on. `schema_version` stays 1.

| field | content | why |
|---|---|---|
| `issued_for` | the observer contract hash (if any), the reference provider and `operator_sha256` (if any) | the identity of `O` |
| `operator_sketch` | the top-`r` eigenvectors and eigenvalues of the reference operator, `r` the ceiling of its effective rank up to a cap, with the fraction of trace they carry | enough of `P_C` to measure how much of a later operator lies in the certified read subspace |
| `coverage_sketch` | per-channel mean and variance of the certified original sample, and the row count | enough of `E` to measure whether later data sits inside the calibration's coverage, without storing a `D×D` covariance |
| `thresholds` | `min_operator_overlap` (0.85), `max_coverage_divergence` (a diagonal Jeffreys divergence per channel, 0.5) | the numbers the status is decided against, recorded so a reader cannot move them after the fact |

## 2. What `tqp verify` checks (`checks.validity`)

`tqp verify CERT --observer C [--data sample.npy --queries q.npy]`:

1. **source artifact unchanged**: the existing input hashes (with `--original`/`--reconstructed`).
2. **observer unchanged**: the contract hash (from #173).
3. **observer read geometry**: with `--data`, the contract's operator is rebuilt (`refinement.observer_operator`) and its overlap with the sketch is `tr(Uᵀ P' U) / tr(P')`, the fraction of the new operator's sensitivity inside the certified read subspace. Below `min_operator_overlap` the certificate is STALE with reason "consumer read geometry changed" and action REPLAN.
4. **data distribution within coverage**: with `--data`, the diagonal Jeffreys divergence between the sketch's moments and the sample's, averaged per channel. Above the threshold: STALE, reason "data outside calibration coverage", action RECERTIFY. Phase 2: the check abstains when its own sampling noise could reach the bar (section 5).
5. **strata coverage** (phase 2, section 5): the share of the sample in regions the certificate did not cover. Above baseline plus tolerance: STALE, reason "data in strata the certificate did not cover", action RECERTIFY.

The result carries `status` (VALID, STALE, INCONCLUSIVE when a check abstained
and none failed, or UNCHECKED when neither `--data` nor a sketch is
available), `reason`, `action`, and the measured numbers.
`verified` keeps its meaning (the certificate is what it says it is);
`applicable` is the new field, and the exit code is 1 when either is false.

```
CERTIFICATE STATUS
  source artifact unchanged         ok
  observer contract unchanged       ok
  observer overlap 0.71 < 0.85      FAIL
  data coverage divergence 0.12     ok
  strata coverage                   not checked
STATUS: STALE   reason: consumer read geometry changed   action: REPLAN
```

## 3. What Phase 1 did not do

- Monitor integration and strata checks: phase 2, section 5.
- No codec-identity check: the certificate does not record which codec
  produced the reconstruction; the plan record does (#169), and binding the
  two is the pipeline composition of #182.

## 4. Phases

1. The `validity` section and the `tqp verify` checks above, with tests that
   turn a passing certificate STALE by rotating the operator, and by shifting
   the data, and keep it VALID otherwise.
2. Monitor integration and strata coverage (section 5). Shipped.
3. Capability discovery (#178) reads the status to decide what an index is
   currently certified for.

## 5. Phase 2: strata coverage, noise units, the monitor

**The strata sketch.** `tqp certify --strata kmeans:N` (or a saved
`tqp-area-map/1` map built on `--original`, or `--by KEY --labels FILE`)
records the certified sample's areas in `validity.strata_sketch`. The sample is
split: the even rows fit one centroid per area; the odd rows are assigned to
their nearest centroid, exactly as a later row will be, and give each area its
row count and radius, the conformal `q`-quantile (`q = 0.99`) of their distances
to it. A map built on another corpus is refused by its fingerprint.

**The check.** A new row is uncovered when it lands in an area the odd half saw
fewer than `n_min = 100` times, or beyond its area's radius. For rows
exchangeable with the certified sample the uncovered fraction is at most
`baseline = (thin + 1)/(n + 1) + (1 - q)`, recorded at issue with a tolerance
(0.05). A Wilson interval on the new sample's fraction decides: wholly above
`baseline + tolerance` is FAIL, wholly below is ok, and straddling abstains.
The report names the areas the uncovered rows came from and why (beyond
radius, or thin at issue). This finds what the moment check cannot: in the
test, 10% of rows from a new region leave the per-channel divergence under
its bar and are called stale here.

**Noise units for the moment check.** The phase 1 check compared the plug-in
divergence with an absolute 0.5. Its expectation under exchangeability is
about `(1/n + 1/m)(1 + (kappa - 1)/2)` per channel (`2 (1/n + 1/m)` for a
Gaussian), which at twelve rows is a third of the bar: a small sample from the
certified distribution could be called stale by its own noise. The check now
reports this floor (`coverage_noise_floor`, estimated with the sample's own
kurtosis and verified against repeated exchangeable draws) and abstains when
it exceeds a fifth of the bar.

**Uncertain means no verdict.** A check that abstains makes the status
INCONCLUSIVE unless another fails. `applicable` stays true (nothing was shown
stale), but capability discovery (#178) now reports CERTIFIED only for VALID;
UNCHECKED and INCONCLUSIVE are CONDITIONAL.

**The monitor.** `QualityMonitor(certificate=doc)` keeps a window of recent
originals (`validity_window`, default 2000), re-runs `check_validity` on it
every `validity_every` records (default 500), fires the alert callback once on
the move into STALE with the reason and action, and exports
`turboquant_certificate_valid`, `_stale`, `_operator_overlap`,
`_coverage_divergence`, `_uncovered_fraction` and `_window_rows`, with NaN for
a check that did not run. A certificate with no `validity` section is refused.

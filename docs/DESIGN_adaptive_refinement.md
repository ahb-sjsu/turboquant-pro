# Adaptive certified refinement (#175)

A fixed `rerank=r` reads `k * r` original rows for every query and promises
nothing about recall. This design replaces it with a band calibrated to a
declared recall, with a guarantee that holds, and measures whether letting the
band adapt per query reads fewer rows than the best fixed depth.

## 1. The procedure

After the compressed scan returns the top `max_candidates` for a query, a
candidate enters the band when its compressed score is within `epsilon` of the
`k`-th compressed score (`band="score"`), measured in units of 1 under cosine
and of `||q||` under inner product and l2. If the band holds exactly `k` rows
the compressed top `k` is returned and no original is read; otherwise the band
is rescored exactly and its best `k` returned. `band="rank"` replaces the score
margin by a count: the first `k + epsilon` candidates for every query. That is
a fixed rerank depth, and calibrated the same way it is the baseline an
adaptive band has to beat.

## 2. The guarantee

`calibrate` runs the procedure on `n` calibration queries against the exact
top `k` and picks the smallest `epsilon` with
`(n R_n(epsilon) + 1) / (n + 1) <= 1 - target`, where `R_n` is the mean of
`1 - recall@k`. The loss is bounded by one and non-increasing in `epsilon`
(the band only grows), so conformal risk control (Angelopoulos et al., ICLR
2024) gives `E[recall@k] >= target` for a new query exchangeable with the
calibration queries, on the same index, `k` and cap.

What it is not: a per-query guarantee, or a statement about queries from
another distribution. What it needs: `n >= 1/(1 - target) - 1` calibration
queries, and a target the cap can reach. Either failing is refused with the
reason (`InfeasibleTarget`), never answered with a weaker policy.

## 3. Provenance

A policy (`adaptive_policy.schema.json`) records `k`, the cap, the band, the
calibration it came from and the index it belongs to: identity plus the sha256
of the stored row norms. `search` refuses a policy calibrated on another index.
Exchangeability of the live query stream is the operator's claim; when it
drifts, certificate expiry (#177) is the check.

## 4. First numbers (synthetic, not the acceptance test)

Atlas, 20 000 rows, d = 96 → 48 PCA dims, cosine, k = 10, cap 300, 1 000
calibration and 2 000 held-out queries. Recall is measured on the held-out
queries; rows are original rows read per query (mean, p99).

| data | bits | target | score band | rank band |
|---|---|---|---|---|
| Gaussian | 2 | 0.99 | 0.9897, 119.9 (190) | 0.9902, 126 (126) |
| Gaussian | 3 | 0.99 | 0.9919, 47.8 (77) | 0.9916, 50 (50) |
| Gaussian | 3 | 0.995 | 0.9960, 57.6 (93) | 0.9968, 62 (62) |
| clustered | 3 | 0.95 | 0.9520, 97.7 (186) | 0.9525, 95 (95) |
| clustered | 3 | 0.99 | 0.9918, 164.1 (297) | 0.9897, 149 (149) |

Every target was kept on held-out queries by both bands. The adaptive band
read 3 to 7 % fewer rows on the Gaussian source and up to 10 % more on the
clustered one, with a heavier tail everywhere. A band scaled by each query's
predicted score noise `sqrt(q' Sigma_e q)` in place of the constant margin
changed rows read by under 1 % and was not kept. On these sources the
certified fixed depth is as good as adaptivity; the stop at the scan (a band
of exactly `k`) never fired at useful targets with `k = 10`.

## 5. Phases

1. **The certified band.** Shipped: `turboquant_pro.adaptive_rerank`
   (`calibrate`, `search`, both bands, the policy and its schema),
   `ADCIndex.metric`, `tests/test_adaptive_rerank.py`.
2. **The acceptance measurement.** Both bands against fixed `rerank=5` on the
   six public arms of `benchmarks/rabitq_public/`, preregistered before it is
   run: recall on held-out queries against the target, rows and bytes read
   (mean and p99), and the stage fractions.
3. **The refinement layer as a middle stage.** Once the layered container of
   #174 phase 2 exists, a band first rescored at base plus layer, then exactly
   only where the layer's margin is still not decisive.
4. **Operation.** A `tqp` command to calibrate and save a policy, and the
   console's `candidates` channel fed from `search`, so rows read per query is
   a trace on the scope.

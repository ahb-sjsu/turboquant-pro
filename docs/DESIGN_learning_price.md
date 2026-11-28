# Design: the learning price of a sampled read operator

`turboquant_pro.learning_price`, `tqp observer price`, `tqp observer probes`.
Source: the T1 note of the rate-leakage-distortion series,
geometric-observation `paper/t1-learn-then-serve` (v4.1), and its addendum
`paper/t1-moving-kernel`. Statement numbers below are the notes'. The note's
proofs have fresh-context audits on record (VI-23, VI-24); its custody table
is still owed (frontier register G6), so it is cited as a working note, not a
theorem of record.

## 0. Thesis

Every consumer-aware allocation in this package spends bits against a weight
`Q`, and almost every such weight is an average of samples:

| weight | averaged over |
|---|---|
| `attention_analytic` | the Jacobian Gram at `n_probe_keys` probed keys (default 32) |
| a top-k retrieval consumer, `E[q qᵀ]` | a sample of queries |
| a blind recovery (readscope) | finite differences at probe points |

The allocation then treats the average as the truth. The note says what that
costs, and the cost is not small in high dimension. This module makes it a
number, with a verdict, before any bits are spent.

## 1. What the note proves, and what is used

**Lemma 1 (the band).** For i.i.d. PSD samples `P_1..P_m` with mean `Q` and an
a priori sandwich `(1 - nu-) Q <= P_i <= (1 + nu+) Q`, with probability at least
`1 - delta`,

    (1 - eps-) Q <= Q_hat <= (1 + eps+) Q,
    eps± = sqrt(2 nu+ nu- L / m) + 2 nu± L / (3 m),   L = ln(2p / delta).

The proof is the matrix Bernstein inequality with the matrix Bhatia-Davis
variance bound. A read metric is PSD, so `nu- = 1` always holds: only `nu+` must
be declared. The exact Bernstein roots `t±` may replace `eps±`
(`exact_roots=True`).

**The learning price.** `tau = (p/2) log2((1 + eps+)/(1 - eps-))` bits per
vector. What it bounds depends on what the samples are, which the caller
declares (`kind`):

- **`linear`**: terms of a linear consumer's weight, such as a retrieval
  consumer's queries, whose weight is `E[q qᵀ]`. The design for `Q_hat` at
  distortion `D (1 - eps-)` is safe for every linear consumer consistent with
  the band. Its regret is at most `tau`, in rate and in leakage at once
  (Proposition 12's closing remark, by (P1)-(P3)). Here `tau` is the whole
  gap.
- **`points`**: read metrics of a nonlinear consumer at probe points (T1's
  (A2): `attention_analytic`, readscope). Here `tau` is only the *learning
  part*. Corollary 8's whole gap is
  `tau_bar = (p/2) log2((1+eps+)/((1-eps-)(1 - rho_bar(D)/D))) = tau + O(D)`
  at the design distortion `D_hat = (1 - eps-)(D - rho_bar(D))`. It needs the
  consumer's derivative bounds `M1, M2, M3`, and `rho_bar(D) < D`.
  `LearnedWeight.gap` computes it and refuses when `rho_bar(D) >= D`. The
  corollary's setting is paper I's: a memoryless Gaussian source that the
  decoder reproduces, with the probe law the source law. In the process
  settings, the transfer to codes is open (Remark 6). Theorem 11's gap,
  `tau_nu + tau`, needs both sandwich constants below one, which a declared
  `nu+` of 900 is not.

For an encoder given only `Q_hat` and the band, `tau` is sharp
(Proposition 12).

**Also implemented.** Corollary 3, the probe count for a target price (exact
read metrics, and with central differences together with the largest step);
Proposition 2, the deterministic error `eps_h` of central differences, which
adds to both `eps±`; Proposition 4, the one-time calibration message that
hands a decoder the weight in `O(p² log(p kappa / eps_q))` bits, which carries
no information about the source.

**Several consumers.** If each consumer `c` of a contract satisfies its own
band, the Loewner order survives non-negative sums, so the weighted sum
satisfies the band with the largest of each error. The events must hold
together, so `delta` is split across the sampled consumers (a union bound).
`tqp observer price --observer` does exactly this.

## 2. Beyond transcription: a sample can refute the declared constant

`nu+` is a claim about the consumer, made before the sample. On the band's
event, `Q <= Q_hat / (1 - eps-)`, so under the declared sandwich every sample
satisfies

    lambda_max(Q_hat^-1/2 P_i Q_hat^-1/2) <= (1 + nu+) / (1 - eps-).

A sample above that bound is impossible unless the event failed, which has
probability at most `delta`. So observing one is a level-`delta` test that
rejects the declared constant, and `learn_weight` returns `REFUTED` with no
price. The mirror test applies to a declared `nu- < 1`, through
`Q >= Q_hat / (1 + eps+)`. With central differences the bounds widen by
`eps_h`. The test is conservative: the event is a worst case. The observed
spread is reported as a diagnostic, and it is **never substituted** for the
declared constant. A constant estimated from the same sample would make the
band circular, and a sample can refute a constant but never confirm one.

## 3. Scope: lower rank, and a kernel that moves

Lemma 1 and Corollary 8 assume `Q > 0`. The note's Remark 14 extends **only
Theorem 11** to a singular weight. It does so under the sandwich (A1), with
both constants below one and the loss sandwich (A1a) known in advance; (A1b)
then *forces* the kernel to be fixed. The addendum is explicit: T1's price for
codes "rests on the lower quadratic sandwich, and a fixed kernel alone does
not give that sandwich."

A kernel that looks fixed in a sample is the module's own rule in reverse:
evidence that could only confirm a hypothesis, never refute it. So a weight
whose numerical rank falls short of `p` gets `NO_CERTIFICATE`, with no price
and no refutation test, since the band's event is not available to test
against. That includes a full-rank but ill-conditioned weight whose smallest
eigenvalues fall under the rank cut, because the refutation test cannot see
those directions. (An earlier draft priced a lower-rank weight on its `r`
read directions when the kernel looked fixed in the sample; an independent
review of the code against the note showed that this overreached.)

The same samples give the addendum's diagnostic, for read metrics at points
(`kernel_geometry`, item #5 of the plan). With `q = rank Q` (Lemma 1 of the
addendum) and `r` the rank of a sample's read metric, the kernel moves when
`q > r`. Then fixed-covariance Gaussian test channels (the class a single
weight serves) exceed the consumer's own rate-distortion function by
`((q - k)/2) log2(1/D) + O(1)` bits as the loss `D -> 0`, with `k` the
consumer's output dimension (Corollary 9). Two cautions:

- `q` here is the rank of the *sample* mean, which is at most `rank Q`, so the
  reported coefficient is a lower bound on the corollary's.
- The corollary's other hypotheses (a measurable right inverse on a closed
  convex image, finite differential entropy, bounded derivatives, `Q` under
  the source law) are the caller's to establish.

The corollary also needs the output to have a density in `R^k`, so the coefficient is reported
only when some sample reaches rank `k`. It is a statement about one class of
test channels, not about every code, and it says when per-region read
operators (per IVF cell, per stratum) can pay at high fidelity.

## 4. What the numbers look like

Least `m` for a target price (`delta = 0.05`, `nu- = 1`), computed with
`probes_for_price` on Atlas. "Corollary" is the note's closed form; "minimal"
is the least `m` at which Lemma 1's own band reaches the target.

| p | nu+ | target (bits/vector) | minimal m | Corollary 3 m |
|---:|---:|---:|---:|---:|
| 16 | 1 | 1 | 7,092 | 36,759 |
| 16 | 1 | 2 | 1,829 | 9,190 |
| 128 | 1 | 1 | 584,623 | 3,109,651 |
| 128 | 1 | 16 | 2,417 | 12,148 |
| 128 | 10 | 16 | 23,537 | 121,471 |
| 1024 | 1 | 128 | 3,006 | 15,105 |
| 1024 | 10 | 128 | 29,268 | 151,045 |

At head dimension 128 the default `n_probe_keys = 32` leaves a learning part
of **281 bits per vector even at `nu+ = 1`**: more than half of a 4-bit code
of that dimension (512 bits). The attention consumer is nonlinear, so
Corollary 8's whole gap is larger still. 10,000 samples bring the learning
part to 7.7 bits.

These are worst-case guarantees of the confidence statement, sharp only for
an encoder that sees nothing but `Q_hat` and the band. The average loss from
using `Q_hat` is typically smaller, and measuring that loss is a separate
claim.

## 5. What is not claimed

- Nothing here measures a codec. The price bounds a frontier gap.
- `nu+` is the caller's, and must be fixed before the sample. A priori routes
  are in the note: `nu+ = M1² / lambda - 1` from the consumer's Lipschitz
  constant and an a priori `lambda <= lambda_min(Q)` (Corollary 8).
- Noisy consumers: output noise adds a bias of order `sigma²/h²` to central
  differences, and with Gaussian noise the samples are unbounded, which
  Lemma 1 does not cover (Remark 17). Not handled.
- The event is a confidence statement about the learned operator. It is not
  a monitor's false-clear rate (Remark 19).

## 6. Tests

`tests/test_learning_price.py`:
- The note's own numbers. For `x + 0.2 sin x`, `m = 32`, `delta = 0.1`:
  `eps± = (0.186, 0.208)`, and exact roots `(0.177, 0.189)`.
- Coverage of at least `1 - delta` at the two-point law where the variance
  bound is attained, with an **I0 self-test**: a band a quarter as wide is
  caught.
- True constants are refuted at no more than the test's level; a false one is
  refuted.
- Invariance under rotation and scale; the factor and metric forms agree.
- No certificate for a lower-rank weight, whether its kernel looks fixed or
  moves, nor for an ill-conditioned one cut by the numerical rank.
- Query samples price a linear consumer, with a regret-bound scope. Point
  metrics with a moving kernel at full rank are priced as a learning part, and
  Corollary 8's gap is checked against its formula, including its refusal
  when `rho_bar(D) >= D`.
- The addendum's example `x1 + a sin x2` (`q = 2`, `r = k = 1`, coefficient
  1/2).
- Proposition 2's bound on a `C³` consumer; Corollary 3(ii)'s step.
- Proposition 4's sandwich and bit count.
- `attention_analytic`'s operator is unchanged bit for bit, and its new
  `point_factors` average to it.
- The CLI's exit codes, the contract path's `delta` split, and the refusal
  when a sampled consumer has no declared constant.

## 7. Next

- Record the learning price in plan records and certificates whose read
  operator was sampled, so a certificate states what its weight cost.
- readscope's blind provider: expose `point_factors`, and use Proposition 2
  for the step.
- Proposition 4 as the format for shipping a learned weight to a decoder that
  cannot probe the consumer.

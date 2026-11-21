# Theory: low-rank weights when the deploying consumer is not the calibrating one

This note is the theoretical spine of Part IV
(`docs/PREREG_consumer_mismatch_lowrank.md`). It states what a consumer-relative view predicts
about low-rank compression of a model's linear maps when the text a model is calibrated on
differs from the text it serves. Every statement below is checked numerically, exactly where
the second-order model is exact, by `benchmarks/consumer_mismatch/theory_check.py`.

## 0. What is known, and what is claimed here

- **The factorization is not new.** Truncating `P^{1/2} W Σ^{1/2}` is the optimal low-rank
  approximation under the Kronecker-factored second-order loss. It is the two-sided whitening of
  Optimal Brain Decomposition (arXiv 2604.00821) and IO-SVD (arXiv 2605.15626). Its one-sided
  form (`P = I`) is SVD-LLM. Data Mining as Observation (ch. 4, eq. for the truncation floor
  `tr(P̃ Π)`, Lean `ReadDistortion.lean`) states the same floor in consumer-relative terms. IO-SVD
  also allocates rank across layers under a global budget.
- **What these sources do not treat** (from their abstracts and summaries; full texts to be
  checked before any claim of priority) is calibration on one consumer and deployment on
  another. Sections 2 to 5 below give a decomposition of the resulting loss and bounds on each
  part. The propositions are elementary. Their use is that they turn the programme's empirical
  finding about blind read operators (readscope, GO-P-2026-041: directions transfer across
  domains, magnitudes do not) into specific, testable predictions about compression.

## 1. Setting

A linear map `y = W x` with `W` of size `m × n`. A **consumer** is the pair of second moments
it induces at this map,

    Σ = E[x xᵀ]        (inputs it sends)
    P = E[g gᵀ]        (how it reads the output; g = gradient of the log-likelihood of a label
                        sampled from the model, so P is the true Fisher at y)

and the consumer-relative distortion of a weight error `Δ` is the Kronecker-factored second-order
loss

    D(Δ) = tr(P Δ Σ Δᵀ) = ‖P^{1/2} Δ Σ^{1/2}‖²_F .

(The KL of the model is `½ D` to second order when inputs and read-outs are independent and
matrices do not interact. Both assumptions are approximations in a real model, and the
experiment measures KL directly.) `P` and `Σ` are taken positive definite, after damping. A
calibrating consumer is written `c` and a deploying consumer `d`.

## 2. The calibrated solution is optimal for its own consumer

**Proposition 1 (weighted Eckart and Young).** Let `σ₁ ≥ σ₂ ≥ …` be the singular values of
`W̃ = P^{1/2} W Σ^{1/2}`, with singular vectors `u_k, v_k`. Then

    min over rank(Ŵ) ≤ r of D(W − Ŵ) = Σ_{k>r} σ_k²,

attained by `Ŵ_r = P^{-1/2} [W̃]_r Σ^{-1/2}`.

*Proof.* `M ↦ P^{1/2} M Σ^{1/2}` is a bijection of the matrices of rank at most `r`, and it maps
`D` to the Frobenius norm. Apply Eckart and Young. ∎

The solution is a sum of components `Ŵ_r = Σ_{k≤r} a_k b_kᵀ` with `a_k = σ_k P^{-1/2} u_k` and
`b_k = Σ^{-1/2} v_k`. Under the calibrating consumer the components are orthogonal, and dropping
component `k` costs exactly `σ_k²`.

## 3. A change of consumer within one map

**Proposition 2 (order monotonicity).** If `α₋ P_c ⪯ P_d ⪯ α₊ P_c` and `β₋ Σ_c ⪯ Σ_d ⪯ β₊ Σ_c`,
then for every `Δ`

    α₋ β₋ D_c(Δ) ≤ D_d(Δ) ≤ α₊ β₊ D_c(Δ).

*Proof.* For positive semidefinite `B`, `Δ B Δᵀ ⪰ 0`, so `tr(A Δ B Δᵀ)` does not decrease when
`A` grows in the semidefinite order. The same holds in `B` for fixed `A`. Apply each side's
bounds in turn. ∎

**Corollary 3 (mismatch bound for one map).** Let `Ŵ_c` be the rank-`r` solution calibrated on
`c` and `D*_d(r)` the best rank-`r` distortion for `d`. Then

    D_d(W − Ŵ_c) ≤ κ_P κ_Σ · D*_d(r),

where `κ_P = α₊/α₋` is the condition number of `P_c^{-1/2} P_d P_c^{-1/2}` and `κ_Σ` likewise.

*Proof.* `D_d(W − Ŵ_c) ≤ α₊β₊ D_c(W − Ŵ_c) ≤ α₊β₊ D_c(W − Ŵ*_d) ≤ (α₊β₊ / α₋β₋) D_d(W − Ŵ*_d)`,
using Proposition 2, then the optimality of `Ŵ_c` for `c`, then Proposition 2 again. ∎

**Corollary 4 (a change of scale costs nothing within one map).** If `P_d = a P_c` and
`Σ_d = b Σ_c`, then `κ_P = κ_Σ = 1`. The calibrated solution is optimal for `d` at every rank, and
`D_d = a b D_c`.

The bound depends only on the relative spectra of the two consumers. A consumer that changes how
strongly each direction is read, but not which directions are read, has a small `κ`.

## 4. A change of consumer across maps, under a global budget

A model has maps `ℓ = 1 … L`. Keeping one component of map `ℓ` costs `w_ℓ = m_ℓ + n_ℓ`
parameters and multiply-adds per token. A **global allocation** chooses ranks `r_ℓ` to minimize
`Σ_ℓ D_ℓ` subject to `Σ_ℓ w_ℓ r_ℓ ≤ B`. A **uniform allocation** fixes each `r_ℓ` to the same
fraction of the map's own break-even rank `m_ℓ n_ℓ / (m_ℓ + n_ℓ)`.

**Proposition 5 (global allocation is water filling).** Each map's tail `Σ_{k>r} σ_{ℓ,k}²` has
non-increasing decrements in `r`. Keeping every component with `σ_{ℓ,k}² / w_ℓ ≥ τ` for a single
threshold `τ` solves the budget's linear relaxation exactly. The integer allocation obtained by
filling greedily in that order is feasible, and its loss exceeds the relaxation's, and so the integer optimum's, by at most the
`σ_{ℓ,k}²` of the first component that does not fit, a bound the harness reports with every
allocation.
(`turboquant_pro.weight_plan` solves the bit-width version of this knapsack exactly on an integer
lattice. A rank lattice is too fine for that dynamic program, so this greedy rule with its
reported bound is used instead.)

**Proposition 6 (the mismatch loss splits into a within-map part and an allocation part).** Let
`r_c` be the global allocation computed on `c`, `Ŵ_{c,ℓ}` the calibrated solutions at those
ranks, and `D*_{d,ℓ}(r)` the best rank-`r` distortion of map `ℓ` for `d`. The deployment loss
above the deployment optimum is

    Σ_ℓ D_{d,ℓ}(W_ℓ − Ŵ_{c,ℓ}) − min over budget-feasible r of Σ_ℓ D*_{d,ℓ}(r_ℓ)
      = SUB + ALLOC,
    SUB   = Σ_ℓ [D_{d,ℓ}(W_ℓ − Ŵ_{c,ℓ}) − D*_{d,ℓ}(r_{c,ℓ})]                  ≥ 0,
    ALLOC = Σ_ℓ D*_{d,ℓ}(r_{c,ℓ}) − min over budget-feasible r of Σ_ℓ D*_{d,ℓ}(r_ℓ) ≥ 0.

*Proof.* Add and subtract `Σ_ℓ D*_{d,ℓ}(r_{c,ℓ})`. `SUB ≥ 0` term by term by optimality of
`D*`, and `ALLOC ≥ 0` because `r_c` is feasible. ∎

By Corollary 3, `SUB ≤ Σ_ℓ (κ_{P,ℓ} κ_{Σ,ℓ} − 1) D*_{d,ℓ}(r_{c,ℓ})`. Under a uniform allocation
the ranks do not depend on the consumer, so `ALLOC = 0` and the whole mismatch loss is `SUB`.

**Proposition 7 (allocation loss under a change of scale per map).** Suppose `P_{d,ℓ} = a_ℓ
P_{c,ℓ}` and `Σ_{d,ℓ} = b_ℓ Σ_{c,ℓ}`, and write `s_ℓ = a_ℓ b_ℓ`. Then `SUB = 0`, the deployment
spectra are `s_ℓ σ_{ℓ,k}²`, and

    ALLOC = Σ_ℓ s_ℓ tail_ℓ(r_{c,ℓ}) − min over budget-feasible r of Σ_ℓ s_ℓ tail_ℓ(r_ℓ),

which is computable from the calibration spectra and the scalars `s_ℓ` alone. `ALLOC = 0` when
every `s_ℓ` is equal. When they differ, the two allocations keep different components exactly
where the ordering of `s_ℓ σ_{ℓ,k}² / w_ℓ` differs from that of `σ_{ℓ,k}² / w_ℓ` across the
threshold.

*Proof.* Corollary 4 within each map gives `SUB = 0` and the scaled spectra. Substitute into
Proposition 6. Equal `s_ℓ` rescale every ratio by one constant and preserve the ordering. ∎

**Remark (size, not proved).** Components that change sides lie near the threshold, so for a
small spread of `log s_ℓ` both their number and their individual cost difference are small, and
`ALLOC` is expected to grow faster than linearly in that spread. The experiment does not rely on
this.

## 5. What the decomposition predicts, and a cheap correction

The programme's measurement of blind read operators across domains (readscope, GO-P-2026-041)
found that the read **directions** transfer (AUROC 0.975) while their **magnitudes** do not. In
the terms above, a domain shift keeps `κ_{P,ℓ}` and `κ_{Σ,ℓ}` moderate and moves the per-map
scales `s_ℓ`. The decomposition then predicts:

1. **Mismatch hurts a global allocation more than a uniform one.** Uniform ranks carry only
   `SUB`, and global ranks carry `SUB + ALLOC`.
2. **Re-estimating magnitudes on the deploying text recovers most of the loss, without
   recomputing any subspace.** Keep the calibrated components `a_k b_kᵀ` and rescore each with
   the deploying consumer,

       γ_k^d = (a_kᵀ P_d a_k)(b_kᵀ Σ_d b_k) = E_d[(a_kᵀ g)²] · E_d[(b_kᵀ x)²],

   then reallocate by water filling on `γ_k^d / w_ℓ`. The rescoring needs only the projections
   `a_kᵀ g` and `b_kᵀ x` on a few deploying sequences, with labels sampled from the model, so no
   labelled deploying data, no `m × m` or `n × n` statistic and no new decomposition. Under the
   per-map scale model of Proposition 7 it is exact. A coarser version uses one scalar per map,
   `ŝ_ℓ = (E_d‖g‖² / E_c‖g‖²)(E_d‖x‖² / E_c‖x‖²)`, which is also exact under that model.
3. **With few deploying sequences, rescoring beats recalibrating from scratch.** A full
   recalibration must estimate `n × n` and `m × m` matrices. Rescoring estimates one variance per
   kept component.
4. **The second-order distortion should rank the measured damage.** `Σ_ℓ D_{d,ℓ}` computed on
   held-out deploying statistics is a prediction made before the model is run, and it should
   order the arms by their measured KL.

Prediction 1 is the decomposition's own test. Predictions 2 and 3 are the practical claims.
Prediction 4 checks that the second-order model is good enough for the first three to matter.
A failure of 1 with 4 holding would mean domain shift moves directions, not magnitudes, for these
maps, and would contradict the transfer of the blind directions to this setting.

## 6. What is exact and what is approximate

- Propositions 1 to 7 are exact statements about `D`.
- `D` is the second-order loss under two assumptions (independent inputs and read-outs, and no
  interaction between maps). Part III found that the full `P_y` adds nothing over its diagonal in
  ranking quantization damage. Low-rank error is structured rather than noise-like, so that
  result does not carry over automatically, and Part IV measures KL rather than assuming `D`.
- `theory_check.py` verifies 1 to 7 on synthetic consumers where `D` is the exact loss: the
  bound of Corollary 3 holds on every draw, and is attained up to the expected slack. Corollary 4
  and Proposition 7 hold to rounding. The decomposition of Proposition 6 is an identity, and the
  allocation loss of Proposition 7 matches a brute-force allocation. The rescoring of section 5 is
  exact under per-map scaling, and its error grows with `κ` when directions also change.

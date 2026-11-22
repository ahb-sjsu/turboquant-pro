/-
Formal checks for "Quantization Sensitivity Follows the Consuming Operator".

Each theorem is one mathematical statement the paper uses. They are elementary; the point
is that the text's identities and its sufficient condition are machine-checked, with no
proof left open. Nothing here is a statement about a trained model: the empirical claims are in the
JSON records and the notebook.

Lean 4 with Mathlib. Section references are to the paper.
-/
import Mathlib

open Finset

namespace OperatorChecks

/-! ## Section 3: decision-level agreement -/

section Agreement

variable {ι : Type*} [Fintype ι] (f g : ι → Prop) [DecidablePred f] [DecidablePred g]

/-- Items the base model gets right split into those the quantized model also gets right
(Correctness Agreement) and the regressions. So CA = base accuracy − r. -/
theorem base_correct_split :
    (univ.filter f).card
      = (univ.filter fun i => f i ∧ g i).card + (univ.filter fun i => f i ∧ ¬ g i).card := by
  classical
  rw [← Finset.card_union_of_disjoint]
  · congr 1
    ext i
    simp only [mem_filter, mem_univ, true_and, mem_union]
    tauto
  · rw [Finset.disjoint_left]
    intro i hi hj
    simp only [mem_filter, mem_univ, true_and] at hi hj
    exact hj.2 hi.2

/-- The same split for the quantized model: agreements on correct items and recoveries. -/
theorem quant_correct_split :
    (univ.filter g).card
      = (univ.filter fun i => f i ∧ g i).card + (univ.filter fun i => ¬ f i ∧ g i).card := by
  classical
  rw [← Finset.card_union_of_disjoint]
  · congr 1
    ext i
    simp only [mem_filter, mem_univ, true_and, mem_union]
    tauto
  · rw [Finset.disjoint_left]
    intro i hi hj
    simp only [mem_filter, mem_univ, true_and] at hi hj
    exact hj.1 hi.1

/-- Regressions minus recoveries is the drop in the number of correct items, so
`r − c` is the accuracy change. -/
theorem regressions_sub_recoveries :
    ((univ.filter fun i => f i ∧ ¬ g i).card : ℤ) - (univ.filter fun i => ¬ f i ∧ g i).card
      = (univ.filter f).card - (univ.filter g).card := by
  have h1 := base_correct_split f g
  have h2 := quant_correct_split f g
  push_cast [h1, h2]
  ring

end Agreement

/-! ## Section 4: scores, softmax and the bit budget -/

section Softmax

variable {ι : Type*} [Fintype ι]

/-- The softmax cancels a constant added to a whole row of scores. -/
theorem softmax_row_constant (x : ι → ℝ) (c : ℝ) (i : ι) :
    Real.exp (x i + c) / ∑ j, Real.exp (x j + c) = Real.exp (x i) / ∑ j, Real.exp (x j) := by
  simp only [Real.exp_add]
  rw [← Finset.sum_mul]
  exact mul_div_mul_right _ _ (Real.exp_pos c).ne'

/-- A row of score errors splits into its row-constant part and its centered part, and the
squared norms add. This is the decomposition behind the row-constant share. -/
theorem row_constant_decomposition (d : ι → ℝ) (c : ℝ)
    (hc : ∑ s, d s = (Fintype.card ι : ℝ) * c) :
    ∑ s, (d s) ^ 2 = (Fintype.card ι : ℝ) * c ^ 2 + ∑ s, (d s - c) ^ 2 := by
  have h : ∑ s, (d s - c) ^ 2
      = ∑ s, (d s) ^ 2 - 2 * c * ∑ s, d s + (Fintype.card ι : ℝ) * c ^ 2 := by
    have : ∀ s, (d s - c) ^ 2 = (d s) ^ 2 - 2 * c * d s + c ^ 2 := fun s => by ring
    simp only [this, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum,
      Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  rw [h, hc]
  ring

end Softmax

/-- One eighth of the key rows at 8 bits and the rest at 3 bits average 3.625 bits. -/
theorem key_bit_budget : (1 / 8 : ℚ) * 8 + (7 / 8) * 3 = 3.625 := by norm_num

/-! ## Section 6: top-k routing -/

section Routing

variable {ι : Type*}

/-- The sufficient condition of Section 6, for one selected expert `i` and one unselected
expert `j`: if the logits are at least `m` apart and both errors are within `m/2` of a
common constant, the order is kept. -/
theorem order_kept_of_centered_error (l δ : ι → ℝ) (c m : ℝ) (i j : ι)
    (hgap : m ≤ l i - l j) (hi : |δ i - c| < m / 2) (hj : |δ j - c| < m / 2) :
    l j + δ j < l i + δ i := by
  rw [abs_lt] at hi hj
  linarith [hi.1, hj.2]

/-- The top-k set is kept: every selected expert stays above every unselected one. -/
theorem topk_set_kept (l δ : ι → ℝ) (c m : ℝ) (S : Set ι)
    (hgap : ∀ i ∈ S, ∀ j ∉ S, m ≤ l i - l j) (herr : ∀ i, |δ i - c| < m / 2) :
    ∀ i ∈ S, ∀ j ∉ S, l j + δ j < l i + δ i :=
  fun i hi j hj => order_kept_of_centered_error l δ c m i j (hgap i hi j hj) (herr i) (herr j)

/-- The spread form used for the measured ratio `R`: if every error lies in `[lo, hi]` and
the spread `hi − lo` is below the margin, the top-k set is kept. So `R < 1` means no flip. -/
theorem topk_set_kept_of_spread (l e : ι → ℝ) (lo hi m : ℝ) (S : Set ι)
    (hgap : ∀ i ∈ S, ∀ j ∉ S, m ≤ l i - l j)
    (hlo : ∀ i, lo ≤ e i) (hhi : ∀ i, e i ≤ hi) (hR : hi - lo < m) :
    ∀ i ∈ S, ∀ j ∉ S, l j + e j < l i + e i := by
  intro i hiS j hjS
  linarith [hgap i hiS j hjS, hlo i, hhi j]

/-- Scaling a router's logits and its quantization error by the same positive constant
changes no routing comparison. So absolute margins are not comparable across models. -/
theorem routing_scale_invariant (l e : ι → ℝ) (s : ℝ) (hs : 0 < s) (i j : ι) :
    s * l j + s * e j < s * l i + s * e i ↔ l j + e j < l i + e i := by
  constructor
  · intro h
    by_contra hcon
    have hle : l i + e i ≤ l j + e j := not_lt.mp hcon
    nlinarith [mul_le_mul_of_nonneg_left hle hs.le]
  · intro h
    nlinarith [mul_pos hs (sub_pos.mpr h)]

end Routing

/-! ## Section 7: state decay -/

section Decay

/-- Mamba's parameterization: with `A = −exp A_log` and `a = exp (Δ A)`, the decay rate is
`−log a = Δ · exp A_log`. -/
theorem decay_rate (Δ Alog : ℝ) :
    -Real.log (Real.exp (Δ * -Real.exp Alog)) = Δ * Real.exp Alog := by
  rw [Real.log_exp]
  ring

/-- Hence `log (−log a) = log Δ + A_log`: a uniform grid on `A_log` is a uniform grid on the
logarithm of the decay rate. -/
theorem log_decay_rate (Δ Alog : ℝ) (hΔ : 0 < Δ) :
    Real.log (-Real.log (Real.exp (Δ * -Real.exp Alog))) = Real.log Δ + Alog := by
  rw [decay_rate, Real.log_mul hΔ.ne' (Real.exp_pos Alog).ne', Real.log_exp]

/-- The steady state of `h = a h + b` is `b / (1 − a)`. -/
theorem steady_state (a b h : ℝ) (ha : a ≠ 1) (hfix : h = a * h + b) : h = b / (1 - a) := by
  have h1 : (1 - a) ≠ 0 := sub_ne_zero.mpr (Ne.symm ha)
  rw [eq_div_iff h1]
  linarith

/-- The change in steady-state gain from a change in the decay: the factor
`1 / ((1 − a')(1 − a))` grows without bound as both decays approach 1. -/
theorem gain_difference (a a' : ℝ) (ha : a ≠ 1) (ha' : a' ≠ 1) :
    1 / (1 - a') - 1 / (1 - a) = (a' - a) / ((1 - a') * (1 - a)) := by
  have h1 : (1 - a) ≠ 0 := sub_ne_zero.mpr (Ne.symm ha)
  have h2 : (1 - a') ≠ 0 := sub_ne_zero.mpr (Ne.symm ha')
  field_simp
  ring

end Decay

end OperatorChecks

#print axioms OperatorChecks.base_correct_split
#print axioms OperatorChecks.quant_correct_split
#print axioms OperatorChecks.regressions_sub_recoveries
#print axioms OperatorChecks.softmax_row_constant
#print axioms OperatorChecks.row_constant_decomposition
#print axioms OperatorChecks.key_bit_budget
#print axioms OperatorChecks.order_kept_of_centered_error
#print axioms OperatorChecks.topk_set_kept
#print axioms OperatorChecks.topk_set_kept_of_spread
#print axioms OperatorChecks.routing_scale_invariant
#print axioms OperatorChecks.decay_rate
#print axioms OperatorChecks.log_decay_rate
#print axioms OperatorChecks.steady_state
#print axioms OperatorChecks.gain_difference

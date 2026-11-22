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

/-! ## Section 3: the arrangement of ties is a Whitney stratification

The strata are the faces of the arrangement of hyperplanes `l i = l j`. A face is an open
piece of its flat, the linear subspace of logit vectors with a given set of ties, so its
tangent space at every point is that flat. If a face `F` lies in the closure of a face `G`,
every tie of `G` is a tie of `F`. Whitney's conditions then reduce to two statements about
flats, proved here: the flat of `F` is contained in the flat of `G` (condition (a), since
the tangent planes along `G` are constant), and every limit of rescaled secants between
points of the two faces lies in the flat of `G` (condition (b)). The identification of
tangent spaces with flats is the part stated in prose. -/

section Strata

open Filter Topology

variable {ι : Type*}

/-- The flat of a set of ties: the logit vectors on which each listed pair is tied. -/
def flat (Z : Set (ι × ι)) : Submodule ℝ (ι → ℝ) where
  carrier := {l | ∀ p ∈ Z, l p.1 = l p.2}
  zero_mem' := by
    intro p _
    rfl
  add_mem' := by
    intro a b ha hb p hp
    simp only [Pi.add_apply, ha p hp, hb p hp]
  smul_mem' := by
    intro c a ha p hp
    simp only [Pi.smul_apply, ha p hp]

theorem mem_flat {Z : Set (ι × ι)} {l : ι → ℝ} : l ∈ flat Z ↔ ∀ p ∈ Z, l p.1 = l p.2 :=
  Iff.rfl

/-- More ties give a smaller flat. -/
theorem flat_mono {Z₁ Z₂ : Set (ι × ι)} (h : Z₁ ⊆ Z₂) : flat Z₂ ≤ flat Z₁ :=
  fun _ hl p hp => hl p (h hp)

/-- A flat is closed. -/
theorem flat_isClosed (Z : Set (ι × ι)) : IsClosed {l : ι → ℝ | l ∈ flat Z} := by
  have h : {l : ι → ℝ | l ∈ flat Z} = ⋂ p ∈ Z, {l : ι → ℝ | l p.1 = l p.2} := by
    ext l
    simp [mem_flat]
  rw [h]
  exact isClosed_biInter fun p _ => isClosed_eq (continuous_apply p.1) (continuous_apply p.2)

/-- Whitney (a) for the arrangement: if every tie of `G` is a tie of `F`, the tangent space
of `F` lies in the (constant) tangent space of `G`. -/
theorem whitney_a_flats {ZF ZG : Set (ι × ι)} (h : ZG ⊆ ZF) : flat ZF ≤ flat ZG :=
  flat_mono h

/-- Whitney (b) for the arrangement: a limit of rescaled secants `c n • (y n - x n)`, with
`x n` in the smaller face and `y n` in the larger, lies in the tangent space of the larger.
Both sequences lie in the flat of the larger face, since the smaller face is in its closure. -/
theorem whitney_b_secants (Z : Set (ι × ι)) (x y : ℕ → ι → ℝ) (c : ℕ → ℝ) (u : ι → ℝ)
    (hx : ∀ n, x n ∈ flat Z) (hy : ∀ n, y n ∈ flat Z)
    (hu : Tendsto (fun n => c n • (y n - x n)) atTop (𝓝 u)) : u ∈ flat Z := by
  have hmem : ∀ n, c n • (y n - x n) ∈ {l : ι → ℝ | l ∈ flat Z} :=
    fun n => (flat Z).smul_mem (c n) ((flat Z).sub_mem (hy n) (hx n))
  exact (flat_isClosed Z).mem_of_tendsto hu (Eventually.of_forall hmem)

/-- The smaller face lies in the flat of the larger: with `whitney_b_secants`, this is the
hypothesis `hx` whenever every tie of `G` is a tie of `F`. -/
theorem smaller_face_in_larger_flat {ZF ZG : Set (ι × ι)} (h : ZG ⊆ ZF) {l : ι → ℝ}
    (hl : l ∈ flat ZF) : l ∈ flat ZG :=
  flat_mono h hl

end Strata

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
#print axioms OperatorChecks.flat_mono
#print axioms OperatorChecks.flat_isClosed
#print axioms OperatorChecks.whitney_a_flats
#print axioms OperatorChecks.whitney_b_secants
#print axioms OperatorChecks.smaller_face_in_larger_flat
#print axioms OperatorChecks.mem_flat

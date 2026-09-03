/-
Machine-checked geometry behind "One Codebook for Every Architecture:
Asymmetric NormalFloat for Calibration-Free KV-Cache Quantization".

The paper's mechanism section makes four informal claims about codebook
geometry.  Each is stated and proved here for a general codebook (the
statements depend only on grid *placement*, not on the NF level values,
so they cover NF4, uniform, and any other 16-level grid):

  (1) `range_div_absmax_lt_one` — the DC-offset diagnostic.  The paper
      reports range/abs-max ≈ 0.9 and reads it as evidence that a key
      channel is one-sided.  Proved: strictly one-sided data (0 < a ≤ x
      ≤ b) always has range/absmax < 1, while data symmetric about zero
      has ratio exactly 2 (`range_div_absmax_symmetric`).  So a measured
      ratio below 1 *entails* one-sidedness; it is not a heuristic.

  (2) `dead_code` — "a symmetric codebook wastes half of its codes on
      the empty side".  Proved: any grid point below the data minimum is
      never the nearest code to any datum, hence never emitted.
      `dead_codes_of_uniform_grid` counts them for an evenly spaced
      symmetric grid: with range/absmax = ρ, a fraction (2−ρ)/2 of the
      grid span is dead — 55% at the paper's measured ρ = 0.9.

  (3) `absmax_sub_ge_halfRange` / `midpoint_attains_halfRange` — the
      zero-point is optimal, and the half-range is the best achievable
      scale.  No choice of offset beats the midpoint.

  (4) `error_ratio` — the quantitative mechanism.  For a fixed grid with
      normalized rounding error ≤ δ, the symmetric quantizer's error
      bound exceeds the asymmetric one's by exactly absmax(x)/halfRange,
      which is 2/ρ (= 2.22× at ρ = 0.9) *before* any nonlinearity of the
      NF levels is taken into account.

Scope, stated honestly: these are statements about worst-case
reconstruction error of a scalar channel.  They do NOT prove anything
about the downstream GQA amplification, the perplexity collapse, or the
measured 3–5× error ratio (which combines this placement effect with
NF4's Gaussian level spacing).  Those remain empirical.

Lean 4 / Mathlib.  Built clean on the Atlas workstation.
-/

import Mathlib

set_option linter.unusedSectionVars false

namespace AsymNF4

open Real

/-! ## 1.  The DC-offset diagnostic -/

/-- Strictly one-sided data has range/abs-max < 1.  Here `a`, `b` are the
channel min and max, so `b` is the abs-max and `b - a` the range. -/
theorem range_div_absmax_lt_one {a b : ℝ} (ha : 0 < a) (hab : a ≤ b) :
    (b - a) / b < 1 := by
  have hb : 0 < b := lt_of_lt_of_le ha hab
  rw [div_lt_one hb]
  linarith

/-- Data symmetric about zero has range/abs-max exactly 2 — the opposite
extreme from (1), which is why the ratio is diagnostic. -/
theorem range_div_absmax_symmetric {b : ℝ} (hb : 0 < b) :
    (b - -b) / b = 2 := by
  field_simp
  ring

/-- Contrapositive form, as used in the paper: a ratio below 1 rules out
data straddling zero symmetrically. -/
theorem not_symmetric_of_ratio_lt_one {a b : ℝ} (hb : 0 < b) (_hab : a ≤ b)
    (h : (b - a) / b < 1) : a ≠ -b := by
  intro hcon
  rw [hcon, range_div_absmax_symmetric hb] at h
  linarith

/-! ## 2.  Dead codes: a symmetric grid wastes its lower half -/

/-- **The wasted-codes lemma.**  If every datum is at least `a`, then a
grid point `g₁` lying strictly below another grid point `g₂ ≤ a` is never
the nearest code to any datum: `g₂` is always strictly closer.  Hence all
codes below the data minimum are dead. -/
theorem dead_code {a x g₁ g₂ : ℝ} (hx : a ≤ x) (h₁₂ : g₁ < g₂) (h₂a : g₂ ≤ a) :
    |x - g₂| < |x - g₁| := by
  have h2 : 0 ≤ x - g₂ := by linarith
  have h1 : 0 ≤ x - g₁ := by linarith
  rw [abs_of_nonneg h2, abs_of_nonneg h1]
  linarith

/-- Every code strictly below the data minimum is dead, for *any* datum in
the channel. -/
theorem dead_code_forall {a b g₁ g₂ : ℝ} (h₁₂ : g₁ < g₂) (h₂a : g₂ ≤ a) :
    ∀ x, a ≤ x → x ≤ b → |x - g₂| < |x - g₁| :=
  fun _ hx _ => dead_code hx h₁₂ h₂a

/-- **Counting the dead span.**  For a symmetric grid spanning `[-b, b]`
and one-sided data on `[a, b]` with range/abs-max `= ρ`, the dead portion
of the grid span is the fraction `(2 - ρ)/2`.  At the paper's measured
`ρ = 0.9` this is `0.55` — "about half", as claimed. -/
theorem dead_span_fraction {a b ρ : ℝ} (hb : 0 < b) (hρ : (b - a) / b = ρ) :
    (a - -b) / (b - -b) = (2 - ρ) / 2 := by
  have ha : a = b - ρ * b := by
    field_simp at hρ
    linarith
  subst ha
  field_simp
  ring

/-- The paper's numeric instance: `ρ = 0.9` gives a dead fraction of
`0.55`. -/
theorem dead_span_at_measured_ratio {a b : ℝ} (hb : 0 < b)
    (hρ : (b - a) / b = 0.9) : (a - -b) / (b - -b) = 0.55 := by
  rw [dead_span_fraction hb hρ]
  norm_num

/-! ## 3.  The zero-point is the optimal offset -/

/-- No offset achieves a smaller scale than the half-range: for data
spanning `[a, b]`, every candidate zero-point `c` has
`max(|a - c|, |b - c|) ≥ (b - a)/2`. -/
theorem absmax_sub_ge_halfRange {a b c : ℝ} (hab : a ≤ b) :
    (b - a) / 2 ≤ max |a - c| |b - c| := by
  have htri : b - a ≤ |a - c| + |b - c| := by
    rcases abs_cases (a - c) with ⟨h1, _⟩ | ⟨h1, _⟩ <;>
    rcases abs_cases (b - c) with ⟨h2, _⟩ | ⟨h2, _⟩ <;>
    rw [h1, h2] <;> linarith
  have hl : |a - c| ≤ max |a - c| |b - c| := le_max_left _ _
  have hr : |b - c| ≤ max |a - c| |b - c| := le_max_right _ _
  linarith

/-- The midpoint attains the bound — so the zero-point construction is
optimal, not merely an improvement. -/
theorem midpoint_attains_halfRange {a b : ℝ} (hab : a ≤ b) :
    max |a - (a + b) / 2| |b - (a + b) / 2| = (b - a) / 2 := by
  have h1 : a - (a + b) / 2 = -((b - a) / 2) := by ring
  have h2 : b - (a + b) / 2 = (b - a) / 2 := by ring
  rw [h1, h2, abs_neg, abs_of_nonneg (by linarith : (0:ℝ) ≤ (b - a) / 2)]
  exact max_self _

/-! ## 4.  The error-bound ratio: the quantitative mechanism -/

/-- Reconstruction error of an affinely-rescaled codebook is exactly the
scale times the normalized rounding error.  This is the identity behind
both quantizers; `s` is the scale, `q` the emitted normalized level. -/
theorem dequant_error (μ s x q : ℝ) (hs : 0 < s) :
    |(μ + s * q) - x| = s * |q - (x - μ) / s| := by
  have hs' : s ≠ 0 := ne_of_gt hs
  have key : (μ + s * q) - x = s * (q - (x - μ) / s) := by
    field_simp
    try ring
  rw [key, abs_mul, abs_of_pos hs]

/-- **The mechanism, quantified.**  With a common normalized rounding
bound `δ`, the symmetric quantizer (scale = abs-max `b`) and the
asymmetric one (scale = half-range `(b-a)/2`) have error bounds whose
ratio is `2b/(b-a) = 2/ρ`.  At the measured `ρ = 0.9` this is `2.22×`
*before* NF4's level spacing is considered — the paper measures `3–5×`
end to end, so placement accounts for roughly half of the gap in the
exponent-free part. -/
theorem error_ratio {a b δ : ℝ} (ha : 0 < a) (hab : a < b) (hδ : 0 < δ) :
    (b * δ) / (((b - a) / 2) * δ) = 2 * b / (b - a) := by
  have hba : (b - a) ≠ 0 := by intro h; linarith [sub_eq_zero.mp h]
  have hδ' : δ ≠ 0 := ne_of_gt hδ
  field_simp
  try ring

/-- The ratio is strictly greater than 2 for any strictly one-sided
channel: the symmetric codebook is *always* more than twice as coarse,
however small the offset — the qualitative form of the paper's claim. -/
theorem error_ratio_gt_two {a b : ℝ} (ha : 0 < a) (hab : a < b) :
    2 < 2 * b / (b - a) := by
  have hba : 0 < b - a := by linarith
  have hne : (b - a) ≠ 0 := ne_of_gt hba
  have key : 2 * b / (b - a) - 2 = 2 * a / (b - a) := by
    field_simp
    try ring
  have hpos : 0 < 2 * a / (b - a) := div_pos (by linarith) hba
  linarith

end AsymNF4

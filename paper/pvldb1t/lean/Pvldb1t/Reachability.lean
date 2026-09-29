/-
The reachability identity behind Table 2 of "Recall Does Not Move at a Trillion Rows".

Setting. A query has a reference top-k, the finite set `N` of its neighbours under the
exact scan of the compressed index. Every neighbour lives in one inverted-file cell, and
the router orders the cells for the query, so each neighbour `x` has a cell rank
`rank x` in that order. A routed pass of width `p` scans the cells of rank below `p`.

Hypothesis. Inside a probed cell the routed pass scores with the same asymmetric
distance the exact scan used, so a reference neighbour is returned exactly when its cell
was probed. `h` states that for every member of the reference set. `hR` says the routed
result set `R` is a subset of the reference set, which is what "recall of R against N"
counts.

Conclusion. The routed set is the set of neighbours whose rank is below `p`, so recall
at width `p` equals the share of neighbours with rank below `p`.
-/
import Mathlib

open Finset

namespace Pvldb1t

variable {ι : Type*}

/-- The routed result is exactly the reference neighbours whose cell rank is below `p`. -/
theorem routed_eq_filter (N R : Finset ι) (rank : ι → ℕ) (p : ℕ)
    (hR : R ⊆ N) (h : ∀ x ∈ N, x ∈ R ↔ rank x < p) :
    R = N.filter (fun x => rank x < p) := by
  ext x
  constructor
  · intro hx
    exact mem_filter.mpr ⟨hR hx, (h x (hR hx)).mp hx⟩
  · intro hx
    obtain ⟨hxN, hlt⟩ := mem_filter.mp hx
    exact (h x hxN).mpr hlt

/-- Recall at width `p`, as a count, equals the number of neighbours of rank below `p`. -/
theorem recall_card [DecidableEq ι] (N R : Finset ι) (rank : ι → ℕ) (p : ℕ)
    (hR : R ⊆ N) (h : ∀ x ∈ N, x ∈ R ↔ rank x < p) :
    (R ∩ N).card = (N.filter (fun x => rank x < p)).card := by
  rw [inter_eq_left.mpr hR, routed_eq_filter N R rank p hR h]

/-- Recall at width `p` as a fraction of the reference set equals the share of neighbours
whose cell rank is below `p`. This is the identity the prediction row of Table 2 uses. -/
theorem recall_eq_share [DecidableEq ι] (N R : Finset ι) (rank : ι → ℕ) (p : ℕ)
    (hR : R ⊆ N) (h : ∀ x ∈ N, x ∈ R ↔ rank x < p) :
    ((R ∩ N).card : ℚ) / N.card = ((N.filter (fun x => rank x < p)).card : ℚ) / N.card := by
  rw [recall_card N R rank p hR h]

/-- Widening the probe never loses a neighbour, so the predicted recall is monotone in `p`. -/
theorem share_mono (N : Finset ι) (rank : ι → ℕ) {p q : ℕ} (hpq : p ≤ q) :
    (N.filter (fun x => rank x < p)).card ≤ (N.filter (fun x => rank x < q)).card := by
  apply card_le_card
  intro x hx
  obtain ⟨hxN, hlt⟩ := mem_filter.mp hx
  exact mem_filter.mpr ⟨hxN, lt_of_lt_of_le hlt hpq⟩

/-- A width above every neighbour's rank returns every neighbour. -/
theorem share_full (N : Finset ι) (rank : ι → ℕ) (p : ℕ)
    (hp : ∀ x ∈ N, rank x < p) :
    (N.filter (fun x => rank x < p)).card = N.card := by
  congr 1
  exact filter_true_of_mem hp

end Pvldb1t

/-
The upper-bound lemma behind the rerank bound in "Recall Does Not Move at a Trillion Rows".

Setting. `U` is the corpus, `S ⊆ U` a shortlist, `A ⊆ S` the compressed (ADC) top-k, which
the shortlist contains by construction. `topk U score k` is the float top-k over the whole
corpus and `topk S score k` the float top-k within the shortlist, both defined as the
elements with fewer than `k` strictly better elements in the set searched.

Conclusion. The number of ADC neighbours that are true float neighbours is at most the
number that survive the float rerank of the shortlist. So the survival rate measured on
the shortlist is an upper bound on ADC-only true recall, without knowing the float top-k
over the corpus.

The general lemma `card_inter_le_of_inter_subset` states the set-theoretic fact with the
ordering hypothesis explicit, and `topk_inter_subset` discharges that hypothesis for the
top-k definition above.
-/
import Mathlib

open Finset

namespace Pvldb1t

variable {ι : Type*} [DecidableEq ι] {α : Type*} [LinearOrder α]

/-- The elements of `U` with fewer than `k` strictly better elements in `U`. With no ties
this is the usual top-`k`; with ties it is the set of elements that some top-`k` contains. -/
def topk (U : Finset ι) (score : ι → α) (k : ℕ) : Finset ι :=
  U.filter (fun x => (U.filter (fun y => score x < score y)).card < k)

/-- The set-theoretic fact. If `A ⊆ S` and every member of `T` that lies in `S` is in
`T_S`, then `A` meets `T` in no more elements than it meets `T_S`. -/
theorem card_inter_le_of_inter_subset (A S T TS : Finset ι)
    (hAS : A ⊆ S) (hT : T ∩ S ⊆ TS) :
    (A ∩ T).card ≤ (A ∩ TS).card := by
  apply card_le_card
  intro x hx
  obtain ⟨hxA, hxT⟩ := mem_inter.mp hx
  exact mem_inter.mpr ⟨hxA, hT (mem_inter.mpr ⟨hxT, hAS hxA⟩)⟩

/-- A corpus top-`k` element that lies in the shortlist is a shortlist top-`k` element,
because the shortlist holds no more elements that beat it than the corpus does. -/
theorem topk_inter_subset (U S : Finset ι) (score : ι → α) (k : ℕ) (hS : S ⊆ U) :
    topk U score k ∩ S ⊆ topk S score k := by
  intro x hx
  obtain ⟨hxT, hxS⟩ := mem_inter.mp hx
  obtain ⟨_, hcard⟩ := mem_filter.mp hxT
  refine mem_filter.mpr ⟨hxS, lt_of_le_of_lt ?_ hcard⟩
  exact card_le_card (filter_subset_filter _ hS)

/-- The rerank bound. For a shortlist `S` containing the ADC top-k `A`, the ADC members
that are corpus float neighbours are at most the ADC members that survive the float
rerank within `S`. -/
theorem rerank_upper_bound (U S A : Finset ι) (score : ι → α) (k : ℕ)
    (hS : S ⊆ U) (hA : A ⊆ S) :
    (A ∩ topk U score k).card ≤ (A ∩ topk S score k).card :=
  card_inter_le_of_inter_subset A S _ _ hA (topk_inter_subset U S score k hS)

/-- The same bound as a fraction of `k`, the form the paper quotes. -/
theorem rerank_upper_bound_ratio (U S A : Finset ι) (score : ι → α) (k : ℕ)
    (hS : S ⊆ U) (hA : A ⊆ S) :
    ((A ∩ topk U score k).card : ℚ) / k ≤ ((A ∩ topk S score k).card : ℚ) / k := by
  apply div_le_div_of_nonneg_right _ (by positivity)
  exact_mod_cast rerank_upper_bound U S A score k hS hA

end Pvldb1t

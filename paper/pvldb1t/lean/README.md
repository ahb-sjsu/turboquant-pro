# Lean 4 check of the two identities the paper rests on

Built on Atlas against Mathlib at Lean `v4.32.2` (manifest pinned, Mathlib rev
`905b95818eb3`), 2026-09-29. `AXIOMS.txt` is the output of `lake env lean Axioms.lean`
after `lake build`; every theorem depends on `propext`, `Quot.sound` and, where `Finset`
intersection needs decidable equality, `Classical.choice`, and on nothing else.

| file | what it states |
|---|---|
| `Pvldb1t/Reachability.lean` | `recall_eq_share`: if the routed result is a subset of the reference top-k and a reference neighbour is returned exactly when its cell rank is below the width, then recall at that width equals the share of neighbours with rank below the width. Also monotonicity in the width and the full-recall width. |
| `Pvldb1t/RerankBound.lean` | `rerank_upper_bound`: for a shortlist `S ⊆ U` containing the ADC top-k `A`, the ADC members in the corpus float top-k are at most the ADC members in the float top-k within `S`. `topk` is the set of elements with fewer than `k` strictly better elements in the set searched, and `topk_inter_subset` discharges the ordering hypothesis. |

Rebuild on Atlas per the project's Lean recipe: pin the toolchain, symlink `.lake/packages`
to an existing Mathlib checkout of the same revision, `lake build`, then
`lake env lean Axioms.lean > AXIOMS.txt`.

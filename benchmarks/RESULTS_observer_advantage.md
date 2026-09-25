# Results: observer advantage, Part I (retrieval embeddings)

Registration: `docs/PREREG_observer_advantage.md` (merged #200, amendments 1 and 2 before any
scoring; Part II registered first, #209, as the registration required). Runs: 281 NRP jobs,
2026-09-23/24, code `68066f7`, 1654 result files. Scorer: `python -m observer_advantage.score` at
master `821c7bd`, run on Atlas on a byte-verified copy of the campaign volume (archive sha256
`1c400ef5fc4c3f37cc5e979180ea5d7391281638814dda40cd1d85b8fdeef1af`; sha256 of the sorted per-file
sha256 list of `oa/results` `d96495876012985a5d8fb73c2437d119437f79e5e58560c48949286a4051f4b1`).
Output: `benchmarks/observer_advantage/results/` (`scorer_output.md`, `scorer_output.json`).

Data: Cohere embed-english-v3 (1024-d) BEIR arms. O is the consumer (query-observer) basis, P the
corpus PCA basis, O_foreign (Of) the basis fitted to a different arm's queries. Codecs: TQ
(TurboQuant ADC), RBQ (faiss RaBitQ), OPQ (faiss OPQ), each at matched stored bytes. Endpoint:
single-pass recall@10, paired per query, 10,000-resample bootstrap, BETTER/WORSE at ±0.01.

## Gates

G0 (exact search reproduces the consumer-basis campaign to 0.002, and the O/Oey split is invisible
to exact search), G1 (O beats P under exact search in 6 of 6 cells), G2 (every codec below its
exact recall and above 0.01) and G3 (same scan path on both sides of every pair) all **PASS**.

## Verdicts

| family | H1 O beats P (asymmetric) | H2 O ties P (symmetric control) | H3 mismatch orders the gain | H4 O beats the wrong observer |
|---|---|---|---|---|
| **TQ** | **HOLDS** (12 of 12 BETTER) | **HOLDS** (6 TIE) | MIXED, ρ = +0.21 | **HOLDS** (6 of 6) |
| **OPQ** | **HOLDS** (12 of 12 BETTER) | **HOLDS** (6 TIE) | MIXED, ρ = +0.21 | **HOLDS** (6 of 6) |
| **RBQ** | **REFUTED** (8 BETTER, 2 TIE, 2 WORSE) | **HOLDS** (6 TIE) | MIXED, ρ = +0.07 | **HOLDS** (6 of 6) |

**Platform claim, by the registered rule: HOLDS FOR TQ AND OPQ ONLY.** The codec-independence
statement ("the observer layer is codec-independent") needed H1 and H2 in all three families, so it
is **not** made. It is not withdrawn either: H1 is refuted in one family, not two.

Effect sizes (single-pass recall@10, O − P, 95% CI):

- **TQ and OPQ, asymmetric arms:** +0.011 to +0.016 on msmarco and +0.016 to +0.042 on hotpotqa,
  every interval clear of zero. The gain is largest on hotpotqa at k = 64.
- **Symmetric control (msmarco-sym):** every family ties, within −0.003 to +0.0004. The gain
  appears where queries and documents differ and not where they do not, as the mechanism
  predicts.
- **Wrong observer (H4):** O beats O_foreign by +0.023 to +0.062 (TQ, OPQ) and +0.038 to +0.127
  (RBQ). The advantage belongs to the reader the basis was fitted to, not to having any
  query-derived basis.
- **RBQ's refutation:** it is on hotpotqa at 2 bits, where O is **worse** at k = 128 (−0.025) and
  k = 256 (−0.016), while at 4 bits it is better or tied in every cell. Why RaBitQ at 2 bits loses
  here is not established; the rerank recovers it (below).
- **H3 (mismatch orders the gain) is MIXED in every family.** The ordering breaks on **nq**:
  mismatch 0.507, among the highest, but O is worse than P there (−0.015 to −0.016 in every family). The
  mismatch index does not predict where the observer basis pays; it predicts where queries and
  documents differ, which is necessary (H2) but not sufficient.

## Reported, not scored

- **After 5× exact rerank** the same hypotheses read: TQ and OPQ H1, H2, H4 HOLD; RBQ H1 becomes
  MIXED (the rerank recovers its 2-bit losses), H2 and H4 HOLD; H3 MIXED throughout (ρ = +0.32).
- **The Observer Advantage table** (bytes P needs over bytes O to reach a recall level) is mostly
  "not measured": single-pass recall rarely reaches Q ≥ 0.80 at the measured byte levels. At the
  rerank endpoint, where it is measured, P/O is 1.00 in every cell. O's gain at matched bytes does
  not become a byte saving at these quality levels, because both sides cross each threshold at
  the same measured byte level.

## What it means, stated with its limits

The registered consequence (`PREREG_observer_advantage.md` section 5) for this outcome is that the
consumer basis O becomes an opt-in planner transform stage **for TQ and OPQ**, fitted from a query
sample, and not a codec-independent layer. RaBitQ is excluded at 2 bits on hotpotqa-like data.

Unlike Parts II and III, the mechanism tests pass here. The symmetric control ties, and the wrong
observer loses by more than the right one wins over PCA. Where queries and documents live in
different parts of the space, compressing in the coordinates the queries read improves ranking
at the same bytes, for two unrelated codecs. The scope is asymmetric text retrieval, one
embedding family (Cohere v3), seven BEIR arms. The mismatch index is not a usable predictor of
where the gain appears (nq), so a deployment has to measure it per corpus, which the planner's
held-out verification already does.

Across the three parts, the same line holds so far. Weighting error by what the reader uses pays:
here the observer basis in retrieval, and in Part III the reader-weighted diagonal Fisher over raw
error. Factorizing or rotating into the reader's coordinates where the codec relies on native
structure does not: the K-FAC form in Part III, and interim Part II, where the observer basis does
not beat native `nf4a` keys at 4 bits.

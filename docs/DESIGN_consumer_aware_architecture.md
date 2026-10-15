# Design: a consumer-shaped transformer, and when its shape survives a change of consumer

**Not implemented, and not registered.** This note states the question, what earlier parts
already settle, what is prior art, and a two-phase test whose first phase trains nothing. It
waits on Part V (`docs/PREREG_full_consumer_kv.md`): phase 1 should not be registered until
Part V's verdicts are scored, because two of them decide whether it is worth running (section 3).

## 0. The question

Parts II to V of the observer-advantage series take a trained model and spend a fixed budget
(key bits, weight rank) where the model's consumer reads, through the read operator
`P_C = Jᵀ G J`. This note asks whether the same quantity can set the **shape** of a model
before it is trained: the rank of each head's query and key maps, and the width of each
block, with parameters moved from where the consumer reads little to where it reads much.

An architecture is structure fixed once, but its consumer is not fixed. The read operator
moves during training, differs between domains, and changes again under fine-tuning. So the
useful question is not whether consumer-shaped structure helps at the consumer it was fitted
to. It is **whether one can predict, before training, how much a shape fitted to one consumer
loses when another consumer reads it.** That prediction is what earlier work does not offer
and what Part IV's theory gives (section 2).

## 1. What is prior art, and what is not claimed

- Sizing structure by sensitivity is not new.
  - Hessian-aware mixed precision (HAWQ and its successors) allocates bits by curvature.
  - Head pruning removes heads whose loss sensitivity is small (Michel, Levy and Neubig,
    "Are Sixteen Heads Really Better than One?", 2019; Voita et al., 2019).
  - Low-rank attention is in production: grouped-query attention (Ainslie et al., 2023) and
    multi-head latent attention (DeepSeek-V2, 2024).
  - Whitened low-rank weight factorisation, the two-sided form Part IV uses, is published
    (OBD-LLM, arXiv 2604.00821; IO-SVD, arXiv 2605.15626).
- So "allocate parameters by a sensitivity measure" is claimed by nobody here. A phase-1 arm
  that beats uniform shaping is a replication of that family unless it also beats the
  sensitivity controls in section 4.
- What may be new is the mismatch statement: a bound, in measurable condition numbers, on
  what a fitted structure loses under a different consumer, and its split into a within-map
  part and an allocation part.

## 2. What the series already establishes

- **Part II (keys, `benchmarks/RESULTS_observer_advantage_keys.md`).**
  - Coding keys in the observer's coordinates is refuted (K1, K1_low and K2 fail; K3 does not
    hold). A shape built by rotating into a read basis has no support.
  - Allocating key bits by logit reads lowered perplexity on all three 7B models but improved
    qasper only on Qwen2.5-7B (K4 inconclusive).
  - The read term did not beat key variance alone (K4b fails). Any phase-1 test must
    therefore include a variance-shaped control, or it cannot tell reads from scale.
- **readscope (C-3b, `ahb-sjsu/readscope`).** For softmax attention the read subspace of a
  head with respect to its keys is spanned by that head's queries, exactly
  (`∂p_i/∂k_s = p_i (e_s − p_is) q_i / √d`), and the instrument recovers it at resolution
  1.000 on 36 real heads across Llama, Qwen and Mistral. A head's key read operator can
  therefore be measured from its queries alone, without probing.
- **Part IV (`docs/THEORY_consumer_mismatch.md`, branch `feat/consumer-mismatch`, draft
  PR #316).** For a low-rank map fitted under consumer `c` and read by consumer `d`:
  - Proposition 2: if `α₋ P_c ⪯ P_d ⪯ α₊ P_c` and likewise for the input covariances, the
    fitted map's distortion under `d` is at most `κ_P κ_Σ` times the best possible at that
    rank, with `κ_P = α₊/α₋`.
  - A pure change of scale (`P_d = a P_c`) costs nothing within a map (`κ = 1`).
  - Proposition 6: under a global budget the mismatch loss splits exactly into a within-map
    part (SUB, directions moved) and an allocation part (ALLOC, the budget split moved).
  - Proposition 7: when each map's consumer changes only by a scale, ALLOC has a closed form.
  - These are exact statements about a fixed structure with fitted weights. A model trained
    from scratch inside a shape co-adapts its weights to it, so the bounds carry over to
    architecture only as a hypothesis. Phase 1 tests that hypothesis; it is not a theorem.
- **Part V (running).** Two of its verdicts bear directly on this note:
  - V1 and V3 ask how far downstream the consumer must be taken (logits, head output,
    the model's output) before allocation by reads pays. A shape should be set by the
    consumer that wins there, not chosen here.
  - V4 asks whether a consumer fitted on one text serves another. If it fails at the
    scale of one KV cache, a shape frozen from one consumer will fail the same way.

## 3. Phase 0: does the read operator hold still? (no training)

The idea is dead before any training if the read operator rotates freely during training or
between domains. That is measurable on public checkpoints.

- **Models.** The Pythia suite (EleutherAI), which publishes checkpoints through training at
  fixed sizes. Start at 160M; add 410M if the 160M answer is borderline.
- **What is measured, per head and per block.**
  - The key read operator of each head from its queries (readscope C-3b), and each block's
    output read operator `E[Jᵀ G J]` by readscope's Jacobian probe, at k/d ≥ 1.25 (below the
    probe's bandwidth cliff at k = d the subspace is not recovered; readscope C-2e).
  - Between consecutive checkpoints, and between two domains at the final checkpoint
    (one held-out web text and one code or scientific text): the leading-subspace
    resolution `(overlap − chance)/(1 − chance)` at the rank the shape would use, and
    `κ_P` restricted to that subspace.
  - The allocation a water-filling rule (Part IV Proposition 5) would assign from each
    operator, and how far that allocation moves.
- **What it decides.** Phase 0 reports, it does not pass or fail. The owner reads three
  curves: when (in training steps) the read subspaces and the allocation stop moving, how
  large `κ_P` is across domains, and whether the cross-domain change is mostly scale
  (ALLOC-like, which Proposition 7 corrects in closed form) or mostly direction (SUB-like,
  which nothing cheap corrects).
- **Cost.** Forward and backward passes on a few hundred sequences per checkpoint. A single
  A10 or Atlas's free GPU is enough; no training.

If the subspaces settle early and change mostly in scale across domains, phase 1 is worth
registering. If they keep rotating to the end of training, the note records that and stops.

## 4. Phase 1: shaped against uniform at matched parameters (sketch, for a later prereg)

- **Model.** A decoder of about 100M parameters (12 layers, 12 heads, width 768), trained on
  a public corpus with a fixed tokenizer, every arm on the same tokens in the same order.
- **Shape variables.** Per-head query/key rank (values and output projection left full, since
  Part II's lesson is about keys), and optionally per-block feed-forward width. Parameters
  are held equal across arms to within one head's worth.
- **Arms.**
  - `U`: uniform, the standard architecture.
  - `C`: consumer-shaped. Ranks and widths water-filled from the read operators of a pilot
    `U` run, measured at the training step phase 0 says the allocation has settled by.
  - `P`: `C`'s rank profile assigned to heads in a random permutation. Separates "a
    non-uniform profile helps" from "this assignment helps".
  - `S`: shaped by activation variance alone, no consumer. Part II's K4b control.
  - `F`: shaped by a Fisher or Hessian diagonal. The prior-art control; `C` must beat it for
    anything here to be new.
- **Outcomes.** Held-out loss in the training domain; loss in a second domain never seen
  when the shape was fitted; and the gap between the two. Three seeds per arm.
- **The prediction, written before training.** From the pilot `U` run, measure `κ_P` per map
  between the training-domain and second-domain read operators, and the ALLOC term of
  Proposition 6. The registered prediction is the ordering of `C`'s second-domain penalty
  (relative to `U`) across the two domains and the share of it the ALLOC correction
  recovers, with bars in units of the across-seed spread (as in Parts II and V).
- **Cost (an estimate, not measured).** About 6 × 10⁸ × 2 × 10⁹ ≈ 1.2 × 10¹⁸ FLOPs per run
  at 2B tokens, or roughly 7 A10-hours at 40% of bf16 peak; 5 arms × 3 seeds ≈ 100 A10-hours.
  Training holds an NRP GPU well above the 40% utilisation floor. The real number comes from
  a smoke run before registration.

## 5. Decisions for the owner

1. Whether phase 0 runs now, while Part V's waves run (it needs one GPU and no NRP jobs), or
   after Part V is scored.
2. The phase-1 corpus and the second domain.
3. Whether phase 1 shapes keys only, or keys and feed-forward width. Keys alone ties it to
   Parts II and V; width is the larger architectural claim and the larger cost.
4. Where the work lives: here beside Parts II to V, or in its own repository as readscope was.

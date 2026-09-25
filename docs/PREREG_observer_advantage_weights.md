# Pre-registration: observer advantage, Part III (weights)

**Status: REGISTERED on commit to master, before either registered model has been run.** The
only runs before registration are the wiring and sizing pilot on Qwen2.5-0.5B (section 6), a model
outside the registered set whose numbers set nothing below. Changes after registration go in the
amendment log with a date and a reason.

> **The question.** At a fixed rate, the damage weight quantization does to a language model's
> behaviour depends on where the bits go. **Which cheap distortion, computed before the model is
> ever run quantized, ranks that damage best: the observer-relative one, or the ones in use?**

## 0. What is known, and what is not claimed

- **The operator is not new.** For a linear map `y = W x` quantized with error `D`, the model's
  second-order behavioural damage is `(1/2) E[(D x)ᵀ R (D x)]`, with `R` the read operator of
  everything downstream of `y`. Treating inputs and read-outs as independent gives
  `tr(P_y D Σ_x Dᵀ)`, the observer predictor here. That is K-FAC's Kronecker factorization
  (Martens and Grosse 2015), and its diagonal-in-`P_y` form is BRECQ's output-Fisher weighting (Li
  et al. 2021). The competitors are its reductions: GPTQ and AWQ optimize `tr(D Σ_x Dᵀ)` (`P_y = I`),
  SqueezeLLM weights `D` by the diagonal weight Fisher, and raw distortion is `‖D‖²`.
- **What is tested is predictive validity:** which cheap proxy ranks behavioural damage across
  many quantized models at one rate, with the hard competitors present. A win over raw alone would
  mean little; the bar that matters is the diagonal Fisher.
- **Programme context.** For attention keys, consumer-relative distortion `tr(P̂ Σ_δ)` picked the
  worse quantizer in 16 of 16 cells where reconstruction error picked it in 2 (readscope
  GO-P-2026-021, one 3B model), and Part II tests allocation by reads in native channels. Weights
  have not been tested.
- **Not a budget claim.** GET's G3 found that weight precision is not a resolution budget at 8
  and 4 bits (thresholds identical at bf16 and int8); the programme files it under reliability.
  Nothing here reads weight bits as an evaluator's budget.
- **Tools cited.** Spectral summaries and the random-graph null for hub degree follow Blum,
  Hopcroft and Kannan, *Foundations of Data Science* (2018), chapters 3 and 8; they enter only the
  reported atlas.

## 1. Design

- **Models.** Qwen2.5-1.5B and Llama-3.2-3B (`unsloth/Llama-3.2-3B`), base models, fp16. Two
  families; every verdict must hold on both.
- **Matrices.** Every decoder projection (`q, k, v, o, gate, up, down`) in every layer.
  Embeddings and the output head stay fp16.
- **Codec.** Round-to-nearest, asymmetric, per group of 128 input columns, at 2, 3, 4, 5, 6 or 8
  bits (`benchmarks/weight_observer/quant.py`). One codec for every arm.
- **Variants.** Four rates (2.5, 3.0, 3.5, 4.0 code bits per parameter, parameter-weighted over the
  matrices, ±0.02) × 50 variants, drawn by a seeded process that never sees a statistic, a
  predictor or a measurement: a random start, then random single-matrix moves toward the rate
  (`variants.py`, seed 20261001). Within a rate every variant stores the same bits.
- **Statistics** (`tables.py`), on WikiText-2 train, 128 sequences of 1,024 tokens, with labels
  **sampled from the model** (seed 20261002), so the Fisher quantities are the true Fisher:
  `Σ_x = E[x xᵀ]` per matrix input; `P_y = E[g gᵀ]` per matrix output, `g` the backpropagated
  gradient of the sampled log-likelihood; the diagonal weight Fisher from per-sequence gradients
  `Gᵀ X`, squared and averaged.
- **Predictors**, as tables over (matrix, bits), summed over a variant's matrices:

  | name | formula | stands for |
  |---|---|---|
  | **observer** | `tr(P_y D Σ_x Dᵀ)` | the observer-relative form (K-FAC) |
  | fisher | `Σ F_ij D_ij²` | SqueezeLLM's diagonal weight Fisher |
  | act | `tr(D Σ_x Dᵀ)` | GPTQ / AWQ's layer objective |
  | raw | `‖D‖²_F` | parameter-space distortion |
  | outdiag | `tr(diag(P_y) D Σ_x Dᵀ)` | BRECQ's output Fisher (reported) |

- **Behaviour.** KL(p_fp16 ‖ p_variant) per token on WikiText-2 test, 48 sequences of 1,024 tokens
  (disjoint from the calibration text), and top-1 agreement (reported).
- **Blind recovery is not an arm.** readscope recovers `P_y` without gradients, but only at about
  `2 d` calls per operating point (its budget cliff), which is prohibitive at `d = 8,960`. It is
  named as a limitation, not run.

## 2. Hypotheses and bars

Per model and rate, Spearman ρ between a predictor and the measured KL over the rate's 50
variants; `ρ̄` is the mean over rates (the fixed-rate comparison). For the observer against each
competitor, the difference in `ρ̄` with a 95% percentile bootstrap resampling variants within rates
(10,000 resamples, seed 0).

| id | observer against | HOLDS | FAILS |
|---|---|---|---|
| **W1** (primary) | fisher | CI above 0 on both models | CI below 0 on either model ("reversed"), or point ≤ 0 on both |
| **W2** | act | same | same |
| **W3** | raw | same | same |

Anything else is INCONCLUSIVE. **Gate (anti-vacuity):** the KL of a rate's variants must be
reliable. The Spearman ρ between even-sequence and odd-sequence KL must be at least 0.8, or that
rate is excluded from `ρ̄` and reported. **Reported, not scored:** outdiag against the observer
(does the off-diagonal of `P_y` matter), each rate's ρ, the pooled ρ across rates, top-1
agreement, and the atlas.

**The atlas** (`atlas.json`, reported, never scored) covers, per matrix:
- the effective rank and top spectrum of `Σ_x` and of `P_y`;
- massive-activation channels in `Σ_x`;
- k-occurrence hubness of the weight rows (k = 10).

It shows where each layer is read, and where hubs sit. Nothing in it chooses what is tested here.
A pattern it shows becomes a hypothesis only in a later registration.

## 3. Execution

- NRP (`ssu-atlas-ai`) through nats-bursting (`weight_observer/nrp.py`):
  - CPU staging jobs in the exempt class build the environment tar, the pinned code tar, the
    weights and the WikiText text on the volume.
  - Each model then runs as one GPU pod on an **A10** (one GPU product per model).
  - The pod installs and downloads nothing.
  - Its CPU and memory come from measurement: a model's own earlier run, or the pilot scaled
    by size.
  - The utilization guard watches CPU and memory; `watch_gpu.sh` deletes a pod whose GPU stays
    under 40%.
- **Pilot (before registration, not scored):** Qwen2.5-0.5B, the full pipeline, to check the wiring
  at scale and measure the resources the registered pods are sized from.
- Each run records its GPU and package versions and refuses to resume elsewhere; variants resume
  after an interruption. A run is never repeated because of its result.
- Scorer: `python -m weight_observer.score --runs <dir> --out results_weights.json`; results in
  `benchmarks/RESULTS_observer_advantage_weights.md` whichever way they fall.

## 4. Consequences (decided now)

| outcome | consequence |
|---|---|
| W1 HOLDS | the observer form becomes the planner's cheap behaviour proxy for weight codecs (the certification tier's first predictor), scoped to the two families and RTN |
| W1 fails, W2/W3 hold | Fisher is the proxy the product uses; the observer claim for weights is scoped to "better than activation-aware" |
| all fail | none of the cheap proxies ranks damage at fixed rate; the planner keeps measuring behaviour directly |

## 5. Limitations

- RTN only: a codec with Hessian rounding (GPTQ) changes `D` itself, and so possibly the
  ranking.
- The additive form ignores interactions between matrices; the measured KL carries them, so any
  shortfall of every predictor is partly that.
- WikiText only; two models at 1.5B and 3.2B.

## 6. Amendment log

(none)

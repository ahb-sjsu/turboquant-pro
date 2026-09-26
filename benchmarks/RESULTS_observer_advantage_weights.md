# Results: observer advantage, Part III (weights)

Registration: `docs/PREREG_observer_advantage_weights.md` (master `b106f25`, PR #218). Runs:
2026-09-25 on NRP, one A10 per model, torch 2.8.0+cu128, transformers 4.56.1, harness `b106f25`.
Scorer: `python -m weight_observer.score` at `b106f25`. Data: `benchmarks/weight_observer/results/`.

## Verdicts

| id | observer against | Qwen2.5-1.5B Δρ̄ [95% CI] | Llama-3.2-3B Δρ̄ [95% CI] | verdict |
|---|---|---|---|---|
| **W1** (primary) | diagonal weight Fisher | −0.224 [−0.316, −0.131] | −0.065 [−0.122, −0.020] | **FAILS (reversed)** |
| W2 | activation-aware | +0.051 [−0.107, +0.208] | +0.847 [+0.687, +0.993] | INCONCLUSIVE |
| W3 | raw | +0.027 [−0.147, +0.199] | +0.767 [+0.604, +0.918] | INCONCLUSIVE |

Gate: split-half reliability of the KL was 0.987–0.999 at every rate of both models; all eight
strata were kept.

**The primary hypothesis is refuted, and reversed.** At fixed rate, SqueezeLLM's diagonal weight
Fisher ranks behavioural damage better than the observer predictor in its registered K-FAC form
`tr(P_y D Σ_x Dᵀ)`, on both models. Against activation-aware and raw distortion the observer form
wins by a landslide on Llama-3.2-3B and ties on Qwen2.5-1.5B, so W2 and W3 are inconclusive.

## Rank correlation with damage (ρ̄, mean over rates, and per rate)

| predictor | Qwen ρ̄ | 3.5 | 4.0 | 4.5 | 5.0 | Llama ρ̄ | 3.5 | 4.0 | 4.5 | 5.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| **fisher** (diagonal weight Fisher) | **0.656** | 0.674 | 0.660 | 0.492 | 0.798 | **0.913** | 0.726 | 0.965 | 0.972 | 0.990 |
| observer (K-FAC) | 0.432 | 0.527 | 0.368 | 0.256 | 0.576 | 0.848 | 0.622 | 0.906 | 0.941 | 0.924 |
| outdiag (BRECQ, reported) | 0.432 | 0.529 | 0.366 | 0.254 | 0.581 | 0.847 | 0.616 | 0.908 | 0.941 | 0.922 |
| raw ‖D‖² | 0.405 | 0.355 | 0.269 | 0.441 | 0.554 | 0.081 | 0.025 | 0.176 | 0.094 | 0.030 |
| act (GPTQ/AWQ objective) | 0.381 | 0.455 | 0.366 | 0.292 | 0.411 | 0.001 | −0.153 | 0.017 | 0.052 | 0.088 |

## Reported, not scored

- **The off-diagonal of `P_y` adds nothing:** the observer and its diagonal (outdiag) differ by at
  most 0.001 in ρ̄ on both models.
- **Activation-aware distortion does not rank damage at fixed rate on Llama-3.2-3B (ρ̄ = 0.001).**
- **Controls** (uniform bits; KL in nats per token):

  | uniform bits | Qwen2.5-1.5B | Llama-3.2-3B |
  |---|---|---|
  | 8 | 0.0005 | 0.0004 |
  | 6 | 0.0085 | 0.0066 |
  | 5 | 0.036 | 0.028 |
  | 4 | 0.134 | 0.129 |
  | 3 | 0.749 | 1.096 |

- **Median KL by rate** (nats per token):

  | rate (bits) | Qwen2.5-1.5B | Llama-3.2-3B |
  |---|---|---|
  | 3.5 | 0.505 | 0.764 |
  | 4.0 | 0.370 | 0.263 |
  | 4.5 | 0.254 | 0.196 |
  | 5.0 | 0.172 | 0.120 |

  Top-1 agreement at 4.0 bits is 0.72 and 0.73. At a mean of 4 bits, random placement costs
  2–3× the KL of uniform 4-bit, so where the bits go matters.
- **Atlas** (`results/*/atlas.json`):
  - Layer 1's `down_proj` input carries massive-activation channels about 3.4 million times the
    median channel in both models; Qwen also has smaller ones in layers 2 and 26.
  - The most hub-skewed weight rows are in MLP `gate_proj` matrices.
  - These are descriptions. A plausible reading of the activation-aware failure is that `Σ_x`
    is dominated by a few channels the downstream read does not weight, but that is a
    hypothesis for later work, not a finding here.

## What it means, stated with its limits

Both predictors that win weight the error by what downstream reads. The diagonal weight Fisher
does it per sequence, `E_s[(Gᵀ X)²]`, which keeps the coupling between a sequence's inputs and its
output gradients. The K-FAC form replaces that coupling with the product of two independent
averages, `Σ_x ⊗ P_y`, and loses ranking accuracy doing so, on both models. The observer idea, weight the error by the reader,
is not refuted; its registered factorization is. The consequence fixed in advance applies: the
diagonal weight Fisher, not the K-FAC form, is the cheap behaviour proxy for weight codecs. The
claim is scoped to round-to-nearest group quantization, two models and WikiText.

**Next (exploratory, then a new registration):** the exact Fisher quadratic form along the actual
quantization error, per matrix, `E_s[⟨D, ∇_W log p_s⟩²]` (no diagonal and no factorization), and
at the model level, `E_s[(Σ_m ⟨D_m, ∇_{W_m} log p_s⟩)²]`, which also carries the interactions
between matrices that every registered predictor omits. Whatever wins that exploration is
confirmed on fresh variants and fresh models before it is claimed.

## Exploration after scoring: what is left for a better per-matrix predictor (not a verdict)

Everything below is exploratory. It uses the two models Part III selected on, and no bar was
set in advance. A planning claim needs fresh models under a new registration. Data:
`weight_observer/results/<model>/explore/`. Analysis: `python -m weight_observer.headroom
--model <model>`, which reads only committed files.

**What was measured (NRP, one A10 per model, the Part III environment):**
- the KL of every decoder matrix quantized alone at 3, 4, 5 and 6 bits (196 matrices × 4
  widths per model, `sensitivity.py`);
- the KL of the exact knapsack plan of each of seven predictors at each Part III rate
  (`plans.py`);
- the plan the measured single-matrix KLs choose (`oracle`), measured in one pod beside the
  Fisher plan and its two nearest rivals (`nrp oracle`).

**1. The diagonal Fisher is calibrated matrix by matrix. The observer table is not.**
Measured single-matrix KL against each table's prediction, as a log ratio over 784 cells:

| table | Qwen sd | Llama sd | share of 4-bit damage misjudged by more than 2× (Qwen, Llama) |
|---|---|---|---|
| diagonal Fisher | 0.31 | 0.31 | 0%, 0% |
| observer (K-FAC) | 0.51 | 0.87 | 20%, 49% |

The Fisher's largest misses are early-layer `q_proj` and `k_proj` at 6 bits, where the damage
is smallest. It under-predicts attention `q`/`k` by about 35% on average, and MLP matrices
not at all. The observer misjudges the matrices that carry the damage, which accounts
for W1's reversal.

**2. Better ranking of random variants does not buy a better plan.** The sum of measured
single-matrix KLs ranks the 200 random variants far better than the Fisher on Qwen (ρ̄ 0.935
against 0.656), and equally on Llama (0.913 against 0.913). The plan it chooses is no better
than the Fisher plan. Measured in the same pod (KL per token, 48 sequences, paired 95%
bootstrap):

| rate | Qwen oracle | Qwen Fisher | Δ | Llama oracle | Llama Fisher | Δ |
|---|---|---|---|---|---|---|
| 3.5 | 0.2295 | 0.2297 | −0.1%, tie | 0.1565 | 0.1560 | +0.3%, tie |
| 4.0 | 0.1017 | 0.1018 | −0.1%, tie | 0.0657 | 0.0660 | −0.4%, tie |
| 4.5 | 0.0497 | 0.0497 | 0.0%, tie | 0.0327 | 0.0325 | +0.7%, tie |
| 5.0 | 0.0238 | 0.0242 | −1.9%, better | 0.0148 | 0.0148 | +0.3%, tie |

Ranking the random variants turns on small differences spread over many matrices. The
optimum is decided by the few matrices with the most damage per bit, and item 1 shows the
Fisher judges those correctly (none of the 4-bit damage sits in a matrix it misjudges by more
than 2×). Item 6 measures that shape directly.

**3. Among the seven predictors, the Fisher plan is best or tied at every rate on both models, with one exception.**
Against it (paired bootstrap):
- the observer and BRECQ-diagonal plans are 2–7% worse at 3.5–4.5 bits and tie at 5.0 on Qwen;
  on Llama they are 5–7% worse at every rate;
- the exact-block and token-level Fisher plans tie or lose by up to 5%. The one exception is
  `fisher_tok` on Qwen at 4.5 and 5.0 bits, which is 1–2% better;
- the activation-aware and raw plans cost 1.4–5.6× the Fisher plan's KL, against a uniform
  4-bit KL of 0.134 (Qwen) and 0.129 (Llama).

**4. Damage adds across matrices near the optimum and not far from it.** A plan's measured KL
is about 0.90 to 1.11 times the sum of its matrices' single KLs for every plan within 2× of the
Fisher plan. The activation-aware and raw plans at 3.5 bits on Llama measure 1.7–1.8× their
sum. Interaction between matrices matters only where no planner would go.

**5. Plan measurements reproduce exactly across pods.** Twelve plans measured again in a
second pod gave identical per-sequence KL (largest change 0.0 nats per token).

**6. The optimum is not broad and flat. It is flat in most directions and has a few cliffs.**
Around the Fisher plan, `k` swaps of bit widths between two matrices of the same type keep the
stored bytes exactly (`flatness.py`, 3 seeded draws per `k`, all measured in one pod). Mean KL
rise over the Fisher plan:

| `k` swaps (share of stored bits moved) | Qwen 3.5 | 4.0 | 4.5 | 5.0 | Llama 3.5 | 4.0 | 4.5 | 5.0 |
|---|---|---|---|---|---|---|---|---|
| 1 (0.0–0.4%) | +0.3% | +3.6% | +0.3% | +5.0% | +0.4% | +28.5% | +12.0% | +1.2% |
| 4 (0.5–0.8%) | +4.7% | +5.3% | +8.6% | +7.9% | +11.8% | +31.7% | +1.5% | +27.5% |
| 16 (1.7–3.7%) | +25.4% | +23.9% | +34.5% | +23.2% | +19.6% | +35.9% | +19.0% | +38.4% |
| 32 (3.4–6.2%) | +33.9% | +38.5% | +58.1% | +46.3% | +50.9% | +101.7% | +122.5% | +90.6% |

The means hide the shape. Of the 24 single swaps, 20 cost under 3.5%. The other four each
downgrade one of a few matrices: layer 1's `down_proj` (Llama, 8 to 4 bits: +81%; 8 to 5: +36%;
Qwen, 6 to 4: +8%) and layer 0's `up_proj` (Qwen, 6 to 4: +14%). Layer 1's `down_proj` is the
matrix whose input carries the massive-activation channels in the atlas above. The additive
prediction from the single-matrix sweep tracks every row (for example +25.6% predicted
against +28.5% measured at 4.0 bits on Llama), so the cliffs are properties of single
matrices, not interactions. This replaces the flat-optimum explanation of item 2: the plan is
decided by a few cliff matrices, the diagonal Fisher protects every one of them (it misjudges
none of the 4-bit damage by 2×, item 1), and elsewhere the landscape is flat enough that a
better table's corrections cost almost nothing to ignore.

**What it means, stated with its limits.** For round-to-nearest group quantization on these
two models, the diagonal Fisher with the exact knapsack (`tqp plan weights`) is within 2% of
any plan a per-matrix predictor can reach. That is not the global optimum. The oracle is the best
additive plan built from matrices measured one at a time, over the widths 3, 4, 5, 6 and 8 bits
and this codec; a plan that exploits interactions between matrices, another width or another
codec lies outside what it bounds. A better per-matrix table has nothing left to win.
Further gains would have to come from the codec (error-compensating quantizers such as GPTQ)
or from terms that couple matrices, and item 4 says the second is small near the optimum. The
planner's weight cost table should be the diagonal Fisher. Everything here is scoped to two
models, one codec and WikiText, and the models are the ones the exploration selected on.

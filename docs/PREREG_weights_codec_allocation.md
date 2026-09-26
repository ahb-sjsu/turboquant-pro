# Pre-registration: allocation and codec, Part III-c (weights)

**Status: DRAFT, not registered.** Registration is this file merged to master after the pilot
(section 4) and before any registered model has been run quantized. The pilot runs on a model
outside the registered set and sets no bar. Changes after registration go in the amendment log
with a date and a reason.

> **The question.** At matched stored bytes, how much of a quantized model's behavioural damage
> is removed by **where the bits go** (a Fisher-planned mixed precision) and how much by **how
> each bit is encoded** (an error-compensating codec), and does planning still pay once the
> codec compensates?

## 0. What is known, and what is not claimed

- **Part III (registered, #219).** At a fixed rate the diagonal weight Fisher ranks behavioural
  damage better than the observer's K-FAC form, on two models (W1 reversed).
- **The headroom exploration (#240, exploratory, the same two models).** For round-to-nearest
  group quantization (RTN), the plan the diagonal Fisher chooses with the exact knapsack
  (`tqp plan weights`) is within 2% of the best plan any per-matrix predictor can reach: the
  oracle built from measured single-matrix KL ties it in 7 of 8 cells. That bound is additive
  and per matrix; it is not the global optimum. It suggests the allocation problem is close to
  saturated for RTN and that the remaining gains lie in the codec. That is the hypothesis here,
  on fresh models, with the codecs in use.
- **The codecs are not new.** GPTQ (Frantar et al. 2022) compensates each column's rounding
  error in the not-yet-quantized columns using the input second moment `H = E[x xᵀ]`. AWQ (Lin
  et al. 2023) rescales input channels by activation magnitude before rounding. Mixed precision
  planned by a sensitivity table is SqueezeLLM's and HAWQ's idea; the exact knapsack is ours.
- **What is claimed if it holds:** a decomposition of the gain into allocation and codec at
  matched bytes, with their interaction, on three models and one corpus. Nothing about
  inference speed, kernels or other corpora.

## 1. Design

**A 2 × 3 factorial at matched stored bytes.**

| | uniform widths | Fisher-planned widths |
|---|---|---|
| RTN (`quant.rtn`, group 128) | `rtn_u` | `rtn_f` |
| GPTQ (group 128) | `gptq_u` | `gptq_f` |
| AWQ (group 128) | `awq_u` | `awq_f` |

- **Budgets.** Two: the stored bytes of uniform 3-bit and of uniform 4-bit (codes plus the
  per-group fp16 minimum and scale, `quant.stored_bits`). Every arm at a budget stores at most
  those bytes; a planned arm counts one byte per matrix for its width map. AWQ's channel scales
  fold into the preceding operation in deployment and are counted as zero, stated here so the
  comparison cannot be read as hiding them.
- **Widths.** Planned arms choose per matrix from {2, 3, 4, 5, 6, 8} bits.
- **One cost table per codec.** The planning cost of matrix `m` at width `b` is the diagonal
  Fisher weighting of that codec's own error, `Σ F_m ⊙ D_m(b)²`, with `D_m(b)` the difference
  the codec actually makes at `b` bits. GPTQ's error is shaped by `H`, so reusing RTN's table
  would plan GPTQ with the wrong costs. `F` is Part III's statistic (sampled labels, 128 × 1024
  WikiText-2 train tokens).
- **GPTQ form.** One-shot: `H` from the full-precision model's inputs, damping 1% of the mean
  diagonal, block 128, columns in natural order, the RTN grid per group. The sequential form
  (inputs taken from the already-quantized layers below) is a reported arm (`gptq_seq_u`), not a
  factor, because it couples matrices and would break the per-matrix cost table.
- **AWQ form.** Per matrix, scales `s = mean|x|^α` over input channels with `α` from a grid of
  20 values in [0, 1] chosen by output MSE on the calibration inputs, then RTN on `W diag(s)`.
- **Models (fresh: none was explored on).** Qwen2.5-3B, Gemma-2-2B and Llama-3.1-8B, base
  models in bf16, staged from ungated mirrors (`unsloth/gemma-2-2b`,
  `unsloth/Meta-Llama-3.1-8B`, as Part III did for Llama-3.2) with the mirror's commit recorded.
  Every projection's input width is a multiple of the 128-column group. The harness holds two
  copies of a model, so the 8B model runs on an A6000 (48 GB) and the others on an A10; one GPU
  product per model.
- **Endpoint.** KL per token from the full-precision model on WikiText-2 test, 48 sequences of
  1024 tokens (Part III's measure). Reported, not scored: perplexity, top-1 agreement, and
  LAMBADA last-word accuracy.

## 2. Hypotheses and bars

For arm X against arm Y at one budget on one model, paired over the 48 sequences: the relative
KL difference `mean(KL_X) / mean(KL_Y) − 1` with a 95% percentile bootstrap interval (10,000
resamples, seed 0). **Better** means the interval lies below 0 and the point estimate is at
least 5% lower. **Worse** is the mirror. Verdicts need all three models.

| id | X against Y | HOLDS | FAILS |
|---|---|---|---|
| **C1** (primary: planning still pays) | `gptq_f` against `gptq_u` and against `awq_u` | better than both at both budgets on ≥ 2 of 3 models, worse on none | better on none |
| **C2** (the codec pays) | `gptq_f` against `rtn_f` | same rule | same rule |
| **C3** (the per-codec table matters) | `gptq_f` against GPTQ planned with RTN's table (`gptq_frtn`) | same rule | same rule |

Anything else is INCONCLUSIVE; "reversed" when worse on ≥ 2 models.

**Reported, not scored:**
- the interaction: the allocation gain `KL(codec_u) / KL(codec_f)` for each codec, and whether
  it is smaller under GPTQ and AWQ than under RTN;
- `awq_f` against `awq_u`, and every other cell of the factorial;
- **the per-matrix bound, probed directly:** from `gptq_f` at the 4-bit budget, a local search
  that swaps widths between same-type matrices (budget exact) and keeps a swap only if the
  measured KL falls, for 64 accepted-or-rejected steps. What it finds bounds the headroom beyond
  any per-matrix table for that codec;
- additivity under GPTQ: measured KL of each planned arm against the sum of its matrices'
  single-matrix KL (a single-matrix sweep at the planned widths only);
- the flatness curve around `gptq_f` (`flatness.py`, as for RTN).

## 3. Gates

- **G0, the codecs are what they claim.** GPTQ with `H = I` and no damping reproduces RTN
  exactly; AWQ with `α = 0` reproduces RTN exactly. Pinned by tests before registration.
- **G1, the budget is matched.** Every arm's stored bytes, recomputed from its widths by the
  scorer, are at most the budget, and a planned arm is within one matrix's granularity of it.
- **G2, the measurement reproduces.** One arm per model is measured twice in separate pods;
  the per-sequence KL must agree to 1e-6 nats per token (it agreed exactly in #240).

A gate failure stops scoring until explained.

## 4. Execution

- **Pilot (not scored).** Qwen2.5-0.5B: every arm at both budgets, for wiring, sizing (CPU,
  memory and GPU utilization per phase, measured) and the cost of the GPTQ and AWQ cost tables.
  Nothing from it sets a bar. The 8B model's memory on an A6000 is checked on the pilot's
  scaling before any 8B job is sent.
- NRP through `weight_observer.nrp`, the Part III discipline: staged environment, pinned code,
  requests from measured usage, GPU jobs that install nothing, no sleep, one product per model.
- Order: G0 tests; the pilot; registration; per model the cost tables, the plans (exact
  knapsack, `turboquant_pro.weight_plan`), then the arms; then the reported probes.
- A cell is never rerun because of its result; an operational failure is rerun unchanged.
- Scorer: `python -m weight_observer.score_codec --results <dir> --out results_codec.json`,
  written into `RESULTS_weights_codec_allocation.md` with every verdict as it falls.

## 5. Consequences (decided now)

| outcome | consequence for turboquant-pro |
|---|---|
| C1 and C2 hold | `tqp plan weights` gains a GPTQ cost table and becomes the default path: plan with the codec's own table, encode with GPTQ |
| C2 holds, C1 fails | the codec carries the gain; planning is dropped from the default weight path and kept as an option |
| C1 holds, C2 fails | planning over RTN stays the default; GPTQ is not added |
| C3 fails with C1 holding | one Fisher table serves every codec; the per-codec tables are not shipped |

## 6. Limitations

Three models, one corpus, KL as the scored endpoint, group-128 min/max grids, one-shot GPTQ as
the scored form. The per-matrix bound in section 2 is a local search, not a global optimum.

## 7. Amendment log

(none yet)

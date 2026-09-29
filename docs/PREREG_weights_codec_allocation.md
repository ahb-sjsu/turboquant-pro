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

- **Budgets, in logical stored bytes.** Two: the stored bytes of uniform 3-bit and of uniform
  4-bit. Logical stored bytes are the packed codes at `b` bits per weight, plus the per-group
  fp16 minimum and scale (`quant.stored_bits`), plus, for a planned arm, one byte per matrix for
  its width map. Every arm at a budget stores at most those bytes. Serialized bytes (a
  container's header, alignment and padding) are not part of the match: every arm is written by
  the same container, and each arm's serialized size is reported beside its logical size. AWQ's
  channel scales fold into the preceding operation in deployment and are counted as zero, stated
  here so the comparison cannot be read as hiding them.
- **Widths.** Planned arms choose per matrix from {2, 3, 4, 5, 6, 8} bits.
- **One cost table per codec.** The planning cost of matrix `m` at width `b` is the diagonal
  Fisher weighting of that codec's own error, `Σ F_m ⊙ D_m(b)²`, with `D_m(b)` the difference
  the codec actually makes at `b` bits. GPTQ's error is shaped by `H`, so reusing RTN's table
  would plan GPTQ with the wrong costs. `F` is Part III's statistic (sampled labels, 128 × 1024
  WikiText-2 train tokens).
- **One calibration set, pinned.** Every statistic a codec or a planner reads, Part III's
  Fisher `F`, GPTQ's `H` and AWQ's choice of `α`, comes from the same calibration windows: the
  first 128 consecutive 1024-token windows of the staged WikiText-2 *train* text
  (`weight_observer.run.chunks`), disjoint from the scored test text. `F` uses labels sampled
  from the model (Part III's rule). Before registration this section records the sha256 of the
  staged `train.txt` and, per model, of its 128 token-id windows, computed by the pilot's
  staging: `train.txt` sha256 `aee724fa58bfbdeb3fc6803297fb6bab27b203d7c40b39ddef9b9770e5d52fe5`;
  Qwen2.5-3B `adda89efcc73d3ef9070f4cc151e86fae9b6460e6c26c224a10d2a8fd9798a7b`;
  Gemma-2-2B `ee9caaca9a27035a72debf59f018165600f23d36438f01615f7ce0ae590633a1`;
  Llama-3.1-8B `7dcba79b76b0e185ab48d7e27697db01da6345122b6fd0f000a3d64ae9229db3`.
  The scorer refuses a cell calibrated on other windows. (How these were computed: "Sample
  identities" below.)
- **GPTQ form.** One-shot: `H` from the full-precision model's inputs, damping 1% of the mean
  diagonal, block 128, columns in natural order, the RTN grid per group. The sequential form
  (inputs taken from the already-quantized layers below) is a reported arm (`gptq_seq_u`), not a
  factor, because it couples matrices and would break the per-matrix cost table.
- **AWQ form.** Per matrix, scales `s = mean|x|^α` over input channels with `α` from a grid of
  20 values in [0, 1] chosen by output MSE on the calibration inputs, then RTN on `W diag(s)`.
- **Models (fresh: none was explored on).** Qwen2.5-3B, Gemma-2-2B and Llama-3.1-8B, base
  models in bf16, staged from ungated mirrors (`unsloth/gemma-2-2b`,
  `unsloth/Meta-Llama-3.1-8B`, as Part III did for Llama-3.2) with the mirror's commit recorded:
  `Qwen/Qwen2.5-3B` `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`, `unsloth/gemma-2-2b`
  `25319945f7fd83b8b903e12081777b7eef2ba993`, `unsloth/Meta-Llama-3.1-8B`
  `e9a141a2091ea561b96483212645a2a05e6f99fc`. Staging must fetch exactly these commits
  (`stage_models --revision`); the staged manifest records the revision it resolved, and a
  model whose manifest names another commit is not run.
  Every projection's input width is a multiple of the 128-column group. One GPU product per
  model, from measured peaks (`weight_observer.sizecheck memprobe`: the harness's own per-group
  code on the model's shapes with random weights, 2026-09-29; reserved plus CUDA context).
  The cost tables hold only the Fisher statistic they read (the accumulator's float64 input and
  output second moments were unused; dropping them left every cost bit-identical and took the
  8B's cost-table peak from 50.2 to 33.1 GiB). Qwen2.5-3B on an A10: cost tables 14.1 GiB, arms
  17.0, of 22.1. Gemma-2-2B on an A10 (eager attention): arms 15.1 GiB; cost tables 20.6 before
  the change, re-measured after it before its first cost-table job. Llama-3.1-8B on an A40
  (48 GB; the draft said RTX A6000, none of which came free in 2.5 h): cost tables 33.1 GiB, arms
  43.0 (GPTQ; AWQ 39.4) of about 44.9 usable, measured on a Colab A100 80 GB with the same
  pinned code and packages (the peaks depend on tensor shapes, not on the card). Host memory
  stays in NRP's exempt class (1 CPU, 2 GiB): measured peaks 1.1 to 1.6 GiB anonymous.
- **Harness precision and attention (settled 2026-09-29, before registration).** The harness
  computes in fp16 with each architecture's own attention kernel (`run.attention`): sdpa,
  except eager for Gemma-2, whose logit soft-cap the sdpa path drops. Evidence, all on the 48
  scored windows or the pilot, with each rule fixed in code before its data:
  - Gemma-2-2B against an fp32 eager reference: fp16 with sdpa missed by 7.4e-4 nats/token, of
    which fp32 sdpa alone accounts for 7.2e-4 (the kernel, not the precision); fp16 with eager
    attention is within 4.0e-5, inside the rule's 1.5e-4. No layer uses more than 4.8% of the
    fp16 range on any model; no non-finite value anywhere.
  - A first rule against a bf16 reference was badly designed (bf16 is coarser than fp16, so
    the gap could not be attributed); it is recorded, not used.
  - Pilot invariance (the ten arms of the four scored comparisons, fp16 against fp32): seven of
    the eight scored ratios within 0.8%, C1a at the 4-bit budget off by 1.27%, so the fixed 1%
    tolerance FAILED. Round-to-nearest arms differ by 0.03 to 0.05% (the fp16 evaluation error),
    GPTQ and AWQ arms by up to 0.84%: their codes are built from calibration statistics taken in
    the harness precision, so fp16 and fp32 yield slightly different, equally valid codes. fp16
    is kept because fp32 is not feasible for the 8B model (two fp32 copies, 64 GB, exceed the
    largest GPU available to the project); the tolerance is not moved, and section 2 adds a
    numerics-sensitivity flag. Outputs and code: PR #268 (`weight_observer.sizecheck`).
- **Endpoint.** KL per token from the full-precision model on WikiText-2 test, 48 sequences of
  1024 tokens (Part III's measure: the first 48 consecutive 1024-token windows of the tokenized
  test text, `weight_observer.run.chunks`). Reported, not scored: perplexity, top-1 agreement,
  and LAMBADA last-word accuracy.
- **The scored sequences are fixed by hash.** Before registration this section records the sha256
  of the staged `test.txt` and, per model, the sha256 of its 48 token-id sequences (the windows
  depend on each tokenizer), computed by the pilot's staging. The scorer recomputes both and
  refuses a cell whose sequences differ: `test.txt` sha256
  `696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83`;
  Qwen2.5-3B `8c11bf4fd44c94b7c919fb3ebb82bf24a81fb6cc00b5a864df33dbab1f48fd91`;
  Gemma-2-2B `29e2864161ba47b9b266af5839a76e3632544e4079876090a4327b87512a7645`;
  Llama-3.1-8B `2eda697bba5e21a5f25b95c50528b3325025ffafc2c7ef020a1e2211af163223`.
- **Sample identities, how they were computed (2026-09-28).** With the harness's own function
  (`weight_observer.codec_run.windows` at commit `215d5a2`), transformers 4.56.1 and
  tokenizers 0.22.2, on each model's tokenizer at the mirror commit recorded under Models; the
  text restaged with `weight_observer.stage_text` (datasets 4.3.0) and byte-identical to the
  pilot's (both sha256 above). As a control, Qwen2.5-0.5B's windows computed the same way
  reproduce the pilot's `hashes.json` exactly. Qwen2.5-3B shares the Qwen2.5 tokenizer, so its
  windows equal the pilot's; that equality was computed, not assumed. Every job recomputes its
  hashes and refuses a mismatch, so a tokenizer drift fails closed.

## 2. Hypotheses and bars

For arm X against arm Y at one budget on one model, paired over the 48 sequences: the relative
KL difference `mean(KL_X) / mean(KL_Y) − 1` with a 95% percentile bootstrap interval (10,000
resamples, seed 0). **The resampling unit is the sequence:** each resample draws 48 sequence
indices with replacement and evaluates both arms on those same indices, so the ratio of means is
computed on paired draws; KL values are never resampled independently per arm. **Better** means
the interval lies below 0 **and** the point estimate is at least 5% lower: the first is
statistical direction, the second practical size. The 5% is set from the exploration of #240, on
models not used here, before any registered result. There, intervals for plans near the Fisher
plan were about ±1% of KL, and the most any better per-matrix table bought over the Fisher plan
was 1.9%; a 5% bar sits above both, so an effect counts only if it exceeds measurement noise and
anything re-tuning the table could buy. The bar is within reach of the effect sought: under RTN
the Fisher plan was 24% and 49% below uniform 4-bit on the two explored models. **Worse** is the
mirror. Verdicts need all three models.

| id | X against Y | HOLDS | FAILS |
|---|---|---|---|
| **C1a** (primary, the mechanism: planning still pays under an error-compensating codec) | `gptq_f` against `gptq_u` | better at both budgets on ≥ 2 of 3 models, worse on none | better on none |
| C1b (secondary, the product claim: the planned GPTQ path beats the other codec's uniform path) | `gptq_f` against `awq_u` | same rule | same rule |
| **C2** (the codec pays) | `gptq_f` against `rtn_f` | same rule | same rule |
| **C3** (the per-codec table matters) | `gptq_f` against GPTQ planned with RTN's table (`gptq_frtn`) | same rule | same rule |

Anything else is INCONCLUSIVE; "reversed" when worse on ≥ 2 models. C1a and C1b are separate
questions and get separate verdicts: C1a asks whether allocation adds anything once the codec
compensates its own error (same codec, different widths); C1b asks whether the whole planned
path beats a competing uniform one at the same bytes (different codecs and widths), which can
hold even if C1a fails.

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
- the flatness curve around `gptq_f` (`flatness.py`, as for RTN);
- **a numerics-sensitivity flag, not a bar:** a scored comparison whose point estimate lies
  within 1.3 points of the 5% line (a relative difference between -6.3% and -3.7%, or the
  mirror for worse) is reported as numerics-sensitive, because calibration statistics taken in
  fp16 rather than fp32 moved the pilot's scored ratios by up to 1.27% (section 1). The verdict
  itself follows the rule above unchanged.

## 3. Gates

- **G0, the codecs are what they claim.** GPTQ with `H = I` and no damping reproduces RTN
  exactly; AWQ with `α = 0` reproduces RTN exactly. Pinned by tests before registration.
- **G1, the budget is matched.** Every arm's stored bytes, recomputed from its widths by the
  scorer, are at most the budget, and a planned arm is within one matrix's granularity of it.
- **G2, the measurement reproduces.** One arm per model is measured twice in separate pods;
  the per-sequence KL must agree to 1e-6 nats per token (it agreed exactly in #240).

- **Every tolerance is in noise units.** Any gate that compares a number with a reference
  (a recorded value, another pod's measurement) states its tolerance as a multiple of a
  run-to-run floor measured on the pilot before any registered cell runs, never as an absolute.
  This is Part II's lesson (its Amendment 7): an absolute 1.0-point tolerance sat within twice
  the movement of a harness that had not drifted.

A gate failure stops scoring until explained. Gate statuses are machine-readable (PASS,
PENDING, FAIL_UNEXPLAINED, FAIL_EXPLAINED, FAIL_EXPLAINED_POSTHOC) through the same machinery as
Part II (`score_keys.decide`, proven in `tests/test_gate_proof.py`), and an explanation is a
disposition that pins the numbers it explains.

## 4. Execution

- **Pilot (not scored).** Qwen2.5-0.5B: every arm at both budgets, for wiring, sizing (CPU,
  memory and GPU utilization per phase, measured) and the cost of the GPTQ and AWQ cost tables.
  Nothing from it sets a bar. The 8B model's memory on its 48 GB GPU is checked on the pilot's
  scaling before any 8B job is sent. Done 2026-09-28/29 on the grid-fixed code: all 16 arms
  measured; its plans regenerated from the grid-fixed cost tables (only the GPTQ-planned arms
  moved, 2 and 3 of 168 matrices); G0 passed on the A10 at every registered model's shapes.
  Two-point scaling of the pilots could not settle the registered models' GPU memory (it
  over-predicted Part III's Llama-3.2-3B by 43%), so each registered model's peak was measured
  directly on its own shapes (section 1).
- NRP through `weight_observer.nrp`, the Part III discipline: staged environment, pinned code,
  requests from measured usage, GPU jobs that install nothing, no sleep, one product per model.
  Each GPU job copies its checkpoint to the pod's own disk while G0 keeps the GPU busy and loads
  only that copy (CephFS served the loaders' scattered reads at 50 to 80 MB/s, idling the GPU for
  minutes per load).
- Order: G0 tests; the pilot; registration; per model the cost tables, the plans (exact
  knapsack, `turboquant_pro.weight_plan`), then the arms; then the reported probes.
- A cell is never rerun because of its result; an operational failure is rerun unchanged.
- Scorer: `python -m weight_observer.score_codec --results <dir> --out results_codec.json`,
  written into `RESULTS_weights_codec_allocation.md` with every verdict as it falls.

## 5. Consequences (decided now)

| outcome | consequence for turboquant-pro |
|---|---|
| C1a and C2 hold | `tqp plan weights` gains a GPTQ cost table and becomes the default path: plan with the codec's own table, encode with GPTQ |
| C2 holds, C1a fails | the codec carries the gain; planning is dropped from the default weight path and kept as an option |
| C1a holds, C2 fails | planning over RTN stays the default; GPTQ is not added |
| C1b holds | the documentation may state that the planned GPTQ path beats uniform AWQ at matched bytes on these models; if C1b fails, it says it does not |
| C3 fails with C1a holding | one Fisher table serves every codec; the per-codec tables are not shipped |

## 6. Limitations

Three models, one corpus, KL as the scored endpoint, group-128 min/max grids, one-shot GPTQ as
the scored form. The per-matrix bound in section 2 is a local search, not a global optimum.
fp16 throughout: GPTQ and AWQ codes depend on the precision of the calibration pass, which moved
the pilot's scored ratios by up to 1.27% against fp32; the bootstrap over sequences does not
include that component, hence the numerics-sensitivity flag.

## 7. Amendment log

(none yet)

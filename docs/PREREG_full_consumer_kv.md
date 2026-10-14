# Pre-registration: observer advantage, Part V (where the key consumer ends)

**Status: DRAFT for review. Not registered.** Registration is the merge of this file to master,
after the wiring smoke of section 6 and before any scored cell runs. Changes after registration go
in the amendment log with a date and a reason.

> **The question.** Part II spent a fixed key-bit budget by how strongly the **queries** read each
> key channel and found the gain promising but not established (K4 inconclusive, K4b failed). A
> query reads a key only through the attention logits. The model's output reads it through the
> softmax, the values the attention mixes and the head's output projection, and then through
> everything downstream. **How far downstream must the consumer be taken before spending key bits
> by its reads beats spending them uniformly, and does a consumer fitted on one text still serve
> another?**

## 0. What the project already establishes, and what is not claimed

- **Part II's verdicts (`RESULTS_observer_advantage_keys.md`, final with a post-hoc explanation).**
  Coding keys in the observer's coordinates is refuted (K1, K1_low, K2 fail, K3 does not hold).
  Allocation by reads in native channels lowered perplexity materially on all three 7B models but
  improved qasper only on Qwen2.5-7B (K4 inconclusive). The read term did not beat key variance
  alone (K4b fails). Part V therefore stays in native channels and changes only the consumer whose
  reads set the allocation.
- **Why the consumer may have stopped too early.** Part II's read weight for key channel `j` is
  `E[q_j²]`, the sensitivity of the attention logits. Two keys whose values are identical can
  exchange attention weight without changing the head's output, so logit sensitivity can
  overstate a channel whose errors only move attention among tokens with similar values, and it
  ignores how strongly attention concentrates on each key. The full read operator `P_C = Jᵀ G J` of the foundational paper
  takes `J` through the softmax, the values and `W_O`, and `G` from what reads the head's output.
- **The consumer chain.** For one head with queries `q_i`, keys `k_t`, values `v_t`, attention
  `a_it`, output `o_i = Σ_t a_it v_t`, and a key error `δk_t`, the logit error is
  `q_i · δk_t / √d`, and the softmax carries it to the output as
  `δo_i = Σ_t a_it (q_i · δk_t / √d)(v_t − o_i)`. With errors independent across keys and
  channels, the expected error energy that each reader in turn sees from channel `j` at a given
  noise level is proportional to

  | consumer | read weight `w_j` of key channel `j` | arm |
  |---|---|---|
  | logits (Part II) | `Σ_i q_ij²` | `read` |
  | head output after `W_O` | `Σ_i Σ_t a_it² q_ij² ‖W_O (v_t − o_i)‖²` | `head` |
  | model output | `E[(∂ℓ/∂k_tj)²]`, labels sampled from the model (true Fisher) | `fisher` |

  summed over a KV head's query readers (GQA). Each row adds the next stage of the same chain. The
  `head` weight needs only the prefill's own queries, keys, values and attention, so like `read` it
  is calibration-free. The `fisher` weight needs a backward pass and is fitted once per model on
  calibration text. The `head` row is the key term of AATC's distortion theorem (below), and
  the `fisher` row is the standard true-Fisher sensitivity. The chain, not any one row, is what Part V tests.
- **The allocation rule is unchanged.** Bits per channel solve
  `min Σ_j w_j Var(k_j) 4^(−b_j)` at a fixed total of `D × 3` bits per head, by the harness's exact
  greedy water filling (`key_coding.water_fill`), widths 1 to 8.
- **The question of future readers is the consumer-mismatch question.** A cache is coded at
  prefill for readers that come later: decoding queries, and in deployment text unlike any
  calibration text. `docs/THEORY_consumer_mismatch.md` (branch `feat/consumer-mismatch`, not yet
  merged; its propositions are checked numerically there) shows that a change of consumer that
  rescales reads without rotating them costs nothing within a coded map, but miscalibrates an
  allocation across coordinates (Proposition 7). In native channels each channel is its own
  one-dimensional map, so every change of consumer is a change of scale per channel and all of its
  cost is allocation cost. So a consumer fitted on other text should serve it worse than one fitted on the
  prompt itself. The programme's blind-probe measurements across domains (readscope,
  GO-P-2026-041: read directions transfer, magnitudes do not) point the same way.
- **The head-output distortion is published; AATC allocates on less of it.** Attention-Aware
  Transform Coding (AATC; Laus, Verdun, Wang, Calmon and Krahmer, arXiv 2608.14191, August 2026)
  bounds the expected squared error of one query's attention output through the softmax, the
  values and `W_O`. Its key term weights each element by the squared attention weight, the value
  residual through `W_O`, the squared query component and the key's noise variance, which for
  one query is the `head` row of the table above. For one query the term factors into a token
  part and a channel part, and AATC allocates its key bits on the channel part alone:
  `w_c = mean_i (B_cᵀ q_i)²` over calibration tokens (its equation for the key water filling).
  That is the `read` weight of the table, in AATC's whitened basis. Summed over the many queries
  that read a cache, the token part differs per query and does not factor out: the `head` row is
  `Σ_i S_i q_ic²` with `S_i = Σ_t a_it² ‖W_O(v_t − o_i)‖²`, each query's read weighted by how
  much its output moves. Part V allocates on that sum. (Reading the paper's method section in
  full, 2026-10-13, corrected this paragraph's earlier claim that the `head` row was AATC's
  allocation weight.) AATC codes keys before RoPE after a whitening transform with per-channel
  widths from 0 to 16 bits (reverse water filling, so a channel can be dropped), calibrated on
  256 sequences, on Llama-3.1-8B and Qwen2.5-7B. Part V does not claim the distortion term. It
  asks four things AATC does not: whether allocating on the summed `head` weight beats the
  read weight AATC allocates on (V1); whether the weight helps in
  native channels with a per-channel codebook, where Part II found no change of basis better than
  native channels (and the observer basis worse on Mistral-7B);
  whether the consumer must reach past the head's output (V3); and whether a consumer fitted on
  calibration text serves a prompt as well as the prompt's own (V4).
- **Future readers have a published estimator.** Expected Attention (Devoto, Jeblick and Jégou,
  arXiv 2510.00636) evicts keys and values by an attention score estimated from a Gaussian model
  of future queries, weighted by the value norm. It evicts rather than quantizes.
- **Logit-product transform coding.** NOVA-KV ("Spend Bits Where Queries Look",
  Fernández-Menduiña, Ziashahabi, Pavez, Ortega and Avestimehr, arXiv 2608.04074, August 2026)
  derives calibrated non-orthogonal transforms under which the error in the attention products
  becomes mean squared error, then vector-quantizes the coefficients at two bits per element. Its
  consumer is the logits (Part II's row of the chain) and its codec is a vector quantizer the
  harness does not have, so it is related work here, not a scored baseline.
- DeepSeek's multi-head latent attention stores a compact latent instead of keys and values, an
  architectural precedent, not a baseline at fixed architecture.
- **Not claimed.** Nothing here is a new codec or a new cache format. A positive result would say
  where an allocation's consumer has to end, and would make the `head` weight a planner stage.
- **Acceptance endpoints for keys, as in Part II:** WikiText-2 perplexity and LongBench qasper
  carry the bars. Reconstruction error and cosine are never acceptance metrics.

## 1. Arms

All arms run the shipped uniform codebook at a mean of 3 key bits per coordinate in native
channels, with Part II's sink, outliers and 4-bit value path, through the Part II harness
(`benchmarks/kvquant_matrix/`). Only `KEY_ALLOC` and its fit change.

| arm | `KEY_ALLOC` | fit | role |
|---|---|---|---|
| `u3` | uniform | none | the shipped allocation (reference) |
| `u3_jit` | uniform | none | the noise floor (one-ulp key jitter, Part II's floor arm) |
| `u3_key` | key | prefill | variance alone (Part II's K4b control) |
| `u3_read` | read | prefill | logit consumer (Part II's K4 arm) |
| **`u3_head`** | head | prefill | head-output consumer (new) |
| `u3_head_cal` | head | calibration | head-output consumer fitted on calibration text |
| `u3_fisher_cal` | fisher | calibration | model-output consumer fitted on calibration text |
| `u3_read_cal` | read | calibration | reported |
| `u2_*`, `u4_*` | as above | as above | reported, the same arms at 2 and 4 bits |
| `u3_rep`, `u3_jit_rep` | | | G1's second runs of `u3` and `u3_jit` (same env lines, own cells) |
| `fp16`, `g0_*` | | | reference and the identity-codebook gates: `g0_native`, and `g0_read`, `g0_head`, `g0_head_cal`, `g0_fisher_cal`, `g0_key`, `g0_read_cal`, one per allocation path |
| baselines | | | section 3 |

Calibration fits for `head_cal` and `fisher_cal` use the same 16 consecutive windows of 1,024
tokens (`PART5_CALIB_N`, `PART5_CALIB_SEQ`) from the start of WikiText-2 train joined by blank
lines, so V3 compares the two weights fitted on exactly the same tokens. Part II's calibration
passages vary in length and are often shorter than the 128-token hot window, where the harness
captures no queries; the first wiring smoke failed on them. The 7B wiring smoke measures the
Fisher backward's peak memory at this length on an A10 and, if it does not fit, sets the
length before registration. (Two 7B smokes first ran out of memory at 1,024 and at 512
tokens because the harness left the weights' gradients on; the Fisher pass now freezes the
weights, and the length is decided on the corrected measurement, never on the bug.)
Labels for `fisher` are sampled from the model with seed 20261013 (`FISHER_SEED`).
`u3_read_cal` keeps Part II's 2,048. Every calibrated weight is fitted
once per model and stored. Each calibrated cell records the sha256 of the table it coded with
(`calib.json`, written after the model loads, since the config sidecar is written before),
and the scorer requires every cell of one model and one weight to have used the same table. For a calibrated arm only the
weights `w_j` come from calibration; `Var(k_j)` is always the prompt's own, so calibrated and
prefill arms differ only in the consumer.

Stored bytes. Every non-uniform arm stores the same allocation table (4 bits per coordinate per
head, Part II's accounting), so all allocation arms are matched to one another exactly, and the
table costs them `4 / n` bits per key element against `u3` for a prompt of `n` tokens, which every
cell records.

## 2. Models, endpoints, gates

- **Models.** Tier A, which carries every verdict: Llama-2-7B-chat (MHA), Mistral-7B-Instruct-v0.2
  (GQA 4:1), Qwen2.5-7B-Instruct (GQA 7:1), Part II's models, harness and prompts. Tier B,
  reported: Qwen2.5-1.5B-Instruct and Llama-3.2-3B-Instruct.
- **Endpoints.** LongBench qasper (all 200 documents, greedy, official metric, per document) and
  WikiText-2 test perplexity (SEQLEN 2048, every chunk, per-chunk NLL) carry the bars. Reported:
  LongBench trec and triviaqa; per-chunk KL to `fp16` on WikiText-2 (the final-output divergence
  the bars do not read); the prefill time spent fitting each allocation. KL is measured in the
  same process: before each chunk's forward the chunk runs once with quantization off (the
  `fp16` arm's forward), and `KL(fp16 ‖ arm)` is summed over the same 2,047 positions as the
  NLL (`wikitext_ppl.py`, `KL_REF=1`; a unit test pins that the reference pass is the `fp16`
  forward and moves the arm's own logits by no bit). Fitting time is the wall time of
  everything a key-coding stage does before its codebook runs (weights, basis, allocation),
  between CUDA synchronizations, per qasper document and per chunk, beside the document's
  generation time.
- **G0, the injection is exact.** As Part II: every identity-codebook arm through each new
  allocation must reproduce `g0_native` bit for bit.
- **G1, the harness has not drifted, in noise units** (the obligation Part II's Amendment 7 placed
  on the next registration that reproduces a recorded number). Before any scored cell, `u3` and
  `u3_jit` run twice on Tier A. On each model `u3` must reproduce Part II's scored `u3` cell within
  three times the larger of (a) the difference between the two `u3` runs and (b) the jitter arm's
  distance from `u3`, on qasper and on perplexity. A failure stops scoring until explained, and
  the explanation's timing (before or after any verdict) is recorded as Part II's Amendment 5
  requires.
- **One environment per comparison.** As Part II's Amendment 2: every cell records its GPU and
  package versions, and an arm is compared only with a reference and floor recorded in the same
  environment.

## 3. Baselines

Both are run in this harness, on the same prompts, at matched stored bytes under one accounting
rule fixed here: every stored bit counts (codes, scales, offsets, outliers, sink and window
entries, allocation tables, and any per-model transform matrices amortized over the evaluated
prompts, reported both amortized and not).

| baseline | what is run | matched to `u3` by |
|---|---|---|
| `aatc` | AATC reimplemented from the paper (`aatc.py`; no public code was found), details below | average code bits = `u3`'s measured mean stored key bits per element on the same model's qasper documents minus AATC's per-token scale (16 bits per 512-channel group); the realized bits are recorded per cell and must match `u3`'s within 1%; values at Part II's 4 bits, as in every arm |
| `aatc_native` | the same with the transform replaced by the identity (native channels), its weights, scales and allocation recomputed there | reported; isolates the transform |
| `ea` | Expected Attention through NVIDIA's `kvpress` (Apache-2.0), keys and values evicted, kept entries in fp16 | eviction ratio set so total stored KV bytes equal `u3`'s (about 80% evicted) |
| `ea_u3` | Expected Attention eviction composed with `u3` key coding | reported |

AATC as reimplemented, from the paper's sections 4 and 5 and its appendix "Experimental
setting":

- **Transform.** Per layer and head group, `k = h B` with `B = S^½ Pᵀ`, `P` and `S` from the
  eigen-decomposition of the keys' second moment, which the paper's appendix proves equal to
  its whitened SVD of `W_K` (`W_V` for values). Fitted on 64 windows of 2,048 WikiText-2 train
  tokens (the paper names WikiText-2 but not the amount). A head group is 512 channels: the
  paper's `G = 1` for Qwen2.5-7B and `G = 2` for Llama-3.1-8B, so `G = 2` for Mistral-7B and
  `G = 8` for Llama-2-7B. Keys are coded before RoPE, as AATC caches `h = xA`; for Qwen2.5,
  whose key projection has a bias, the moment includes it.
- **Weights and allocation.** Key weight `mean (B_c q)²` over calibration tokens and the
  group's query heads, using pre-RoPE queries; value weight `Σ_r ‖(B^V W_O^r)_c‖²`. σ² is each
  coefficient's mean square. Both are divided by their layer's 95th percentile. Integer widths
  0 to 16 come from exact reverse water filling on `w σ² 4^(−b)` over all unprotected layers.
  Layers 0 to 2 get the average plus one bit (keys) and the average (values).
- **Quantizer.** Per-channel scale `s_c` is the 99th percentile of `|h_c|` on calibration.
  Per token and group, `α = max_c |h_c / s_c|` over all the group's channels, step
  `2 α s_c / 2^b`, and codes are clamped to `[−2^(b−1), 2^(b−1) − 1]`. The first 4 and last
  128 tokens stay in fp16.
- **Two readings the paper's text leaves open, fixed by a check on the model.** The first
  wiring smoke (Qwen2.5-7B, A10, 2026-10-13) gave `aatc2` a perplexity of 393,759 against
  `kivi2`'s 6.91. A CPU diagnosis on Atlas (`part5_aatc_debug.py`: one 512-token chunk, a
  small calibration, uncoded perplexity 7.80) found the cause and settled both readings:
  - *Zero mean.* The paper develops transform coding for a zero-mean source (its section
    2.1), but its quantizer has no zero point. Qwen2.5's keys carry a large constant offset
    (layer 0's leading coefficient has variance 604 against about 2 for the rest).
    - Coded uncentred, layer 0's keys alone at 3 bits gave perplexity 170,050, while every
      other single layer stayed within 7.78–7.83.
    - Centred on the calibrated mean, which is subtracted before the transform and added
      back after decoding (a per-model constant), all keys gave 7.81 at 4 bits and 7.89 at
      2 bits, and keys and values gave 9.89 at 2 bits.
    - Centring is the implementation.
  - *Reconstruction levels.* The text's rounding `⌊h/Δ + ½⌋` with that clamp cannot
    represent a positive value at 1 bit (its levels are `{−αs, 0}`). The step `2αs/2^b`
    divides `[−αs, αs]` into exactly `2^b` cells, which is the mid-rise quantizer:
    `q = clamp(⌊h/Δ⌋)`, reconstruction `(q + ½)Δ`. Mid-rise is the implementation; the
    written form stays selectable (`AATC_QUANT=written`).
  - Taking α over the coded channels only changed nothing (151,596 against 148,381,
    uncentred), so α is taken over all channels, as written.
- **Calibration text.** 128 windows of 2,048 tokens from FineWeb sample-10BT and 128 from
  OpenR1-Math-220k. An OpenR1 row is its problem followed by its first generation. The text is
  staged once (`part5_aatc_text.py`, 2,600 rows, sha256 `c1dc9d17…`).
- **Stored bits.** Codes, one fp16 `α` per token and group, and the sink in fp16. The
  per-model transform, scales and widths are reported separately (amortized and not).
- **Harness-wide difference.** As for every arm, the harness codes the settled prefill once.
  The paper re-codes its window in blocks of 16 during decoding.

**Fidelity check, decided before registration.** A reimplementation can be unfaithful. On
Qwen2.5-7B-Instruct (Part V's harness, A10), `aatc2` must beat `kivi2` on LongBench qasper, the
direction the paper reports (0.386 against 0.187).
- `aatc2` is AATC at an average of 2 bits on keys and values.
- `kivi2` is KIVI as the paper ran it: keys per channel in groups of 32 tokens, values per
  token in groups of 32 channels, 2 bits each, 4 sink and 128 recent tokens in fp16.
- **PASS:** the paired 95% bootstrap interval of the per-document F1 difference, over all 200
  documents, lies above 0. trec and triviaqa are reported.
- The outcome is written once to `part5_aatc_fidelity.json` (`score_part5.py --fidelity`).
- If the check fails, V5 is scored against `ea` alone, and that is recorded.

## 4. Hypotheses and bars

For arm X against reference Y, paired over documents (qasper) and chunks (perplexity): mean
difference with a 95% percentile bootstrap interval (10,000 resamples, seed 0) and the floor
`|mean(u3_jit) − mean(u3)|`. **Better** = interval above 0 and mean above twice the floor; **worse**
symmetrically. Verdicts use Tier A only and require G0 for every arm involved. The rule is Part
II's, unchanged, so the two parts read on one scale.

| id | X vs Y | the question | HOLDS | FAILS |
|---|---|---|---|---|
| **V1** (primary) | `u3_head` vs `u3_read` | does taking the consumer through the softmax, values and `W_O` beat the logits? | better on qasper **and** perplexity in ≥ 2 of 3 models, worse on either in none | better in none; "reversed" if worse in ≥ 2 |
| **V2** | `u3_head` vs `u3` | does the head-output consumer's allocation beat uniform bits (Part II's K4, with the fuller consumer)? | same | same |
| **V3** | `u3_fisher_cal` vs `u3_head_cal` | does anything downstream of the head's output matter, at equal fit? | same | same |
| **V4** | `u3_head` vs `u3_head_cal` | is the prompt's own consumer better than one fitted on other text (the mismatch prediction)? | same | same |
| **V5** | `u3_head` vs `aatc` and vs `ea` | does native-channel allocation by the head-output consumer match or beat the published methods at matched bytes? | better on qasper or perplexity in ≥ 2 models against every baseline, worse on neither in any | worse on either in ≥ 2 models against any baseline |

Anything else is INCONCLUSIVE. V5 is scored only if V2 holds.

**Reported, not scored.** Every arm at 2 and 4 bits; `u3_read_cal` against `u3_read` (the mismatch
of the logit consumer); Tier B throughout; trec and triviaqa; KL to `fp16`; fitting time; stored
bits per element by component; and, per (layer, head), the rank correlation between the three
consumers' weights (where the chain changes which channels matter).

**Stated expectations, not bars.** The `head` weight differs most from `read` in heads whose
attention spreads over tokens with similar values, so V1's effect, if any, is larger on qasper
(long documents) than on perplexity. V4 favours the prompt's own fit more at 2 bits than at 4.

## 5. What each outcome would mean

| outcome | consequence |
|---|---|
| V1 and V2 HOLD | the head-output consumer becomes the key-path allocation stage, calibration-free; Part II's logit weight is retired |
| V2 HOLDS, V1 fails | allocation by reads works, and the logits already carry what the head output reads; the cheaper `read` weight ships |
| V3 HOLDS | the consumer must reach past the head; a calibrated Fisher table becomes a model-specific planner input |
| V4 HOLDS | allocation weights are fitted per prompt, never shipped from calibration; the mismatch theory's allocation prediction is confirmed for caches |
| V4 reversed | a calibrated consumer serves better than the prompt's own (the prompt's queries are a poor estimate of its future readers), which points to estimating future reads, as Expected Attention does |
| V2 fails | spending key bits by any of these consumers does not beat uniform at 3 bits; the observer claim for key allocation is withdrawn |

## 6. Execution

- **Code.** `key_coding.py` gains `KEY_ALLOC=head` (computed from the prefill's queries, keys,
  values, attention and `W_O`, in chunks so no full attention matrix is held) and `KEY_ALLOC=fisher`
  (a stored table), and `BASIS_FIT=calib` extends to both. A unit test pins `head` against a
  brute-force finite difference of one head's output on a small random head, and `fisher` against
  autograd on a two-layer toy.
- **Smoke before registration, wiring only.** Every harness arm on documents 0, 50, 100 and 150
  of each task and 8 perplexity chunks, on NRP A10 pods. No comparison is computed from it.
  What it established, 2026-10-13 (harness at `05f4ca6`):
  - G0 holds for every new allocation: on Qwen2.5-7B-Instruct `g0_head`, `g0_head_cal` and
    `g0_fisher_cal` give perplexity 6.0343, equal to `g0_native` (and 8.5306 on
    Qwen2.5-1.5B-Instruct).
  - Every arm a verdict reads ran to completion on Qwen2.5-7B.
  - The Fisher calibration at 16 x 1,024 tokens peaks at 20.20 GiB on the A10's 22.06 GiB,
    so the registered length stands.
  - Four defects were found and fixed before any scored cell: a grid attribute the runner
    reads; calibration passages shorter than the hot window; per-cell checkpoint loads of
    2 min 41 s from the network volume (now one load per pod into a local fp16 copy verified
    bit-identical); and weight gradients built in the Fisher backward (now frozen). An
    out-of-memory at 1,024 tokens was the last of these, not the window length.
  - At 1.5B the harness keeps an A10 only ~20% busy, under NRP's floor, so Tier B moves to
    L4 pods (next item).
- **Hardware, decided 2026-10-12 by the repository owner: NRP A10 pods** (namespace
  `ssu-atlas-ai`) for Tier A, every arm, so everything a verdict reads is one GPU product.
  **Amended 2026-10-13 by the repository owner: Tier B runs on NRP L4 pods**, every arm, so
  each Tier B model is also one GPU product and no comparison mixes A10 and L4 cells.
  - The open L4 nodes are in zone `fullerton`, which no linstor class reaches. A Tier B
    model's weights sit on single-attach RBD volumes of Fullerton's Ceph
    (`rook-ceph-block-fullerton`), one per weight file, with every pod of the set pinned to
    that zone (`part5_nrp.py`, `PLACEMENT`).
  - Tier B carries no verdict, so the change moves no bar. Its pods are held to the same
    40% GPU floor, and the Tier B wiring smoke on an L4 measures it before any Tier B run.
  - The 1.5B smoke at 4 documents and 8 chunks ran at 90–100% inside each arm, but each
    arm is one process lasting 10–40 s, so the model loads between arms pulled the
    5-minute mean to 34% and pod_guard stopped it (53% over its 12.5 minutes). The
    utilization is measured again with arms of 20 documents and 40 chunks.
  - **Tier B runs before registration, on the owner's instruction (2026-10-14),** once
    that lengthened smoke holds the L4 above 40%. Every wave but the fidelity check runs,
    in order, through `part5_chain.py`. These cells were run before this document was
    registered; Tier B is reported, carries no verdict and enters no gate.
  - Submission is through nats-bursting with Part III's pattern (`benchmarks/weight_observer/nrp.py`): CPU staging
  jobs in the exempt class put one environment tar, the pinned code tar, the weights, the text and
  the calibration tables on the volume; each GPU pod installs and downloads nothing, reaches GPU
  work within its first 40 seconds, is sized from measurement, and is watched by the utilization
  guard and `part5_watch_gpu.sh`. Every submission is scored against the programme's NRP preflight
  checklist first.
- **G1 across hardware.** Part II's cells ran on GV100 and A100, Part V's run on A10, so G1
  compares across environments. Its tolerance is therefore three times the larger of section 2's
  same-environment floor and the cross-GPU difference Part II measured for one cell (Amendment 8:
  0.09 qasper points between A100 and GV100 on Mistral-7B `nf4a`; for perplexity, 0.1%), and every
  comparison that carries a bar stays within the A10 environment. G1 runs on Tier A only, so
  Tier B's L4 cells enter no gate.
- A cell is never rerun because of its result; an operational failure is rerun unchanged.
- Scorer: `score_part5.py`, which imports Part II's registered `score_keys.py` (unchanged)
  for its cell readers, statistics, judgement and gate-status rules and adds Part V's gates
  and verdict rules. A cell counts only when every document and every chunk ran; an arm is
  gated by the G0 arm of its own allocation path; G1 reads Part II's scored `u3` from
  `results/results_keys_20260927.json`, pinned by hash. Every comparison's floor is the
  jitter arm against the arm it jitters (`|mean(u3_jit) − mean(u3)|`, section 4), whatever
  the reference: Part II's scorer measured it against the reference, which for V1's
  `u3_read` or V5's baselines would put the reference's own effect into the floor.
  Results in `benchmarks/RESULTS_full_consumer_kv.md` whichever way they fall.
- Execution plan: `docs/PART5_RUN_DESIGN.md` (pods per model and wave, order, sizing,
  deadlines, what is measured first).

## 7. Limitations, stated before the data

- **Independence.** All three weights assume key errors independent across tokens and channels,
  as Part II's did. The per-channel codebook's errors are correlated within a channel through its
  shared range, which this does not model.
- **Future readers.** The prefill's queries stand in for the decoding queries. V4 tests a
  calibrated alternative, not a model of future queries; Expected Attention's estimate is the
  comparison for that.
- **Keys only.** Values are coded at Part II's 4 bits in every arm. A consumer for values
  (`∂o_i/∂v_t = a_it W_O`) is the natural next registration and is not tested here.

## 8. Amendment log

(empty)

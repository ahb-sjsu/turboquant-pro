# Results: observer advantage, Part V (where the key consumer ends)

Registration: `docs/PREREG_full_consumer_kv.md` (registered 2026-10-14 by its merge to master,
`56b8181`). Its amendment log is empty. One owner-approved change to the registered harness
was made after registration and is not logged there; it is described under *Harness* below.
Scorer: `score_part5.py` (last changed at `d388fc3`), run from master `3b1e55f` on Atlas on
2026-10-20 21:09 UTC over one scoring root, `/archive/ahb-sjsu/tqp_part5/interim/final/root`,
into which every Tier A model's run root was fetched fresh from its volume set after the last
wave finished (Mistral-7B's `head` wave, 21:06 UTC). Scorer output:
`benchmarks/kvquant_matrix/results/results_part5_tierA_20261020.json` (sha256
`a5cfd4bb5fa00c7aefa7d65b9024d965a7fcee86c4ad5bdf05d609737fbe28cb`). G1 reads Part II's scored
`u3` from `results/results_keys_20260927.json` (sha256 `c5c11656…`).

**Status: FINAL** (the scorer's `verdict_status`). Every cell a verdict reads ran to
completion and verified on all three Tier A models. V1 to V4 fail. V5 is not scored, because
the registration scores it only if V2 holds. The reported waves at 2 and 4 bits and Tier B are
not in this scoring (see *Not yet scored*).

Units, as in Part II. qasper, trec and triviaqa: the paired mean difference of the per-document
score (points), arm minus reference. Perplexity and KL to `fp16`: the paired mean difference of
per-chunk NLL (or KL) per token, reference minus arm. So **positive means the arm is better** on
every endpoint. Intervals: 95% percentile bootstrap, 10,000 resamples, seed 0. A difference
counts (**better** / **worse**) only when its interval excludes 0 and its mean exceeds twice the
floor `|mean(u3_jit) − mean(u3)|`.

| floor (one-ulp key jitter) | Llama-2-7B | Mistral-7B | Qwen2.5-7B |
|---|---|---|---|
| qasper (points) | 0.76 | 0.26 | 2.90 |
| perplexity (NLL per token) | 0.10e-3 | 0.33e-3 | 0.08e-3 |

Qwen2.5-7B's qasper floor is large. Jittering `u3`'s keys by one ulp moves its qasper by 2.90
points (39.12 to 42.02), so a qasper difference on Qwen must exceed 5.79 points to count.

## Harness

Every cell ran on NRP A10 pods with torch 2.10.0+cu128 and transformers 5.5.0, and every
comparison below is between cells recorded in one environment. Cells record one of three
harness commits.

- `2e8956a`, the gate cells (`u3`, `u3_jit`, `u3_rep`, `u3_jit_rep`) on Llama-2-7B and
  Mistral-7B, submitted before registration as the prereg allows.
- `d388fc3`, which adds Expected Attention. Its added lines are gated on `EA` and no existing
  arm's code changes (`tests/test_part5_ea.py`).
- `47f4033` (2026-10-16, #324), the owner-approved fix that gives `aatc_native` AATC's
  calibration windows. With the transform off, the harness had passed no windows, AATC's second
  pass raised `KeyError`, and `aatc_native` produced no cell on any model. The change is one line
  in `tq_paper_lb_shard.ensure_aatc`. It leaves `aatc` (transform on) unchanged and touches no
  other arm. Cells run after it record `47f4033`, including Llama-2-7B's `u3_head`,
  `u3_fisher_cal` and baselines and Mistral-7B's `u3_head` and `u3_fisher_cal`. Their code path
  is the same as at `d388fc3`.

This change belongs in the prereg's amendment log with its date and reason. It is recorded here
because the log was not updated when it was made.

A cell is never rerun because of its result. Waves stopped for operational reasons several
times: pods lost on the cluster or stopped at their deadline, arms that exited with an error
before finishing (Llama-2-7B's `u3_head_cal` and `u3_fisher_cal` after 20 s on 2026-10-16,
Qwen2.5-7B's `aatc_native` before the fix above), and one submission the submitter's own
preflight refused (Mistral-7B's `head`, 2026-10-20). Each wave was resubmitted unchanged, and its
cells resumed per document and per chunk where the last pod stopped. Mistral-7B's `head` wave,
the slowest at about 14 s of fitting per qasper document, ran from 2026-10-15 to 2026-10-20 over
many such submissions.

## Gates

| gate | Llama-2-7B | Mistral-7B | Qwen2.5-7B |
|---|---|---|---|
| G0, every new allocation path exact through the identity codebook (`g0_read`, `g0_head`, `g0_head_cal`, `g0_fisher_cal` against `g0_native`) | PASS | PASS | PASS |
| G1, `u3` reproduces Part II's scored `u3` within three times the larger noise | PASS | PASS | PASS |
| G1 detail, qasper now vs recorded (tolerance) | 21.57 vs 21.59 (2.28) | 28.60 vs 28.42 (0.78) | 39.12 vs 41.42 (8.69) |
| G1 detail, perplexity now vs recorded (relative difference; tolerance 0.3%) | 6.9800 vs 6.9801 (0.003%) | 5.9667 vs 5.9682 (0.026%) | 7.5340 vs 7.5306 (0.046%) |

The two runs of `u3` agreed bit for bit on every model (repeat difference 0). The qasper
tolerance is therefore three times the jitter arm's distance from `u3`, and the perplexity
tolerance is three times the 0.1% cross-GPU difference of section 6, which is larger than the
jitter. Qwen2.5-7B's `u3` scores 2.30 qasper points below Part II's
record, inside its tolerance of 8.69, which the large jitter on Qwen sets.

## Verdicts

Each cell gives qasper (q) and perplexity (nll), with the interval where it helps the reading.

| id | arm vs reference | Llama-2-7B | Mistral-7B | Qwen2.5-7B | verdict |
|---|---|---|---|---|---|
| **V1** (primary) | `u3_head` vs `u3_read` | q +0.82 [−0.49, +2.37]; nll +0.51e-3 | q +0.16; nll −0.09e-3 | q −1.12; nll +0.02e-3 | **FAILS** |
| **V2** | `u3_head` vs `u3` | q −0.36; nll +2.93e-3 **better** | q +1.38 [−0.24, +3.12]; nll +2.05e-3 **better** | q +5.04 [+2.30, +8.00]; nll +3.40e-3 **better** | **FAILS** |
| **V3** | `u3_fisher_cal` vs `u3_head_cal` | q +1.26; nll −1.16e-3 **worse** | q −0.42; nll −0.42e-3 | q +0.20; nll +0.11e-3 | **FAILS** |
| **V4** | `u3_head` vs `u3_head_cal` | q +0.65; nll +0.15e-3 | q +0.09; nll −0.22e-3 | q +0.64; nll +0.10e-3 | **FAILS** |
| **V5** | `u3_head` vs `aatc` | q −0.42; nll −0.36e-3 | q +0.29; nll −1.21e-3 **worse** | q −0.17; nll −3.64e-3 **worse** | **NOT SCORED** (V2 does not hold) |
| | `u3_head` vs `ea` (qasper alone) | q +3.59 [−0.25, +7.26] | q +4.50 [+2.45, +6.62] **better** | q +6.75 [+4.06, +9.65] **better** | |

Where no judgement is written, the difference did not count. Under the registered rule (Part
II's, unchanged), a model counts as better only when it is better on qasper **and** perplexity,
and a verdict fails when no model is.

**The head-output consumer is not better than the logit consumer (V1).** Allocating 3-bit key
widths by the head-output weight instead of the logit weight changes neither endpoint by more
than its noise on any model. The two weights give perplexities within 0.6e-3 nats per token of
each other on all three models.

**Allocation by the head-output consumer does not beat uniform widths on both endpoints (V2).**
It lowers perplexity on all three models, by 2.05e-3 to 3.40e-3 nats per token (about 0.2% to
0.35% lower perplexity), each interval clear of 0 and of twice its floor. It does not move qasper
by a counting amount on any model. Qwen2.5-7B comes closest, +5.04 points with an interval
above 0, short of its 5.79-point bar. Part II's K4 (`u3_read` against `u3`) showed the same
perplexity gain on all three models and a counting qasper gain on Qwen2.5-7B. With the fuller
consumer, qasper counts on none.

**Reaching past the head's output does not help (V3).** At equal fit on the same calibration
tokens, the model-output (Fisher) weight is materially worse than the head-output weight in
perplexity on Llama-2-7B and indistinguishable on the other two.

**The prompt's own consumer is not better than a calibrated one (V4).** The head-output weight
fitted on the prompt and the same weight fitted on 16 windows of WikiText-2 train give
endpoints within noise of each other on all three models. The consumer-mismatch prediction
that a consumer fitted on other text serves a prompt worse is not supported for this weight in
native channels. It is not reversed either.

**Against the published methods (V5, reported only).** At matched stored bits (AATC 0.2% to
0.3% above `u3`'s key bits), the AATC reimplementation is better than `u3_head` in perplexity
on Mistral-7B and Qwen2.5-7B, and closer to `fp16` in KL on all three models (next section).
Against Expected Attention at matched KV bytes, `u3_head` is better on qasper on Mistral-7B and
Qwen2.5-7B. These rows carry no verdict.

## Reported, not scored

Absolute values per arm (qasper, perplexity, trec, triviaqa, KL to `fp16` per token).

| arm | Llama-2-7B | Mistral-7B | Qwen2.5-7B |
|---|---|---|---|
| `fp16` | 22.06, 6.9425, 64.0, 83.26, 0 | 29.63, 5.9381, 71.0, 86.23, 0 | 43.36, 7.4574, 71.5, 89.97, 0 |
| `u3` | 21.57, 6.9800, 63.0, 82.34, 0.0147 | 28.60, 5.9667, 66.0, 72.29, 0.0094 | 39.12, 7.5340, 66.5, 83.04, 0.0189 |
| `u3_jit` | 22.33, 6.9807, 64.0, 84.01, 0.0146 | 28.34, 5.9686, 66.5, 75.11, 0.0095 | 42.02, 7.5346, 64.5, 81.73, 0.0190 |
| `u3_read` | 20.39, 6.9631, 64.0, 84.14, 0.0127 | 29.82, 5.9539, 71.0, 85.96, 0.0069 | 45.28, 7.5086, 73.0, 90.12, 0.0139 |
| `u3_head` | 21.21, 6.9595, 62.5, 82.56, 0.0128 | 29.98, 5.9545, 71.0, 86.68, 0.0069 | 44.16, 7.5085, 73.0, 88.69, 0.0127 |
| `u3_head_cal` | 20.56, 6.9606, 63.0, 84.15, 0.0127 | 29.89, 5.9532, 71.5, 86.29, 0.0068 | 43.52, 7.5092, 72.5, 89.27, 0.0127 |
| `u3_fisher_cal` | 21.82, 6.9686, 64.0, 83.31, 0.0131 | 29.47, 5.9557, 71.5, 86.34, 0.0068 | 43.72, 7.5083, 73.0, 89.32, 0.0122 |
| `aatc` | 21.63, 6.9570, 64.0, 84.39, 0.0059 | 29.69, 5.9473, 71.0, 85.97, 0.0029 | 44.33, 7.4812, 72.5, 89.48, 0.0065 |
| `aatc_native` | 21.74, 6.9823, 64.5, 82.29, 0.0104 | 29.60, 5.9516, 71.0, 86.24, 0.0055 | 43.25, 7.4832, 72.5, 88.85, 0.0079 |
| `ea` | 17.63, two-pass, 62.0, 80.06, two-pass | 25.48, NaN, 67.5, 84.20, NaN | 37.41, two-pass, 62.5, 65.19, two-pass |
| `ea_u3` | 19.30, two-pass, 62.5, 77.88, two-pass | 24.00, two-pass, 63.0, 70.61, two-pass | 34.37, two-pass, 60.5, 60.09, two-pass |

`ea`'s perplexity follows the two-pass convention of prereg section 3, in which an early query
loses most of its own context, and is not comparable with the coded arms' (Llama-2-7B 35,391,
Qwen2.5-7B 1,431). On Mistral-7B one of its 163 chunks (chunk 0) recorded a NaN NLL and two
recorded a NaN KL, so its means are NaN. The cause has not been diagnosed. It does not enter
any verdict, since V5 judges `ea` on qasper alone and V5 is not scored.

**Uniform 3-bit keys lose most on triviaqa and trec, which carry no bar.** On Mistral-7B, `u3`
scores 72.29 on triviaqa against `fp16`'s 86.23, and every allocation arm scores 85.96 to 86.68.
On Qwen2.5-7B the corresponding numbers are 83.04, 89.97, and 88.69 to 90.12. trec shows the same
pattern by 5 to 6.5 points.

| arm vs reference | endpoint | Llama-2-7B | Mistral-7B | Qwen2.5-7B |
|---|---|---|---|---|
| `u3_head` vs `u3` | trec; triviaqa; KL | −0.5; +0.22; +1.85e-3 b | +5.0 b; +14.39 [+9.62, +19.40] b; +2.55e-3 b | +6.5 b; +5.65 b; +6.25e-3 b |
| `u3_head` vs `u3_read` | trec; triviaqa; KL | −1.5; −1.58; −0.06e-3 | 0; +0.71; −0.02e-3 | 0; −1.43; +1.23e-3 b |
| `u3_fisher_cal` vs `u3_head_cal` | trec; triviaqa; KL | +1.0; −0.84; −0.41e-3 w | 0; +0.05; −0.01e-3 | +0.5; +0.06; +0.46e-3 b |
| `u3_head` vs `u3_head_cal` | trec; triviaqa; KL | −0.5; −1.59; −0.08e-3 | −0.5; +0.38; −0.05e-3 | +0.5; −0.58; −0.01e-3 |
| `u3_head` vs `aatc` | trec; triviaqa; KL | −1.5; −1.83; −6.86e-3 w | 0; +0.71; −3.97e-3 w | +0.5; −0.80; −6.14e-3 w |
| `u3_head` vs `ea` | trec; triviaqa | +0.5; +2.49 | +3.5 b; +2.48 | +10.5 b; +23.49 b |
| `aatc_native` vs `u3_head` | q; nll; KL | +0.53; −3.27e-3 w; +2.40e-3 b | −0.37; +0.48e-3; +1.33e-3 b | −0.91; +3.37e-3 b; +4.81e-3 b |
| `ea_u3` vs `u3` (fewer bytes than `u3`) | q; nll | −2.27; two-pass | −4.61 w; two-pass | −4.75; two-pass |

b better, w worse. The triviaqa floor on Mistral-7B is 2.8 points (`u3_jit` scores 75.11 against
`u3`'s 72.29), so uniform 3-bit keys are unstable on this task there, and the +14.39 exceeds
twice that floor by a wide margin.

**AATC is closer to `fp16` than `u3_head` on every model.** Its KL to `fp16` is 42% to 51% of
`u3_head`'s (0.0059 against 0.0128, 0.0029 against 0.0069, 0.0065 against 0.0127). With the
transform replaced by the identity (`aatc_native`), AATC's KL rises to 0.0104, 0.0055 and 0.0079,
still below `u3_head`'s. Replacing the transform gives up about two thirds of AATC's lead over
`u3_head` on Llama-2-7B and Mistral-7B and about a quarter on Qwen2.5-7B.
`aatc_native` and `u3_head` differ in their weight, their quantizer and their bit range, so this
comparison does not isolate any one of those.

**Fitting time** (seconds per qasper document spent before the codebook runs, and its share of
the document's generation time).

| arm | Llama-2-7B | Mistral-7B | Qwen2.5-7B |
|---|---|---|---|
| `u3_read` | 2.19 s (30%) | 2.45 s (42%) | 2.41 s (56%) |
| `u3_head` | 5.27 s (52%) | 13.90 s (81%) | 9.37 s (84%) |
| `u3_head_cal` | 1.41 s (22%) | 1.32 s (29%) | 1.23 s (40%) |
| `u3_fisher_cal` | 1.39 s (22%) | 1.32 s (28%) | 1.20 s (39%) |

The calibrated arms still compute each prompt's key variance and its allocation. AATC's and
Expected Attention's per-prompt fitting is not recorded by this field.

**Stored key bits per element.** `u3` 4.6231 (Llama-2-7B), 4.6079 (Mistral-7B), 4.6113
(Qwen2.5-7B). Every allocation arm stores 0.0009 to 0.0011 more, its allocation table. `aatc`
stores 4.6369, 4.6177 and 4.6220, 0.21% to 0.30% above `u3`. Every baseline cell passed the
scorer's check that its stored bits match `u3`'s within 1%. Expected Attention keeps a fraction of
each prompt's entries in fp16 so that its total KV bits equal `u3`'s (on the perplexity chunks,
5.164 bits per element for both, 67.7% of entries evicted).

**The stated expectations (not bars).**
- *V1's effect, if any, is larger on qasper than on perplexity.* There is no V1 effect on either
  endpoint.
- *V4 favours the prompt's own fit more at 2 bits than at 4.* Not testable here. The 2- and
  4-bit waves have not run on Tier A.

## Not yet scored

- **Tier A reported waves.** `reported3` (`u3_key`, `u3_read_cal`), `reported2`, `reported4`,
  `head2` and `head4` were not part of the Tier A chains, which ran the verdict waves and V5's
  baselines. The prereg reports every arm at 2 and 4 bits and `u3_read_cal` against `u3_read`.
  None of these is in this scoring.
- **Tier B.** Qwen2.5-1.5B-Instruct (finished 2026-10-18) and Llama-3.2-3B-Instruct
  (finished 2026-10-19) ran every wave on NRP L4 pods. Their cells are on their Fullerton volume
  sets and have not been fetched or scored. Tier B carries no verdict.
- **Rank agreement between the head and Fisher weights per (layer, head).** The scorer computes
  it from the calibration tables, which were not in the scoring root. It is not computed.

## What it means, stated with its limits

The prereg fixed the consequence of this outcome in advance (section 5). **V2 fails, so the
observer claim for key allocation is withdrawn.** Taking the key consumer past the logits,
through the softmax, the values and the head's output projection, does not change which 3-bit
key widths serve the model better than the logit weight does (V1). The model-output Fisher
weight does no better at equal fit (V3). A weight fitted on the prompt is no better than the same
weight fitted on calibration text (V4). Key allocation stays uniform in what ships.

The data do not support calling allocation by reads useless. On all three models `u3_head`
lowers perplexity and KL to `fp16` against uniform widths by amounts that count, and the other
allocation arms' absolute perplexities and KLs sit at its level (their comparisons with `u3`
are not in the registered or reported set, so none is judged here). On Mistral-7B and
Qwen2.5-7B the allocation arms' absolute triviaqa and trec scores also recover most of what
uniform 3-bit keys lose there, and `u3_head`'s gains on both count. These endpoints were reported, not registered,
and qasper, a registered endpoint, shows no counting gain on any model. A claim for allocation
on retrieval-style tasks would need its own registration on fresh models with those endpoints
named in advance. The consumer chain tested here gives no reason to prefer the costlier
head-output weight over the logit weight. Its prefill fitting takes 2.4 to 5.7 times as long
(5.3 to 13.9 s per qasper document against 2.2 to 2.5 s).

AATC, a calibrated whitening transform with reverse water filling over 0 to 16 bits, is closer
to `fp16` than every native-channel allocation arm at matched stored bits, and better in
perplexity than `u3_head` on two of three models. Part II found no change of basis better than
native channels. These results do not overturn that finding, because AATC changes its
quantizer and bit range along with its basis.

The claims are scoped to three 7B instruction models on A10 pods, 3-bit keys with values at
4 bits, LongBench qasper, trec and triviaqa, and WikiText-2 perplexity at 2,048 tokens. They
assume, as all three weights do, key errors independent across tokens and channels (prereg
section 7).

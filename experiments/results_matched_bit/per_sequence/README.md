# Matched-bit projection sensitivity with per-sequence output (Table 2 intervals)

The four runs behind the matched-bit table of `paper/tnnls_operator`, repeated with
per-sequence output so that the Q/K-over-V/O ratio gets a sequence-bootstrap interval. The
records one folder up (`../matched_bit_<model>.json`, July 2026, Atlas GV100) are unchanged.

## What ran

- Script: `experiments/matched_bit_projection_sensitivity.py`, which takes the same sequences
  as the July runs (its own WikiText-2 chunking, window 256) and records:
  - `out_kl_seq` and `out_top1_flip_seq`: the mean KL and the flip rate of each sequence,
    for every projection and bit width;
  - `intervals`: a paired bootstrap over sequences of mean(out_kl Q, K) / mean(out_kl V, O),
    with 2000 resamples and seed 0. One set of resamples serves all four projections, and
    each projection's KL is weighted by positions, as out_kl is.
- Where: NRP, one A10 Job (`tmlr-t2`, 2026-11-22 12:40Z), 1 CPU and 2 GiB, all four models in
  sequence; `experiments/nrp_tmlr/driver.py submit t2`. Measured usage was 0.85 CPU, 1.52 GiB
  mean and 1.57 GiB peak (`../nrp_usage_observed.json`). Software: torch 2.10.0+cu128 and
  transformers 5.5.0, the Atlas environment.
- Weights and inputs: staged copies (`experiments/nrp_tmlr/`). Each model's weight stripes match
  Atlas's reference stripes by sha256. Those load into a model equal to `from_pretrained`'s:
  every tensor and the logits were checked for Qwen, Llama, Mistral and the Gemma text model.
  The sequences were pre-tokenized on Atlas with the script's own chunking (`prep.py mbtokens`).
- Dtypes: Gemma-3-4B-it runs in bfloat16, the others in float16, as in July.
- `NB_WEIGHT_METRICS=0`: the weight-space columns (w_*) are null here. They run in numpy on the
  CPU, would leave a one-CPU pod's GPU idle, and do not depend on the sequences. The July
  records hold them, and the weights are the same checkpoints.

## Checks before the run (Atlas GPU 1, Qwen2.5-1.5B, 4 sequences)

- The staged path (stripes and pre-tokenized inputs) equals the Hub-and-datasets path exactly:
  356 of 356 values.
- Against the committed July script, out_kl and amplification differ by at most 1.3e-6
  relative, the only differences. That is float32 summation order: the new script takes the
  perturbed model's KL one sequence at a time so that Gemma's 262k vocabulary fits an A10.
- With weight metrics off, every KL, flip, per-sequence value and interval is equal. Only the
  null w_* fields differ.

## Result, against July

| model | bits | Q/K over V/O [95% interval] | July |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 8 | 0.378 [0.361, 0.396] | 0.383 |
| | 4 | 0.264 [0.248, 0.283] | 0.265 |
| | 3 | 0.174 [0.166, 0.182] | 0.174 |
| | 2 | 0.266 [0.259, 0.273] | 0.266 |
| Llama-3.2-3B | 8 | 0.390 [0.368, 0.419] | 0.401 |
| | 4 | 0.364 [0.343, 0.384] | 0.363 |
| | 3 | 2.776 [2.557, 3.013] | 2.774 |
| | 2 | 1.129 [1.107, 1.152] | 1.129 |
| Mistral-7B-v0.1 | 8 | 0.283 [0.265, 0.304] | 0.291 |
| | 4 | 0.293 [0.265, 0.323] | 0.293 |
| | 3 | 0.445 [0.421, 0.468] | 0.445 |
| | 2 | 1.050 [1.017, 1.081] | 1.050 |
| gemma-3-4b-it | 8 | 0.876 [0.815, 0.939] | 0.835 |
| | 4 | 0.421 [0.387, 0.455] | 0.425 |
| | 3 | 0.414 [0.375, 0.451] | 0.429 |

Every July ratio lies inside the new interval. Per projection and bit width, the largest relative
difference in out_kl from July is 1.7% for Qwen, 3.7% for Llama, 3.0% for Mistral and 11% for
Gemma. Those runs used a different GPU (A10 against GV100) and newer software, and July's Gemma
ran bfloat16 on Volta, which has no native bfloat16 arithmetic.

sha256 (first 16 hex digits): Qwen 4413293b34abae9e, Llama f007eb712f6e5ecb,
Mistral 2416c16faedc8b8a, Gemma dc7c6e793eb77868.

# Llama-3.2-3B: the factorial, the score-error decomposition and the wavelength probe

Added 2026-11-22 (owner's request, for a reviewer's intervals). The same three harnesses as
the committed records, with per-sequence output and a paired sequence bootstrap (2000
resamples, seed 0, whole sequences, position-weighted KL; one set of resamples serves every
condition of a record). They ran on Atlas GPU 1, a GV100, with the environment of the
committed runs (torch 2.10.0+cu128, transformers 5.5.0), on cores 12 to 23 with 8 threads.

| record | settings | against the committed record |
|---|---|---|
| `llamacpp_factorial_Llama-3.2-3B.json` | E4, 32 sequences, Q2_K and Q3_K_M | 519 of 519 shared values equal |
| `score_error_Llama-3.2-3B.json` | E3, 8 sequences (the record's) | 470 of 470 shared values equal |
| `score_error_Llama-3.2-3B_n32.json` | E3, 32 sequences | no committed record (intervals at 8 sequences are thin) |
| `k_wavelength_Llama-3.2-3B.json` | 3 bits, 32 sequences | July record: largest relative difference 8.6e-8 |

New fields:
- **Factorial:** each projection's `functional` gains `kl_seq`, `flip_seq`, `positions` and
  `kl_ci95`, and each arm gains `functional_qk_over_vo_ci95`.
- **Decomposition:** each layer gains `seq`, its sums per sequence, and each condition gains
  `intervals` for const_share, centered_rel, attention_tv and the layer median of
  const_share.
- **Wavelength probe:** `full_k` and every bucket gain per-sequence KL and flip rates, and
  `intervals` covers full-K KL, each bucket's ONLY and SPARED KL, the share of damage in the
  two longest-wavelength buckets, the longest-over-shortest ONLY ratio, and the bucket
  Spearman.

The wavelength probe now keeps its base log-probabilities on the CPU and compares one sequence
at a time: Llama's 128k vocabulary over 32 x 256 positions (4.2 GB) does not fit beside the
model in the memory GPU 1 has free. The 8.6e-8 difference from July is that summation order.

Factorial, functional Q/K over V/O [95% interval]:

| arm | Q2_K | Q3_K_M |
|---|---|---|
| kquant_mix | 0.912 [0.836, 1.022] | 1.118 [1.070, 1.169] |
| kquant_pure | 0.170 [0.155, 0.190] | 0.227 [0.215, 0.240] |
| chan_mix | 31.65 [29.89, 33.46] | 15.35 [14.04, 16.69] |
| chan_equal | 1.129 [1.107, 1.152] | 2.774 [2.555, 3.011] |

chan_equal is the matched-bit quantizer at 2 and 3 bits on the same 32 sequences, and it
matches the Llama rows of the Table 2 records above (1.129 [1.107, 1.152] at 2 bits; 2.776
[2.557, 3.013] at 3 bits on an A10).

Score-error decomposition, const_share [95% interval] (8 sequences; 32 sequences):

| | 8 sequences | 32 sequences |
|---|---|---|
| Q 4-bit | 0.461 [0.454, 0.468] | 0.458 [0.453, 0.462] |
| Q 3-bit | 0.674 [0.670, 0.678] | 0.670 [0.666, 0.673] |
| K 4-bit | 0.361 [0.342, 0.378] | 0.354 [0.346, 0.362] |
| K 3-bit | 0.731 [0.726, 0.734] | 0.726 [0.723, 0.729] |

Wavelength probe, 3 bits:
- **Full-K KL:** 0.7407 [0.670, 0.818].
- **Share of ONLY damage in the two longest-wavelength buckets:** 0.722 [0.706, 0.737].
- **Longest over shortest bucket (ONLY KL):** 95% interval [26.1, 34.6].
- **Bucket Spearman:** 0.905 [0.786, 0.929].

sha256 (first 16 hex digits):
- llamacpp_factorial b5d22fe720574a28
- score_error a66e9b1b72202fc1
- score_error_n32 4fd69655ee8f9bff
- k_wavelength 2ff4dd7ee5bfb043

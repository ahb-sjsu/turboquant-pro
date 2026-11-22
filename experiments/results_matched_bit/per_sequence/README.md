# Matched-bit projection sensitivity with per-sequence output (Table 2 intervals)

The four runs behind the matched-bit table of `paper/tmlr_operator`, repeated with
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

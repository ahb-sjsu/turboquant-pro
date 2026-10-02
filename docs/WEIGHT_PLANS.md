# Mixed-precision weight plans

`tqp plan weights` chooses one bit width per decoder matrix under a stored-byte budget,
exactly (a multiple-choice knapsack with a Lagrangian dual bound). `tqp plan encode-weights`
encodes a Hugging Face causal LM with GPTQ at the plan's widths and stores the result
packed, at exactly the plan's byte count (`weights.tqpw`).
`tqp plan decode-weights` turns that file and the base checkpoint back into a runnable
model.

```
tqp plan weights --costs fisher_costs.json --bits-per-weight 4 --out plan.json
tqp plan encode-weights --plan plan.json --model-path Qwen/Qwen2.5-3B \
    --calib-text wikitext2/train.txt --out qwen-planned-gptq
tqp plan decode-weights --packed qwen-planned-gptq/weights.tqpw \
    --model-path Qwen/Qwen2.5-3B --out qwen-planned-gptq-hf
```

## What the evidence supports

Part III-c (`docs/PREREG_weights_codec_allocation.md`, results in
`benchmarks/RESULTS_weights_codec_allocation.md`) measured the KL from the full-precision
model on 48 WikiText-2 sequences, for Qwen2.5-3B, Gemma-2-2B and Llama-3.1-8B, at the stored
bytes of uniform 3-bit and 4-bit.

- **Plan from one diagonal-Fisher cost table, encode with GPTQ.** A plan solved from the
  table GPTQ's own errors produce was no better than one solved from the round-to-nearest
  table: no cell differed by 5% (C3 failed). So the planner takes one table and the codec is
  chosen at encoding time. Per-codec tables are not shipped.
- **At matched stored bytes, the Fisher-planned GPTQ path beats uniform-width AWQ** on these
  three models and this corpus: 31% to 71% lower KL, every 95% interval below the 5% bar (C1b).
- It also beats the same plan encoded by round-to-nearest (35% to 75% lower KL, C2), and
  beats uniform-width GPTQ (C1a) on Qwen2.5-3B and Llama-3.1-8B at both budgets and on
  Gemma-2-2B at the 4-bit budget. At 3 bits on Gemma-2-2B the planned and uniform GPTQ
  arms are level.

None of this is measured beyond those models, that corpus and the KL observer.

## The encoder

`turboquant_pro.weight_codec` is the registered harness codec
(`benchmarks/weight_observer/quant.py`) ported line for line. That is one-shot GPTQ on a
min/max grid per group of 128 input columns, damping 0.01 of the mean Hessian diagonal, with
the Hessian taken from the full-precision model's inputs on 128 windows of 1024 tokens.
`tests/test_weight_codec.py` checks that `encode_model` writes the same weights, bit for
bit, as the harness's encoding of the same plan.

## The stored form

`weights.tqpw` ([PACKED_WEIGHTS_SPEC.md](PACKED_WEIGHTS_SPEC.md)) holds each matrix's
codes at its planned width and, for every group of 128 input columns, a float16 minimum
and step. Its payload is exactly the plan's `stored_bits`, and the command refuses a plan
whose count disagrees with the model. Next to it, `weight_encoding.json` records:

- the plan's hash, the codec and its parameters
- the identity of the calibration windows
- the file's size
- `grid_rounding`

The codes are the codec's own. The grid is rounded from float32 to float16 to fit the
32 bits per group the plan counts, so decoded weights differ slightly from the codec's
output. `grid_rounding` records the largest difference in units of the group's step.
The results above were measured on the codec's output, not on the decoded weights.
`--save-model` also saves the model with the stored weights decoded into it, so what
runs is what is stored.

## Where cost tables come from

From the weight-observer harness under `benchmarks/`:

```
python -m weight_observer.codec_run tables --model-path M --text wikitext2 --out O
python -m weight_observer.codec_run cost-table --out O --model-key M   # O/cost_table_rtn.json
```

Its damage model is the diagonal Fisher weighting of each width's round-to-nearest error,
`sum F * D^2`: the one table the results above planned GPTQ with (`gptq_frtn`). The study's
planned arms also paid one byte per matrix for the width map. To compare with them at the
same stored bytes, lower `--bytes` by the matrix count.

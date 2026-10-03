# Check: do Part III-c's results hold for the stored weights?

**Stated 2026-10-02, before any measurement.** Data, code and verdicts are added below this
statement afterwards, without editing it.

## Question

Part III-c (`docs/PREREG_weights_codec_allocation.md`, results in
`benchmarks/RESULTS_weights_codec_allocation.md`) measured each arm's KL from the
full-precision model on the GPTQ codec's output. In that output, every group of 128 input
columns is decoded with a float32 grid. The product's stored form (`weights.tqpw`,
`docs/PACKED_WEIGHTS_SPEC.md`) keeps the codec's codes but rounds that grid to float16,
because the plan's byte count holds 32 bits per group. This check asks one question: do
the stored weights reproduce the measured KL closely enough that the registered verdicts
stand for what the product actually stores?

## What is measured

For each model, each arm the product ships, and the registered plans
(`benchmarks/weight_observer/results/codec/<model>/arms.json`):
`gptq_u3`, `gptq_u4`, `gptq_f3`, `gptq_f4`, `gptq_frtn3`, `gptq_frtn4`. The steps are:

1. Encode the full-precision model with the product encoder
   (`turboquant_pro.weight_codec.encode_model`, GPTQ). Measure the per-sequence KL on the
   48 registered scored windows with the harness's `kl_per_sequence` (the **codec**
   measurement).
2. Write the stored form to a `.tqpw` file, read it back (CRC-checked), and decode it into
   the same model (`apply_packed`). Measure again (the **decoded** measurement).

The inputs are the registered models, text and windows, with every hash checked. Each
model runs on its registered GPU class: Qwen2.5-3B and Gemma-2-2B on Atlas's Quadro GV100
(GPU 1), Llama-3.1-8B on a Colab A100 80 GB. The pins are torch 2.8.0 and transformers
4.56.1, as registered. The runner is `python -m weight_observer.packed_check run`, the
scorer `python -m weight_observer.packed_check score`.

## Criteria

- **E (the product encoder is the registered codec on real models).** For every arm, the
  codec measurement reproduces the registered arm's per-sequence KL to within 1e-6 nats
  per token on every sequence (the registered G2 tolerance). Whether it is also bit for
  bit identical is reported.
- **S (storage costs nothing that matters).** For every arm, the decoded mean KL relative
  to the codec mean KL, `mean(decoded) / mean(codec) - 1`, has its 95% paired bootstrap
  interval inside **[-1%, +1%]**. The bootstrap is the registered scorer's (10,000
  resamples of the 48 sequences, seed 0). 1% is a fifth of the registered 5% bar.
- **V (the verdicts stand).** Re-score C1a, C1b, C2 and C3 with the registered scorer's
  rules and resampling order. Each decoded GPTQ arm replaces its registered measurement;
  the RTN and AWQ arms stay as registered, since the product does not ship them. Every
  cell's judgement (better / worse / neither) and every verdict must match the registered
  ones. The scorer first replays the registered data and must reproduce
  `results_codec.json` exactly, or it stops.
- Also checked for every arm: the stored payload equals the plan's stored bits less its
  width map, bit for bit.

**PASS** = E, S and V on all three models. Then `docs/WEIGHT_PLANS.md` and the spec may
say the results hold for the stored weights on these models and this corpus. If S or V
fails, the docs say what the stored form costs, and the C1b sentence is scoped to the
codec output. If E fails, nothing about the stored form is concluded until the product
encoder is reconciled with the harness.

## Amendment 1 (2026-10-02, before any measurement)

Atlas GPU 1 is occupied by another of the owner's workloads. On the owner's decision, all
three models run on a Colab A100 80 GB
(`benchmarks/weight_observer/colab/packed_check_all.ipynb`), not only Llama-3.1-8B.
Qwen2.5-3B and Gemma-2-2B were registered on a GV100, so for them a codec measurement on
the A100 is not expected to match the registered numbers to 1e-6.

- **E** is judged for Llama-3.1-8B only, the one model measured on its registered GPU
  class. For Qwen2.5-3B and Gemma-2-2B the largest deviation is reported, not judged.
  For those two models, the evidence that the product encoder is the registered codec
  is the 8B and the CPU test in which it reproduces every harness arm bit for bit
  (`tests/test_weight_codec.py`).
- **S** is unchanged and judged on all three models. It compares two measurements from
  the same run on the same GPU.
- **V** substitutes each decoded GPTQ arm as `registered * decoded / codec`, per
  sequence. That is the storage effect measured within one run, applied to the
  registered measurement, so every comparison stays within its registered GPU class.
  Where the codec measurement equals the registered one, as E requires on the 8B, this
  is the decoded measurement itself.

## Results

(added after the measurements)

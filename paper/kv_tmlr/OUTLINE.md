# TMLR paper outline — *KV-cache quantization is model-dependent*

**Status:** outline / data-mapping. Prose + LaTeX written once the matrix (#10) is final.
Numbers cited here are current as of the matrix + rescue sweeps; `[running]` = uniform
cross-model sweep in flight. Data backing each table: `benchmarks/kvquant_matrix/*.json`.

## Working title
"There Is No Universal KV-Cache Codebook: Architecture-Dependent 4-bit Quantization of
the Attention KV Cache"

## Thesis (one sentence)
The best calibration-free 4-bit KV-cache codebook is **not universal**: NF4 is optimal on
Llama-family MHA models and **catastrophic** on high-ratio-GQA models (Qwen2.5), where
asymmetric uniform is required — and we explain the collapse quantitatively as
(codebook error on DC-offset keys) × (model error tolerance set by the GQA ratio).

## Contributions
1. **A negative result with teeth:** NF4 — the default data-free KV codebook — silently
   collapses Qwen2.5-7B (43.8→4.7 qasper, degenerate repetition), while a *lower-precision*
   3-bit uniform codebook works fine. Bit-depth is not the axis; codebook shape is.
2. **A mechanism, measured:** NF4 costs ~4× more reconstruction error than asymmetric
   uniform on DC-offset KV keys (both Llama and Qwen); but NF4's key error is *equal*
   across the two models — the collapse is driven by **error tolerance**, which tracks the
   GQA ratio (Qwen 7:1 amplifies each KV error into 7 query heads; Llama 1:1 MHA does not).
3. **A practitioner decision tree** (companion guide) + a single reproducible harness/notebook.
4. **Correcting an over-claim** (ours and, implicitly, the field's): single-family
   benchmarking (Llama-only) hides codebook fragility. Pre-vs-post-RoPE, often presented as
   a clean win, is model-specific and marginal in our matrix.

## Section map

- **§1 Intro.** KV cache dominates long-context memory; 4-bit quant is standard; papers
  benchmark mostly on Llama and imply universality. We show that's false. Lead figure:
  NF4 vs uniform qasper per model (the 4.7-vs-35 cliff on Qwen).
- **§2 Background.** KV-cache quant (per-channel keys, per-token values), NF4 vs uniform,
  outliers/sink, RoPE, MHA vs GQA. Prior work: KVQuant (Fisher NUQ, calibrated), KIVI,
  KV-cache quant blogs. Our setting: **calibration-free.**
- **§3 Setup.** Models (Llama-2-7B/13B MHA, Mistral-7B 4:1 GQA, Qwen2.5-7B 7:1 GQA),
  LongBench tasks, single-harness rule, the fast prefill-once cache (deployable, ~fp16 speed).
- **§4 Main result — codebook choice is model-dependent.**
  - Table 1: fp16 / NF4 / uniform × 4 models × tasks. `[uniform row running]`
  - Qwen collapses under NF4; uniform required. Llama: NF4 > uniform.
- **§5 Anatomy of the collapse (the diagnosis).**
  - Table 2: Qwen rescue sweep — outliers/short-ctx/8-bit-values don't help; uniform does;
    3-bit uniform > 4-bit NF4. (`results_rescue.json`)
  - Fig: per-channel NF4 vs uniform reconstruction error, Qwen vs Llama (≈equal NF4 error →
    it's tolerance, not representation).
  - Argument: error(codebook, offset keys) × tolerance(GQA ratio).
- **§6 What does generalize.** Calibration-free uniform + 2% outliers + sink is the robust
  default (never collapses); NF4 is a quality upgrade *only* for error-tolerant MHA models.
  Pre-RoPE is a model-specific micro-opt (ablation table), not a headline.
- **§7 Recommendations / decision tree.** (mirrors the guide)
- **§8 Limitations.** 4 models, LongBench English subset (paper expands tasks + WikiText
  ppl + same-harness KVQuant/KIVI), no >13B, GQA-ratio hypothesis tested on 4 points.
- **§9 Reproducibility.** One harness, JSON results, notebook (single-GPU subset + full).

## Tables/figures needing data still
- T1 uniform row (Llama-7B/13B, Mistral): **uniform sweep running now.**
- (paper-grade, optional) more LongBench tasks; WikiText-2 ppl; KVQuant + KIVI in-harness;
  Qwen error-tolerance threshold sweep; GQA-ratio test on a 3rd GQA model (e.g. Llama-3-8B 4:1).

## Open decisions for the user
- Scope before writing LaTeX: ship with **4 models / 3 tasks** (tight, fast) or **expand**
  to ~8 tasks + WikiText + KVQuant/KIVI baselines (stronger, ~half a day more compute)?
- Venue style: TMLR (`tmlr.sty`) — confirm.

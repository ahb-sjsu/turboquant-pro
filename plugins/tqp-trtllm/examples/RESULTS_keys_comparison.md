# Keys comparison: fp8 / nvfp4 / per-channel — the prediction was wrong

**Run:** NRP L40S, 2026-07-16 · real post-RoPE keys from `unsloth/Llama-3.2-3B`
(320 tokens, layers 0/13/27) · attention KL and top-1 agreement vs exact keys
under random unit queries · values exact.

**Pre-registered prediction** (design doc §3): damage ordering
`fp8_tensor > fp8_head > nvfp4 ≈ per_channel` — coarser scale granularity
destroys the DC key offsets softmax reads.

**Measured (mean attention KL, worst first):**

| format | bits | mean KL | mean top-1 |
|---|---|---:|---:|
| nvfp4 block-16 | 4 | 1.16e-04 | 0.80 |
| per-channel asym-NF4 + 2% outliers | 4 | 5.1e-05 | 0.82 |
| fp8 per-tensor | 8 | 1.0e-05 | 0.92 |
| fp8 per-head | 8 | 1.0e-05 | 0.92 |

**The prediction is refuted, and the reason is the finding.** The
registration conflated scale granularity with format class. e4m3 is a
*floating-point* grid: every value carries its own exponent, so an outer
per-tensor scale does not destroy per-channel offsets — fp8's format IS a
per-value scale. The DC-offset fragility mechanism (measured for the
KV keys finding and the §2.3 W^K rows) applies to **fixed-point grids at
matched bit-width**, where a shared scale wastes levels on the offset. And
fp8 here has 2× the bits of the 4-bit rows — this was never a matched-bit
comparison for the fp8 rows.

What survives at matched 4 bits: **per-channel asym-NF4 beats block-16
e2m1 by ~2.2× KL** on these keys — the zero-point + outlier treatment
matters more than block-local scaling at equal width.

**Corrected claim for the record:** fp8 KV (either scale mode) is
near-lossless on DC-offset key families at 8 bits — scale granularity is
a non-issue for float formats; granularity claims belong to fixed-point
grids at matched width. All KLs here are small in absolute terms
(1e-05–1e-04); task-level (LongBench) separation between the 4-bit
formats is the follow-up.

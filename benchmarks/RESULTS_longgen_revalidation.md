# Long-generation degradation: a re-validation that does not reproduce it

**Headline: the recorded long-generation negative in `results_longgen.json` does
not reproduce.** Re-running the harness's own driver on the committed `NF4A`
configuration gives a gap of **-0.31** where the record says **13.7**, and it
fails at every long-generation cell tested, on two architectures and two tasks.
Every fp16 control reproduces the recorded value. The finding is a positive
statement about the record, not a null result about the re-run.

Run 2026-08-08. Harness `tq_paper_lb_shard.py` at `a04d71a`, unmodified.
Evidence under `/archive/c12/reval/` on Atlas, scored with LongBench's own
`metrics.py`, the same module `tq_enh_agg.py` uses.

## The matrix

Forty documents per cell, `NUM_SHARDS=5 SHARD_ID=0` so indices 0, 5, ... 195, an
unbiased subsample. Both arms see identical documents.

| dataset | model | max_gen | fp16 | quant | gap | recorded gap | |
|---|---|---:|---:|---:|---:|---:|---|
| gov_report | Qwen2.5-7B | 512 | 31.83 | 32.14 | **-0.31** | 13.70 | does not reproduce |
| gov_report | Qwen2.5-7B | 256 | 27.72 | 27.77 | **-0.05** | 9.60 | does not reproduce |
| multi_news | Qwen2.5-7B | 512 | 24.24 | 23.73 | **0.51** | 7.50 | does not reproduce |
| gov_report | Llama-2-7B | 512 | 27.42 | 28.25 | **-0.83** | 12.13 | does not reproduce |
| multi_news | Llama-2-7B | 512 | 25.53 | 25.88 | **-0.35** | 9.00 | does not reproduce |
| 2wikimqa | Qwen2.5-7B | 32 | 47.83 | 44.08 | **3.75** | 1.90 | reproduces, exceeds |
| gov_report | Qwen2.5-7B | 512 | 31.83 | 5.19 | **26.64** | (symmetric NF4) | collapses, 2x past 13.7 |

## Why this is a finding and not a broken re-run

Four positive controls, all of which had to behave before the negatives meant
anything.

**The fp16 arm reproduces the record at every length and both models.** 31.83
against a recorded 31.80 at 512, and 27.72 against 27.31 at 256, on Qwen. On
Llama-2, 27.42 against 26.75 and 25.53 against 26.18. The unquantized arm also
reproduces the recorded length scaling, 27.3 to 31.8. For the environment to
explain the discrepancy it would have to leave the unquantized arm exact at two
lengths on two models while moving the quantized arm by 10 to 14 points.

**The quantized arm does real damage where the record says it should.** On
2wikimqa at 32 generated tokens the nf4a arm costs 3.75 points against a
recorded 1.9. An arm that was silently unquantized could not do that.

**The pipeline detects catastrophic collapse.** Symmetric NF4 on the same cell
scores 5.19, a 26.6 point drop. Absence of a gap at 512 is therefore a
measurement of absence, not an absence of measurement.

**Quantization is confirmed applied at the tensor level.** The
`DynamicCache.update` hook fires on all 28 layers during `generate()`, the
in-place assumption still holds under transformers 4.56, and the stored settled
region differs from clean fp16 with relative key error up to 0.28 and value
error up to 0.43, propagating across layers.

## The specific document the record quotes

`results_longgen.json` quotes gov_report index 0, with an asym-NF4 tail of
`"...for ( (,. real ( ( ( ( the: a for a the like the for and aL..."`. On that
same document, through the harness's own `main()`, the fp16 tail reproduces the
recorded fp16 tail near verbatim while the nf4a tail is coherent prose about the
Offshore Patrol Cutter. The nf4a arm also emits EOS early on 9 of 15 documents
in an earlier pass, where the record says generation "often" runs to max length.

## What the 18.1 is not

The natural hypothesis was a mislabeled arm, since `results_matrix.json` records
that Qwen "COLLAPSES under 4-bit NF4" and that "symmetric NF4 ppl EXPLODES 10x,
asym-NF4 holds at fp16". That hypothesis is dead. Symmetric NF4 gives **5.19**,
not 18.1. The recorded value sits between the two codebooks and matches neither.

Provenance could not be settled from the repository. `tq_focus.sh` and
`tq_abl.sh` run Qwen gov_report only at `MAXGEN` 64, 128 and 256, never 512, and
`results_matrix.json` has no gov_report entry for Qwen at all. The 512 row most
likely came from `tq_expand.sh` or `tq_phase3.sh`, which run `fp16`, `nf4` and
`nf4a` arms over a task list including gov_report, but the raw aggregates lived
under `/root` on the original machine and were never committed, so which arm
supplied the number cannot be recovered from here.

## Ruled out

The nf4a path in `tq_paper_lb_shard.py` is unchanged since the finding was
recorded, +282/-6 lines, all additions for kvquant and kivi. Config matches the
committed `NF4A` line exactly, verified from the run's own printed banner.
Prompt construction, chat template, 31500 truncation and generation config are
all confirmed by the fp16 reproduction. The `_qdone` guard, the HOT window and
the outlier fraction behave as documented.

## Caveats, stated plainly

**Different stack from the original.** Quadro GV100, torch 2.8.0+cu128,
transformers 4.56.1. The original ran on Ampere with an older transformers. This
is weak as an explanation given the fp16 controls, but it is not zero.

**One accommodation is ours.** Volta has no flash kernel and the memory
efficient kernel refuses `enable_gqa`, so the default path materialises a
28 x T x T attention matrix and OOMs at 12.78 GiB. The run forces the
`repeat_kv` path under `EFFICIENT_ATTENTION`. That selects a kernel, not an
algorithm, and the fp16 arm lands on 31.83 under it.

**Forty documents, not two hundred.** The fp16 match to within 0.03 indicates
the subsample is representative, but the quantized arm's variance at n=40 is not
characterised.

**ROUGE-L is forgiving.** gov_report and multi_news are scored by ROUGE-L over
long summaries, 2wikimqa by QA-F1 which punishes a few wrong tokens. Real
degradation could be present at 512 and invisible to ROUGE-L. This does not
rescue the recorded figures, which were measured with the same ROUGE-L, but the
defensible claim is "the recorded ROUGE-L gap does not reproduce", not "4-bit KV
is harmless at long generation".

## Recommended action

Not taken here. The gov_report and multi_news 512 rows in
`results_longgen.json`, and the same table in `KV_QUANT_GUIDE.md` line 148,
should be marked as not reproducing pending resolution. The `_about` field's
claim that the effect is "distinct from the GQA codebook collapse" and affects
MHA and GQA alike is the part that fails hardest, since Llama-2 shows -0.83 and
-0.35.

Downstream, `readscope`'s C-12 calibration was declared entirely on this
phenomenon and its A0 anti-vacuity gate failed on the same data, so that
calibration should be withdrawn rather than re-run.

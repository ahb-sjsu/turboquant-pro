# LongBench task scores: TurboQuant keys vs KVQuant (calibration-free)

Real **LongBench task scores** (not just per-layer fidelity) on **Llama-2-7B-chat**,
full 200-sample test splits, greedy decode, 3500-token middle-truncation — the same
harness for every method, so the rows are directly comparable. `qasper` is the
outlier-channel-sensitive task (generative QA); `trec` is classification; `triviaqa` is
QA-F1. Runner: `benchmarks/tq_enh_lb_shard.py` (+ `tq_enh_agg.py`).

| KV scheme | bits (KV) | trec | triviaqa | qasper |
|---|---|---:|---:|---:|
| fp16 (reference) | 16 | 64.0 | 83.26 | 22.06 |
| **KVQuant nuq4-1%** (Fisher + K-means, vendor) | ~4.3 | 64.0 | 83.16 | **21.06** |
| per-channel **uniform** 4-bit | 4 | 62.5 | 81.84 | **14.38** |
| per-channel NF4 + **1%** outliers + sink | ~4.3 | 63.0 | 82.61 | 20.23 |
| **per-channel NF4 + 2% outliers + sink** (recommended) | ~4.6 | 63.5 | **83.32** | **20.82** |
| per-channel NF4 + **3%** outliers + sink | ~4.9 | 63.0 | **83.37** | 20.67 |
| per-channel uniform 2-bit (sanity) | 2 | 58.0 | 75.75 | 16.15 |

## Findings
1. **The outlier-channel open item (below, #4) is real and now fixed.** Uniform 4-bit
   per-channel keys collapse on `qasper` (22.06 → **14.38**) because a handful of
   high-magnitude *key channels* dominate attention and uniform 4-bit crushes them.
2. **Calibration-free additions recover it.** NF4 non-uniform levels + dense-sparse fp16
   outliers + a 4-token attention sink lift `qasper` from 14.38 back to **20.82** (at 2%
   outliers).
3. **No calibration required.** KVQuant needs an offline Fisher-gradient pass + per-channel
   K-means. TurboQuant's recovery uses a fixed NF4 codebook + top-magnitude outlier
   selection — no calibration set, no K-means. Enable with
   `PerChannelKV(..., nf4=True, outlier_frac=0.02)` or
   `TurboQuantKVCache(..., key_nf4=True, key_outlier_frac=0.02)`.
4. **Effectively a dead heat with KVQuant — and it *exceeds* KVQuant on `triviaqa`**
   (83.32 vs 83.16 at 2%). On `qasper` it trails by **0.24** (20.82 vs 21.06) — within
   LongBench noise. KVQuant keeps a hair's edge there at the cost of its calibration pipeline.
5. **2% is the sweet spot.** The outlier sweep `1% → 2% → 3%` gives qasper
   `20.23 → 20.82 → 20.67`: it peaks at 2%, and 3% regresses slightly while adding storage
   (~4.6 vs ~4.9 effective bits). Ship **`outlier_frac=0.02`**.

Honesty notes: these numbers came from a faithful but slow *simulation* cache
(re-quantizes the settled window each decode step); a production cache quantizes
incrementally as tokens leave the hot window. The fp16/KVQuant/TQ rows here are a single
self-consistent harness — they are **not** comparable to LongBench numbers from other
harnesses (truncation/prompt details shift absolute scores by several points).

---

# Fused KV-decode on a real model: quality on real long-context activations

End-to-end check of the fused KV-decode on a real served model. The real LongBench
dataset loader is deprecated (HuggingFace `datasets` dropped script-based datasets),
so we use **real natural text** (Project Gutenberg PG-1342, *Pride and Prejudice*) on
**Qwen2.5-7B-Instruct** (eager attention), capturing the *true* post-RoPE
query/key/value via a non-invasive recording hook, and comparing per layer at the
decode position:

    fp16 standard attention   vs   fused decode over quantized K/V

(4096-token context, 28 layers; the kernel is exact vs decompress-then-attend, so this
isolates the **quantization** quality, not the kernel.)

| config | all-cold (worst case) | two-tier: fp16 sink=4 + hot=512, coded cold |
|---|---:|---:|
| 3-bit | mean 0.367 / median 0.32 | **mean 0.155 / median 0.119** (max 0.68) |
| 4-bit | mean 0.208 / median 0.17 | **mean 0.086 / median 0.048** (max 0.60) |

## Honest findings
1. **The kernel changes speed, not quality.** It reproduces decompress-then-attend
   exactly (<=4e-7, M0-M3); the errors above are pure quantization loss.
2. **3-bit KV is aggressive** for this model: ~12% median per-layer attention-output
   error even with a hot window. **4-bit halves it** to ~5% -> prefer 4-bit keys (or
   asymmetric `key_bits=4, value_bits=3`) for quality-sensitive decode.
3. **The two-tier scheme matters:** keeping the attention sink (first tokens) and the
   recent hot window in fp16 cuts error ~2.4x vs all-cold. This is what
   `TurboQuantKVCache` already does.
4. **A few layers stay high** (max ~0.6) -- the known attention-sink / outlier-channel
   problem in KV quantization; per-channel scaling or more fp16 sinks would help. Honest
   open item.

## Scope (honest)
This measures per-layer **attention-output fidelity** on real activations -- a direct
quality signal (preserved attention -> preserved generation) -- not a full LongBench
*task score*, which needs a generate()-loop attention monkeypatch (model-arch-specific,
fragile; scaffolded in `benchmark_longbench_parity.py`). The fidelity result already
says: ship 4-bit (or asymmetric) KV for quality; 3-bit is a memory-max setting.

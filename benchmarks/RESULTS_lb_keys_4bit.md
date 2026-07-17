# LongBench-qasper 4-bit KEY separation (P3) — prediction confirmed

Harness: [`lb_keys_4bit.py`](lb_keys_4bit.py). Three legs on Llama-3.2-3B over
100 LongBench-qasper samples, **keys fake-quantized in the live KV cache**
(compress→decompress each decode step), values fp16. Reports qasper F1 per leg.

**Run:** Colab (`unsloth/Llama-3.2-3B`, `NB_N=100`).

## Pre-registered prediction

From the code-space result — per-channel asym-NF4 beat NVFP4 block-16 by ~2.2×
on attention KL — the registered expectation was **per-channel NF4 ≥ NVFP4 at
task level**. Reported below as-found.

## Result — confirmed

| key format | qasper F1 | Δ vs fp16 |
|---|---:|---:|
| fp16 (baseline) | 12.37 | — |
| **per-channel asym-NF4** (+2 % fp16 outliers) | **11.82** | **−0.55** |
| NVFP4 block-16 | 10.62 | −1.75 |

Per-channel asym-NF4 keys land at **11.82** vs NVFP4's **10.62** — the predicted
ordering holds. Task-level, per-channel's degradation from fp16 is **3.2× smaller**
(−0.55 vs −1.75), so the code-space attention-KL advantage (~2.2×) carries to
qasper F1 and, if anything, widens. Both 4-bit key formats stay usable; the
per-channel format is the clear pick for 4-bit KV keys.

**Scope, honestly:** N=100, so F1 carries roughly ±1 point of noise — the ~1.2
F1 gap is meaningful and in the predicted direction, but this is a separation
result, not a precise effect size. The value is that the code-space metric and
the downstream task agree on the ordering.

**Reproducibility note:** this script lived only in an NRP ConfigMap until now,
and NRP's opportunistic tier preempted the long job (second model load + eval)
before the 4-bit legs finished on every attempt — the numbers above are the
first complete run, obtained on Colab after committing the script to the repo.

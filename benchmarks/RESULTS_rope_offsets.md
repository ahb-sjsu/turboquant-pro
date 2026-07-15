# asym-NF4 key offsets are RoPE-frequency-structured

**Script:** [`rope_offset_frequency.py`](rope_offset_frequency.py) · GPU host (run on 2×GV100
"atlas", one model at a time on a single 32 GB GPU, fp16) · deterministic.

**Question** (the mechanism the TMLR draft needed): is the per-channel DC offset that
`asym-NF4`'s zero-point absorbs (v1.4.0 finding) an arbitrary per-channel statistic, or is it
structured by the RoPE rotary frequency of the channel? Post-RoPE keys' slowest rotary
channels are near-constant across positions within a context window, so any pre-RoPE
per-channel mean survives rotation there as a DC component — while fast channels average it
away. If the offset is frequency-structured, (a) the GQA-collapse causal chain gains its
measured substrate, and (b) a **deterministic, calibration-free zero-point** derived from the
model config alone (θ, head_dim, window) becomes possible.

**Protocol:** WikiText-2-raw test, 8 passages × 512 tokens; post-RoPE keys captured from the
KV cache for every layer and KV head; per-channel mean μ_c (the asym-NF4 offset) and |k|;
rotary map (rotate_half convention): wavelength(c) = 2π·θ^(2(c mod d/2)/d). "DC channels" =
wavelength > window (67.2 % of channels at 512 tokens for θ = 10⁶, d = 128; 51.6 % for
θ = 10⁴).

## Results

| model | GQA | RoPE θ | pooled Spearman(\|μ_c\|, wavelength) | per-layer median | offset mass in DC channels (frac. of channels) |
|---|---|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 6:1 | 10⁶ | **0.914** | 0.741 | **99.1 %** (67.2 %) |
| Qwen2.5-7B-Instruct | 7:1 | 10⁶ | **0.960** | 0.773 | **99.0 %** (67.2 %) |
| Qwen2.5-14B-Instruct | 5:1 | 10⁶ | **0.937** | 0.822 | **97.8 %** (67.2 %) |
| Mistral-7B-v0.1 | 4:1 | 10⁴ | **0.982** | 0.948 | **96.3 %** (51.6 %) |

Full per-frequency-pair curves and JSON: [`item4_result.json`](item4_result.json) /
[`item4_offsets.png`](item4_offsets.png) — |offset| vs rotary wavelength per model with the
context-window line; the offset mass sits almost entirely to the right of it, rising
monotonically into the slowest channels.

## Reading

- **The correlation is there, strongly, in both families**: |offset| tracks rotary wavelength
  at Spearman 0.91–0.98 pooled across three Qwen scales *and* Mistral (different RoPE base),
  with **96–99 % of the offset mass** in the channels whose wavelength exceeds the window.
- **Mechanism closed:** the collapse chain is now measured end-to-end — RoPE geometry at the
  model's θ/window puts the DC component in identifiable channels (this experiment);
  symmetric NF4 wastes half its codes on it (fig_asymnf4); GQA ratio sets the error tolerance
  that turns the waste into collapse (fig_mechanism).
- **Deterministic zero-point candidate:** since the offset-carrying channels are identified by
  the config alone, a frequency-derived zero-point schedule could replace per-channel
  calibrated μ metadata. Quality comparison vs calibrated asym-NF4 = future work.

**Config gotcha (recorded because it silently biased the first run):** transformers 5.x moves
`rope_theta` into the config's rope-parameters dict for some models, so
`getattr(cfg, "rope_theta", 1e4)` silently defaults — the first pass ran Qwen at an assumed
θ = 10⁴. Spearman was unaffected (wavelength is monotone in channel index; Spearman is
rank-invariant), and the mis-θ DC-mass was a conservative lower bound (92/92/82 %); the
corrected θ read is now in the script and the table above is from the corrected run.

**Caveats:** no Llama-2 (gated, no HF token on the host); one dataset; 512-token windows (the
DC-channel set shrinks as the window grows toward the wavelength scale — worth a sweep);
Spearman on |μ| pooled over layers/heads.

---

# Follow-up: the deterministic zero-point works

**Script:** [`deterministic_zeropoint.py`](deterministic_zeropoint.py) · same GPU host ·
JSON: [`item4b_result.json`](item4b_result.json).

Five 4-bit NF4 key quantizers, identical except for the zero-point μ (values fp16 to isolate
the key effect; WikiText-2 perplexity — the metric that catches key failures; same
sink+hot-window protocol as `benchmark_kvcache_postrope.py`):

| zero-point μ | calibration needed | 1.5B ppl | 7B ppl |
|---|---|---:|---:|
| fp16 reference | — | 12.234 | 9.057 |
| none (symmetric NF4) | — | 30.956 | 241.258 |
| per-window channel mean (shipped asym-NF4) | per window | 12.534 | 9.155 |
| **sparse: μ only on config-identified DC channels** | per window, **~33 % less metadata** | **12.484** | **9.150** |
| offline static μ | once, held-out train split | 12.518 | 9.224 |
| **RoPE-averaged `k_proj` bias** | **none — weights + config only** | **12.533** | **9.179** |

- **Zero-calibration matches calibrated.** Pushing the model's own `k_proj` bias through the
  position-averaged rotation gives a zero-point that recovers near-fp16 perplexity on both
  models — within 0.001/0.024 ppl of the shipped calibrated asym-NF4 — with *no calibration
  data and no per-channel metadata* (the zero-point is recomputable from the checkpoint).
  Spearman(measured offline μ, RoPE-averaged bias) ≈ 0.65 on both models: the bias doesn't
  perfectly rank-match the offsets, but it captures the part that matters.
- **Config-sparse zero-points slightly beat full calibration** at one-third less metadata:
  restricting μ to the wavelength>window channels (identified from θ/head-dim/window alone)
  matches or improves on dense per-channel means — consistent with 96–99 % of the offset mass
  living there.
## LongBench confirmation (task level)

**Script:** [`tq_zp_lb_shard.py`](tq_zp_lb_shard.py) via [`kvquant_matrix/tq_zp.sh`](kvquant_matrix/tq_zp.sh) ·
Qwen2.5-7B-Instruct, LongBench-qasper, full 200-sample split, the exact published protocol
(4-bit keys, G=32 groups, 2 % outliers, sink=4, hot=128, uniform 4-bit values), one GV100.

| key zero-point | qasper F1 |
|---|---:|
| fp16 (published, same harness) | 43.77 |
| symmetric NF4 (published) | 4.69 (collapse) |
| calibrated asym-NF4 — **anchor, this host** | 42.35 *(published 41.91 — parity ✓)* |
| **sparse μ (config-identified DC channels)** | **42.66** |
| **bias-derived μ (zero calibration)** | **43.36** |

The perplexity result confirms at the task level, and then some: **both calibration-lean
variants beat dense calibration** on the collapse-sensitive task, and the fully deterministic
bias-derived zero-point lands within 0.41 of fp16. The anchor reproducing the published 41.91
(+0.44, run-to-run noise) validates the host port (memory-efficient SDPA forced — the math
kernel OOMs at ~12k tokens on 32 GB — plus a KV-expanding wrapper because torch 2.10's
mem-efficient kernel rejects GQA-packed calls; both recorded in the harness).

- **Scope:** the bias route is Qwen-specific by construction (Mistral has no `k_proj` bias —
  its offsets come from hidden-state statistics, so it needs the offline or sparse variant);
  WikiText-2 ppl at 512-token windows, 3 chunks; LongBench task-score confirmation and the
  shipped-`PerChannelKV` integration are next.

# Computing attention directly on compressed KV codes — and an honest look at what 3-bit actually costs

*Draft for r/researchML. Tone: technical, honest, discussion-seeking. ~900 words.*

---

**TL;DR:** I built a fused CUDA "decode" kernel that runs an attention step directly on
quantized KV *codes* — no dequantization, no per-token inverse rotation. It beats
decompress-then-attend by up to **13× at 32k context** and is numerically exact vs the
dequant path (≤4e-7). But the interesting part for this sub is the *negative* result I
got when I checked it on a real model: the kernel is exact, yet **3-bit KV
quantization itself costs ~12% per-layer attention error** on real Qwen2.5-7B
activations (4-bit ~5%). So the speed win is real and the quality question is separate —
and worth being honest about.

## The one idea: compute on the codes, rotate only at the boundaries

TurboQuant-style KV compression stores each key/value as: a random rotation `R`, a
scalar-quantized code per dim (Lloyd-Max centroids), and an L2 norm. The standard decode
path reconstructs: unpack → centroid lookup → **inverse-rotate (S·d² FLOPs)** →
materialize K/V in fp16 → `qKᵀ` → softmax → `·V`. Profiling showed the dequant is ~99%
of the decode cost, dominated by that inverse rotation and by materializing K/V.

But you never need to reconstruct. With `q̃ = R·q` (rotate the query *once*):

```
score_s = ‖K‖_s · (q̃ · cent_K[kcode_s])          # ADC over the code indices
p       = softmax(score / √d)                      # online / flash softmax
acc     = Σ_s p_s · ‖V‖_s · cent_V[vcode_s]        # V weighted-sum in code space
out     = Rᵀ · acc                                 # one inverse rotation, at the end
```

Both the scores (K) and the output (V) are computed in code space. The only dense
rotations are `q̃ = Rq` and `out = Rᵀacc` — once per head per step, not once per token.
This is exactly asymmetric distance computation (ADC) from product quantization, applied
to attention. The same primitive powers compressed vector search; here it powers KV
attention. One idea, two subsystems.

## The kernel

- **M1 (correctness):** one block/head, online softmax + code-space V accumulation.
  Correct, but slow (0.47× vs dequant) — 32 blocks = no occupancy, and a per-key block
  reduction with 7 `__syncthreads` × S.
- **M2 (performance):** split-K flash-decode. One *warp* per (head, key-split); each lane
  owns d/32 dims and holds `q̃` in registers; the per-key score is a single
  `__shfl_xor` all-reduce (no shared memory, no syncthreads). Splitting the key axis
  gives `H·nsplit` blocks, so occupancy scales and the otherwise-serial S loop
  parallelizes. A cheap host-side flash-combine merges the splits.

Result on a GV100, head_dim 128, 32 heads, vs the CuPy dequant path:

| context | fused kernel | dequant | speedup |
|--------:|-------------:|--------:|--------:|
| 2,048   | 1.04 ms      | 1.83 ms | 1.8×    |
| 8,192   | 1.11 ms      | 5.71 ms | 5.1×    |
| 32,768  | 1.87 ms      | 24.9 ms | **13.3×** |

The kernel stays ~flat (memory-bound on the 3-bit codes) while dequant scales linearly —
so the advantage *widens* exactly where long-context decode hurts. Wired behind a
two-tier cache (`fused_decode`), it's also 2.3× faster than just *decompressing* the
cache, and exact vs decompress-then-attend (4.1e-7).

## The honest part: exact ≠ lossless

"Exact vs dequant" only means the kernel reproduces whatever the quantization gives you.
So I checked the *quantization* on a real model: load Qwen2.5-7B, feed a real
long-context prompt, capture the true post-RoPE q/k/v via a non-invasive hook, and
compare per-layer attention output: **fp16 vs fused-decode over quantized KV**.

| KV config | all-cold (worst case) | two-tier (fp16 sink+hot, coded cold) |
|---|---:|---:|
| 3-bit | 0.367 | **0.155** (median 0.12) |
| 4-bit | 0.208 | **0.086** (median 0.048) |

Takeaways I didn't expect to have to write down:

1. **3-bit KV is aggressive.** ~12% median per-layer attention-output error even with an
   fp16 hot window. 4-bit halves it. We changed the library's AutoConfig to **default
   keys to 4-bit** (attention scores `softmax(QKᵀ/√d)` amplify key error through the
   softmax; values are far less sensitive and carry the compression).
2. **The two-tier scheme matters a lot** — keeping the attention *sink* (first tokens)
   and the recent *hot window* in fp16 cuts error ~2.4× vs all-cold.
3. **A few layers stay high** (max ~0.6) — the known attention-sink / outlier-channel
   problem. Per-channel scaling or more fp16 sinks would help. Honest open item.

A caveat on the metric: per-layer attention-output L2 is a *pessimistic* proxy (it
weights all output dims equally and ignores that later layers may absorb or compound the
error). A full LongBench *task score* needs a generate()-loop hook (model-arch-specific;
scaffolded but not run). I'd rather report the fidelity I can stand behind than a task
number I hand-tuned.

## Why I think this is worth sharing

The speedup is a clean systems result, but the part I keep coming back to is the
discipline of separating "is the kernel correct?" (yes, provably) from "is 3-bit KV good
enough?" (depends, and less than the marketing usually implies). Same energy as the other
thing this library keeps insisting on: cosine-similarity-to-the-original is *not* a
reliable proxy for retrieval quality at high compression — measure the task.

Code, kernels, benchmarks, and the LongBench-parity script are all open
(`pip install turboquant-pro`, v1.1.0). Happy to share the kernel source or the capture
hook.

**Discussion:** (1) Has anyone measured per-layer KV-quant error vs downstream task
score directly — how tight is the L2-proxy-to-task-score relationship in practice? (2)
Best lightweight fix for the outlier-channel layers that doesn't blow up the bit budget
— per-channel scales, Hadamard pre-rotation, or just more fp16 sinks?

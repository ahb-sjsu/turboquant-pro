# Design Doc — Fused KV-Decode CUDA Kernel

**Status:** proposed · **Owner:** TBD · **Goal:** a single CUDA kernel that computes
one autoregressive attention step directly over compressed K/V codes — no
reconstruction, no inverse-rotation per token — turning the 99%-of-decode dequant
cost (`RESULTS_decode_overhead.md`) into a fused, memory-bound flash-decode.

## 1. Problem & success criteria

Today's decode path (cold pages): `unpack 3-bit → centroid lookup → inverse-rotate
(S·d² FLOPs) → materialize K in fp16 → qKᵀ → softmax → ·V`. Measured: dequant is
**~99%** of the CPU decode step, dominated by the **inverse-rotation matmul** and by
**materializing K/V**. We have proven (`benchmark_kv_adc.py`, exact to 3.3e-7) that
scores need neither:

```
score_s = q · K̂_s = ‖K‖_s · ( rotate(q) · cent_K[code_s] )          # ADC, no inverse-rotation
out     = Σ_s p_s K̂… no: out = unrotate( Σ_s p_s ‖V‖_s cent_V[vcode_s] )   # one unrotate, at the end
```

so **both** K (scores) and V (weighted sum) are computed in code space; only `rotate(q)`
(once/head) and `unrotate(out)` (once/head) touch the rotation.

**Success criteria.**
1. **Correctness:** output matches the reference dequant attention to fp16 tolerance
   (the math is exact; only LUT/accumulator quantization differs).
2. **Latency:** ≥ the throughput of an fp16 flash-decode at equal context, with KV in
   **3-bit** (≈5× less KV traffic) — i.e. memory-bound on the 3-bit codes, not on
   reconstructed fp16.
3. **Drop-in:** behind `TurboQuantKVCache`, composing with the two-tier hot/cold cache
   (hot window stays fp16; the kernel serves cold pages).

## 2. Math (code-space attention)

Per head, decode step with query `q ∈ ℝ^d`, `S` cached keys/values. Keys/values stored
as tq-pro codes: `K̂_s = ‖K‖_s · R^T cent_K[kcode_s]`, likewise `V`. With `q̃ = R q`
(R = tq-pro rotation, applied **after** RoPE — see §6):

```
score_s = ‖K‖_s · Σ_j q̃_j · cent_K[kcode_s,j]                     (ADC over d code indices)
p       = softmax( score / √d )                                    (online / flash)
acc_j   = Σ_s p_s · ‖V‖_s · cent_V[vcode_s,j]      (j = 1..d)       (accumulate in code space)
out     = R^T acc                                                  (one inverse-rotation)
```

No per-token reconstruction, no per-token inverse-rotation; the only dense rotations
are `q̃ = Rq` and `out = R^Tacc`, each `d²` once per head per step.

## 3. Kernel design (flash-decode + ADC)

One thread block per `(batch, head)` (decode batch is small; grid = B·H).

1. **Build the score LUT.** From `q̃` (d floats) and the shared K-codebook
   `cent_K` (2^b entries), compute `LUT[j,s] = q̃_j · cent_K[s]` (d × 2^b). For
   `b=3` that's d×8 floats; quantize to int8 with a per-(head) scale for the
   `pshufb`/`__byte_perm` fast path (reuse the embedding-kernel LUT logic).
2. **Streaming flash loop over key tiles.** For each tile of `T` keys:
   unpack codes (3/4-bit), accumulate `score_s = Σ_j LUT[j, code_{s,j}]` via shared-mem
   table lookups; maintain running `m` (max) and `ℓ` (sum) for **online softmax**;
   and accumulate the **V code-space sum** `acc_j += p_s·‖V‖_s·cent_V[vcode_{s,j}]`
   incrementally with the same rescaling trick FlashAttention uses on `m` updates.
3. **Finalize.** `acc /= ℓ`; apply `out = R^T acc` (one `d×d` apply, or the structured
   sign-flip+permutation for `d>4096`); write fp16 output.

**Layouts.** Repack cold-page codes into the kernel-friendly blocked layout at
eviction time (when L1→L2 demotion happens), amortizing the repack the same way the
embedding `ADCIndex` does at build. K and V share the codebook tables in shared memory.

**Precision.** Scores accumulate in fp32 (or int32 with the int8 LUT + a dequant
scale); the V code-space accumulator is fp32 (d floats/head in registers/shared).
3-bit unpack via shift/mask; ship a **4-bit "kernel mode"** first (byte-aligned,
`pshufb`-friendly), add 3-bit after — mirroring the embedding-kernel plan.

## 4. Integration

- `TurboQuantKVCache.fused_decode_step(q, layer)` → calls the kernel over cold pages,
  combines with the fp16 hot-window attention (standard) via one more online-softmax
  merge (hot and cold partials combined by `(m, ℓ, acc)`).
- Reuses the existing CuPy `RawKernel` plumbing (`cuda_kernels.py`) and the rotation
  (`_Pi`) / codebook (`centroids`) already on `TurboQuantPGVector`/`TurboQuantKV`.
- CPU fallback: the exact numpy path in `benchmark_kv_adc.py` (already validated).

## 5. Validation

- **Correctness:** per layer, compare fused output vs `dequant→qKᵀ→softmax→·V` (fp16
  tolerance) on random and real activations; assert score path exact vs §2.
- **Latency/throughput:** tokens/s and time-to-first-token vs an fp16 flash-decode at
  2k/8k/32k context, 3-bit and 4-bit; report KV bytes moved.
- **End-to-end quality:** LongBench / GSM8k perplexity-neutrality at 3-bit (sanity vs
  the KV-compression claims), so the kernel changes speed, not accuracy.

## 6. Risks & mitigations
- **RoPE interaction.** RoPE is applied to q,k by the model *before* tq-pro
  quantization; the cache stores post-RoPE, tq-pro-rotated key codes. The decode query
  must be RoPE'd at the current position *then* tq-pro-rotated (`q̃ = R·RoPE(q)`)
  before the kernel — handled in the wrapper, not the kernel. Document and test this
  ordering explicitly (it is the most likely correctness bug).
- **3-bit unpack in-kernel** → 4-bit mode first; 3-bit via RAM-side repack.
- **Online-softmax + V-rescale** is the subtle part → start from a vetted flash-decode
  template (e.g. FlashAttention-2 decode) and swap the K/V loads for ADC/code-space.
- **Two codebooks (K,V) + LUT in shared memory** → fits easily (d×2^b is KB-scale);
  watch occupancy if d is large.

## 7. Effort & milestones
1. **M0 — DONE.** CuPy/NumPy reference of the *full* fused step (scores+softmax+V in
   code space, one unrotate) in `turboquant_pro/kv_fused.py`; validated **exact** vs the
   dequant path (3.4e-7 CPU, 4.0e-7 GPU) by `benchmark_kv_decode.py` + `tests/test_kv_fused.py`.
   Array-level CuPy is 1.36x (still materializes `cent[codes]`); the raw kernel (M1)
   removes that for the memory-bound win.
2. **M1 — DONE (correctness).** Single-head CUDA kernel (CuPy RawKernel,
   `turboquant_pro/kv_kernel.py`): one block/head, fp32 online softmax + code-space V,
   validated **2.5e-6** vs the M0 reference (`benchmark_kv_kernel.py`). It is
   deliberately *not yet fast* — 0.47× vs dequant — because of low occupancy (32
   blocks = one per head) and a per-key block reduction (7 syncthreads × S). Those are
   M2's targets, not bugs.
3. **M2 — DONE (performance).** Split-K flash-decode (`fused_decode_split` in
   `turboquant_pro/kv_kernel.py`): one warp per (head, key-split), `__shfl_xor` score
   reduction (no `__syncthreads`/shared LUT), host-side flash-combine across splits.
   Correct (5–7e-7 vs reference) and **beats the dequant path 1.8× @2k, 5.1× @8k,
   13.3× @32k** context — the kernel stays ~1–2 ms while dequant scales with context,
   so the win widens where long-context decode hurts. Remaining M2 polish (int8-LUT
   `pshufb` score path, hot/cold online-softmax merge) folds into M3.
4. **M3 (3–5 d):** 3-bit support, `TurboQuantKVCache` integration, LongBench/GSM8k
   neutrality, docs.

**Net:** ~2–3 weeks. This shares the asymmetric-distance primitive with the shipped
embedding kernel (`turboquant_pro/_adc`): one idea — *compute on the codes, rotate at
the boundaries* — serves both vector search and KV-cache attention.

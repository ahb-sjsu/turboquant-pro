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
4. **M3 — core DONE.** (a) **3-bit** confirmed (kernel handles ncent=8: 4.4e-7 vs
   reference, 4.9× over dequant) alongside 4-bit. (b) **Hot/cold online-softmax merge**
   (`kv_fused.fused_decode`): fp16 hot window + coded cold pages combined exactly —
   validated == full attention over [dequant-cold ; hot] (CPU) and CPU↔GPU to 2.7e-7.
   **(c) `TurboQuantKVCache.fused_decode` — DONE.** Feeds the cache's packed
   `CompressedKV` cold chunks to the kernel (`return_partials=True`) and merges with the
   fp16 hot window via `merge_partials`. Validated on GPU: **exact vs decompress-then-
   attend (4.1e-7)** and **2.3× faster than decompressing the cache alone** (32 ms vs
   75 ms at 4k context, head_dim 128, 8 heads) — the full fused attention beats just the
   reconstruction step. CPU path uses the numpy reference. **Only remaining:** the
   LongBench/GSM8k end-to-end run on a served model — whose quality is already
   *determined* (every step is exact vs the dequant path; the kernel changes speed, not
   accuracy). The kernel and its cache integration are shipped.

**Net:** ~2–3 weeks. This shares the asymmetric-distance primitive with the shipped
embedding kernel (`turboquant_pro/_adc`): one idea — *compute on the codes, rotate at
the boundaries* — serves both vector search and KV-cache attention.

## 8. M4 — Per-channel keys with dense-sparse fp16 outliers (design)

The kernels above decode the **PolarQuant** format (global centroid table, per-token
norms, rotation at the boundaries) — the *values* path and legacy keys. The
recommended **keys** path since v1.2.0 is `PerChannelKV`: per-channel asym-NF4 with
optional top-2% fp16 outliers and, since the zero-point work, three zero-point modes.
Today outlier-bearing keys therefore go through decompress-then-attend; M4 fuses them.
The reformulation that makes it clean: **keys only enter attention through `q·k`, so
every piece of the per-channel format becomes a term of the score** — nothing is ever
scattered into a reconstructed K.

### 8.1 Score decomposition (dense part)

Per-channel keys are quantized in the **original post-RoPE basis — no rotation** —
so the decode query is used directly (contrast §6's RoPE+rotate ordering for
PolarQuant). For asym-NF4, `k̂_j = μ_j + a_j · NF4[c_j]`, hence

    score_s = q·μ(h)  +  Σ_j (q_j · a_j) · NF4[c_js]
              ─────────    ────────────────────────────
              bias(h):     w_j = q_j·a_j (H,d), computed once per decode
              folded once  step host-side; the 16-entry NF4 table lives in
              per head     registers — NO shared-memory LUT is needed.

Uniform per-channel keys are the same shape (`w_j = q_j·scale_j`, bias `q·zero`).
**All three zero-point modes land in the same `q·μ` bias term**: calibrated and
sparse μ are stored per block; the "bias" mode's μ is recomputable host-side from
(k_bias, θ, block position) — one dot product per head per block, folded into the
bias. The kernel never branches on the mode.

### 8.2 Outliers as sparse score deltas (the contiguity/divergence answer)

The container stores outliers as flat `(outlier_idx int32, outlier_val fp16)` chosen
**per channel** (top-2% over tokens). The fused form stores, per outlier, the *delta*
against the dense dequant:

    Δ_(s,j) = v_fp16 − (μ_j + a_j·NF4[c_js])        (fp16)
    score_s += Σ_{j ∈ outliers(s)}  q_j · Δ_(s,j)

- **Contiguity:** selection is channel-major but consumption is token-major, so a
  one-off build-time transpose packs a **token-major CSR**: `row_ptr (H, S+1) int32`
  plus entry arrays `(col uint16, delta fp16)`. At 2% × d=128 that is ~2.6 entries
  per (head, token), read contiguously.
- **Divergence:** the dense loop stays completely branch-free (it never checks for
  outliers). The sparse correction is a separate short pass per token in the same
  warp: **lanes stride the row's entries** (`for e = row_ptr[s]+lane; e < row_ptr[s+1];
  e += 32`), each computing `q[col_e]·Δ_e`, then one `__shfl_xor` reduction adds the
  correction to the score before the online-softmax update. Divergence is bounded to
  the ragged tail of a ~3-entry list; rows can be padded to warp multiples if
  profiling ever shows it matters.
- **Why deltas, not values:** the dense pass needs no masking and the sparse pass is
  pure addition — no code re-reads, no scatter, and it composes with the flash
  accumulator because the correction lands before that token's max/renormalize step.

### 8.3 Kernel shape and V path

`fused_decode_pck` clones the M2 split-K skeleton (one warp per (head, key-split),
`__shfl_xor` reductions, unnormalized (m, l, acc) partials, host flash-combine): only
the score computation is swapped. Values are untouched — PolarQuant code-space
accumulation exactly as M1/M2 (the recommended config quantizes values with
`TurboQuantKV`), so the V accumulator and the host-side unrotate are shared code.

### 8.4 Validation ladder

1. NumPy/CuPy reference (`kv_fused_pck.py`): fused per-channel scores + CSR deltas
   **exact vs `PerChannelKV.decompress` → attention**, for uniform / NF4 / asym-NF4 ×
   {calibrated, sparse, bias} zero-points × outlier_frac ∈ {0, 2%}.
2. CUDA kernel vs reference to ≤1e-5 (fp32 accumulators), 4-bit first.
3. Cache integration (`TurboQuantKVCache.fused_decode` dispatching per-channel key
   blocks to the new kernel) + the same exactness gate as M3(c).

### 8.5 M4 status — DONE (incl. cache dispatch)

Reference (`turboquant_pro/kv_fused_pck.py`) and CUDA kernel
(`fused_decode_pck` + `fused_decode_pck_cuda` in `turboquant_pro/kv_kernel.py`)
are implemented and validated: **exact vs decompress-then-attend** on CPU for
uniform / NF4 / asym-NF4 × {calibrated, sparse, bias} zero-points ×
outlier_frac ∈ {0, 2%, 10%} × {packed, unpacked} × hot/cold merge
(`tests/test_kv_fused_pck.py`), and **exact vs the reference on GPU**
(max err ≤ 8e-8 on GV100). End-to-end one-shot wrapper timing vs
decompress-then-attend (H=8, d=128, 2% outliers): **6.0× @2k, 2.1× @8k,
2.1× @32k** — with the wrapper rebuilding the CSR and re-uploading codes every
call.

**Cache dispatch (validation-ladder step 3) is in.**
`TurboQuantKVCache.fused_decode` now routes per-channel key pages through the
fused path via a `PreparedPCKBlock` per cold page: the query-independent work
(key-code unpack, grid parameters, token-major outlier CSR, value codes/norms
— device-resident on GPU) is built **once per cold flush**, lazily on the
first fused decode that sees the page, and reused for every subsequent decode
step; per call only the O(H·d) query projections (`w = q ⊙ a`, `bias = q·μ`)
are computed. Pages are immutable once flushed, so the cache is append-only
(`clear()` drops it). Each page contributes an `(m, l, acc)` partial — the
kernel on GPU, the reference einsum on CPU — merged exactly with the hot
window by the shared online-softmax `merge_partials`. Fallback to
decompress-then-attend remains for grids/configs with no fused form: nuq
(data-fit quantile) tables, structured rotations, and GPU head dims outside
the kernel's `d % 32 == 0, d ≤ 512` envelope. Memory note: prepared pages hold
bit-unpacked (uint8) codes — the materialization the kernel consumes; the
packed container remains the storage of record.

Measured (GV100, H=8, d=128, asym-NF4 + 2% outliers, hot_window=512, exact to
≤6e-8 vs the reconstruct path): steady-state fused decode **2.0× @2k, 5.6×
@8k, 12.5× @32k** over decompress-then-attend — the amortization the one-shot
wrapper couldn't show (it plateaued at 2.1× @32k rebuilding the CSR per
call). The first decode after a flush pays the page build (~1.4–1.8× one
reconstruct decode across all pages, then amortized). Next (kernel work, not
dispatch): batch the per-page launches (126 pages @32k) into one
variable-page kernel, and a packed-code (`uint4`-per-code) dense loop to drop
prepared-page memory 2×.

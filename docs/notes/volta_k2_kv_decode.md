# Volta K2 — fused KV attention-on-codes decode (P0)

Part of the Volta-kernel track (extend the existing CuPy/NVRTC kernel layer for
`sm_70`; do not fork CUDA). K2 is the memory-bandwidth play — fuse the
per-channel-KV dequant into the score so fp16/fp32 K is never materialized. It
does **not** use tensor cores (`wmma`/`HMMA.884` is K1's concern), so it is the
low-risk first kernel.

## What it computes

Per-channel asym-NF4 keys enter attention only through `q·k`, so:

    score[h,s] = bias[h] + sum_d w[h,d] * grid[code[h,s,d]]
      w    = q * per-channel weight   (H,D)
      bias = sum_d q * per-channel mu (H,)   -- zero-point folded
      grid = NF4 / uniform codebook   (L,)

`kv_fused_pck.pck_key_scores` expresses this as `einsum(w, grid[codes])`, which is
correct but materializes `grid[codes]` — an `(H,S,D)` fp32 tensor, the same
bandwidth as decompressing. `volta_kernels.k2_key_scores` reads the 1-byte codes
directly and does the LUT-dequant + dot in registers.

## Kernel (v1)

One warp per `(h,s)`: 32 lanes stride the head dim `D` (coalesced byte reads of
`codes`), the NF4 grid lives in shared memory, a warp shuffle reduces the partial
dot, lane 0 writes `acc + bias[h]`. Compiled lazily via CuPy NVRTC (arch auto =
`compute_70` on a GV100). No `cp.async`/`ldmatrix`/tensor cores — none needed for
a memory-bound GEMV.

## Measured (Quadro GV100, uncontended, fp32 codes intermediate for baselines)

| shape (H,S,D) | kernel | decompress+attend | einsum(codes) | speedup |
|---|---|---|---|---|
| 8, 4096, 128  | 0.016 ms | 0.397 ms | 0.333 ms | 24.8× / 20.8× |
| 32, 4096, 128 | 0.056 ms | 1.094 ms | 0.316 ms | 19.5× / 5.6× |
| 32, 8192, 128 | 0.125 ms | 1.538 ms | 0.622 ms | 12.3× / 5.0× |

Exact vs the NumPy reference and vs `pck_key_scores` (asym-NF4, dense) to fp32
rounding (~5e-6). Kernel sustains ~270–300 GB/s of codes traffic, ≈31% of the
GV100's ~870 GB/s HBM2 peak — so there is tuning headroom.

The K1 kill-criterion bar (≥1.3× over dequant+cuBLAS) does not apply to K2, but
for reference K2 clears it by 10×+.

## P1 finding — packed codes are a storage win, not a speedup

`k2_key_scores_packed` reads the container's `packed=True` bytes (the bit-reversed
`_pack_indices` layout). Correct to fp32 (5.7e-6). Measured on an uncontended
GV100 (bits=4):

| shape (H,S,D) | unpacked | packed | vs unpacked | vs decompress+attend |
|---|---|---|---|---|
| 32, 4096, 128 | 0.071 ms | 0.079 ms | 0.89× | 11.0× |
| 32, 8192, 128 | 0.125 ms | 0.139 ms | 0.90× | 11.4× |
| 8, 8192, 128  | 0.030 ms | 0.031 ms | 0.97× | 13.1× |

**Packing does not speed decode** — K2 is latency/occupancy-bound (~31% of HBM2
peak), so reading half the bytes doesn't cut time. Two dead ends were measured and
discarded honestly: naive per-bit reconstruction was **3× slower** than unpacked
(4 byte-loads + bit ops per code); a single-load-per-code variant was still ~3×.
The shipped bits=4 kernel uses a 256-entry format-decode LUT (byte → its two
codes) with contiguous channel-pair assignment, which reaches **parity** (~0.9×)
at **half the KV-cache storage**. So packing buys memory, not latency; the speedup
lever is occupancy tuning (below), after which packing would also help.

## P2 finding — occupancy tuning: ~34% → ~60% of peak (1.6–1.9×)

The scalar one-warp-per-`(h,s)` kernel did too little work per warp (four 32B
byte-loads + a shuffle-reduce) and sat at ~34% of HBM2 peak. Two changes, measured
independently on an uncontended GV100 (unpacked, exact 5.7e-6):

| shape (H,S,D) | scalar | vec4 | vec4+ns4 | vec4+ns4 vs scalar |
|---|---|---|---|---|
| 32, 4096, 128  | 0.0552 ms (35%) | 0.0537 (36%) | 0.0347 ms (**56%**) | 1.59× |
| 32, 8192, 128  | 0.1139 ms (34%) | 0.1058 (37%) | 0.0644 ms (**60%**) | 1.77× |
| 32, 16384, 128 | 0.2340 ms (33%) | 0.2831 (27%) | 0.1229 ms (**63%**) | 1.90× |

- **vec4** — each lane reads a `uint32` (4 codes) → one coalesced 128B transaction
  per warp instead of four 32B ones. *Alone it barely helps* (even regresses at
  large S): fewer instructions but the warp is still latency-stalled.
- **vec4+ns4** — each warp also keeps **NS=4 independent `s` rows** in flight, so
  four loads overlap before any reduction. This memory-level parallelism is the
  real lever → ~55–63% of peak. `ns8` gave nothing over `ns4` and costs registers.

Shipped as `_KERNEL_SRC_VEC4NS` (default when `D % 4 == 0`; bounds-checked so S
need not be a multiple of 4 — validated at S=1023/1021/777 and D=64/96 on the
GV100). The scalar kernel remains the odd-`D` fallback. K2 is now closer to
bandwidth-bound, so packing (P1) should begin to pay as a speedup too — an open
follow-up (the packed LUT kernel has not yet had the vec/ns treatment).

## P3 finding — vec/ns on the packed kernel: ~1.5× over baseline packed, but still
## not a speedup over unpacked

Applied the same vec+ns4 recipe to the packed path: each lane reads a `uint16`
(2 bytes = 4 codes) per row, the format-decode LUT turns each byte into its two
codes, and NS=4 `s` rows stay in flight. Uncontended GV100 (exact 5.7e-6):

| shape (H,S,D) | unpacked (tuned) | packed base (P1) | packed vec+ns4 |
|---|---|---|---|
| 32, 4096, 128  | 0.040 ms | 0.063 ms | 0.046 ms (0.86× unpacked) |
| 32, 8192, 128  | 0.073 ms | 0.121 ms | 0.087 ms (0.84×) |
| 32, 16384, 128 | 0.159 ms | 0.256 ms | 0.179 ms (0.89×) |
| 8, 8192, 128   | 0.023 ms | 0.030 ms | 0.020 ms (**1.15×**) |

**~1.5× faster than the P1 packed baseline**, near-parity with the tuned unpacked
(even faster at small H). But packing still does **not** beat unpacked decode: a
D=128 packed row is only 64B — half a 128B transaction — so the packed kernel
tops out at ~24% of peak and can't go bandwidth-bound the way unpacked (uint32,
full line) does. Confirms the P1 verdict *after* tuning: packing is a **storage**
win (half the KV cache) at ~parity decode. A `gridLUT` variant (byte → grid-value
pairs, killing the code→grid indirection) was slightly faster at small S but its
2 KB shared table **regressed at large S** (occupancy), so `vec+ns4` (small shared,
consistent) shipped instead. Both dead ends recorded honestly.

## P4 — values leg + outlier CSR: the whole cold-block decode fused

The decode has three code-reading terms; P0–P3 did the keys, P4 adds the other two
so the entire `pck_cold_partials` runs on-GPU with no `(H,S,D)` fp32 intermediate.

**Values leg** (`value_accum`) — the transpose of the key score:
`acc_code[h,d] = sum_s (e·norm_v)[h,s] · cent[vcode[h,s,d]]`, reduced over S with a
scalar PolarQuant codebook. Block per `(h, s-chunk)`, one thread per `d` so the
`vcodes` reads coalesce, S split across blocks with `atomicAdd` (SCH=128 best).
Uncontended GV100 vs the cupy `einsum(cent[vcodes])`:

| shape (H,S,D) | value kernel | einsum | speedup |
|---|---|---|---|
| 32, 4096, 128 | 0.066 ms (30% peak) | 0.77 ms | **11.8×** |
| 32, 8192, 128 | 0.093 ms (42% peak) | 1.46 ms | **15.8×** |

**Outlier CSR** (`apply_outlier_csr`) — dense-and-sparse keeps the top-magnitude
key entries in fp16; `build_outlier_csr` turns them into token-major score deltas.
One thread per `(h,s)` row adds `sum_k q[h,cols[k]]·deltas[k]` on-GPU, so the sparse
correction no longer round-trips to a NumPy scatter.

**End-to-end**: a test composes `k2_key_scores` → `apply_outlier_csr` → softmax
(cupy, cheap on `(H,S)`) → `value_accum` → `@ pi` and matches `pck_cold_partials`
`(m, lsum, acc)` to fp32 tolerance on the GV100. The two heavy code-reading einsums
are now fused kernels; only the softmax and the tiny `(H,D)@(D,D)` unrotation stay
in cupy (neither materializes a large tensor).

## Status / next

- [x] v1 kernel (unpacked uint8 codes), exactness tests, benchmark.
- [x] **Occupancy tuning** — vec4+ns4, ~34% → ~60% of peak (1.6–1.9×); shipped.
- [x] **Packed 4-bit codes + vec/ns** — uint16+LUT+ns4, ~1.5× over naive packed,
      near-parity with unpacked at half storage; general per-bit kernel for 2/3-bit.
      Packing is a storage win, not a decode speedup (64B-row transaction limit).
- [x] **Values leg** (`value_accum`) — 12–16× over cupy einsum; fused.
- [x] **Outlier CSR deltas** (`apply_outlier_csr`) — on-GPU sparse key correction.
- [x] **Whole cold-block decode** validated against `pck_cold_partials`.
- [ ] **Integration** — dispatch these from `kv_fused_pck`/`kv_kernel` when CuPy +
      an `sm_70` device are present, else keep the einsum reference. (Wiring only;
      the kernels and the end-to-end equivalence are done.)

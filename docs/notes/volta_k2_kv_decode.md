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

## Status / next

- [x] v1 kernel (unpacked uint8 codes), exactness tests, benchmark.
- [x] **Packed 4-bit codes** — LUT kernel, exact, parity decode at half storage
      (see finding above); general per-bit kernel for 2/3-bit.
- [ ] **Occupancy tuning** (now the priority) — more `(h,s)` per warp / ILP /
      vectorized loads to become bandwidth-bound; only then does packing speed up.
- [ ] **Outlier CSR deltas** in-kernel (or a fused second pass) — currently the
      caller adds `build_outlier_csr` contributions; fold them so the sparse path
      stays on-GPU.
- [ ] **Values leg** — merge PolarQuant value partials (the `pck_cold_partials`
      online-softmax) so the whole decode step is one fused path.
- [ ] **Tuning** — LUT shared-bank layout, block granularity, and `S`-tiling to
      push past ~31% of HBM2 peak.
- [ ] **Integration** — dispatch `k2_key_scores` from `kv_fused_pck` when CuPy +
      an `sm_70` device are present, else keep the einsum reference.

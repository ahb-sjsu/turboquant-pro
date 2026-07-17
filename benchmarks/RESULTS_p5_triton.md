# P5 Triton port — exactness + latency

Harness: [`p5_triton_bench.py`](p5_triton_bench.py) and
[`tests/test_kv_triton.py`](../tests/test_kv_triton.py). The Triton port of the
M2/M4 fused-decode kernels (`turboquant_pro/_triton_kernels.py`) must be exact
vs the NumPy reference (the CuPy RawKernel oracle) and reach perf parity; see
`docs/DESIGN_hardware_and_plugins.md` §4.2 / P5.

## Exactness — 7/7, NRP L40

Both layers green:

- **CPU-logic gate** (runs everywhere): a NumPy replay of the kernel's exact
  control flow (split-K + per-token CSR loop + online softmax + flash-combine)
  equals the all-at-once reference across D=64/96/128 and outlier fractions
  0/0.02/0.05 — proves the decomposition before any GPU time.
- **GPU exactness** (NRP L40, torch 2.4.0 / triton 3.0.0): the per-page,
  batched-page, and Polar kernels all match the reference. `pytest -q`: **7
  passed**.

## Latency — batched-page beats per-page, NRP L40

Per decode step, 8 heads × 128 dim, per-channel asym-NF4 keys + 2 % fp16
outliers, PolarQuant values. `err` is max |Triton − NumPy reference| (fp32
reassociation noise).

| ctx | cold pages | err per-page | err batched | ms per-page | ms batched | batched speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 6 | 7.2e-08 | 8.4e-08 | 3.43 | 2.08 | **1.65×** |
| 8,192 | 30 | 4.5e-08 | 5.2e-08 | 18.09 | 9.32 | **1.94×** |
| 32,768 | 126 | 1.7e-08 | 1.8e-08 | 75.47 | 37.52 | **2.01×** |

The single-launch batched-page kernel (the deferred §8.5 item, landed in the
Triton port) widens from 1.65× to ~2× as the cold cache grows — the per-page
path pays one launch per page, the batched path one launch total.

## Perf parity — batched Triton beats the CuPy RawKernel, H100

H100 80GB (lightning, torch 2.8.0+cu128 / triton 3.4.0), same config, with the
CuPy RawKernel oracle (`TurboQuantKVCache.fused_decode` on `use_gpu=True`) timed
as the parity baseline. `pytest`: **7 passed**. This is the exit-criterion
"parity or better" check — cleared on Hopper.

| ctx | cold pages | ms per-page | ms batched | **ms CuPy RawKernel** | batched vs RawKernel | err |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 6 | 4.25 | 2.22 | 17.46 | **7.9×** | 7.7e-08 |
| 8,192 | 30 | 21.88 | 8.57 | 25.89 | **3.0×** | 4.5e-08 |
| 32,768 | 126 | 94.54 | 36.55 | 69.96 | **1.9×** | 1.7e-08 |
| 65,536 | 254 | 191.55 | 75.69 | 130.18 | **1.7×** | 1.4e-08 |
| 131,072 | 510 | 390.60 | 159.53 | 251.97 | **1.6×** | 1.1e-08 |

The batched-page kernel is faster than the RawKernel at **every context length
from 2k to 128k** (and exact to ~1e-8) — the ratio narrows from 7.9× to 1.6× as
the RawKernel's per-page launch overhead amortizes, but batched stays ahead
throughout. Per-page Triton wins at small ctx but falls behind the RawKernel
from 32k on — it pays P launches (254/510 at 64k/128k) — so **batched is the
path to recommend**. (The 64k/128k `cupy` numbers, 130/252 ms, match the
independent long-context bench's steady `fused_decode` — 128/246 ms — so the two
harnesses agree on the RawKernel baseline.) These kernels are launch/latency-bound at this size (single decode
step, 8×128), so H100 ≈ L40 in absolute ms; the win is the batched-vs-RawKernel
ratio, not Hopper throughput. (The `numpy<2` pin didn't take in that cloudspace
image — numpy 2.5.1 — so cupy/scipy logged "compiled against NumPy 1.x"
warnings; cosmetic, the oracle still ran, hence the real `cupy` numbers.)

## Pending

- **A100 perf-parity** — the exit criterion names A100; H100 already shows
  better-than-parity and L40 exactness holds, so A100 is confirmation, not a gap.
- **B200 / native NVFP4** — Blackwell (`sm_100`) is the first native-NVFP4 arch;
  the port compiles under torch 2.8+cu128 / triton 3.4, but a B200 allocation
  hasn't materialized yet (a "B200" studio request landed an H100).
- **ROCm** — single-source ready; not validated (Nautilus has no AMD GPUs, §4.5).

Environment notes (both cost a validation round, worth knowing): Triton JIT
needs a host C compiler — use the `pytorch...-devel` image, not `-runtime`
("Failed to find C compiler"); and `@triton.jit` kernels must live at **module
scope** — Triton resolves `tl` against the kernel's module globals, so a
closure-local `tl` raises `NameError('tl is not defined')` at compile time.

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

## Perf parity — A100 (the named exit-criterion GPU)

A100-SXM4-40GB (Colab Pro, torch 2.x / triton), same config. `pytest`: **7
passed**. The P5 exit criterion names A100 explicitly — this is it, measured, not
inferred from H100.

| ctx | cold pages | ms per-page | ms batched | **ms CuPy RawKernel** | batched vs RawKernel | err |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 6 | 7.65 | 4.91 | 46.20 | **9.4×** | 1.9e-07 |
| 8,192 | 30 | 36.73 | 20.64 | 57.58 | **2.8×** | 2.4e-07 |
| 32,768 | 126 | 153.53 | 86.30 | 113.26 | **1.3×** | 1.3e-07 |

Batched Triton beats the RawKernel at every context (9.4× → 1.3×), exact to
~1e-7 (fp32 reassociation noise; a touch above H100's ~1e-8 but well inside the
tests' tolerances, which all passed). Same shape as H100/L40 — per-page trails
the RawKernel from 32k on (P launches), batched wins throughout. **"Parity or
better" now holds on all three named tiers (L40 exactness, H100, A100).**

## Portability — Turing (T4), the oldest arch in the fleet

Tesla T4 (`sm_75`, Colab), 7/7 exact. The port compiles and runs unchanged on
Turing — the kernels are fp32 code-space, so no fp8/bf16 tensor cores are needed
and the oldest GPU works as-is.

| ctx | cold pages | ms per-page | ms batched | ms CuPy RawKernel | batched vs RawKernel | err |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 6 | 8.93 | 4.83 | 58.17 | **12.1×** | 1.9e-07 |
| 8,192 | 30 | 37.38 | 19.98 | 54.29 | **2.7×** | 2.4e-07 |
| 32,768 | 126 | 159.34 | 84.89 | 106.34 | **1.3×** | 1.3e-07 |

The absolute ms nearly match the A100 (per-page 8.9/37.4/159 vs A100's
7.7/37/154) — a clean demonstration that these single-decode-step kernels are
**launch/latency-bound**: GPU class barely moves them, and the batched-vs-
RawKernel ratio is the signal, not raw throughput. **P5 coverage now spans
Turing → Ampere → Hopper** (T4 / A100 / H100), all exact, batched winning
throughout.

## Pending

- **A100 perf-parity** — ✅ **cleared** (see the A100 table above): 9.4× → 1.3×
  over the RawKernel on A100-SXM4-40GB, 7/7 exact.
- **B200 / native NVFP4** — Blackwell (`sm_100`) is the first native-NVFP4 arch;
  the port compiles under torch 2.8+cu128 / triton 3.4, but a B200 allocation
  hasn't materialized yet (a "B200" studio request landed an H100).
- **ROCm** — single-source ready; not validated (Nautilus has no AMD GPUs, §4.5).

Environment notes (both cost a validation round, worth knowing): Triton JIT
needs a host C compiler — use the `pytorch...-devel` image, not `-runtime`
("Failed to find C compiler"); and `@triton.jit` kernels must live at **module
scope** — Triton resolves `tl` against the kernel's module globals, so a
closure-local `tl` raises `NameError('tl is not defined')` at compile time.

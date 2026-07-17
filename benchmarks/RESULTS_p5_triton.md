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

## Pending

- **CuPy RawKernel perf-parity column** — the oracle needs CUDA toolkit headers
  for its own JIT (`pip install cupy-cuda12x[ctk]`); the plain wheel on the
  `-devel` image raised "Failed to find CUDA headers", so `ms_cupy_raw` is null
  here. Exactness vs the NumPy reference (which the RawKernel is itself gated
  against) is unaffected.
- **A100 perf-parity** — the exit criterion's A100 run; L40 exactness + the
  batched speedup already hold.
- **ROCm** — single-source ready; not validated (Nautilus has no AMD GPUs, §4.5).

Environment notes (both cost a validation round, worth knowing): Triton JIT
needs a host C compiler — use the `pytorch...-devel` image, not `-runtime`
("Failed to find C compiler"); and `@triton.jit` kernels must live at **module
scope** — Triton resolves `tl` against the kernel's module globals, so a
closure-local `tl` raises `NameError('tl is not defined')` at compile time.

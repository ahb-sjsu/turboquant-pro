# Hopper fp8 compute — measured, and the verdict is a boundary

**Run:** user's lightning.ai trial, NVIDIA H100 80GB HBM3, torch 2.x,
`hopper_fp8_compute_bench.py` (H=32, Q=64, D=128; DC-offset-shaped keys in
`FP8NativeKV` containers; error vs fp64 reference).

| S | fp16 | fp8 storage+upcast | fp8 `_scaled_mm` (per-head loop) |
|---:|---|---|---|
| 2k | 0.026 ms · 41 TF/s | 0.086 ms · 12 TF/s | 0.646 ms · 1.7 TF/s |
| 8k | 0.040 ms · 107 TF/s | 0.325 ms · 13 TF/s | 0.645 ms · 6.7 TF/s |
| 32k | 0.141 ms · 122 TF/s | 1.248 ms · 14 TF/s | 0.649 ms · 26 TF/s |

**Findings, honestly framed:**

1. **Naive fp8 compute loses to fp16 at every size.** `torch._scaled_mm`
   has no batched form, so per-head Python-loop launches cost a flat
   ~0.65 ms — pure launch overhead below 32k (1.7 TF/s at 2k), and still
   4.6× behind fp16's tensor cores at 32k. The fp8 *compute* win on Hopper
   requires FA3-class fused attention or grouped GEMMs, not `_scaled_mm`
   loops. That is the measured boundary; the storage passthrough
   (validated Ada + Hopper) remains the win available from our layer.
2. **The upcast path is a memory play, not a speed play** — materializing
   fp16 keys per call costs 3–9× fp16 matmul latency; in a real decode the
   upcast fuses into the attention kernel. Halved KV bytes is the claim;
   these numbers say don't claim speed for it.
3. **fp8 max|err| ~4–7 (vs fp16's 0.06)** on scores of O(100) is e4m3 key
   precision showing through — the storage format's ~1–2 % score error, as
   expected, orthogonal to the compute question. (And consistent with the
   keys-comparison finding: fp8 keys were near-lossless on attention
   *distributions*.)

House lesson repeated: the trial hour that refutes a hoped-for win is as
valuable as one that confirms it — nobody at this layer should burn time
on `_scaled_mm` loops for decode attention.

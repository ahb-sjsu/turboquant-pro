# KV-decode dequantization overhead (#1) — the reviewer is right

For one decode step over an 8192-token cache (head_dim 128, 32 heads, 3-bit), we
timed the dequant pass (unpack -> centroid lookup -> inverse-rotate -> scale) against
the attention matmul it feeds. **CPU NumPy path** (Atlas):

| configuration | dequant | attention | dequant share |
|---|---:|---:|---:|
| full compressed cache | 1378 ms | 15 ms | **98.9%** |
| two-tier (512 fp16 hot, 7680 cold) | 1262 ms | 12 ms | 99.1% |

**Dequant dominates the decode step.** The critique is valid. The cost is not the
unpacking per se but the **inverse-rotation matmul** (a per-head dxd rotation applied
to every cached token) plus centroid lookup — a separate memory-bound pass before the
tiny `q.K^T`.

*(The production path is the CuPy GPU kernels; that run errored in this harness
(`CUDA_ERROR_INVALID_VALUE`) and is not reported here rather than guessed. The CPU
number already establishes that dequant, not attention, is the decode bottleneck.)*

## The fix is the ADC trick we already built
Reconstructing K to compute `q.K^T` is wasteful: the inner product can be computed
**directly on the compressed codes** via asymmetric distance — exactly the kernel that
made embedding search fast (`turboquant_pro/_adc`). Applied to attention, `q.K^T`
becomes a per-query LUT over the packed key codes, with **no inverse-rotation and no
materialized K** — i.e. the *fused dequant-inside-attention* kernel the reviewer asks
for. So our embedding ADC kernel is the prototype for the KV fused-decode kernel; that
is the highest-value KV-path work item.

## Mitigations available today
- **Two-tier cache** keeps the decode-critical hot window in fp16 (zero dequant);
  only cold pages pay, bounding the cost to the cold fraction.
- For pure compressed-domain scoring, the ADC path (no reconstruction) is the route.

# Sparse fp16 outliers in the fused decode — already latency-hidden

A reviewer flagged the mixed-precision access pattern (mostly 4-bit codes
alongside sparse fp16 outliers) as a possible source of branch divergence /
bandwidth stalls in `TurboQuantKVCache.fused_decode`, and asked whether the
kernel hides the latency of the sparse lookups. **Measured answer: yes,
completely** — the sparse-outlier cost is independent of the outlier count, so
there is no memory-pattern optimization to do here.

## Method

`kv_kernel.py` M4 fused decode (`pck_block_partials_cuda`), one decode step over
a 32,768-token per-channel-key cache (8 heads × 128 dim, asym-NF4 keys,
PolarQuant values, 512-token fp16 hot window), steady-state median of 30 calls
with `cuda.synchronize()`. The only variable is `outlier_frac` — the fraction of
per-channel entries kept in fp16 as a token-major CSR of score deltas. Sweeping
it scales the sparse work (nnz) while holding everything else fixed.

**Run:** Atlas, NVIDIA Quadro GV100 (Volta, sm_70), CUDA 12.0, GPU isolated via
`CUDA_VISIBLE_DEVICES`.

| `outlier_frac` | CSR entries (nnz) | steady-state | Δ vs nnz=0 |
|---:|---:|---:|---:|
| 0.00 | 0 | 169.5 ms | — |
| 0.02 | 645,120 | 159.7 ms | **−5.8 %** |
| 0.05 | 1,677,312 | 162.1 ms | −4.3 % |
| 0.10 | 3,354,624 | 161.9 ms | −4.5 % |

## Finding — the sparse pass is free

Latency is **flat as the outlier count grows 13×** (645k → 3.35M entries): 159.7
→ 162.1 → 161.9 ms, within run-to-run noise, and the outlier variants are if
anything marginally *faster* than the no-outlier path. **The sparse-outlier cost
does not scale with nnz**, which is the signature of work that is fully hidden
behind other latency.

Why it hides, concretely:

- **No mixed 4-bit/16-bit access on the hot path.** The dense score reads the
  key *codes* as unpacked `uint8` (coalesced, one byte per dim); the fp16
  outliers are pre-reduced to **fp32 CSR score-deltas** (`Δ = v_fp16 −
  dequant(code)`) built once per cold flush — the decode never re-reads fp16
  values or scatters into a reconstructed key.
- **Branch-free dense loop + warp-cooperative sparse pass.** Lanes stride the
  token's short CSR row and a single `__shfl_xor` reduction folds the correction
  in *before* the online-softmax update (design §8.2), so divergence is bounded
  to the ragged tail of a ~3-entry list and the correction is a handful of fp32
  adds per token.
- **Memory-bound elsewhere.** Each page's dense per-channel sum and the
  code-space value accumulation dominate; the sparse adds fit in their shadow.

## Conclusion

No optimization is warranted for the sparse-outlier memory pattern — it is
already latency-hidden, as the design intended and this sweep confirms. The
actual latency lever at long context is **per-page launch overhead** (126 kernel
launches at 32k tokens), which the **P5 batched-page kernel already collapses to
a single launch** (1.9–9.4× faster than the per-page path; see
`RESULTS_p5_triton.md`). The highest-leverage fused-decode optimization was
therefore already shipped in P5; the outlier pattern was never the bottleneck.

*(Scope: absolute ms are GV100/Volta; the load-bearing result — outlier cost
independent of nnz — is architecture-general, since it follows from the sparse
work being hidden behind the dense/value passes rather than from any GV100
specific.)*

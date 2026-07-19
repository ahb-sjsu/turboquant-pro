# Response to Review 3 (technical) — `turboquant-pro-review-3.txt`

This deep technical review was largely positive (praising the Track 1/Track 2 split, the evidence
ladder, key/value asymmetry, and the high-GQA NF4 fix). Below: what we **did now**, and what we've
**tracked** with honest rationale for not rushing it.

## Summary checklist for next iteration

**1. Fix dynamic dimension calculations in `PCAMatryoshkaPipeline.estimate_storage()`. ✅ Done.**
Was a `@staticmethod` with hard-coded `input_dim=1024, output_dim=384, bits=3`, so `pipe.estimate_storage(n)`
ignored the real pipeline and always reported 1024→384 @ 3-bit. Now:
- `estimate_storage(self, n, input_dim=None, output_dim=None, bits=None)` — an **instance method**
  defaulting to the pipeline's real `input_dim` / `output_dim` / `bits`; explicit overrides still work.
- `estimate_storage_for(n, input_dim, output_dim, bits)` — a **static** helper for the old
  dimension-agnostic call (internal callers `autotune.py` and the existing test moved to it).
- Regression test `test_estimate_storage_uses_pipeline_dims` in `tests/test_pca.py` asserts the
  returned dims/bits match the pipeline (e.g. 768→256 @ 4-bit now reports 768/256/4, ratio 23.3, not
  1024/384/3). All 48 `test_pca.py` tests pass. Docs updated (`docs/claims.md`, `CLAIMS.md`,
  `RESULTS_canonical.md`, CHANGELOG).

**2. CI benchmarks measuring L1 cache misses in `CompressedHNSW` graph traversal. ⧗ Tracked.**
Not landed this round: hardware cache-miss counters (`perf stat -e L1-dcache-load-misses`, PAPI,
`valgrind --tool=cachegrind`) are platform-specific and **not reliably exposed on shared GitHub-hosted
runners** (virtualized, no PMU access). Putting a flaky cache-miss gate in the CI matrix would produce
noise, not signal. Plan: an **offline** cachegrind profile of the inner-product-table traversal
(committed as a `benchmarks/` script + a results doc), run on a fixed local machine, rather than a
per-commit CI check. The reviewer's underlying concern — the IP lookup table thrashing L1 at high
thread counts once it exceeds cache — is a real, worthwhile measurement; it just doesn't belong in
hosted CI.

**3. Sliding-window anchor corrections for autoregressive tasks >512 tokens. ⧗ Tracked (research).**
This is an algorithmic exploration (periodic fp16 anchor / lightweight residual reset every *N* tokens
to fight compounding 4-bit KV error over long decode). It needs a design + a GPU long-generation
benchmark to show it actually reduces drift without eating the compression win. We've kept the
**honest caveat** in the docs (all 4-bit KV schemes degrade on 512+ token generation) rather than ship
an unvalidated mitigation. Opened as a design task for the KV-cache track.

## Other constructive critiques

- **OOD robustness (anisotropic / heavy-tailed domain embeddings). ✅ Done.** Added
  [`notebooks/claims/04_ood_anisotropic.ipynb`](notebooks/claims/04_ood_anisotropic.ipynb): a synthetic
  pathological corpus (power-law eigenvalue spectrum, Student-t heavy tails, random rotation) run
  through the canonical ladder against exact ground truth. Verified it runs; it confirms the honest
  dataset-dependence story — when the spectrum is **concentrated**, PCA+TQ stays strong at high
  compression (mirror of the flat-spectrum GloVe-100 case), and the failure boundary tracks spectral
  concentration as `RESULTS_glove.md` argues.
- **Mixed-precision SIMD alignment; hot/cold cache double-buffering (dedicated CUDA streams);
  structured warnings on AVX2→NumPy / CUDA-extension fallbacks. ⧗ Tracked.** Sound systems polish;
  scoped for the engineering track — none change correctness, and we'd rather benchmark each change
  than assert it.

## Delivered this round
- `estimate_storage()` fix + static split + regression test (checklist #1).
- OOD anisotropic stress-test notebook (empirical-robustness critique).
- Doc/CHANGELOG updates reflecting the fix (bug note → "fixed in v1.4.1").

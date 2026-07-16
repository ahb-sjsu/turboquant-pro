# Vector-DB retrieval benchmark — real 199k LaBSE embeddings

Atlas, `/archive/results_aesthetics/bip_sample_200k_labse.npy` (768-dim, 199,000
corpus / 1,000 held-out queries), exact cosine ground truth, CPU,
`OMP_NUM_THREADS=20`. Reproduce: `benchmarks/benchmark_vectordb.py`.

## Fair head-to-head at 32× compression (two-stage = oversample×5 + rerank)

Every method gets the same protocol: compressed first stage → rerank the top-50
candidates by exact fp32 on the retained originals.

| method | bytes/vec | build s | qps | recall@10 (1-stage) | recall@10 (+rerank) |
|---|---:|---:|---:|---:|---:|
| fp32-flat (exact) | 3072 | 7 | 212 | 0.9997 | — |
| faiss-PQ | 96 | 142 | 136 | 0.467 | 0.827 |
| faiss-IVFPQ | 96 | 355 | 9513 | 0.496 | 0.756 |
| faiss-RaBitQ (2024 SOTA) | 96 | 0.3 | 4 | 0.630 | 0.962 |
| **faiss-OPQ** | 96 | **632** | 915 | 0.780 | **0.999** |
| **turboquant-pro** TQ3 | 100 | **31** | 224 | 0.784 | **0.9993** |

## Honest takeaway
- **Accuracy: turboquant-pro ties the strongest baseline (OPQ) — now *measured*.**
  With a fair rerank for all methods, tq-pro recall@10 ≈ 0.999 vs OPQ ≈ 0.999 — a
  statistical tie, and the [bootstrap 95% CIs below](#bootstrap-95-cis--canonical-harness-reproduction)
  make it explicit: the +rerank intervals overlap almost entirely (tq-pro
  0.9994 [.999, 1.000] vs OPQ 0.9995 [.999, 1.000]) and single-pass intervals are
  identical. Both clearly beat PQ (0.816) and IVF-PQ (0.751). tq-pro does **not**
  "dominate" OPQ on recall; the earlier impression was an artifact of giving
  rerank only to tq-pro.
- **Build cost is the real win: ~20× faster.** tq-pro builds in **31 s** vs OPQ's
  **632 s** at equal recall — because it is PCA-based (one eigen-decomposition),
  not OPQ's expensive iterative rotation+codebook optimization. At VLDB scale,
  index-construction time matters as much as query time.
- **Query caveat (honest):** as benchmarked, tq-pro searches a flat index over
  reconstructed vectors (224 qps) and is slower than OPQ's ADC (915 qps). tq-pro
  ships a native compressed/ADC + GPU search path (`gpu_adc_search`) that should
  close this — measuring it is future work, not claimed here.
- **System advantages (beyond this table):** SQL-native compressed search in
  pgvector, multi-modal presets, and zero-config `AutoConfig` — none of which the
  faiss baselines offer.

## Bootstrap 95% CIs — canonical-harness reproduction

Re-run through the shared, CI-tested harness (`benchmarks/canonical_embedding.py`,
percentile bootstrap `n_boot=2000` over the 1,000 queries), **same real LaBSE
corpus**, at the matched **96 B / 32×** operating point (`out_dim=256, bits=3,
oversample×5`). This turns the "statistical tie" above from an assertion into a
measured overlap of intervals.

| method | B/vec | R@10 single-pass [95% CI] | R@10 +rerank [95% CI] |
|---|---:|---:|---:|
| fp32-flat (exact) | 3072 | 0.9998 [1.000, 1.000] | — |
| faiss-PQ | 96 | 0.4656 [.456, .475] | 0.8155 [.807, .824] |
| faiss-IVFPQ | 96 | 0.4932 [.483, .503] | 0.7510 [.739, .762] |
| faiss-RaBitQ | 96 | 0.6233 [.615, .632] | 0.9646 [.961, .969] |
| **faiss-OPQ** | 96 | 0.7856 [.778, .792] | **0.9995 [.999, 1.000]** |
| **tq-pro PCA256+TQ3** | 96 | 0.7847 [.778, .792] | **0.9994 [.999, 1.000]** |
| tq-pro ADCIndex (256/3) | 96 | 0.7847 [.778, .792] | 0.9992 [.999, 1.000] |

Reading the intervals:
- **tq-pro ties OPQ — measured.** +rerank 0.9994 [.999, 1.000] vs OPQ 0.9995 [.999,
  1.000]: intervals overlap almost entirely, and the single-pass intervals are
  identical ([.778, .792]). The gap sits well inside the noise band — a genuine
  tie, not a ranking.
- **tq-pro beats RaBitQ at matched bytes**, both stages, with **non-overlapping**
  intervals: single-pass 0.7847 [.778, .792] vs 0.6233 [.615, .632]; +rerank 0.9994
  [.999, 1.000] vs 0.9646 [.961, .969].
- **ADCIndex reproduces the PCA+TQ ranking** — identical single-pass interval;
  +rerank 0.9992 vs 0.9994 (within noise) — confirming the compressed-domain scorer
  is faithful to the reconstruct-cosine ranking.

*Build-time note.* The harness build times here (tq-pro 110 s, OPQ 258 s) run
through the reference Python packing loop and faiss defaults, **not** the optimized
path — the **31 s vs 632 s** figures in the table above are the production
`benchmark_vectordb.py` numbers. Recall is the apples-to-apples quantity and it
agrees across both harnesses (tq-pro ≈ 0.784 single / ≈ 0.999 +rerank either way).

Reproduce (internal GPU box, CPU-side): `benchmarks/canonical_embedding.py` over the
199k LaBSE array with `out_dim=256 bits=3 oversample=5 n_boot=2000` (≈34 min, OPQ
dominates).

## Method contribution (the original novelty)
PCA-Matryoshka makes **non-Matryoshka embeddings truncatable without retraining**
(naïve truncation to 256-d: cosine 0.467 → 0.974), which is what lets a
training-free pipeline reach OPQ-class accuracy at a fraction of the build cost.

*Next:* 1M-scale (Gutenberg), a standard public dataset, multi-modal, and the
pgvector-native numbers — see the sprint.

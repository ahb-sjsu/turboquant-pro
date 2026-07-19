# Embedding-retrieval benchmarks

Compressed-domain retrieval numbers for turboquant-pro, compared like-for-like against PQ, IVF-PQ, RaBitQ, and OPQ across LaBSE and BGE-M3 corpora.

Part of [TurboQuant Pro](../../README.md).

### Retrieval (embeddings)

> **Reading these numbers — avoid the classic apples-to-oranges.** Quantized-retrieval recall has **two** operating points: **single-pass** (scan the compressed codes once) and **+rerank** (re-score the top candidates with exact vectors). Compare like-for-like on **both** axes, and match on **bytes/vector** — *not* bits or nominal dim, since the PCA front-end changes the dimension. The frequent misread is to place a *single-pass* recall next to RaBitQ's headline **90–99%, which is itself a *with-rerank* number**. Matched fairly: at equal bytes, *single-pass*, turboquant-pro leads (e.g. **+24 pts @ ~100 B/vec** on LaBSE); *with rerank*, both saturate near 1.0 (tq-pro's own rerank goes 0.819 → 0.9997 at 5×). The estimator-isolated head-to-head — official `rabitqlib` + ScaNN, RaBitQ run exhaustively so recall reflects the *estimator*, not the index — is [`RESULTS_rabitq_comparison.md`](../../benchmarks/RESULTS_rabitq_comparison.md).

At **32× compression**, recall@10 on real LaBSE / multilingual-Gutenberg embeddings ([`RESULTS_labse_199k.md`](../../benchmarks/RESULTS_labse_199k.md), [`RESULTS_gutenberg_1m.md`](../../benchmarks/RESULTS_gutenberg_1m.md)) — all methods reranked identically:

| method | recall@10 (single) | recall@10 (+rerank) | index build |
|---|---:|---:|---:|
| PQ | 0.467 | 0.827 | 142 s |
| IVF-PQ | 0.496 | 0.756 | 355 s |
| RaBitQ (2024 SOTA) | 0.630 | 0.962 | 0.3 s |
| OPQ | 0.780 | 0.999 | 632 s |
| **turboquant-pro** | **0.784** | **0.9992** | **31 s** |

Beats the 2024 binary-quant SOTA (RaBitQ) at both operating points and ties OPQ at **4–20× lower index-build cost** — and this holds at 1M scale (0.989 +rerank, tying OPQ).

**15-method comparison on BGE-M3 (1024-dim, 2.4M vectors)** — *single-pass, no rerank* (the reranked frontier is the next table):

| Method | Compression | Recall@10 (single-pass) | Cosine Sim |
|--------|----------:|----------:|----------:|
| Scalar int8 | 4× | 97.2% | 0.9999 |
| TurboQuant 4-bit | 7.9× | 90.4% | 0.995 |
| TurboQuant 3-bit | 10.6× | 83.8% | 0.978 |
| **PCA-384 + TQ3** | **27.7×** | **76.4%** | **0.979** |
| **PCA-256 + TQ3** | **41.0×** | **78.2%** | **0.963** |
| Binary quantization | 32.0× | 66.6% | 0.758 |
| PCA-128 + TQ2 | 113.8× | 78.7% | 0.924 |
| PQ M=16 K=256 | 256.0× | 41.4% | 0.810 |

Note PCA-256+TQ3 has *lower* cosine (0.963) but *higher* recall@10 (78.2%) than PCA-384+TQ3 — cosine measures per-vector fidelity, recall measures ranking; they diverge at high compression.

**With 5× oversampling + exact reranking (standard production practice), on 50K BGE-M3:**

| Method | Compression | No rerank | Fetch 2× | Fetch 5× | Fetch 10× |
|--------|------------|-----------|----------|----------|-----------|
| Scalar int8 | 4× | 99.0% | 100% | **100%** | 100% |
| TQ3 uniform | 10.5× | 83.4% | 98.2% | **100%** | 100% |
| **PCA-384 + TQ3** | **27.7×** | 79.2% | 96.8% | **99.8%** | 100% |
| PCA-256 + TQ3 | 41× | 75.4% | 91.6% | **98.6%** | 100% |
| Binary | 32× | 54.4% | 69.6% | 85.6% | 93.6% |
| PQ (M=16) | 256× | 38.4% | 53.2% | 73.6% | 84.6% |

**Production deployment (PCA-384 + TQ3, BGE-M3, 3.3M vectors — 27.7× regardless of content):**

| Corpus | Vectors | Original | Compressed |
|--------|--------:|---------:|-----------:|
| Ethics (37 langs) | 2.4M | 9.4 GB | 338 MB |
| Publications | 824K | 3.2 GB | 116 MB |
| Code repos | 112K | 437 MB | 16 MB |
| **Total** | **3.3M** | **13 GB** | **470 MB** |

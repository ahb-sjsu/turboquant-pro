# [P] [R] turboquant-pro autotune: One command finds the optimal compression for your vector database

We just shipped an autotune CLI for [turboquant-pro](https://github.com/ahb-sjsu/turboquant-pro) — it connects to your PostgreSQL database, samples embeddings, sweeps 12 compression configurations, and tells you exactly which one to use.

```bash
turboquant-pro autotune \
  --source "dbname=mydb user=me" \
  --table chunks --column embedding \
  --min-recall 0.95
```

**10 seconds later:**

```
Evaluating 12 configurations...
              Config   Ratio   Cosine   Recall   Var%   Time
--------------------------------------------------------------
       PCA-128 + TQ2  113.8x   0.9237   78.7%  79.9%   2.2s
       PCA-128 + TQ3   78.8x   0.9390   84.0%  79.9%   0.6s
       PCA-256 + TQ3   41.0x   0.9700   92.0%  92.3%   0.7s
       PCA-384 + TQ3   27.7x   0.9823   93.7%  97.3%   0.7s
       PCA-384 + TQ4   20.9x   0.9906   96.0%  97.3%   0.6s
       PCA-512 + TQ4   15.8x   0.9949   96.3%  99.0%   0.6s

Recommendation (min recall >= 95%):
  PCA-384 + TQ4: 20.9x compression
  Cosine similarity: 0.9906
  Recall@10: 96.0%
  Storage: 758 MB -> 36 MB (saves 722 MB)
```

## What it does

The problem: you have a pgvector database with millions of embeddings. You want to compress them. But which configuration? PCA to how many dims? 2-bit, 3-bit, or 4-bit quantization? The quality/compression tradeoff depends on *your specific data*.

Autotune answers this in ~10 seconds:

1. Samples N embeddings from your table (default 5K, we used 2K here)
2. Tries all 12 combinations of PCA dims (128, 256, 384, 512) x bit widths (2, 3, 4)
3. Measures cosine similarity preservation and recall@10 for each
4. Identifies the Pareto-optimal frontier
5. Recommends the highest compression that meets your recall threshold
6. Prints copy-paste code

## Real results on 194K production embeddings

We ran this on a production RAG system (194K BGE-M3 1024-dim embeddings from 33 code repos):

**If you need 95%+ recall:** PCA-384 + TQ4 gives you 20.9x compression at 0.991 cosine and 96.0% recall. Your 758 MB shrinks to 36 MB.

**If you can tolerate 84% recall:** PCA-128 + TQ3 gives you 78.8x compression at 0.939 cosine. 758 MB becomes 10 MB. That's your entire corpus fitting in L3 cache.

**If you want maximum compression:** PCA-128 + TQ2 hits 113.8x — nearly two orders of magnitude. Cosine still at 0.924.

The whole sweep took 10.8 seconds on CPU. No GPU needed.

## How it works under the hood

[PCA-Matryoshka](https://github.com/ahb-sjsu/turboquant-pro) (our IEEE TAI submission) is a training-free technique: fit PCA once on a sample, rotate all vectors so truncation works, then quantize to 2-4 bits via TurboQuant scalar quantization. The "autotune" part is just sweeping the configuration space and measuring quality on your actual data.

The key insight from our 15-method benchmark: PCA-Matryoshka + TurboQuant strictly dominates both binary quantization and product quantization across the practical compression range. Autotune finds exactly where on that frontier your data lives.

## Install & use

```bash
pip install turboquant-pro[pgvector]

turboquant-pro autotune \
  --source "dbname=mydb user=me" \
  --table my_embeddings \
  --column embedding_vector \
  --min-recall 0.95 \
  --output results.json
```

Options: `--sample-size`, `--pca-dims`, `--bits`, `--queries`, `--top-k`. Full results saved as JSON for programmatic use.

```python
# After autotune recommends PCA-384 + TQ4:
from turboquant_pro import PCAMatryoshka, PCAMatryoshkaPipeline, TurboQuantPGVector

pca = PCAMatryoshka(input_dim=1024, output_dim=384)
pca.fit(sample_embeddings)
tq = TurboQuantPGVector(dim=384, bits=4)
pipeline = PCAMatryoshkaPipeline(pca, tq)

compressed = pipeline.compress(embedding)  # 4096 bytes -> 195 bytes
```

---

**GitHub:** https://github.com/ahb-sjsu/turboquant-pro
**PyPI:** `pip install turboquant-pro`
**Paper:** IEEE TAI submission (PCA-Matryoshka: 15-method compression benchmark)

Happy to answer questions about the benchmark methodology or help debug autotune results on your corpus.

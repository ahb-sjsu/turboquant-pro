# User guide — compress embeddings safely in 15 minutes

This is the shortest path from a corpus of embeddings to a compressed index you can
**trust** — with the acceptance check built in. Nothing here needs a GPU.

```bash
pip install turboquant-pro
```

> This workflow and the **`tqp` CLI** used in the [certification](certification.md) and [claim-replay](claim_replay.md) guides both ship in **1.8.0**, the current PyPI release — `pip install turboquant-pro` (add `[torch]` for the tracer).

## 1. Compress and search (the library)

`PCAMatryoshka` reduces dimension by reordered PCA; `.with_quantizer(bits=...)` adds
the low-bit scalar quantizer; `ADCIndex` searches the compressed codes directly and
reranks the top candidates exactly.

```python
import numpy as np
from turboquant_pro import PCAMatryoshka, ADCIndex, certificate_from_embeddings

corpus  = np.load("corpus.npy")        # (N, D) float32
queries = np.load("queries.npy")       # (Q, D) float32

pca  = PCAMatryoshka(input_dim=corpus.shape[1], output_dim=corpus.shape[1]).fit(corpus)
pipe = pca.with_quantizer(bits=3)      # ~10x smaller
index = ADCIndex(pipe).add(corpus)

ids, scores = index.search(queries, k=10)                       # fast, compressed
ids = index.search(queries, k=10, rerank=10, originals=corpus)  # exact rerank -> high recall
```

**Acceptance, not vibes.** Before you ship, certify that the compression preserves
the *ranking* your search depends on:

```python
recon = pipe.decompress_batch(pipe.compress_batch(corpus))
cert  = certificate_from_embeddings(corpus, recon, metric="cosine")
assert not cert.vacuous, "compression too lossy to guarantee ranking"
print(f"Kendall-tau floor >= {cert.tau_floor:.3f}")   # a *guaranteed* rank floor
```

If the certificate is vacuous, use more bits or require exact rerank — see the
[certification guide](certification.md). Do **not** accept on reconstruction cosine;
[here is why](operator_aware_quantization.md).

## 2. The one-command path (the CLI)

`tqp` wraps the same pipeline. Plan a recipe, build a persisted index, certify it:

```bash
tqp plan embeddings --embeddings corpus.npy --target "recall@10 >= 0.90"
tqp index create --embeddings corpus.npy --out corpus.tqe --bits 3
tqp index search corpus.tqe --queries queries.npy --k 10 --rerank 10
tqp index certify corpus.tqe --min-tau 0.5      # exit 1 if the rank floor is not met
```

`tqp plan` reports the highest-compression recipe on the Pareto frontier that meets
a **measured** `recall@k` target (never a reconstruction target). See
[CLI.md](../CLI.md) for every flag.

## 3. Let it adapt

For a production index, hand `search` a runtime policy: it stays single-pass where
the ranking is unambiguous and escalates to exact rerank only where the top-k
boundary is tied — cheap where it can be, careful where it must be.

```python
from turboquant_pro import TQEIndex, TQPRuntimePolicy
idx = TQEIndex.open("corpus.tqe")
ids, scores = idx.search(queries, k=10, policy=TQPRuntimePolicy())
```

## What to read next

- Keep the index fresh as the corpus grows/changes → [production lifecycle](production_lifecycle.md).
- Understand what the certificate guarantees → [certification](certification.md).
- Reproduce the headline GloVe recall number → [claim replay](claim_replay.md).
- Compressing KV caches or model weights instead of embeddings →
  [operator-aware quantization](operator_aware_quantization.md).

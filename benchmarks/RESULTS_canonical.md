# Canonical embedding benchmark — protocol & one-command reproduction

This is the single, apples-to-apples comparison behind the headline embedding claim
(*beats RaBitQ on recall, ties OPQ at scale, builds faster*). One harness, all methods,
identical rerank protocol, public data with provided ground-truth.

## One command (Colab)

Open **[`notebooks/claims/00_canonical_sota_embedding.ipynb`](../notebooks/claims/00_canonical_sota_embedding.ipynb)**
→ *Run all*. It `pip install`s, downloads a public ann-benchmarks dataset, runs the ladder,
and renders the table. No local setup, no private data.

## One command (CLI / local hardware)

```bash
pip install turboquant-pro faiss-cpu h5py
python - <<'PY'
import numpy as np, h5py, urllib.request, os
import benchmarks.canonical_embedding as ce   # or add benchmarks/ to sys.path
url="http://ann-benchmarks.com/glove-100-angular.hdf5"; fn=url.split("/")[-1]
if not os.path.exists(fn): urllib.request.urlretrieve(url, fn)
with h5py.File(fn) as f:
    C=ce.normalize(np.asarray(f["train"],dtype="float32"))
    Q=ce.normalize(np.asarray(f["test"][:2000],dtype="float32"))
    gt=np.asarray(f["neighbors"][:2000,:100],dtype="int64")
rows=ce.run_canonical(C,Q,gt,out_dim=100,bits=3,oversample=5,threads=8)
print(ce.to_markdown(rows))
PY
```

`benchmarks/canonical_embedding.py` is imported by both the notebook and CI, so the numbers
you get are the numbers the tests exercise.

## Protocol (identical across every method)

- **Data:** public ann-benchmarks HDF5 (`glove-100-angular`, `nytimes-256-angular`,
  `deep-image-96-angular`), scored against the dataset's **provided** top-100 neighbours.
  (Subsetting the corpus recomputes exact ground truth so ids stay valid.)
- **Methods:** `fp32-flat`, `faiss-PQ`, `faiss-OPQ`, `faiss-IVFPQ`, `faiss-RaBitQ`,
  `PCA-only` (truncation, fp32), `TQ-only` (scalar-quant, full dim), `PCA+TQ`, `ADCIndex`.
- **Rerank:** every ANN method is measured single-stage **and** +rerank with the **same**
  `oversample` (candidates = `10 × oversample`), reranked by exact fp32 cosine on the
  retained originals — the standard two-stage ANN protocol.
- **Storage:** `bytes/vector` is **analytic** (`out_dim × bits ÷ 8`), not from
  `estimate_storage()` (which was dimension-agnostic before v1.4.1; see `docs/claims.md`).
- **Byte-budget matching:** PQ/OPQ/IVFPQ use `m = out_dim × bits ÷ 8` subquantizers (largest
  divisor of `dim`) so they sit at the same budget as `PCA+TQ`.

## The actual public numbers

Reference results on public datasets (GloVe-100, NYTimes-256, deep-image-96) are recorded in
**[`RESULTS_glove.md`](RESULTS_glove.md)**, including the honest dataset-dependence finding:
PCA *truncation* helps only for high-dim/concentrated-spectrum embeddings; at full dimension /
matched bytes the TurboQuant scalar quantizer wins on GloVe and vision and ties on NYTimes.
Notebook 00 reproduces that full picture (run it; don't take our word).

> Absolute QPS / build-time numbers are hardware-dependent (report your machine). The
> reproducible quantities are **recall at a given compression** and the **relative** build-time
> ranking, both of which the harness measures identically for every method.

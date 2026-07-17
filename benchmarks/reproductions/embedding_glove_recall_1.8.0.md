# Independent reproduction — `embedding_glove_recall` (v1.8.0, from PyPI)

The canonical "trust this project" claim, reproduced from the **published 1.8.0
package on PyPI** — not the development tree — in a clean, isolated virtualenv.
The library under test is the shipped wheel (`site-packages`), installed with
`pip`, so this exercises the artifact a user actually gets, not an editable
`-e .` checkout.

## Method

```bash
python -m venv venv && . venv/Scripts/activate
pip install "turboquant-pro[yaml]==1.8.0"                 # from PyPI, not editable
git clone --depth 1 --branch v1.8.0 \
    https://github.com/ahb-sjsu/turboquant-pro
cd turboquant-pro
tqp replay embedding_glove_recall                          # gated replay
```

`tqp replay` executes the claim's command (`python benchmarks/canonical_glove.py
--small`) and compares the normalized `results.json` against the floors declared
in `claims.yaml` (`recall_at_10_rerank_min: 0.95`, `compression_ratio_min: 9.5`),
exiting non-zero on any regression.

## Environment

| | |
|---|---|
| `turboquant-pro` | **1.8.0** (PyPI wheel, `site-packages` — not editable) |
| Python | 3.12.2 |
| numpy | 2.5.1 |
| PyYAML | 6.0.3 |
| Platform | Windows x86-64, CPU only |
| Date | 2026-07-17 |

## Result — reproduced

```
# tqp replay  claims=1  reproduced=1  regressed=0  error=0
  [reproduced] embedding_glove_recall  (35.5 s, CPU)
```

From [`embedding_glove_recall_1.8.0.results.json`](embedding_glove_recall_1.8.0.results.json)
(sha256 `28847b4186091a95a8e49c6248fa4db1f0d703caa2f8020f7629fedd3b22e2ff`):

| metric | value | floor | pass |
|---|---:|---:|:--:|
| recall@10 (+rerank ×12) | **1.000** | ≥ 0.95 | ✅ |
| compression ratio | **9.639×** | ≥ 9.5 | ✅ |
| recall@10 (single-pass) | 0.788 | — | *(diagnostic)* |
| mean cosine | 0.981 | — | *(diagnostic)* |

The acceptance metric is **reranked recall** — the metric retrieval actually
consumes. The single-pass recall (0.788) and reconstruction cosine (0.981) are
labelled diagnostics: cosine reads ~0.98 while single-pass recall is far lower,
exactly the divergence this project is built to expose. The floors are met, so
the published package reproduces the headline retrieval claim on real public
GloVe data on a stock CPU in about half a minute.

*(Recipe: PCA-100 + 3-bit TurboQuant + compressed-domain ADC search, 12× oversample
+ exact rerank, on the hermetic bundled subset — n=2000, 100 queries. The full
1.18M run is `tqp replay embedding_glove_recall` with `full_command`.)*

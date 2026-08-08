# TurboQuant Pro

[![PyPI version](https://img.shields.io/pypi/v/turboquant-pro.svg)](https://pypi.org/project/turboquant-pro/)
[![PyPI Downloads](https://static.pepy.tech/badge/turboquant-pro)](https://pepy.tech/project/turboquant-pro)
[![Python versions](https://img.shields.io/pypi/pyversions/turboquant-pro.svg)](https://pypi.org/project/turboquant-pro/)
[![Tests](https://img.shields.io/github/actions/workflow/status/ahb-sjsu/turboquant-pro/ci.yml?label=tests)](https://github.com/ahb-sjsu/turboquant-pro/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20660087.svg)](https://doi.org/10.5281/zenodo.20660087)

**Consumer-aware compression for embedding indexes and LLM KV caches.** TurboQuant Pro compresses each vector by the metric its downstream consumer actually uses — retrieval recall for indexes, attention/generation quality for KV caches — **not reconstruction cosine alone**, which is repeatedly shown here to be blind, or even anti-correlated, with quality.

```bash
pip install turboquant-pro
tqp replay embedding_glove_recall --small   # reproduce the headline retrieval claim — CI-gated, runs in seconds
```

- **Embedding retrieval:** **32× compression at recall@10 0.9992** (after matched reranking, on real LaBSE/Gutenberg data) — outperforms RaBitQ and ties OPQ under a matched-byte public benchmark protocol, at **4–20× lower index-build cost**.
- **KV caches:** architecture-aware **key** quantization avoids a failure that is invisible to reconstruction metrics — PolarQuant keys read 0.995 cosine yet blow perplexity to ≈10⁴; per-channel keys keep it near fp16.
- **At scale & in production:** compressed-domain search, persisted / **larger-than-RAM** sharded + memory-mapped indexes, distribution-free rank **certificates**, one-command **replay**, and drift **monitoring**.

> Every headline number — with its reproduction status, dataset, one-click notebook, and hardware — is a row in **[`CLAIMS.md`](CLAIMS.md)**. The acceptance signal everywhere is rank fidelity / a certificate / the consumer's metric — **never reconstruction cosine.**

The current pre-release is **2.0.0a2**, which adds production semantics to the vLLM KV connector, the Postgres track, and self-metering anatomy instruments. The last stable line is **1.9.x** (larger-than-RAM search + index format v3), and the `tqp` CLI and certification platform shipped in 1.8.0. APIs under `connectors/` may still move before 2.0.0. Full notes: [`CHANGELOG.md`](CHANGELOG.md).

## Installation

```bash
pip install turboquant-pro          # core (numpy only) + the `tqp` CLI
pip install turboquant-pro[torch]   # + operator tracer (`tqp trace`)
pip install turboquant-pro[fast]    # + AVX2 ADC kernel (pybind11)
pip install turboquant-pro[gpu]     # + CuPy CUDA 12.x
pip install turboquant-pro[all]     # everything (pgvector, FAISS, NATS, …)
```

## 30-second embedding compression

The central, best-validated contribution — compress a corpus and search the codes directly:

```python
from turboquant_pro import PCAMatryoshka, ADCIndex

pca = PCAMatryoshka(input_dim=768, output_dim=256).fit(train_vectors)
pipeline = pca.with_quantizer(bits=3)                 # PCA rotate/truncate + 3-bit TurboQuant
index = ADCIndex(pipeline).add(corpus)                # compressed-domain index (~63 B/vec)

ids, scores = index.search(queries, k=10)                          # single-pass, fast
ids, scores = index.search(queries, k=10, rerank=5, originals=corpus)   # exact rerank → ~0.9997
```

`PCAMatryoshka.suggest_output_dim(corpus, target_variance=0.95)` picks the truncation dim from the data's spectrum. See the [user guide](docs/guides/user_guide.md).

## Or compress an LLM KV cache

Architecture-aware by quantizer, not just bit-width — per-channel **keys** + PolarQuant **values**:

```python
from turboquant_pro import TurboQuantKVCache

cache = TurboQuantKVCache.robust(head_dim=128, n_heads=32, hot_window=512)  # asym-NF4 keys + 2% outliers, 4-bit K/V
# or auto-configure from a model name:
from turboquant_pro import AutoConfig
cache = AutoConfig.from_pretrained("llama-3-8b", target="balanced").build_cache()   # K4/V3
```

`robust()` is one codebook that stays near-fp16 across every architecture tested (including high-GQA models where symmetric NF4 silently collapses). See the [KV keys finding](docs/KV_KEYS_FINDING.md).

## Choose your workflow

| Goal | Start here |
|---|---|
| Compress a vector index and search it | [User guide](docs/guides/user_guide.md) · [fast ADC design](docs/DESIGN_fast_adc.md) |
| Keep an index larger than RAM (memmap / shards) | [Production lifecycle](docs/guides/production_lifecycle.md) |
| Compress an LLM KV cache correctly | [KV keys finding](docs/KV_KEYS_FINDING.md) · [operator-aware quantization](docs/guides/operator_aware_quantization.md) |
| Compress model weights | [Model-weight guide](docs/guides/model_weight_compression.md) |
| Certify & third-party-verify a deployment | [Certification](docs/guides/certification.md) |
| Reproduce a headline number yourself | [`CLAIMS.md`](CLAIMS.md) · [claim replay](docs/guides/claim_replay.md) |
| Integrate (pgvector, FAISS, NATS, vLLM, …) | [Integrations](docs/integrations.md) |
| Drive it from an agent (LangChain / DSPy / MCP / GPT) | [Agent tools](examples/agentic/) · [`agent_tools`](turboquant_pro/agent_tools.py) |

## Why consumer-aware compression?

One governing principle ties the whole toolkit together:

> **Compress a tensor by the metric its consumer uses. Accept or reject on that metric — recall, perplexity, a rank certificate, an expert-set flip rate — never reconstruction cosine on its own.**

The sharpest illustration is KV-cache **keys**. PolarQuant normalizes each key and quantizes its *direction*, discarding the per-channel scale that `softmax(Q·Kᵀ)` depends on. On Qwen2.5 that reads a reassuring **0.995 key cosine** while perplexity explodes to **≈10⁴**; per-channel key quantization at the same width keeps it near fp16 (≈15). A reconstruction-only benchmark cannot see this. Full write-up: [`docs/KV_KEYS_FINDING.md`](docs/KV_KEYS_FINDING.md).

That boundary is now instrumented, so the principle ships as tooling rather than advice:

- **`rank_certificate`** — turns a measured distortion κ + the corpus's distance-ratio concentration μ̂ into a **distribution-free** rank floor (Kendall τ ≥ 1−2μ̂); a vacuous floor is the per-corpus "exact reranking required" signal. Emit with `tqp certify`, re-check with `tqp verify` (a third party re-hashes the inputs and reproduces the math).
- **`a2_probe`** — selects the quantizer family against the *declared* consumer (cosine / L2 / attention logits) at calibration time; it reproduces the keys catastrophe as a unit test.
- **`operator_trace` / `operator_sensitivity`** — infer each tensor's consumer (softmax score / residual / MoE gate / SSM decay) and apply the discipline that operator needs, validated on real Mixtral, OLMoE, and Mamba models.
- **`read_operators`** — the consumer's read operator `P_C` itself, as a pluggable provider, so the consumer-relative distortion `tr(P_C·Σ_δ)` is a **number you gate on** rather than a principle you appeal to. `identity` (which is reconstruction error, made explicit), `declared`, and `attention_analytic` (a closed form, since a head's read subspace is spanned by its queries) ship in tree; [`tqp-readscope`](plugins/tqp-readscope/) adds blind recovery for consumers with no closed form. Why a registry and not a function: two defensible references for one attention head differ by about **0.3 in overlap**, so a number is not interpretable unless it names the reference it was computed against — `tqp certify --reference PROVIDER` records exactly which one, hashed.
- **`read_allocation`** — spends a bit budget *against* that operator by reverse water-filling, which is the same optimization as power allocation across frequency bins with a task's sensitivity in place of a signal's power. `allocation_report` gives the gain over uniform at a fixed budget, and warns when the operator's spectrum is so concentrated that the split rests on one or two directions — which is correct if the operator is, and is also exactly what a sub-dimensionally recovered operator looks like regardless of the consumer. If you are allocating for a long context, allocate against the operator estimated over the whole sequence rather than a calibration prefix — that was better on every cell C-11c measured.

Backed by the companion theory papers: [the-angular-observer](https://github.com/ahb-sjsu/the-angular-observer) (the rank-certificate and (A2) transfer theory) and [geometric-observation](https://github.com/ahb-sjsu/geometric-observation) — the evidence repository home of **Paper III** (Observation Theory: consumer-relative rate–distortion and the omission floor) and **Paper IV** (the consumer-relative flip). TurboQuant Pro is *Paper II* of that series, the compression-as-observation work.

The measurement side ships separately as [**readscope**](https://github.com/ahb-sjsu/readscope) (`pip install readscope`) — an oscilloscope and spectrum analyzer for model consumers, which recovers `P_C` from a consumer's outputs alone. It is numpy-only and deliberately independent: neither package imports the other, and [`tqp-readscope`](plugins/tqp-readscope/) is the ~100-line adapter that knows about both. Its specification is worth reading before trusting a measured operator — recovery against the direction budget is a **cliff at `k = d`** and the cliff is rank-independent, so a sub-dimensional probe returns the dominant direction and little else however many directions you intended to use.

### The strategic bet

As models and vector databases scale, the binding constraint shifts from *storing the vector* to *preserving what its consumer reads with it*. Reconstruction fidelity — the objective essentially every quantizer optimizes — is increasingly the wrong one: it can show a reassuring 0.995 cosine while the downstream task collapses. TurboQuant Pro is the production embodiment of the alternative: **measure the consumer's read operator, spend bits against it, and ship a certificate that the ranking survives** — turning a theory program (Paper I's transfer/rank theory, Paper IV's consumer-relative flip) into instruments you run in CI. The bet is that *certified, consumer-aware compression* becomes table stakes as ratios climb and silent quality regressions get more expensive to miss. That is the axis this project competes on — not one more point on the compression-vs-reconstruction curve, but the certificate that the compression preserved the thing that mattered.

## How it works

A per-vector flow — extract L2 norm → unit-normalize → random-orthogonal rotate → Lloyd-Max scalar-quantize → bit-pack — compresses embeddings and KV-cache *values* near-losslessly (the **TurboQuant** algorithm, Zandieh et al., ICLR 2026). KV-cache **keys** take the per-channel path instead (above).

```mermaid
flowchart LR
    A["Raw vector<br/>(float32)"] --> B["PCA-Matryoshka<br/>rotate + truncate"]
    B --> C["Random orthogonal<br/>rotation"]
    C --> D["TurboQuant<br/>Lloyd-Max SQ"]
    D --> E["Bit-pack<br/>8×3-bit = 3 B"]
    E --> F["Compressed code"]
    A -. "L2 norm (kept alongside)" .-> F
    classDef out fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px;
    class F out;
```

## Benchmark snapshot

At **32× compression**, recall@10 on real LaBSE / multilingual-Gutenberg embeddings — all methods reranked identically:

| method | recall@10 (single) | recall@10 (+rerank) | index build |
|---|---:|---:|---:|
| PQ | 0.467 | 0.827 | 142 s |
| RaBitQ (2024 SOTA) | 0.630 | 0.962 | 0.3 s |
| OPQ | 0.780 | 0.999 | 632 s |
| **turboquant-pro** | **0.784** | **0.9992** | **31 s** |

Holds at 1M scale (0.989 +rerank, tying OPQ). **Full tables** — the 15-method BGE-M3 comparison, the rerank frontier, KV-cache generation quality & memory, the RaBitQ estimator-isolated head-to-head — are in [**docs/benchmarks/embeddings.md**](docs/benchmarks/embeddings.md) and [**docs/benchmarks/kv.md**](docs/benchmarks/kv.md). Reproduce end-to-end on public data: [`notebooks/turboquant_benchmark.ipynb`](notebooks/turboquant_benchmark.ipynb) · [Colab](https://colab.research.google.com/github/ahb-sjsu/turboquant-pro/blob/master/notebooks/turboquant_benchmark.ipynb).

> **Reading compression ratios.** Ratios vary with source dimension, PCA truncation, code width, retained metadata, and whether exact originals are kept for reranking — so distinguish *compressed payload* vs *all-in index storage* vs *full retrieval-pipeline storage*. The canonical headline is **32× at recall@10 0.9992** above; other figures in the benchmark docs (e.g. 27.7× single-vector, 114× pipeline-storage) are labeled by their accounting basis.

## At scale & in production

**Larger-than-RAM search (1.9.0).** `TQEIndex` persists an index and memory-maps it; a block-streamed path keeps peak RAM at `O(n_queries × block)` at any corpus size. `ShardedIndex` splits a corpus into shards that **share one PCA basis** (scores stay comparable) behind a JSON manifest and fans search across them (parallel across cores; `distributed.py` partitions shards across machines). On disk, **index format v3** bit-packs sub-byte codes — a *lossless* re-encoding (rankings bit-identical to v2) at **24.1 B/row** vs 41 B/row in v2 (2M rows / 4-bit / `--no-originals`).

```python
from turboquant_pro import TQEIndex, ShardedIndex

idx = TQEIndex.open("index.tqe", mmap=True)                 # memory-mapped, read/search only
ids, scores = idx.search(queries, k=10, block=100_000)      # bounded-RAM, block-streamed

ShardedIndex.create(corpus, "shards/", shard_size=500_000, bits=3)   # one shared PCA basis
ids, scores = ShardedIndex.open("shards/manifest.json").search(queries, k=10)
```

**The `tqp` CLI** covers the whole lifecycle — `trace → plan → compress → certify → verify → replay → monitor`, plus a persisted-index workflow:

```bash
tqp plan embeddings --embeddings corpus.npy --target "recall@10 >= 0.90"    # recipe on the Pareto frontier
tqp certify --original corpus.npy --reconstructed corpus_q.npy --min-tau 0.8 \
  --task "recall@10 >= 0.995" --environment --html report.html              # rank floor + provenance envelope
tqp verify certificate.json --original corpus.npy --reconstructed corpus_q.npy   # a third party re-checks it
tqp index create --embeddings corpus.npy --out shards/ --bits 3 --shard-size 500000
tqp index search shards/manifest.json --queries q.npy --k 10 --mmap --block 100000
tqp query "SELECT id, score FROM 'x.tqe' ORDER BY COSINE(:q) LIMIT 10 WITH (RECALL >= 0.95)" \
  --queries q.npy                                       # declare the target; the planner meets it (1.9.1)
tqp anatomy --npy corpus.npy --k 10                     # hub anatomy: what your hubs ARE (1.9.1)
tqp hubdiff --original corpus.npy --reconstructed corpus_q.npy --min-anti-recall 0.9 \
                                                        # the tail mean recall hides (1.9.1)
tqp anatomy --npy corpus.npy --strata kmeans:8 --save-map map.json   # per-stratum, not pooled (unreleased)
tqp hubdiff --original corpus.npy --reconstructed corpus_q.npy --labels lang.txt \
  --abstain-fails                                       # min-over-strata; ABSTAIN (unreleased)
```

New to hubness and anti-hubs? **[`docs/HUBNESS_PRIMER.md`](docs/HUBNESS_PRIMER.md)**
— the ten-minute primer on why aggregate recall can stay green while your
hardest queries collapse, and how `anatomy`/`hubdiff` catch it. **Trust the
tail, not the mean.**

**Stratified instruments (STRATA Phase 1, unreleased).** These landed after
the 2.0.0a2 tag, so they are on `master` and in no published wheel yet. Install
from source (`pip install -e .`) to use them. A pooled hubness number can stay
green while one stratum fails, so the gates run **per stratum and report
the minimum**, never the average. Strata come from k-means, a saved area map,
or a label file. Area maps are content-addressed (`tqp-area-map/1`): an
incomplete profile matches nothing, including itself, and a tampered artifact
refuses to load. A stratum with too few rows returns **ABSTAIN** with a
registered cause rather than a pass, and `--abstain-fails` makes that an error
in CI. The relational surface (`attach_strata`) exposes the same results to
DuckDB for querying.

Full command reference: [`docs/CLI.md`](docs/CLI.md). Also here: `QualityMonitor` (cosine + (A2) tangential drift, Prometheus metrics), `behavioral_agreement` (decision-level flip rate + noise floor), hardware-aware profiles (Volta→Blackwell), a portable Triton fused-decode kernel, and cross-framework export (FAISS / Milvus / Qdrant / Weaviate / Pinecone) — see [Integrations](docs/integrations.md).

## Agents & tool use

Autonomous systems can consume the whole pipeline as tools. `turboquant_pro.agent_tools` is a small JSON-in/JSON-out surface with docstrings written for tool-calling models — wrapped for **LangChain**, **DSPy**, an **MCP** server, and custom-GPT **Actions** in [`examples/agentic/`](examples/agentic/).

```python
from turboquant_pro import best_compression_at_recall, certify_ranking

plan = best_compression_at_recall(corpus, k=10, min_recall=0.99)   # "best ratio at 0.99 recall" — accepts on recall, not cosine
cert = certify_ranking(corpus, reconstructed)                      # the distribution-free rank receipt
```

The **goal is a runtime input**: the agent declares the target recall (or the consumer metric, or `k`) *per task*, and the tool accepts and certifies against **that** goal — never reconstruction cosine. That is the project's one rule expressed as an API, and it is *why* cosine can't be the gate: the coordinate worth keeping is the one that carries the currently-declared goal's geometry. Full guide: [`examples/agentic/README.md`](examples/agentic/README.md).

## Feature & stability matrix

The full table is in [`docs/api-stability.md`](docs/api-stability.md) (the source of truth); component reference in [`docs/API.md`](docs/API.md).

| Tier | Components |
|---|---|
| **Stable** | `PCAMatryoshka`, embedding compression pipeline, basic `TurboQuantKV`, TQE1 format |
| **Beta** | `ADCIndex`, `TQEIndex` (memmap + format v3), `ShardedIndex`, `TurboQuantKVCache`, the rank certificate (`tqp certify`/`verify`), the (A2) probe + quality monitor, the `tqp index` lifecycle, the runtime safe-fallback policy, FAISS / pgvector wrappers |
| **Experimental** | agent tool surface (`agent_tools` + `examples/agentic`), `tqp query` (SQL-ish workload interface), hub anatomy + anti-hub oracle (`tqp anatomy`/`hubdiff`), STRATA stratified instruments (area maps, min-over-strata gates, ABSTAIN, `attach_strata` — unreleased, `master` only), **vLLM V1 KV connector** (`turboquant_pro.connectors` — [2.0 roadmap](docs/ROADMAP_2.0.md)), quantizer plugin registry + conformance kit, read-operator provider registry + conformance kit (`read_operators`), CUDA/Triton fused decode, multi-node shard server (`distributed.py`), vLLM manager, model-weight compressor, PostgreSQL extension, NATS transport |

**Scope & honesty:** results are strongest on **text embeddings and LLM workloads**; multimodal APIs/presets exist but are less validated. "Beats RaBitQ" means under our matched-byte public protocol; "robust across every architecture" means every architecture *tested*. All 4-bit KV quant (asym-NF4 included) still degrades on very-long-generation tasks; readscope's C-11c measures a **candidate mechanism** for that negative rather than an explanation of it — a head's read operator moves along the sequence, so a key compressed against an early operator is later read by a different one, and allocating against the early operator misprices the late consumer by 225% of a uniform split's cost. That was sixteen head-cells over 192 positions of one 3B model, against a paired random-split null that turned out to account for more than half the naive effect. It has never been run against a real degradation curve, so it stays a hypothesis. Negative results and caveats are kept first-class in [`docs/claims.md`](docs/claims.md) and the [soundness audit](docs/soundness_audit.md).

> **Not to be confused with** the similarly-named `turboquant` (the HuggingFace KV-cache implementation of the original ICLR TurboQuant algorithm). TurboQuant Pro is a broader, retrieval-first platform that uses that quantizer as **one component**.

## Documentation & reproducibility

- **[Documentation hub](docs/)** — guides, reference, and the 15-minute reviewer path.
- **Agents & MCP:** [`examples/agentic/`](examples/agentic/) — LangChain / DSPy / MCP / custom-GPT wrappers over [`turboquant_pro.agent_tools`](turboquant_pro/agent_tools.py).
- **Reproduce the claims:** [`CLAIMS.md`](CLAIMS.md) (claim → notebook → hardware → status) · [claim replay guide](docs/guides/claim_replay.md) · [evidence ladder](docs/claims.md).
- **Benchmarks:** [embeddings](docs/benchmarks/embeddings.md) · [KV cache](docs/benchmarks/kv.md) · [release/library growth](docs/RELEASE_HISTORY.md).
- **Formats:** [FORMATS.md](docs/FORMATS.md) (TQE1 / TQIX / certificates at a glance) · [FORMAT_SPEC.md](docs/FORMAT_SPEC.md) · [CERTIFICATE_SPEC.md](docs/CERTIFICATE_SPEC.md).
- **Extending it:** two plugin points, same shape. [`turboquant_pro/plugins.py`](turboquant_pro/plugins.py) registers **quantizers** (what gets certified) — see [`plugins/`](plugins/) for `tqp-bnb`, `tqp-gptq-awq`, `tqp-trtllm`. [`turboquant_pro/read_operators.py`](turboquant_pro/read_operators.py) registers **read operators** (what they are certified against) — see [`plugins/tqp-readscope/`](plugins/tqp-readscope/). Each has its own conformance kit, and in-tree entries register through the same public interface an out-of-tree package uses. Design: [`docs/DESIGN_hardware_and_plugins.md`](docs/DESIGN_hardware_and_plugins.md).
- **Citation:** [`CITATION.cff`](CITATION.cff) (GitHub "Cite this repository") · full BibTeX + acknowledgments in [`docs/CITATION.md`](docs/CITATION.md).
- **Contributing:** CI runs `ruff check` and `black --check` over `turboquant_pro/`, `tests/`, and `benchmarks/`, and both must pass. Install the hooks so a mismatched local formatter version cannot redden CI: `pip install pre-commit && pre-commit install`. The pinned versions live in [`.pre-commit-config.yaml`](.pre-commit-config.yaml) and must stay equal to the dev extras in `pyproject.toml`.

## License

MIT License. See [LICENSE](LICENSE). Author: **Andrew H. Bond**, San Jose State University.

# Release history — library growth

Part of [TurboQuant Pro](../README.md). Full prose release notes are in [`CHANGELOG.md`](../CHANGELOG.md); this table tracks test/module growth per release.

| Version | Tests | Modules | Key Features |
|---------|------:|--------:|-------------|
| v0.5.0 | 175 | 8 | Autotune, FAISS, vLLM, pgext |
| v0.8.0 | 244 | 14 | CUDA kernels, HNSW, cache |
| v0.9.x | 303 | 19 | Asymmetric K/V, RoPE, auto-config |
| v0.10.0 | 351 | 23 | auto_compress, hardware, export |
| v1.0.0 | 397 | 27 | Learned codebooks, multi-modal, observability |
| v1.1.0 | 473 | 32 | ADCIndex, fused KV-decode, TQE1 format |
| v1.2.0 | 489 | 33 | Per-channel KV keys — correct key architecture |
| v1.3.0 | 493 | 33 | Calibration-free NF4 + dense-sparse outliers (≈ KVQuant on Llama) |
| v1.4.0 | 497 | 33 | Asymmetric NF4 — one robust codebook across architectures |
| **v1.4.3** | **514** | **33** | **Docs + reproducibility: canonical benchmark harness, per-claim notebooks, CLAIMS.md; `estimate_storage()` dimension fix** |
| *1.5.0 – 1.7.0* | — | — | *behavioral-agreement, operator tracing, SSM/MoE sensitivity, `tqp index` lifecycle, runtime policy (see [`CHANGELOG.md`](../CHANGELOG.md))* |
| **1.8.0** | **847** | **44** | **Certification platform** (`tqp` CLI, plugin registry + conformance, `tqp certify`/`verify`), **P5 Triton port** of the fused M2/M4 kernels (Turing→Hopper), out-of-tree plugin ecosystem (bnb / trtllm / gptq-awq) |
| **1.9.0** | **919** | **47** | **At-scale search** — memmap + `ShardedIndex` + block-streamed (larger than RAM) + multi-node `distributed.py`; **index format v3** bit-packing (~1.7× smaller `--no-originals`); (A2) ZCA-whitened family; richer `tqp certify` envelope |
| **1.9.1** | **947** | **50** | **Queryable, agent-consumable, tail-aware** — SQL-ish `tqp query` (ANALYZE / EXPLAIN / SELECT … WITH (RECALL ≥ r)); `agent_tools` + LangChain/DSPy/MCP/GPT wrappers; hub anatomy + anti-hub oracle (`tqp anatomy`/`hubdiff`, [primer](HUBNESS_PRIMER.md)); resumable 100B-scale IVF builds + source routing |

> Test counts are **pytest-collected item counts** (parametrized cases count individually), snapshotted at each release — not the raw `def test_` function count, which is lower. The table jumps from v1.4.3 to **1.8.0** (the certification-platform release); the current PyPI release is **1.9.1**; the intervening 1.5.0–1.7.0 (and metadata-only 1.8.1) releases are in [`CHANGELOG.md`](../CHANGELOG.md). The [Tests badge](https://github.com/ahb-sjsu/turboquant-pro/actions) shows CI **pass/fail** status, not a count; for the exact current number run `pytest -q --co | tail -1`. Run the history benchmark: `python benchmarks/benchmark_release_history.py`.

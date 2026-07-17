# Memory-mapped / sharded index at scale

Reproduce: [`benchmarks/bench_index_scale.py`](bench_index_scale.py) (`build` then
`search`, run as separate processes so the search process's peak RSS reflects only
search). Measured on the internal box (Xeon E5-2690 v3, NumPy). Corpus is synthetic
moderate-rank; recall is against the exact cosine top-10.

## 10M vectors (dim 64 → PCA 32 → 3-bit, 10 shards, 2.84 GiB on disk)

| search | mode | peak RSS | RSS ÷ index | recall@10 | QPS |
|---|---|---:|---:|---:|---:|
| single-pass | **memmap** | **2.42 GiB** | 0.85× | 0.586 | 3.6 |
| single-pass | non-mmap (load all) | 5.81 GiB | 2.05× | 0.586 | 3.0 |
| + exact rerank ×10 | **memmap** | 4.62 GiB | 1.63× | **1.000** | 2.9 |

**What it shows.**

- **Bounded RAM.** Memory-mapped single-pass search peaks at **2.42 GiB — below the
  2.84 GiB index and 2.4× under the non-mmap baseline (5.81 GiB)**. The non-mmap path
  must load every array *and* build an N-entry id→pos dict (~2–3 GiB at 10M); the
  memmap path streams the codes in row-blocks and keeps no id map, so its footprint
  is the working set, not the index. Under real memory pressure (RAM < index) the OS
  evicts clean file-backed pages, so the mapped index need never be fully resident —
  the point of memmap: **search an index larger than RAM.**
- **Recall holds at scale.** Sharding across 10 shards with a shared PCA basis keeps
  scores comparable, and exact rerank recovers **recall@10 = 1.0** at 10M vectors.
  (Single-pass 3-bit ADC is 0.586 without rerank, as expected — rerank is the
  high-recall stage.)
- **Build streams.** The 10M index builds in ~55 s shard-by-shard.

## Notes

- QPS here is the pure-NumPy blocked path (no AVX2 kernel under memmap); the shards
  are independent, so throughput scales with the fan-out across cores/nodes.
- The same script runs unchanged at larger scale (e.g. on an NRP node) — bump `--n`
  and `--shard-size`; the memmap footprint stays bounded by `n_queries × block`
  plus the per-query candidate working set.
- Rerank touches the stored originals for the candidate rows; to keep single-pass
  memory minimal on a huge index, build with `--no-originals` (single-pass only) or
  rerank a small oversample.

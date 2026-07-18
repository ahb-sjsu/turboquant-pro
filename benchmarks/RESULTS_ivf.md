# IVF coarse-partition: recall vs fraction of corpus scanned

Reproduce: [`benchmarks/bench_ivf.py`](bench_ivf.py). Ground truth is the **brute-force
ADC top-k** over the whole corpus — the ranking IVF is trying to reproduce — so
`recall@10 = 1.0` means "identical to scanning everything." Synthetic corpus,
n = 100,000, dim 64 → PCA 32 → 4-bit, `nlist = 316` (√n), mean cell radius **65°**.

| mode | recall@10 | scan fraction | speedup vs brute | qps |
|---|--:|--:|--:|--:|
| brute-force ADC (baseline) | 1.000 | 100% | 1× | 709 |
| fixed nprobe=4 | 0.583 | 1.3% | 79× | 3743 |
| fixed nprobe=8 | 0.749 | 2.6% | 39× | 2197 |
| fixed nprobe=16 | 0.881 | 5.1% | 20× | 1133 |
| fixed nprobe=32 | 0.963 | 10.2% | 10× | 615 |
| **weighted A\* β=0.25** | 0.804 | **3.2%** | **31×** | 1145 |
| **weighted A\* β=0.50** | **0.999** | 22% | 4.5× | 56 |
| weighted A\* β=0.75 | 1.000 | 60% | 1.7× | 26 |
| admissible A\* (β=1) | 1.000 | 89% | 1.1× | 19 |

## What it shows

- **Coarse partitioning makes search sublinear.** Best-first `nprobe` recovers **96%
  recall while touching ~10% of the corpus** (10× fewer rows), and 88% at ~5%. The
  partition + best-first order is the workhorse — a clean recall/scan dial.
- **Weighted A\* is the adaptive dial.** Instead of picking `nprobe`, set how much
  recall you'll trade: `β` shrinks the cell radius to prune more. **β=0.5 lands at
  recall 0.999 while scanning 22%**, per-query, with no manual `nprobe`. β=0.25 is
  aggressive (0.80 @ 3%); β→1 is exact.
- **The admissible (worst-case) A\* stop is exact but barely prunes.** At 65° cell
  radii the bound `cos(θ − r)` is ≈1 for almost every cell, so the provably-safe stop
  scans 89%. This is the honest limitation of literal A\* here: an admissible
  heuristic gives no free selectivity in high dimension — you *must* relax it
  (weighted A\*, or a fixed `nprobe`) to get speed. Reported, not hidden.

## Sharded IVF at scale, on local NVMe

The single-node numbers above use `IVFIndex`. The same coarse layer folds into
`ShardedIndex` (`build_ivf` + `search(nprobe=…)`) for indexes that span many shards.
[`benchmarks/bench_ivf_sharded.py`](bench_ivf_sharded.py), **50M vectors** (dim 32 →
PCA 24 → 4-bit, 10 shards, `nlist=2048`, 7.87 GiB index) on a **local NVMe** volume
(Atlas), 200 queries; reference = the full-scan ADC top-k, so recall 1.0 = identical.

Build streamed shard-by-shard in **169 s**; `build_ivf` in **258 s**; peak RSS **6.94
GiB** throughout (< the index).

| search | scan fraction | recall@10 | QPS | speedup vs brute |
|---|--:|--:|--:|--:|
| brute full-scan | 100% | 1.000 | 0.93 | 1× |
| ivf `nprobe=8` | 0.58% | 0.412 | 13.6 | **14.7×** |
| ivf `nprobe=16` | 1.11% | 0.555 | 9.6 | 10.4× |
| ivf `nprobe=32` | 2.03% | 0.702 | 5.4 | 5.8× |
| ivf `nprobe=64` | 3.84% | **0.839** | 3.8 | 4.1× |

- **This is the storage-medium lesson made concrete.** The same search on a 1B index
  over **CephFS** (a network FS) was I/O-bound to impracticality — scoring
  fancy-indexes hundreds of thousands of *random* rows into the codes memmap per
  query × shard, i.e. random page faults over the network (see
  `RESULTS_index_scale.md`). On **local NVMe** those reads are cheap, so the search
  runs at interactive speed and the sublinear scan turns into real wall-clock: brute
  full-scan is **0.93 QPS** (216 s for 200 queries), IVF is **4–15×** faster while
  touching **0.6–4%** of the corpus.
- **Locality optimizations that make NVMe pay:** the fan-out is grouped *by cell* (each
  probed cell's rows + codes read once per shard and scored against all its queries in
  one batched matmul), posting lists are memory-mapped `.npy` (a probe touches only the
  cell slices it needs), and inverted-list members are ascending so code reads within a
  cell are sequential.
- **Tuning:** `nlist` sets cell size (here 50M/2048 ≈ 24k rows/cell); finer cells
  (larger `nlist`) raise recall at a given scan but cost more in `build_ivf`. QPS is the
  pure-NumPy path — the win is *rows scanned*; a kernel/parallel fan-out lifts absolute
  throughput.

## Notes

- Everything derives from the codes the ADC index already stores (the coarse space is
  `normalize(cent[codes])`), so the partition adds no per-row storage.
- QPS here is the pure-NumPy prototype path; the win is *rows scanned*, which is what
  carries to memmap (a probe faults in only its cells' pages) and to sharding (the same
  best-first order across per-shard IVF). This is the layer that takes search from
  O(N) toward the few-percent scans a trillion-vector index needs.
- Acceptance is recall of the shortlist against the exact ADC ranking — never
  reconstruction cosine.

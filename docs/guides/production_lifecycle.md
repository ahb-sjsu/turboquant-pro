# Production lifecycle guide — a mutable, compressed, drift-aware index

A benchmark index lives in RAM and is thrown away. A production RAG index must
persist, grow, forget, stay honest about storage, survive format upgrades, prove it
still ranks well, and notice when its basis goes stale. `TQEIndex` + the `tqp index`
command group cover that whole loop over the corruption-checkable **TQIX** container.

```
ingest → search → update → compact → migrate → certify → monitor
```

## The loop

```bash
# ingest: fit the PCA basis once, persist basis + codes
tqp index create --embeddings corpus.npy --out corpus.tqe --bits 3

# search: compressed, with exact rerank for high recall
tqp index search corpus.tqe --queries q.npy --k 10 --rerank 10

# update: append (same basis, no refit) and tombstone by id
tqp index add    corpus.tqe --embeddings new.npy
tqp index delete corpus.tqe --ids 12,88,90

# compact: physically drop tombstoned rows, reclaim bytes (ids are preserved)
tqp index compact corpus.tqe

# migrate: upgrade the on-disk format under a real version bump
tqp index migrate corpus.tqe --to-version 2

# certify: a rank certificate over the live vectors (gates the exit code)
tqp index certify corpus.tqe --min-tau 0.5

# monitor: is the PCA basis still a good fit for recent data?
tqp index drift corpus.tqe --embeddings recent.npy
```

Same surface from Python via `TQEIndex.create/open/add/delete/compact/migrate/search/certify/drift`.

## What makes it trustworthy

- **Corruption is detected, never silent.** Every section of the TQIX container
  carries a CRC32; a flipped byte is a clean `IndexCorruptionError`, not
  wrong-but-plausible vectors. Writes are atomic (temp file + rename). A
  single-byte-flip fuzzer guards the invariant "detected, or byte-identical."
- **Ids are external and stable.** `create`/`add` assign monotonic ids; they survive
  `compact` (rows are dropped, ids are never renumbered), so external references stay
  valid across maintenance.
- **Exact rerank + certify need the originals.** `create` keeps fp32 originals by
  default (`--no-originals` to skip); they power exact rerank and let the index
  certify itself. Without them, rerank degrades to the reconstruction.

## Closing the loop — adaptive and drift-aware

Two signals turn this from a static store into an adaptive one:

```python
from turboquant_pro import TQEIndex, TQPRuntimePolicy
idx = TQEIndex.open("corpus.tqe")
policy = TQPRuntimePolicy()

# adaptive retrieval: single-pass where margins are wide, exact rerank where tied
ids, _ = idx.search(queries, k=10, policy=policy)

# scheduled maintenance: refit/migrate when the encoder distribution drifts
report = idx.drift(recent_embeddings)
if policy.evaluate_index_drift(report).conservative:
    #  action == "refit_or_migrate"  →  rebuild the basis on fresh data
    ...
```

`drift` compares the variance the stored basis retains on new data to what it
retained at fit time (plus a mean shift); a large drop means the basis is stale and
recall will silently erode. Catch it before your users do. See
[certification](certification.md) for the guarantee behind `certify`, and the
[operator-aware guide](operator_aware_quantization.md) for why acceptance is recall /
a rank certificate, never reconstruction cosine.

## Scaling out — memmap and sharding

A single index scales to as many vectors as one file conveniently holds. Beyond
that, two mechanisms keep RAM bounded and let you go to billion-scale:

**Memory-mapped search.** Open an index with `mmap=True` and the big arrays (codes,
norms, originals) are memory-mapped, not read into RAM — the OS pages in only what a
query touches, and search streams the codes in row-blocks so peak memory is
`O(n_queries × block)`, independent of how many vectors the index holds:

```python
idx = TQEIndex.open("index.tqe", mmap=True)      # bounded RAM, read/search only
ids, _ = idx.search(queries, k=10, rerank=10, block=262_144)
```

A memmap-opened index is read/search only (mutations raise — reopen in RAM to
modify). CRC is checked on the small sections at open; verify the full file with
`index_file.read_container` when you need an integrity pass.

**Sharding.** `ShardedIndex` splits a corpus into `shard_size`-row shards that
**share one PCA basis** (so their scores are directly comparable) plus a small JSON
manifest. Search fans out over the shards — each opened memory-mapped — and merges
the per-shard top-k into a global top-k. The shards are independent, so the fan-out
parallelizes across cores or machines:

```python
from turboquant_pro import ShardedIndex
ShardedIndex.create(embeddings, "corpus.shards", shard_size=1_000_000, bits=3)
sh = ShardedIndex.open("corpus.shards/manifest.json")   # memmap by default
ids, scores = sh.search(queries, k=10, rerank=10)
```

Same from the CLI: `tqp index create --embeddings e.npy --out corpus.shards
--shard-size 1000000`, then `tqp index search corpus.shards --queries q.npy
--k 10 --rerank 10 --mmap` (a directory or `manifest.json` path is searched as a
shard set).

At 10M vectors, memory-mapped single-pass search peaks at ~2.4 GiB vs ~5.8 GiB for
a full RAM load (2.4×), while reranked recall@10 stays 1.0 — measured in
[`benchmarks/RESULTS_index_scale.md`](../../benchmarks/RESULTS_index_scale.md)
(`benchmarks/bench_index_scale.py`, which runs unchanged at larger scale). A 1B-vector
index (157 GiB, 200 shards) builds streaming in a 12 GiB pod at peak RSS 8.9 GiB, and
searches with the fan-out RSS bounded far below the index size — the point of memmap.

**Sublinear search — IVF coarse layer (experimental).** A billion-shard fan-out still
*scans every row* (`O(N)`). `sh.build_ivf(nlist=...)` adds a coarse layer on top of an
existing sharded index: one global k-means quantizer over the quantized directions
(fit once, like the PCA basis) plus per-shard inverted lists as sidecars. Then
`sh.search(queries, k=10, nprobe=32)` selects the best cells once (weighted-A\* order,
`radius_scale`) and scores only those cells' rows across shards — the path from `O(N)`
toward the few-percent scans a trillion-vector index needs (recall/scan tradeoff in
[`benchmarks/RESULTS_ivf.md`](../../benchmarks/RESULTS_ivf.md)).

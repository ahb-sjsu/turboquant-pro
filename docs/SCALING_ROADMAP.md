# Scaling roadmap: from the single-node engine to 10¹²⁺

How turboquant-pro's sharded index scales from one box to a trillion-plus vectors —
and, crucially, **what to reuse rather than rebuild.** The distributed layer is not new
infrastructure: it composes three existing, released projects.

## Where we are (the substrate, validated to 1B)

The **per-node engine** is complete and measured:

- `ShardedIndex` — streaming ingest, memmap search, bounded fds, one shared PCA basis.
- **IVF coarse layer** (`build_ivf` + `search(nprobe=…)`) — sublinear scan (few-% of rows).
- **Hierarchical IVF** (`build_ivf(hierarchical=…)` + `search(top_probe=…)`) — two-level
  (IVF-of-IVF) quantizer: cheaper assign, finer partition, probes cluster into a few top
  cells (locality + server routing).
- **GPU `build_ivf`** — the `O(N·nlist)` coarse-quantizer wall; ~17× on a GV100 at
  identical recall (`RESULTS_ivf.md`).
- **Parallel per-shard fan-out** (`search(workers=N)`) and the **parallel-build API**
  (`write_shard` / `finalize_manifest`).
- **The storage law** (measured): random-access IVF search needs **block / local NVMe**
  (Linstor, RBD, node-local), *not* a network FS — CephFS was 78–105× slower
  (`RESULTS_index_scale.md`, `RESULTS_ivf.md`).

## The distribution substrate: compose, don't reinvent

turboquant-pro is already a citizen of the **nats-bursting** fabric (it is that paper's
transport-compression layer). The distributed stages reuse the fabric's parts:

| need | reuse |
|---|---|
| distributed **build** work-queue | nats-bursting `TaskDispatcher` / ephemeral `Job`s — publish `write_shard` tasks, workers pull |
| distributed **search** shard-servers | nats-bursting **persistent queue-group pools** (`PoolDescriptor`/`Worker`) — warm, no cold start |
| query/result **transport** | NATS request/reply, payloads compressed by **turboquant-pro** itself |
| per-pod **right-sizing** (stay in the util policy) | **batch-probe** — GPU batch + thermal-aware thread control |
| cluster **fairness** | nats-bursting **politeness controller** (probe + back off) |

So turboquant-pro contributes the *transport-agnostic* scatter-gather primitives
(serialize a query, serve a shard-range, merge partials); nats-bursting is the wire.

## Stages (recommended order)

1. **Per-node engine** — ✅ done; **1B validated on Linstor block storage** (38.2 GB
   index, 7.45 GiB peak RSS, recall 0.976 @ 6.9% scan; `RESULTS_ivf.md`).
2. **(c) Scatter-gather search** — ✅ done. `distributed.py`: a shard-server answers
   `bytes → partial top-k bytes` (`ShardServer`), a coordinator scatters over a
   `transport` callable and merges partials (`scatter_gather` + `_merge_partials`).
   Transport-agnostic; the **NATS wire is now built and live-tested** (`nats_transport.py`
   + `nats_worker` entrypoint + `shard_pool_manifest`) — see below.
3. **Distributed build** — ✅ API done (`write_shard` / `finalize_manifest`, incl.
   `basis_from=` / `ids=` for cell-grouped shards); *deployment* over the nats-bursting
   work-queue is the remaining wiring. Build time → `total / n_workers`.
4. **IVF-as-router** — ✅ done (`Router` + `scatter_gather_routed` +
   `build_cell_placement`). The global coarse quantizer routes each query to only the
   servers holding its `nprobe` cells, so a query touches `nprobe/nlist` of the fleet —
   sublinear *in servers*, mirroring the sublinear-in-rows win.
5. **Operational scale-out** — *next.* Storage tiering (hot codes on NVMe, cold
   originals on S3/CephFS for a rerank tier), Linstor-HA replication, distributed
   mutation (delete/compact/re-cluster), monitoring.

### Landed since the plan was written

- **Hierarchical IVF (IVF-of-IVF)** — ✅ `build_ivf(hierarchical=…)` +
  `search(top_probe=…)`. Two-level quantizer (top cells × leaf cells): cheaper
  `O(N·(top+sub))` assignment, finer partition at the same cost, and probes that
  cluster into a few top cells. This is the locality/quality fix the 1B run pointed at
  (single-node speedup *decayed* with `nprobe` because probes hit scattered pages) and
  the structure that lets the router skip servers **without** manual cell-alignment
  (`RESULTS_ivf.md` → "Hierarchical IVF").
- **GPU `build_ivf`** — ✅ 17× on a GV100, identical recall.
- **Parallel per-shard fan-out** — ✅ `search(workers=N)`; ~2.4–2.6× on top of IVF at 1B.
- **nats-bursting deployment adapter** — ✅ `nats_transport.py`: `NatsShardServer`
  (request/reply queue-group subscriber, search off-loaded to a thread), `NatsTransport`
  (sync transport over one background loop for the coordinator's fan-out), the
  `turboquant_pro.nats_worker` pod entrypoint, and `shard_pool_manifest` (one addressable
  Deployment per shard-range). **Live-tested on Atlas**: 4 worker subprocesses on a real
  `nats-server`, routed coordinator recall **1.0000** vs single-node. Subject per
  shard-range for addressability, queue group per subject for replicas; core NATS
  request/reply (crosses leaf links). No hard dep — `nats-py` lazy-imported.
- **Tiered rerank** — ✅ `rerank_tier.py` + `search(rerank=…, rerank_store=…)`. Hot tier
  stays codes-only; the wide IVF/ADC shortlist is exactly re-scored against originals
  fetched from a cold tier (`NpyOriginalStore` or any `fetch(ids)` callable), read bounded
  to the shortlist. Breaks the ADC-vs-truth ceiling (test: strictly beats ADC-only, >0.9
  true-recall at 2-bit). Runs once at the coordinator over the merged shortlist.

**Recommended next:** **operational scale-out** — storage tiering wiring (place the cold
`NpyOriginalStore` on S3/CephFS while codes stay on NVMe), Linstor-HA replication, then
distributed mutation (delete/compact/re-cluster) last. The distributed layer is now
complete end-to-end; a **production multi-node fleet run** (real PVCs, several pods,
cold-tier rerank) is the natural end-to-end validation.

## The numbers (`--no-originals`, ~30 B/row)

| scale | index | fleet (local-NVMe shard-servers) | build | search |
|---|--:|---|---|---|
| 1B | ~30 GB | 1 node | GPU quantizer + streaming, ~1 h | validated (block) |
| **1T** | **~30 TB** | ~50–100 nodes (0.3–0.6 TB each) | GPU quantizer + distributed `write_shard`, **hours** | IVF-router → few nodes/query |
| **5T** | ~150 TB | ~250–500 nodes | same, wider fan-out | same pattern |

## Design principles / honest limits

- **Keep the core SQLite-clean.** The north star is the *embeddable* single-node engine;
  ship distribution as a **separate, optional coordinator layer**, not bolted into the core.
  Most workloads (100M–10B on a few nodes) never need the full fleet.
- **Batch before serving.** Prove 1T *offline* (recall/throughput) before an online-latency
  architecture (replication, hedged requests, straggler mitigation).
- **Recall ceiling.** ADC-only caps recall (~0.92 at coarse nlist); fine nlist (GPU-affordable
  now) helps, and the **tiered rerank** (now implemented, `rerank_tier.py`) closes the rest —
  fetch candidate originals from the cold tier for the shortlist only and exact-rescore.
- **Coarse-quantizer quality at 1T.** A flat `nlist` gets coarse; the fix —
  **hierarchical quantization (IVF-of-IVF)** — is now implemented
  (`build_ivf(hierarchical=…)`): a top layer over leaf cells, cheaper to assign and
  finer at the same cost, and it makes probes local. Residual/PQ quantization is a
  further option if leaf cells stay too wide.
- **Mutation at scale** is real distributed-systems work; defer until adoption demands it.
- Acceptance is always **shortlist recall vs the exact ADC ranking — never reconstruction
  cosine.**

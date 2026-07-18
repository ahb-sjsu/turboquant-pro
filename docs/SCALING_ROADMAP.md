# Scaling roadmap: from the single-node engine to 10¹²⁺

How turboquant-pro's sharded index scales from one box to a trillion-plus vectors —
and, crucially, **what to reuse rather than rebuild.** The distributed layer is not new
infrastructure: it composes three existing, released projects.

## Where we are (the substrate, validated to 1B)

The **per-node engine** is complete and measured:

- `ShardedIndex` — streaming ingest, memmap search, bounded fds, one shared PCA basis.
- **IVF coarse layer** (`build_ivf` + `search(nprobe=…)`) — sublinear scan (few-% of rows).
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

1. **Per-node engine** — done; 1B validated on block storage.
2. **(c) Scatter-gather search** — *next.* A transport-agnostic coordinator in
   turboquant-pro: a shard-server answers `bytes → partial top-k bytes`; a coordinator
   scatters a query over a `transport` callable and merges partials (`_merge_partials`
   already exists). Testable in-process; the NATS adapter is a thin nats-bursting pool.
3. **Distributed build** — deploy `write_shard` over the nats-bursting work-queue: shard 0
   fits the basis, then N workers build the rest. Build time → `total / n_workers`.
4. **IVF-as-router** — the **1T unlock.** The global coarse quantizer routes each query to
   only the servers holding its `nprobe` cells (subject-based routing), so a query touches
   `nprobe/nlist` of the fleet, not all shards — sublinear *in servers*, mirroring the
   sublinear-in-rows win. Reuses the coarse layer + GPU `build_ivf`.
5. **Operational scale-out** — storage tiering (hot codes on NVMe, cold originals on
   S3/CephFS for a rerank tier), Linstor-HA replication, distributed mutation
   (delete/compact/re-cluster), monitoring.

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
  now) helps, full recall needs a **tiered rerank** — fetch candidate originals from cold
  storage for the shortlist only.
- **Coarse-quantizer quality at 1T.** A flat `nlist` gets coarse; the fix is
  **hierarchical / residual quantization** (IVF-of-IVF), another GPU kmeans layer.
- **Mutation at scale** is real distributed-systems work; defer until adoption demands it.
- Acceptance is always **shortlist recall vs the exact ADC ranking — never reconstruction
  cosine.**

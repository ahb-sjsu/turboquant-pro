# DRAFT — NRP large-experiment request: 1T-row compressed-index validation

*Status: **SENT 2026-07-20** by Andrew Bond to the NRP Matrix channel, as a
single combined post covering both the 1T compute allocation and the public
benchmark bucket. Awaiting an admin reply (the process: you post the request,
an admin replies added or denied). The per-post templates below are retained
for future exception requests, since each exception is handled as its own
post.*

## ⏳ While waiting — likely follow-ups, with answers ready

| if they ask | answer |
|---|---|
| "When do you need it?" | Not immediately. Prerequisites (BIGANN build, baseline comparisons, the shard-open fix) are in flight; we'd rather start when we can use the allocation than hold it idle. Happy to take a scheduled window. |
| "Can you use RBD instead of Linstor?" | Yes. Linstor was chosen for local-NVMe-backed latency; RBD is acceptable and we'll re-measure. We would *not* use CephFS for hot codes — measured 78–105× slower for random-access index reads. |
| "Can you use fewer concurrent jobs?" | Yes — the build parallelizes arbitrarily. 64 × 6-CPU for ~2 days, or e.g. 16 × 6-CPU for ~8 days, whichever suits the scheduler. |
| "How much egress will the bucket generate?" | Unknown until adoption; we'll publish a size cap and can rate-limit or move mirrors off-NRP if egress becomes an issue. The design deliberately keeps a durable copy elsewhere. |
| "Why not just use the existing 64Gi PVCs?" | 1T at 24 B/row is ~26 TB; at 64Gi/PVC that's ~430 volumes, which is worse for the scheduler and for us than ~64 × 400 GB. (The 10B run already had to spread over 8 volumes because of this cap.) |
| "Is this GPU work?" | No GPUs at all. Pure CPU batch. |

---

## 💬 Matrix posts (the actual process)

Exceptions are requested by posting in the NRP Matrix channel; an admin
replies that it has been added or denied. Keep each post short, one coherent
ask, concrete numbers, and a link for anyone who wants the detail. **Post them
separately** — a small ask answered "added" is worth more than a large one
that stalls, and they are needed at different times.

### Post A — storage/hosting exception (send when Phase 1 is ready to publish)

> Hi — requesting a storage exception for namespace `ssu-atlas-ai`.
>
> We'd like to host a public, citable benchmark dataset for billion-scale
> vector search in an NRP S3 bucket — ~1–2 TB for the first phase. It's
> distributed as a signed manifest + hashes rather than bulk data, so users
> fetch only the shards they need; that's what makes evaluation possible for
> groups that can't stage 100+ TB locally. Repo:
> github.com/ahb-sjsu/openvector-bench
>
> Three asks:
> 1. Can a bucket that's actively served to external users be exempted from
>    the 6-month inactivity reclamation? You previously added an exception for
>    our `atlas-nats-leaf` Deployment in this namespace (thanks — that's the
>    link our burst jobs run over); if there's an equivalent lever for storage
>    we'd rather use your existing mechanism than ask for something bespoke.
> 2. Can we rely on a stable path/URL? Published manifests and a DOI record
>    will reference it, and a path change silently breaks third-party
>    reproduction.
> 3. What's the S3 quota expectation, and is external egress from a public
>    benchmark bucket acceptable use?
>
> On our side: we'll keep a durable copy off-NRP (Zenodo DOI + institutional
> mirror) so NRP is never a single point of failure, publish a hard size cap,
> prune superseded versions, remove on request, and credit NRP in the paper
> and the dataset record.

### Post B — 1T large-experiment allocation (send when ready to run it)

> Hi — requesting approval for a large experiment in `ssu-atlas-ai`, per the
> policy that large runs need written approval.
>
> Background: we've completed 1B and 10B-row distributed vector-index builds
> and measurements here. The 10B run was 8 shard-ranges × 1.25B on Linstor,
> built as two waves of 4 CPU-pegged batch jobs, 24 B/row, routed-IVF recall
> 0.9988 vs an exact full-scan reference. Harness and results are public:
> github.com/ahb-sjsu/turboquant-pro
>
> We'd like to extend it to **1 trillion rows**. From measured constants
> (24 B/row, ~1.35M rows/min/CPU):
> - ~26 TB block storage (Linstor or RBD) as ~64 volumes of ~400 GB — this
>   needs the per-PVC cap lifted for this namespace (currently 64Gi on
>   `linstor-ha`), which is the main exception we're asking for;
> - ~12,500 CPU-hours, as ≤64 concurrent 6-CPU batch jobs over ~2 days, or
>   fewer jobs over longer if that's easier on the scheduler;
> - ~1 TB CephFS for manifests/results; no GPUs;
> - ~1 week end-to-end, compute storage released afterwards.
>
> It's all batch, CPU-saturated, requests==limits. We restructured our harness
> after utilization enforcement flagged some idle serving pods earlier —
> serving now happens only in short windows with exempt-class pods.
>
> Happy to reshape the concurrency, storage class, or scheduling window to
> whatever suits the cluster. Full detail if useful: <link to this file>

**Sequencing note.** Post A is small, unblocks the benchmark work, and builds
the record for Post B. Post B should wait until the remaining prerequisites
are done (baselines, the shard-open fix, BIGANN) — asking for 12,500 CPU-hours
before we can use them wastes the goodwill.

---

## Who / what

**Namespace:** `ssu-atlas-ai` (Andrew H. Bond, San José State University,
andrew.bond@sjsu.edu). **Project:** [turboquant-pro]
(https://github.com/ahb-sjsu/turboquant-pro) — an open-source (MIT)
compressed vector index (bit-packed ADC codes + IVF + distributed
scatter-gather). Goal of this experiment: validate the index's published
scaling claims at **one trillion rows**, producing an open, reproducible
benchmark (harness and results are public in that repo, and the benchmark artifacts in
github.com/ahb-sjsu/openvector-bench;
prior runs at 50M/1B on Nautilus are already published in
`benchmarks/RESULTS_ivf.md`).

## What we have already validated on Nautilus (within standing limits)

- **1B-row fleet run (2026-07-19, complete):** distributed build (4 jobs ×
  250M rows, ~31 min wall) onto per-server Linstor volumes at a measured
  **24.01 B/row**; served by 4 NATS shard-servers; routed IVF recall 0.999
  vs the exact distributed full-scan reference; a cold-tier exact rerank
  restoring true fp32 recall from 0.592 to 0.991.
- **10B-row run (2026-07-20, complete):** 8 shard-ranges × 1.25B on 8 × 40Gi
  Linstor volumes (the 64Gi per-PVC admission limit is what forced the layout
  into 8 volumes), built as two waves of 4 CPU-pegged batch jobs. Routed IVF
  recall **0.9988** at nprobe=128 (0.982 at 32) against the exact ADC
  full-scan reference, 5.1× faster than full scan, at 24.01 B/row.
- Compliance posture: batch jobs are CPU-saturated start-to-finish with
  requests==limits; serving pods run only in short measurement windows and
  are sized in the exempt class (cpu≤1/mem≤2Gi); no idle pods; TTL cleanup
  on all jobs. (Early serving pods were correctly flagged by utilization
  enforcement; the harness was redesigned around it and the lessons are
  documented in the repo.)

## The 1T request

Using the measured constants (24.01 B/row at 4-bit; build throughput
~1.35M rows/min per CPU):

| resource | ask | basis |
|---|---|---|
| **Block storage (Linstor or RBD)** | **~26 TB** total, as ~64 volumes × ~400 GB (well under the 10 TB/volume platform ceiling; current namespace admission cap is 64Gi/PVC — this is the core exception requested) | 1T × 24 B/row + 10% headroom |
| **CPU** | **~12,500 CPU-hours**, shaped as up to **64 concurrent 6-CPU batch jobs** (384 CPUs) for ~2 days of build, else fewer jobs over proportionally longer | measured build rate |
| **CephFS** | 1 TB (bootstrap artifacts, manifests, results; no fp32 cold store at 1T — acceptance is recall vs the exact distributed ADC reference) | |
| **Duration** | ~1 week end-to-end: ~2 days build, short serving windows for measurement (≤8 exempt-class pods, minutes each), teardown | |
| **GPUs** | none | |

Deliverables back to the community: the complete harness + results published
in the open repo; a writeup of Nautilus as the platform for
trillion-row open-source vector-index benchmarking; storage released
immediately after the measurement windows (nothing archival — consistent
with the 6-month purge policy).

We are happy to adjust shape (fewer concurrent jobs over longer, different
storage class, scheduling windows) to whatever is least disruptive.

## Second ask: hosting a public reference benchmark (storage-policy exception)

Separately from the compute above, we would like to host a **public,
citable reference corpus** for billion-to-trillion-scale ANN research in an
NRP S3 bucket, and we are asking about two policy points.

**What it is.** A benchmark corpus defined by a signed Merkle manifest rather
than by distributed bytes: researchers fetch a small manifest, then either
regenerate shards deterministically or fetch only the shards they need, with
every byte hash-verified. This makes evaluation possible at scales where the
corpus itself (128 TB at 10¹²) can never be downloaded. Ground truth for a
fixed query set — the expensive artifact, and the one that unlocks the
scale for everyone else — would be published alongside it.

**Why NRP.** The measurements that motivate it were made here, the multi-region
S3 pools are well matched to a locator-based fetch design, and NRP would be
credited as the hosting infrastructure in the paper and in the dataset record.
We expect the primary consumers to be academic groups that cannot stage
billion-scale corpora locally — which is the same gap NRP exists to close.

**The two asks.**
1. **Exemption from inactivity-based reclamation** for one designated public
   benchmark bucket. We understand the 6-month purge policy targets abandoned
   volumes; this bucket would be actively served to external users, and we are
   asking for it to be classified accordingly (or for guidance on the right
   mechanism).
2. **A stable path/URL commitment** for that bucket, since published manifests
   and any DOI record will reference those URLs; a path change silently breaks
   third-party reproduction.

**What we commit to in return.** A durable copy independent of NRP (Zenodo
DOI plus institutional mirror), so NRP is never the single point of failure;
a published size budget with hard caps and pruning of superseded versions;
prompt removal on request; and acknowledgement of NRP in all resulting
publications and in the dataset record.

**Scale of this ask (separate from the compute request):** Phase 1 is ~1–2 TB
(a 10⁷–10⁹-row corpus plus ground truth and manifests). We would return with
numbers before growing it, rather than expanding silently.

## Open questions for NRP

1. Preferred mechanism for the per-PVC admission-limit exception (64Gi →
   ~400 GB × 64 volumes) — namespace label, storage-class grant, or a
   different class (RBD)?
2. Is 64 × 6-CPU concurrent batch acceptable under the utilization policy
   (all pods CPU-pegged, requests==limits), or should we cap lower?
3. Any preferred nodes/regions for sustained CephFS + Linstor throughput
   (we measured large node-to-node variance in CephFS client bandwidth)?
4. S3 quota per namespace, and whether a multi-TB **public** benchmark bucket
   is acceptable use — including expectations about external egress if other
   institutions or mirrors pull from it.
5. Is there an existing mechanism for "actively-served public dataset" that we
   should be using instead of requesting an exception?

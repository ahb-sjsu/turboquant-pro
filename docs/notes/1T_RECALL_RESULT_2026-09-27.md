# The 1T recall measurement: 10^12 rows, 500 servers, 2026-09-27

Read after `1T_BUILD_COMPLETE_2026-09-24.md`, which this closes. The build gave 500 servers
x 400 shards x 5M rows of 4-bit ADC index with an IVF layer, 24.0 TB in 500 Linstor volumes. This
note records the recall measurement over that index, how it was obtained, what it cost, and what
it does and does not show.

## Result

Routed IVF recall@10 against the exact ADC full scan, 100 queries (the seeded 1B query set,
4 shards x 25), scored by `benchmarks/fleet/fleet_score10.py` on the exact merge of 500 per-server
partial top-10s (shared basis, so scores are comparable and the merge is exact):

| corpus | nprobe 32 | nprobe 128 |
|---|---|---|
| 1B (20M rows, 2 servers, earlier record) | 0.992 | 1.000 |
| 100B (2026-07-28) | 0.9834 | 0.9986 |
| 698B (interim, first 349 servers, 2026-09-27 00:25Z) | 0.985 | 0.999 |
| **1T (500 servers, 2026-09-27 08:59Z)** | **0.989** | **0.999** |

Recall against the full scan is flat from 20M to 10^12 rows: the IVF routing loses nothing that the
exact scan finds, at either probe width, three orders of magnitude apart. The interim number on
the first 349 servers matched the final one, and recall at nprobe 128 on the first half of those
349 was also 0.999, so the figure is stable inside the run as well as across scales.

Per-server wall time, one CPU per job (see "shape" below):

| phase | median | mean | p90 | min | max | CPU-hours |
|---|---|---|---|---|---|---|
| reference full scan, 2e9 rows | 2664 s | 3008 s | 4013 s | 1062 s | 16308 s | 418 |
| IVF nprobe 32 | 1064 s | 1131 s | | 429 s | 14286 s | 157 |
| IVF nprobe 128 | 1335 s | 1420 s | | 562 s | 14524 s | 197 |

The per-server "speedup" of IVF over the full scan that the score script prints (2.7x and 2.1x on
means) is **not** a latency figure and is not comparable with the 100B run's 14.7x and 8.5x: these
jobs ran one CPU each in the enforcement-exempt class, read-bound on Linstor, with a fixed
per-job cost (image pull, pip, clone, 400 shard opens) that dominates the IVF pass. The number
that this run establishes is recall; the latency story belongs to the serving-window measurements.

Full record: `benchmarks/fleet/record/1t/post/score_1T.log` (RESULT_JSON with every per-server
wall time), `post_state.json`, both driver logs, the shepherd log, the three reference pilots, the
IVF pilot and the interim score.

## Shape: everything ran in NRP's enforcement-exempt class

The first attempt (2026-09-24 20:41Z to 2026-09-25 10:10Z) ran the reference scan at 6 CPU /
8 GiB per job. That is an NRP utilization violation (metered at 8 % CPU / 3 % memory on one pod,
and a 1.5 to 7.5 GiB memory swing on another that the right-sizer scores as "no legal request"),
and the run was stopped at 153/500 with the driver's stop marker. Three changes put the job inside
the exempt envelope (1 CPU, 2 GiB), each one found by a 2 GiB pilot that died without it, in this
order:

1. **Queries from the cache, not regenerated.** `fleet_ref.py` called `queries()`, which
   generates four whole 5M-row blocks (about 640 MB each plus temporaries). OOM-killed 85 s after
   start. It now reads the file the qcache phase wrote (`TQP_QCACHE_NAME`), as `fleet_ivf.py`
   already did.
2. **At most 2 shards open.** A memory-mapped shard still materializes its row-id and tombstone
   arrays, 45 MB per 5M-row shard, and `ShardedIndex` keeps 128 open by default: 5.8 GB across a
   full scan. OOM-killed after 17 minutes, around the 45th shard, and this is the 7.5 GB peak the
   right-sizer had seen on the 8 GiB jobs. `TQP_REF_OPEN_SHARDS=2` (IVF: 8, with 2 workers).
3. **65536-row scan blocks** (`TQP_REF_BLOCK`), so per-block temporaries are tens of MB.

Pilot `aqx-ref1t-499` with all three: 1912 s wall, CPU mean 884m peak 979m, memory mean 1242Mi
peak 1469Mi trough 1051Mi, process RSS 346 MB. Pilot `aqx-ivf1t-0`: 36 min, CPU mean 475m,
memory mean 1258Mi peak 1568Mi, RSS 165 MB; the memory above the process is clean posting-list
page cache, which the cgroup reclaims (anon 224 MB / file 1.9 GB observed at the limit, no OOM).
The score job at 2 CPU / 4 GiB was never created by the controller (held from 08:41Z); the same
job at 1 CPU / 2 GiB was created within seconds and finished in minutes. Commit `8f25bc4` carries
the two scripts and the driver; the score descriptor follows in this note's commit.

## How the pool behaved

The resumed run (2026-09-25 18:01Z to 2026-09-27 08:59Z) submitted 928 jobs for 849 completions:
348 reference scans (152 were done before the stop), 500 IVF passes, one score. 101 recycles,
0 held after the first minutes, 0 gave up, 0 parked, no driver restart. Throughput was 20 wide
(the burst controller's `max_concurrent_jobs`), about 27 reference servers an hour and 28 IVF
servers an hour, so a day and a half in all.

Two failure classes, both benign:

- **One host kills every job placed on it.** `ry-gpu-04.sdsc.optiputer.net` gives SIGBUS
  (exit 135) on the memory-mapped index read within 3 to 13 minutes, every placement, after
  `AttachVolume` succeeds. 24 of the 75 "job failed" recycles are exit 135; the rest are exit 255
  (node went away, 13), sandbox creation failures (2), and pods whose node's kubelet became
  unreachable (13, `exit=None`). Retries landed elsewhere and finished. The cost of the bad host
  was about 130 slot-minutes in the first seven hours.
- **Pending past 45 minutes** (26 recycles): the driver deletes and re-issues; in every case seen
  the original pod had run anyway and the retry found its output within minutes.

Three reference pods got stuck `Terminating` on nodes whose kubelet became unreachable
(`fiona8.ucsc.edu` twice, `k8s-stratix-10-02.sdsc.optiputer.net`); the driver's wedge rule is 16 h,
too slow for that case, so they were force-removed by hand and the job controller replaced them.
Server 495's IVF pod was likewise moved by hand off `ry-gpu-10.sdsc.optiputer.net`, a slow-storage
host that had also held the 4.5 h reference outlier. A "Terminating longer than 30 min" rule
belongs in the driver before the next run.

The first attempt also exposed a driver defect: the burst controller defers Job creation while five
of the namespace's Jobs are pending (other campaigns' GPU jobs tripped it from 03:30Z), and the
driver treated two polls without a Job object as "vanished" and re-issued, queueing a duplicate
behind every deferred original (234 phantom recycles, throughput cut to 3 servers an hour). Fixed
in `de1b7e6`: a submitted Job is waited on for 90 polls before it is treated as lost.

## Provenance

- Index: built 2026-08-04 to 2026-09-24, `1T_BUILD_COMPLETE_2026-09-24.md`.
- Queries: `queries1t.npy` from the qcache phase (2026-09-24 20:41Z), the first 25 rows of the
  seeded shards 0, 50000, 100000, 150000.
- Job containers clone `master` HEAD at run time, so the code that scanned changed as master
  moved. Commits seen in pod logs during this measurement: `c434a99`, `de1b7e6`, `6c314c1`,
  `8f25bc4`, `24ef5fe`, `b106f25`, `821c7bd`, `d0cb2ad`, `13bd727`, `507b902`, `0b5bd07`,
  `c9f1fbe`. The fleet scripts themselves come from the `tqp-fleet-code` ConfigMap, replaced only
  for `fleet_ref.py` (md5 `476a8b9c`) and `fleet_ivf.py` (`28fe069d`) at the resume; the merge
  and recall code (`fleet_score10.py`, `fleet_common.py`) did not change. Pinning the clone to a
  commit belongs in the descriptor before the next run.
- Announcement on the Nautilus Matrix channel before launch
  (`NRP_1T_MEASUREMENT_ANNOUNCEMENT.txt`), acknowledged by an NRP admin.

## What this shows, and what it does not

- Recall of the routed index against the exact scan of the same index is 0.989 (nprobe 32) and
  0.999 (nprobe 128) at 10^12 rows, unchanged from 20M and 100B rows. That is the claim.
- It is a self-consistency measure on the `gen_block` corpus (rank-16 recipe, `FLEET_CORPUS_GEOMETRY.md`),
  which exists to exercise scaling mechanics, not to stand in for real embeddings. Recall on real
  data is a different number (the 15M Cohere pilot: 0.689 ADC-only, 0.981 reranked) and the paper
  must keep the two apart.
- 100 queries. The 100B run used 500; the reference scan is compute-bound in the query count and
  500 would have cost about five times the 418 CPU-hours.

## Next

- The 500 volumes (`tqp-fleet-1t-0..499`, 27.3 TiB) stay bound until the owner says release.
- The measurement is the last number the PVLDB-targeted writeup needs; acknowledgement of NRP in
  the paper is a standing commitment (`NRP_SCALE_REQUEST.md`).

## Follow-up on the built index, 2026-09-27 16:32Z to 21:29Z (no new distance scan)

Three exempt-class jobs read the partials and the index metadata (commit 3660828 and 38198f5;
records `analysis1t_partials.log`, `analysis1t_cells.log`, `driver1t_analysis.log` under
`benchmarks/fleet/record/1t/post/`).

**Per-query recall** (analysis job, 16:34Z): at 32 probes 90 of 100 queries at 1.0, minimum 0.8,
one query below 0.9; at 128 probes 99 of 100 at 1.0, minimum 0.9. The mean 0.999 is one query
missing one neighbour.

**Where the reference neighbours live.** The 1000 reference top-10 slots sit on 233 of the 500
servers and one server holds 16.6 percent of them, because the 100 queries are rows of four seeded
shards and each shard's own basis keeps a query's neighbours near its home shard. Losing one
random server costs 0.002 of recall in expectation; losing that one server would cost 0.166. The
query set is therefore a property of four shards, and a query set drawn from more shards is the
fix, at the cost of a proportionally longer reference scan.

**Neighbour occurrence.** 1000 slots, 1000 distinct rows, no row appears twice. The statistic is
uninformative at 100 queries and is reported as such.

**Cell census** (500 metadata jobs, 31.6 s median each, plus a merge, 21:28Z). The 2048 cells
hold 49.2 million to 2.15 billion rows, mean 488 million, median 418 million, Gini 0.328, the
largest one percent of cells holding 3.7 percent of rows, no empty cell. The imbalance is the
shared coarse quantizer's, fitted once on the bootstrap shard.

**Reachability, which is recall predicted from cell ranks alone.** Every reference neighbour's
cell has a rank in its query's probe order, and recall at width p is exactly the share of
neighbours with rank below p, because scoring inside a probed cell is the same asymmetric
distance the reference used. The merge's prediction against the measured widths:

| width | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| predicted | 0.406 | 0.599 | 0.727 | 0.867 | 0.952 | 0.989 | 0.997 | 0.999 | 1.000 |
| measured | | | | | | 0.989 | | 0.999 | |

Both measured widths match the prediction to the last digit. The neighbours' cell ranks have
median 1, p90 9, p99 32 and maximum 169, so the single neighbour that 128 probes misses sits in
the 170th cell of its query's order, and 256 probes should return every neighbour of every query.
The prediction for 16, 64 and 256 was recorded at 21:39Z, before any probe-sweep partial existed
(the sweep's first jobs were submitted at 21:37Z and take about 40 minutes each); the sweep's
score will be set beside it here.


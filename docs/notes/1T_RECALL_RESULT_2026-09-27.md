# The 1T recall measurement: 10^12 rows, 500 servers, 2026-09-27

> **For the paper session (2026-09-29).** The fleet session no longer edits `paper/pvldb1t`.
> Results after commit 9ecdfb8 (rerank bound, Table 5) land in this note and in
> `benchmarks/fleet/record/1t/post/` only: the nested-scale section below (recall versus N inside
> one index, flat at 128 and 256 probes, rising with N at 16 and 32) and, when it lands, the
> non-member query run (tag `1tnm`, `score_1Tnm.log`, per-volume sha256 in `hash1tnm_part_*.json`).
> Before every paper commit run the tone grep from `SUBMISSION-CHECKLIST.md`.

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
| measured | | | | | 0.952 | 0.989 | 0.997 | 0.999 | 1.000 |

Both measured widths match the prediction to the last digit. The neighbours' cell ranks have
median 1, p90 9, p99 32 and maximum 169, so the single neighbour that 128 probes misses sits in
the 170th cell of its query's order, and 256 probes should return every neighbour of every query.
The prediction for 16, 64 and 256 was recorded at 21:39Z, before any probe-sweep partial existed
(the sweep's first jobs were submitted at 21:37Z and take about 40 minutes each). The sweep's
score, 2026-09-29 12:55Z, is the measured row: 0.952, 0.997 and 1.000, each equal to the
prediction to the last digit. Five widths are now measured and all five agree with the count
of neighbours whose cell rank is below the width, so recall of this index is a property of the
coarse quantizer's probe order alone, and any further width can be read from the cell ranks
without a scan.

## Probe sweep, 2026-09-27 21:37Z to 2026-09-29 12:56Z

Routed passes at nprobe 16, 64 and 256 on every server, one exempt-class job per server
(1 CPU, 2 GiB, `fleet_ivf.py` with `TQP_NPROBES=16,64,256`, 8 open shards, 2 workers), then
one score job. Result `probe_1T.log` (RESULT_JSON with every per-server wall time), driver log
`driver1t_probe.log`, both under `benchmarks/fleet/record/1t/post/`.

| width | recall@10 vs exact scan | median | mean | p90 | min | max | CPU-hours |
|---|---|---|---|---|---|---|---|
| 16 | 0.952 | 916 s | 1071 s | 1440 s | 395 s | 12236 s | 149 |
| 64 | 0.997 | 1219 s | 1366 s | 1743 s | 505 s | 14645 s | 190 |
| 256 | 1.000 | 1594 s | 1719 s | 2199 s | 578 s | 13786 s | 239 |

The wall times are per pass inside one job, so a server's three passes share one image pull,
clone and 400 shard opens; the per-server total was about 40 minutes at the median. The three
maxima are one server whose job landed on the slow-storage host already seen in the main run,
and it alone held the score for the last seven hours of the sweep.

Pool: 574 submissions for 500 completions and one score, 73 recycles (63 servers needed a second
try, 9 a third, 1 a fourth), 0 gave up, 0 parked, no driver restart, 20 wide, about 31 servers an
hour once the pool was full. Recycle causes: 25 SIGBUS on the memory-mapped read (exit 135, the
same host as the main run), 17 node lost (exit 255), 11 kubelet unreachable (`exit=None`),
4 exit 1 from one storage fault at 01:05Z on 2026-09-28 that cleared on retry, 1 sandbox start
error, and the rest pods that failed after their node's volume mount did. One pod sat in
`ContainerCreating` for 17 hours because its node could not mount the server's volume (bad
superblock on that node only); the pod was removed by hand, the Job failed and the driver
re-issued it, and the retry finished elsewhere in 40 minutes. The storage controller was down
from 09:01Z to about 11:30Z on 2026-09-28 and the pool waited it out with no losses. A pod stuck
`ContainerCreating` past an hour should be replaced by the driver the way the 30-minute
`Terminating` rule already does; that is the one rule this sweep adds to the list.

## Rerank bound on regenerated floats, 2026-09-29 13:27Z to 15:52Z (no new scan)

Item 5 of the follow-up plan. Wide-100 shortlists per query from the merge of the 500 per-server
top-10 partials (`fleet_rerank_prep.py`, one job), the float rows they name regenerated from the
corpus seeds in 100 exempt-class slices with no index volume (`fleet_rerank_gen.py`, about 57
shards and ten minutes each away from one slow node), then one score job (`fleet_rerank_score.py`)
that reranks every shortlist to ten by cosine, the metric `fleet_gt.py` used for ground truth at
1B. Record `rerank_1T.log` and `driver1t_rerank.log`; result JSON
`/shared/fleet/results/rerank1t_bound.json`. Commit 39e6075 (code), the two set lemmas the
analysis rests on are checked in Lean under `paper/pvldb1t/lean/`.

Regenerated rows: 11002 in 5662 shards. Survival = share of the ADC
top-10 in the float top-10 of the same shortlist, an upper bound on ADC-only true recall
(a corpus float neighbour inside the shortlist is a shortlist float neighbour). Transfer = share
of the routed shortlist's float top-10 in the reference shortlist's float top-10.

| shortlist | ref | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| survival of the ADC top-10 (mean) | 0.621 | 0.623 | 0.623 | 0.624 | 0.622 | 0.621 |
| survival, minimum over queries | 0.100 | 0.100 | 0.100 | 0.100 | 0.100 | 0.100 |
| queries keeping all ten | 22 | 23 | 22 | 22 | 22 | 22 |
| transfer after rerank (mean) | -- | 0.940 | 0.982 | 0.996 | 1.000 | 1.000 |
| transfer, minimum | -- | 0.500 | 0.700 | 0.900 | 1.000 | 1.000 |
| mean cosine, ADC top-10 | 0.863 | 0.863 | 0.862 | 0.863 | 0.863 | 0.863 |
| mean cosine, reranked top-10 | 0.929 | 0.929 | 0.929 | 0.929 | 0.929 | 0.929 |
| query's own row in ADC top-10 / reranked top-10 | 100/100 | 100/100 | 100/100 | 100/100 | 100/100 | 100/100 |

None of this is true recall. The float top-10 over all 10^12 rows is unknown here (no cold store
at 1T); the 1B run measured it from a cold store (0.592 ADC-only, 0.991 reranked) and the two
must stay apart. The shortlist is the top-100 of the per-server top-10 merge, not the exact ADC
top-100, because a server may hold more than ten of those.

## Nested scale inside one index, 2026-09-29 16:56Z (no new scan)

The per-server partials hold the exact top-10 of the reference scan and of every routed pass
for each query on each server, scored on the shared basis, so the exact top-10 over any subset
of servers is the merge of that subset's partials and recall of routing against the exact scan
of the subset is exact. A subset of k servers is a corpus of 2k billion rows built from the same
law with the same basis, coarse quantizer, router and queries, so along k only N changes. One
exempt-class job, `fleet_nested.py` (commit 4024367, descriptor fix ddf014b), 26 seconds; record
`nested_1T.log`, `driver1t_nested.log`; result `/shared/fleet/results/nested1t.json`. Prefix
subsets are servers 0 to k-1; random subsets are five seeded draws of k servers (seed 1234).
The home servers of the four query shards are 0, 125, 250 and 375; the prefix holds one of
them until k reaches 126, all four at 500, and the random subsets held at most one.

| servers | rows | prefix 16 | 32 | 64 | 128 | 256 | random mean 16 | 32 | 64 | 128 | 256 | random range at 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2e9 | 0.827 | 0.918 | 0.965 | 0.989 | 0.999 | 0.797 | 0.905 | 0.965 | 0.991 | 0.999 | 0.782 to 0.811 |
| 2 | 4e9 | 0.831 | 0.922 | 0.971 | 0.994 | 1.000 | 0.819 | 0.921 | 0.971 | 0.992 | 0.999 | 0.812 to 0.826 |
| 3 | 6e9 | 0.833 | 0.924 | 0.975 | 0.993 | 1.000 | 0.814 | 0.917 | 0.972 | 0.994 | 0.999 | 0.802 to 0.831 |
| 5 | 1e10 | 0.838 | 0.923 | 0.969 | 0.990 | 1.000 | 0.846 | 0.931 | 0.979 | 0.996 | 1.000 | 0.836 to 0.855 |
| 10 | 2e10 | 0.868 | 0.939 | 0.980 | 0.994 | 1.000 | 0.861 | 0.943 | 0.984 | 0.997 | 1.000 | 0.836 to 0.880 |
| 20 | 4e10 | 0.871 | 0.945 | 0.983 | 0.997 | 1.000 | 0.881 | 0.953 | 0.986 | 0.998 | 1.000 | 0.872 to 0.889 |
| 50 | 1e11 | 0.884 | 0.959 | 0.988 | 0.998 | 1.000 | 0.906 | 0.970 | 0.990 | 0.998 | 1.000 | 0.880 to 0.935 |
| 100 | 2e11 | 0.910 | 0.973 | 0.989 | 0.999 | 0.999 | 0.913 | 0.971 | 0.993 | 0.999 | 1.000 | 0.896 to 0.922 |
| 200 | 4e11 | 0.926 | 0.977 | 0.991 | 0.999 | 0.999 | 0.942 | 0.984 | 0.996 | 1.000 | 1.000 | 0.918 to 0.953 |
| 300 | 6e11 | 0.945 | 0.985 | 0.995 | 0.999 | 0.999 | 0.948 | 0.988 | 0.997 | 0.999 | 1.000 | 0.936 to 0.956 |
| 400 | 8e11 | 0.955 | 0.989 | 0.996 | 0.999 | 1.000 | 0.944 | 0.987 | 0.997 | 0.999 | 1.000 | 0.936 to 0.951 |
| 500 | 1e12 | 0.952 | 0.989 | 0.997 | 0.999 | 1.000 |  |  |  |  |  | |

At 128 and 256 probes recall is flat from 2e9 to 1e12 rows (0.989 to 0.999
and 0.999 to 1.000). At narrow widths it rises with N inside the one index:
16 probes from 0.827 at two billion rows to 0.952 at a trillion, 32 probes from
0.918 to 0.989, with nothing changed but N. The random subsets agree with the
prefix within the range shown, so the effect is not the home servers. The cell-rank picture
explains the direction: as the corpus grows a query's ten nearest rows get nearer, and nearer
rows sit in earlier cells of the probe order, so a fixed narrow width reaches more of them. The
slow rise at 32 probes across the separate 1e8, 1e10, 1e11 and 1e12 runs (0.979, 0.982, 0.983,
0.989) is the same effect seen with the corpus confound removed. For the paper: 'recall does
not move' is the statement at 128 probes and above; at 16 and 32 probes recall improves with N.

## Archive of the shared volume, 2026-09-29 17:33Z

Everything the measurement produced off the index volumes (bootstrap with basis, coarse quantizer
and manifest; query caches; every per-server partial of every phase; shortlists and regenerated
rows; every result JSON) is packed by `fleet_archive.py` into one archive with a sha256 manifest
of each member: 3814 files, 182,240,843 bytes in, 147,442,900 bytes out, archive sha256
`02f27a4cd734af4cb224747bb812845537ad15e12bac9c2e8ee38b9372d86d75`. It lives on the shared
volume under `archive/` and, verified by checksum after the copy, on Atlas at
`/archive/experiments/tqp-fleet-1t/shared-fleet-20260929.tar.gz` with its manifest beside it. The
record is therefore self-contained without the cluster; only the index itself (the 500 volumes)
is not copied, and it is a function of the seeds and the code.

## Non-member queries, 2026-09-29 17:45Z to 2026-10-02 04:31Z (run tag `1tnm`)

Every query of the main measurement is a corpus row, so each has an exact match in the index and
its neighbours sit near its home shard. This run repeats the measurement with 100 queries drawn
from the same generator under seeds no corpus shard uses (shards 200000, 250000, 300000 and
350000, 25 rows each, `queries1tnm.npy`, sha256 `7d9e4a48152da1ee...`), so no query is in the
index and none has a home server. Same index, same 500 servers, same reference scan and routed
passes at 32 and 128 probes, same exact merge. Record `score_1Tnm.log` (RESULT_JSON with every
per-server wall time) and `driver1tnm.log`.

| queries | nprobe 32 | nprobe 128 |
|---|---|---|
| corpus rows (2026-09-27) | 0.989 | 0.999 |
| not in the index (2026-10-02) | 0.969 | 0.999 |

At 128 probes recall against the exact scan is 0.999 for queries the index has never seen,
the same figure as for corpus rows. At 32 probes it is lower by 0.02, which is the home-shard
advantage the corpus-row queries had (their reference neighbours lie in the cells their own
shard's basis favours, and the shared coarse quantizer reaches those cells early). That is the
number to quote for the standard queries-not-in-database protocol.

**Two registered predictions preceded this score.** The paper session registered a scale-transfer
model fitted on 20 calibration servers (`docs/PREREG_scale_transfer.md`, script sha256
`81ec880e...`) and a second one fitted on corpora of 1e8 to 1e9 rows rebuilt from the seeds
(`docs/PREREG_scale_transfer_small.md`, commit 108ed8e). The fleet session ran the first on
exactly the 60 calibration partials copied to Atlas and committed its output
(`scale_transfer_predict_1tnm.json`, commit b859fba, with `scale_transfer_calib_1tnm.SHA256SUMS`)
before fetching or reading the score; the score job had run on the cluster by then but its output
stayed unread. Grading of both predictions is the paper session's step, from the copy of all 1500
partials on Atlas (`/archive/experiments/tqp-fleet-1t/scale_transfer/1tnm-all/`, SHA256SUMS over
1997 files).

**Per-server wall time, one CPU per job.** The cluster was slower than during the main run; the
scan work does not depend on the query set.

| phase | median | mean | p90 | max | CPU-hours |
|---|---|---|---|---|---|
| reference full scan + checksum | 2793 s | 3143 s | 4630 s | 8596 s | 437 |
| routed, 32 probes | 1057 s | 1325 s | 1602 s | 9524 s | 184 |
| routed, 128 probes | 1408 s | 1538 s | 1948 s | 7439 s | 214 |

**Index checksums.** Each reference job also wrote a sha256 of every file on its volume
(`hash1tnm_part_S.json`, 1201 files a server, about 65 s). 495 of 500 exist; servers 182, 377,
421, 429 and 484 lack one because their retry found the partial already written and skipped the
scan, and the checksum pass with it. A hash-only pass over those five (`driver1tnm_hash.log`,
2026-10-02 04:43Z to 05:14Z, `fleet_ref.py` now writes the manifest in that path too, 5114c17)
completed the fingerprint: 500 of 500 manifests, 1201 files a server, copied to Atlas beside the
partials with a SHA256SUMS over 2002 files. The 500 manifests cover 600,500 files and
24,004,293,071,106 bytes; the repository record carries their own digests
(`hash1tnm_manifests.SHA256SUMS`, one line a server). The manifests themselves (73 MB) are published
outside git as one asset of the data release `data-1t-hash-manifests-2026-10-02`
(`hash1tnm_manifests.tar.gz`, sha256 `cea55f0e...`; how to verify is in
`record/1t/post/hash1tnm_manifests.README.md`), with a copy on Atlas. The index is verifiable
byte for byte against a rebuild from the seeds, and nothing further needs the 500 volumes.

**Dates of the index.** Volumes provisioned from 2026-08-04; the index complete on all 500 servers
at 2026-09-24 20:05Z; every measurement in this note taken between 2026-09-24 and 2026-10-02;
release of the 500 volumes begun 2026-10-02 07:18Z on the owner's word. The shared record volume
and the Atlas copies remain.

**Grading of the two registered predictions** (paper session, master 85d54af, note
`docs/notes/SCALE_TRANSFER_RESULT_2026-10-02.md`): the 20-server model predicted 0.953 at 32 probes
against 0.969 measured (1.6 standard errors) and 0.994 at 128 against 0.999 (1.7), passing all
three registered tests; the small-corpus model predicted 1.000 at 32 probes, 4.4 standard errors
above the measurement, failing its first test, and passed the other two. The graders reproduced
the score from the copied partials.

**Pool.** 1075 submissions for 1000 server completions plus the query cache and the score, 73
recycles (62 servers needed a second try, 11 a third), 0 gave up, 20 wide. Recycle causes: 15 lost
nodes, 10 memory-mapped read faults on the known host, 4 kubelet unreachable, 3 transient I/O
errors, 2 sandbox start errors, 1 memory kill (the first query-cache attempt, fixed in 5356afe),
and the rest pods replaced by hand off two nodes whose storage path ran three to nine times
slower than the median (a 32-probe pass of 2.6 h against 18 min). A cluster-wide disruption at
23:30Z on 2026-09-30 (seven nodes lost at once, name resolution failing for new pods, storage
warnings) emptied the pool for about an hour; the driver's back-off rode it out. One Job delete
did not cascade during that disruption and its Failed pod blocked the server's re-issue for four
hours under the old rule; the driver now ignores terminated pods (3a0ce76). One submission was
accepted by the controller and never created, which the 90-poll held rule caught.

# Record of the regenerated-index run (tag `1trg`)

This folder holds the record of the run in Section `sec:regen` of `paper/pvldb1t/main.tex`.
That run poses 200 queries, one from each of 200 shards: 100 non-members, drawn from shards
400000 to 400099, and 100 corpus rows, one from each of 100 servers. The queries run against
the trillion-row index, regenerated shard by shard from the seed. Every server's shards are
rebuilt, and each shard is checked against the sha256 the first run recorded and against the
released float32 scores. The shard is then searched exactly and with routing at 32 and 128
probes, and discarded.

A shard counts as **reproduced** when both of its cell files hash equal to the first run's
record and every released neighbour it holds comes back with its released float32 score bit for
bit. Shards that no build reproduced are left out of the comparison and counted. Shards that
reproduce under both a whole write and a streamed write are included.

The code is `benchmarks/fleet/fleet_qcache_regen.py`, `fleet_regen_score.py` and
`fleet_regen_merge.py`. The jobs were submitted by `driver1t_post.py` through nats-bursting. The final merge ran the
code at commit `eade71c` (cluster code map `tqp-fleet-code-1trgfull9`).

## Result

`analysis1trg_final.json` is the merged result, produced by the final merge of 2026-11-20 (job
`aqx-rgmerge1trg`, log `avx/driver_merge4.log`; the JSON is that log's `ANALYSIS_JSON` line).

- Of 200000 shards, 199600 reproduced. 400 shards on 33 servers did not and are left out
  (`excluded_unreproduced.shard_list`).
- The reproduced shards hold 998073 released neighbours, and all of them are bit-equal. The
  left-out shards hold 12.
- At 32 probes, recall is 0.982 for the corpus rows (95% CI 0.969 to 0.993) and 0.977 for the
  non-members (0.967 to 0.986). At 128 probes it is 0.999 and 1.000.
- No reference neighbour of a corpus-row query lies in a left-out shard. One reference
  neighbour of one non-member query does. The other 99 non-member queries give 0.978 at 32
  probes and 1.000 at 128.

`untried1trg_final.json` splits the 400 left-out shards by how far each was tried.
`avx/avx_untried.py` wrote it on Atlas. The copy here was reformatted for the repository's
linters (import order and line wrapping), and its logic is unchanged.

- 291 were also written whole on AVX-512 cluster nodes under all 31 build settings, in the
  first or second block pass below. Those passes reproduced 90 of the 381 shards they reached,
  all on servers 0 to 7, and none of the 196 they reached on other servers.
- 109, on 13 servers, are in the second block pass's plan, but their jobs did not complete
  before the stop. Each of them was rebuilt at least 27 times, on the cluster (10 or 30 builds)
  and on the Atlas workstation (17 builds). The list is `untried_list`.
- The script counts a second-pass job as complete if the driver log shows it `DONE` or the
  cluster reports the Job succeeded. Three jobs (67, 68, 70) finished after the driver was
  stopped. Only the cluster shows them, so the 64 jobs the script found are recorded in
  `second_pass_jobs_completed`. The cluster's Job records expire, so a later rerun of the script
  would find fewer.

## Files, in run order

| File | What it is |
|---|---|
| `driver1trg_pilot.log` | Pilot on the cluster, servers 12, 300 and 1. |
| `atlas/atlas_first_1trg.json` | Wave 1 (servers 0 and 2 to 7) was rebuilt on Atlas first. These are the shards that pass reproduced, by server (1935). |
| `driver1trg_w1.log`, `post_state_1trgw1.json` | Cluster pass over the rest of wave 1 (servers 0, 2, 3, 4, 5, 7). Server 6 was reproduced entirely on Atlas. |
| `driver1trg_pool.log`, `post_state_1trgpool.json` | Cluster pass over servers 8 to 499 except 12 and 300 (490 servers). |
| `atlas/deferred_1trg.json` | Shards the cluster passes deferred, by server (2226 shards on 33 servers). |
| `atlas/atlas_pass.log` | Atlas pass over those deferred shards, 2026-11-10 to 11-11. |
| `driver1trg_merge.log` | First merge, 2026-11-11 (superseded by the final merge). |
| `trace1trg_excluded.json` | Read-only trace of 2026-11-11. One entry per shard still unreproduced then (712), as `[server, shard, {trials, kernel, deferred, hash_ok}]`. |
| `avx/left_after_atlas.json` | 51 shards on servers 3 and 7 still unreproduced after a further Atlas rebuild of 2026-11-11 to 11-13. |
| `avx/plan_stream.json`, `avx/driver_avxs.log` | Streamed-write pass of those 51 shards on AVX-512 cluster nodes (9 jobs). |
| `avx/plan_block.json`, `avx/driver_avxb_pilot.log`, `avx/driver_avxb_rest.log` | First block (whole-write) pass of the same 51 shards on AVX-512 nodes at 1 CPU and 3 GiB (9 jobs). 26 reproduced. |
| `avx/plan_block2.json`, `avx/driver_avxc.log`, `avx/post_state_avxc.json` | Second block pass: 88 jobs over 439 shards, at most three at a time. The driver was stopped by hand at 2026-11-20 17:26Z, with 61 jobs done, 6 given up after 3 tries (58 to 62 and 66; their Jobs never appeared on the cluster), and job 69 waiting to retry. Jobs 67, 68 and 70 were running at the stop and completed afterwards. Jobs 71 to 87 were not submitted. |
| `avx/driver_merge4.log` | Final merge, 2026-11-20 19:00 to 19:07Z. |
| `analysis1trg_final.json` | The merged result (above). |
| `untried1trg_final.json`, `avx/avx_untried.py` | The split of the left-out shards (above). |

Cluster node names in the driver logs are the public names of the National Research Platform
nodes the jobs ran on. They show which jobs ran on AVX-512 nodes.

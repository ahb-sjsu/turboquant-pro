"""What the regenerated-index run (tag 1trg) used, from its record.

    python paper/pvldb1t/figdata/cost_1trg.py benchmarks/fleet/record/1trg \
        > paper/pvldb1t/figdata/cost_1trg.txt

Reads only files in the record folder and prints the figures Section sec:cost
quotes for the run of Section sec:regen. Two sources:

- analysis1trg_final.json, field regen_and_search_wall_s_per_server: one entry
  per partial the final merge read, each the time the job spent in its shard
  loop (regenerate, hash, check, search), summed over the attempts that resumed
  it. fleet_regen_merge.py (eade71c) builds that list as [500 cluster-pass
  partials, one per server] + [Atlas partials] + [AVX-512-pass partials, in
  sorted path order]. The AVX-pass partials staged from Atlas sit under /avxa*
  and the cluster's under /shared, so the Atlas ones come first.
- the driver logs: one line per submission (SUBMIT) and per completed job (DONE).

Atlas is a workstation that ran several rebuild processes at once at one to
three threads, so its partials' walls are process time, not core time, and are
reported apart from the cluster's.
"""

from __future__ import annotations

import json
import os
import re
import statistics as st
import sys

REC = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/fleet/record/1trg"
N_SRV = 500

d = json.load(open(os.path.join(REC, "analysis1trg_final.json"), encoding="utf-8"))
walls = d["regen_and_search_wall_s_per_server"]
n_avx = d["avx512_pass"]["partials"]
n_atlas = len(walls) - N_SRV - n_avx
assert n_atlas > 0, n_atlas


def events(path):
    """Counts of SUBMIT and DONE lines, and the job names each covers."""
    sub, done = 0, set()
    for line in open(os.path.join(REC, path), encoding="utf-8"):
        if m := re.match(r"=== \d\d:\d\d SUBMIT (\S+) try", line):
            sub += 1
        elif m := re.match(r"=== \d\d:\d\d DONE (\S+)", line):
            done.add(m.group(1))
    return sub, done


logs = {
    "pilot": "driver1trg_pilot.log",
    "wave 1": "driver1trg_w1.log",
    "pool": "driver1trg_pool.log",
    "AVX stream": "avx/driver_avxs.log",
    "AVX block pilot": "avx/driver_avxb_pilot.log",
    "AVX block rest": "avx/driver_avxb_rest.log",
    "AVX block second": "avx/driver_avxc.log",
}
counts = {k: events(v) for k, v in logs.items()}
# Three second-pass block jobs (67, 68, 70) finished after its driver was stopped,
# so their DONE is not in the log. untried1trg_final.json records the completed
# jobs as the cluster reported them.
u = json.load(open(os.path.join(REC, "untried1trg_final.json"), encoding="utf-8"))
second_done = u["second_pass_jobs_completed"]
n_second = len(second_done) if isinstance(second_done, list) else int(second_done)
avx_cluster_jobs = (
    len(counts["AVX stream"][1])
    + len(counts["AVX block pilot"][1] | counts["AVX block rest"][1])
    + n_second
)
n_avx_atlas = n_avx - avx_cluster_jobs
assert n_avx_atlas >= 0, (n_avx, avx_cluster_jobs)

cluster_pass = walls[:N_SRV]
atlas = (
    walls[N_SRV : N_SRV + n_atlas]
    + walls[N_SRV + n_atlas : N_SRV + n_atlas + n_avx_atlas]
)
avx_cluster = walls[N_SRV + n_atlas + n_avx_atlas :]
assert len(avx_cluster) == avx_cluster_jobs


def summ(name, v):
    v = [x for x in v if x > 0]
    s = sorted(v)
    n = len(s)
    print(
        f"{name}: partials {n}, sum {sum(s) / 3600:.1f} h, "
        f"median {st.median(s):.0f} s, p10 {s[n // 10]:.0f} s, "
        f"p90 {s[(9 * n) // 10]:.0f} s, max {s[-1]:.0f} s"
    )
    return sum(s) / 3600


print(f"record {REC}")
print(
    f"partials in the final merge {len(walls)}: cluster pass {N_SRV}, Atlas {n_atlas}, "
    f"AVX-512 pass {n_avx} (Atlas {n_avx_atlas}, cluster {avx_cluster_jobs})"
)
print(f"servers whole on Atlas (cluster-pass wall 0): {d['servers_whole_on_atlas']}")
h_pass = summ("cluster pass, 1 CPU 2 GiB", cluster_pass)
h_avx = summ("cluster AVX-512 passes, 1 CPU 2 GiB stream / 3 GiB block", avx_cluster)
h_atlas = summ("Atlas partials (process time)", atlas)
print(f"cluster total {h_pass + h_avx:.1f} h of job time in the shard loop")
print(
    f"builds per shard {d['regeneration']['builds_per_shard_mean']}, "
    f"shards rebuilt on Atlas {d['regeneration']['shards_rebuilt_on_atlas']}"
)
for k, (sub, done) in counts.items():
    print(f"driver {k}: submissions {sub}, jobs done {len(done)}")
print(f"second block pass jobs completed (cluster record) {n_second}")
tot_sub = sum(c[0] for c in counts.values())
tot_done = (
    sum(len(c[1]) for k, c in counts.items() if k != "AVX block second") + n_second
)
print(
    f"all cluster rebuild and search jobs: submissions {tot_sub}, completed {tot_done}"
)

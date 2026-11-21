"""Read-only: split the final merge's excluded shards by how far they were tried.

swept_avx512: excluded after a block build sweep on an AVX-512 node (first block pass plan_block.json,
              or a completed job of the second pass plan_block2.json)
untried:      in plan_block2.json but its job never completed (gave up, or not submitted before the stop)
Writes work/untried1trg_final.json and prints a summary.
"""

import json
import os
import re
import subprocess
from collections import Counter

import numpy as np

D = "/home/claude/tqp_fleet/avx1trg"
W = "/archive/experiments/tqp-fleet-1t/work"
FIN = os.environ.get("FIN", f"{W}/analysis1trg_final.json")
OUT = os.environ.get("OUT", f"{W}/untried1trg_final.json")
fin = json.load(open(FIN))
p2 = json.load(open(f"{D}/plan_block2.json"))
p1 = json.load(open(f"{D}/plan_block.json"))
print("plan entry sample", json.dumps(p2[0])[:160])

done = set(
    int(m)
    for m in re.findall(
        r"DONE aqx-rgavxc(\d+)-1trg",
        open(f"{D}/driver_avxc.log", errors="ignore").read(),
    )
)
kc = subprocess.run(
    [
        "kubectl",
        "-n",
        "ssu-atlas-ai",
        "get",
        "jobs",
        "--no-headers",
        "-o",
        "custom-columns=N:.metadata.name,S:.status.succeeded",
    ],
    capture_output=True,
    text=True,
    env={**os.environ, "KUBECONFIG": "/home/claude/.kube/config"},
)
out = kc.stdout
# fail loud: a silent empty listing undercounts the completed jobs
assert (
    kc.returncode == 0 and "aqx-" in out
), f"kubectl listing failed rc={kc.returncode} err={kc.stderr[:300]}"
for line in out.splitlines():
    m = re.match(r"aqx-rgavxc(\d+)-1trg\s+1\b", line)
    if m:
        done.add(int(m.group(1)))
print("second-pass jobs completed", len(done), "of", len(p2))


def shards(entry):
    sid = int(entry["sid"])
    loc = [int(s) for s in entry["shards"]]
    if any(s >= 400 for s in loc):  # global shard ids g = sid*400 + local
        loc = [s - sid * 400 for s in loc]
    assert all(0 <= s < 400 for s in loc), entry
    return {(sid, s) for s in loc}


excl = {tuple(x) for x in fin["excluded_unreproduced"]["shard_list"]}
plan2 = set().union(*(shards(e) for e in p2))
plan1 = set().union(*(shards(e) for e in p1))
tried2 = set().union(*(shards(p2[k]) for k in done)) if done else set()
print("plan2 shards", len(plan2), "plan1 shards", len(plan1), "tried2", len(tried2))
assert plan2 <= {
    tuple(x)
    for x in json.load(open(f"{W}/analysis1trg_avx512.json"))["excluded_unreproduced"][
        "shard_list"
    ]
}, "plan2 not within the 11-17 excluded list"

untried = sorted((plan2 - tried2) & excl)
swept = sorted(excl & (tried2 | plan1))
other = sorted(excl - set(untried) - set(swept))
repro2 = sorted(plan2 - excl)
assert (
    set(repro2) <= tried2
), "a second-pass shard left the excluded list without a completed job"
# How each untried shard WAS built: cluster pool trials (trace of 11-11) + Atlas deferred-pass trials.
trace = {(e[0], e[1]): e[2] for e in json.load(open(f"{W}/trace1trg_excluded.json"))}
atlas = {}
for sid in sorted({s for s, _ in untried}):
    f = f"{W}/rg_atlas_1trg/regen1trg_atlas_{sid}.npz"
    try:
        z = np.load(f, allow_pickle=False)
    except FileNotFoundError:
        continue
    if sid == sorted({s for s, _ in untried})[0]:
        print("atlas partial keys", sorted(z.files))
    tr = z["trials"]
    for j in z["shards"].tolist() if "shards" in z.files else range(len(tr)):
        atlas[(sid, int(j))] = int(tr[j]) if len(tr) == 400 else None
built = []
for k in untried:
    c = trace.get(k, {}).get("trials")
    a = atlas.get(k, 0)
    built.append((k, c, a))
ct = [c for _, c, _ in built if c is not None]
at = [a for _, _, a in built if a is not None]
tot = [(c or 0) + (a or 0) for _, c, a in built]
print(
    "untried: cluster trials min/median",
    (min(ct), int(np.median(ct))) if ct else None,
    "atlas trials>0",
    sum(1 for x in at if x),
    "min total builds",
    min(tot) if tot else None,
    "no cluster trace",
    sum(1 for _, c, _ in built if c is None),
)
res_build = {
    "untried_cluster_trials": Counter(ct),
    "untried_atlas_trials": Counter(at),
    "untried_total_builds_min": min(tot) if tot else None,
}
res = {
    "untried_builds": {
        k: dict(v) if isinstance(v, Counter) else v for k, v in res_build.items()
    },
    "excluded": len(excl),
    "excluded_servers": len({s for s, _ in excl}),
    "swept_avx512_block": len(swept),
    "untried_avx512_block": len(untried),
    "other": len(other),
    "second_pass_planned": len(plan2),
    "second_pass_tried": len(tried2),
    "second_pass_reproduced": len(repro2),
    "second_pass_reproduced_by_server": dict(Counter(s for s, _ in repro2)),
    # both block passes on AVX-512 nodes together (first: plan_block.json, second: completed plan_block2 jobs)
    "block_avx512_reached": len(plan1 | tried2),
    "block_avx512_reproduced": len((plan1 | tried2) - excl),
    "block_avx512_reached_servers_ge8": len({k for k in plan1 | tried2 if k[0] >= 8}),
    "block_avx512_reproduced_servers_ge8": len(
        {k for k in (plan1 | tried2) - excl if k[0] >= 8}
    ),
    "block_avx512_reproduced_servers": sorted({k[0] for k in (plan1 | tried2) - excl}),
    "untried_by_server": dict(Counter(s for s, _ in untried)),
    "untried_servers": len({s for s, _ in untried}),
    "untried_list": untried,
    "second_pass_jobs_completed": sorted(done),
}
json.dump(res, open(OUT, "w"), indent=1)
print(
    json.dumps(
        {
            k: v
            for k, v in res.items()
            if k not in ("untried_list", "second_pass_jobs_completed")
        },
        indent=1,
    )
)

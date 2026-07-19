# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Split this pod's shard-range into L logical shard-servers (1T/5T shape).

Writes ``server_L{L}_{gid:03d}.manifest.json`` per logical server — round-robin
over the pod's 50 shards — into ``/idx`` (for the workers) and into the pod's
shared export dir (for the coordinator's placement builder, next to the
``.ivf.off.npy`` sidecars already there). ``TQP_SPLITS`` is a comma list of L
values, e.g. ``16,64`` -> 64- and 256-endpoint topologies.
"""

import json
import os
import shutil

SID = int(os.environ["TQP_SERVER_ID"])
SPLITS = [int(x) for x in os.environ.get("TQP_SPLITS", "16,64").split(",")]
IDX = "/idx"
OUT = f"/shared/fleet/server_{SID:03d}"

with open(f"{IDX}/manifest.json", encoding="utf-8") as f:
    manifest = json.load(f)
shards = manifest["shards"]

for L in SPLITS:
    for k in range(L):
        gid = SID * L + k
        assigned = shards[k::L]
        sub = dict(manifest)
        sub["shards"] = assigned
        sub["n_shards"] = len(assigned)
        sub["n_rows"] = sum(s["n_rows"] for s in assigned)
        name = f"server_L{L}_{gid:03d}.manifest.json"
        with open(f"{IDX}/{name}", "w", encoding="utf-8") as f:
            json.dump(sub, f)
        shutil.copy(f"{IDX}/{name}", OUT)
    print(
        f"L={L}: wrote {L} logical manifests (gids {SID * L}..{SID * L + L - 1})",
        flush=True,
    )
print("SPLIT_DONE", flush=True)

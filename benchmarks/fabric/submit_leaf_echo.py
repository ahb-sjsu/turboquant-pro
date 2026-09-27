"""Send the leaf_echo client to NRP, through nats-bursting, to exercise the leaf link.

The pod connects to the in-cluster leaf node (``nats://atlas-nats:4222``), so all
its traffic crosses the leaf link to the Atlas hub, where the responder and the
recorder run. Run this ON ATLAS with the environment that has ``nats_bursting``.

The preflight below is code, not recall (the NRP rules in the agi-hpc memory
``reference_nrp_job_policies.md``), and it refuses to submit if any rule fails:

- exempt class: requests <= 1 CPU and <= 2 GiB, so the 20% utilisation floors
  do not apply; the renderer sets limits == requests;
- memory is sized from a MEASUREMENT (the rehearsal's peak RSS), never a guess;
- ephemeral storage is declared (pip writes to the node's disk);
- no GPU;
- the command terminates by itself and contains no sleep: every phase blocks
  only on the network, each request has a timeout, and the client exits;
- the Job name is unique (timestamped), so a stale completed Job cannot answer
  for it; after submitting, verify its creationTimestamp is fresh.

    /home/claude/env/bin/python benchmarks/fabric/submit_leaf_echo.py \\
        --prefix tqp.fabric.exp.leaf.<ts> --footprint rehearsal/client.json [--submit]

Without ``--submit`` it prints the descriptor and the preflight and sends nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import sys
import time
from pathlib import Path

IMAGE = "python:3.12-slim"
NATS_PY = "nats-py==2.14.0"  # the version the rehearsal ran on Atlas
LEAF_URL = "nats://atlas-nats:4222"
EXEMPT_CPU, EXEMPT_MEM_MIB = 1, 2048
MEM_MIB, EPHEMERAL = 512, "1Gi"
PIP_ALLOWANCE_MIB = 150  # pip resolving and installing one pure-Python wheel


def descriptor(prefix: str, name: str) -> dict:
    script = Path(__file__).with_name("leaf_echo.py").read_bytes()
    run = (
        "import base64,os;"
        "exec(compile(base64.b64decode(os.environ['SCRIPT_B64']),'leaf_echo.py','exec'))"
    )
    return {
        "name": name,
        "image": IMAGE,
        "command": [
            "sh",
            "-c",
            f'pip install --no-cache-dir -q {NATS_PY} && python -c "{run}" client',
        ],
        "args": [],
        "env": {
            "SCRIPT_B64": base64.b64encode(script).decode(),
            "NATS_URL": LEAF_URL,
            "PREFIX": prefix,
            "PYTHONUNBUFFERED": "1",
        },
        "resources": {
            "cpu": str(EXEMPT_CPU),
            "memory": f"{MEM_MIB}Mi",
            "gpu": 0,
            "ephemeral_storage": EPHEMERAL,
        },
        "labels": {"atlas.io/batch": "tqp-fabric-exp"},
        "backoff_limit": 0,
    }


def preflight(d: dict, footprint: dict) -> list[str]:
    """Every NRP rule this job must meet, as failures (empty: it may go)."""
    bad = []
    r = d["resources"]
    if int(r["cpu"]) > EXEMPT_CPU:
        bad.append(f"cpu {r['cpu']} is above the exempt class ({EXEMPT_CPU})")
    mem = int(r["memory"].removesuffix("Mi"))
    if mem > EXEMPT_MEM_MIB:
        bad.append(f"memory {mem} MiB is above the exempt class")
    peak = (footprint.get("usage") or {}).get("peak_rss_mib")
    if not peak:
        bad.append("no measured peak RSS for the client: run the rehearsal first")
    elif mem < 1.25 * (peak + PIP_ALLOWANCE_MIB):
        bad.append(
            f"memory {mem} MiB is under 1.25 x (measured peak {peak:.0f} MiB + pip "
            f"{PIP_ALLOWANCE_MIB} MiB)"
        )
    if int(r.get("gpu") or 0):
        bad.append("requests a GPU")
    if not r.get("ephemeral_storage"):
        bad.append("ephemeral storage is not declared")
    cmd = " ".join(d["command"] + d["args"])
    if re.search(r"\bsleep\b", cmd) or "while true" in cmd:
        bad.append("the command sleeps or loops forever")
    script = base64.b64decode(d["env"]["SCRIPT_B64"]).decode()
    client_src = script.split("async def client", 1)[1].split("async def responder")[0]
    if "sleep" in client_src:
        bad.append("the client code sleeps")
    if not re.search(r"-\d{10}$", d["name"]):
        bad.append("the Job name is not timestamped (a stale Job could answer)")
    return bad


async def submit(d: dict, wait_s: float) -> list:
    """Publish burst.submit with burst.status.> subscribed FIRST: nats-bursting
    reports rejections there and nowhere else."""
    import nats

    nc = await nats.connect("nats://localhost:4222", name="tqp-fabric-submit")
    job_id = d["name"]
    got: list = []

    async def on_status(msg):
        if msg.subject.endswith(job_id):
            got.append(json.loads(msg.data))

    await nc.subscribe("burst.status.>", cb=on_status)
    await nc.flush()
    await nc.publish(
        "burst.submit", json.dumps({"job_id": job_id, "descriptor": d}).encode()
    )
    await nc.flush()
    deadline = time.monotonic() + wait_s
    while not got and time.monotonic() < deadline:
        await asyncio.sleep(0.5)  # on Atlas, waiting for the controller's answer
    await nc.drain()
    return got


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--prefix", required=True)
    p.add_argument("--footprint", required=True, help="the rehearsal's client.json")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--status-wait", type=float, default=60.0)
    args = p.parse_args(argv)
    name = f"tqp-fabric-leaf-{int(time.time())}"
    d = descriptor(args.prefix, name)
    with open(args.footprint, encoding="utf-8") as f:
        foot = json.loads(f.read().strip().splitlines()[-1])
    bad = preflight(d, foot)
    shown = {
        **d,
        "env": {**d["env"], "SCRIPT_B64": f"<{len(d['env']['SCRIPT_B64'])} b64 chars>"},
    }
    print(json.dumps({"descriptor": shown, "preflight": bad or "PASS"}, indent=2))
    if bad:
        print("refusing to submit", file=sys.stderr)
        return 2
    if not args.submit:
        print("dry run: nothing sent (pass --submit)", file=sys.stderr)
        return 0
    status = asyncio.run(submit(d, args.status_wait))
    print(json.dumps({"job": name, "status": status}, indent=2))
    ok = any(s.get("state") not in ("error", "failed") for s in status)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

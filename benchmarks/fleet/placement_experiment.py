"""Attribute the 1T reference-scan wall-time spread to placement, on the 1T volumes.

The 1T measurement ran the same full scan 500 times, one CPU each, and took
1,062 to 16,308 s; its record does not say where each job ran. This re-runs the
scan for 16 servers, read-only, on nodes chosen across zones, with the
placement record (``fingerprint.py``: node, one-core speed, ``r_blk`` to the
volume, phase marks) and an identity check against the 1T partials. The
predictions it is scored against are in ``PLACEMENT_EXPERIMENT.md``, committed
before any job ran.

Run on Atlas with the environment that has nats-py:

    /home/claude/env/bin/python benchmarks/fleet/placement_experiment.py \\
        --out DIR [--submit]

Without ``--submit`` it prints the plan, the descriptors' shape and the
preflight, and creates nothing. With it, at most ``--maxpar`` jobs exist at a
time; each is deleted once collected, and its pod waited out before the next
job may use the claim.

Preflight, in code: the 1T pilot's measured envelope (exempt class 1 CPU /
2 GiB: mean 884m CPU, peak 1469Mi memory, RSS 346 MB), ephemeral storage
declared, no GPU, no sleep, read-only mounts of the server volume and of the
shared results, only ``tqp-fleet-1t-*`` claims, a claim held by a live pod is
never used, the clone pinned to one commit, Job names timestamped.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import random
import subprocess
import sys
import time
from pathlib import Path

NS = "ssu-atlas-ai"
IMAGE = "python:3.12"  # the 1T jobs' image
REPO = "https://github.com/ahb-sjsu/turboquant-pro.git"
BATCH = "tqp-placement"
ZONE_LABEL = "topology.kubernetes.io/zone"

# 16 servers, spread over the index; zones chosen to span distance to the data
# (every volume is one replica in LINSTOR pool "unl") and per-core CPU speed.
SERVERS = [11 + 31 * i for i in range(16)]
ZONES = (
    ["unl"] * 3
    + ["mizzou"] * 2
    + ["sdstate", "usd"]
    + ["mghpcc"] * 2
    + ["ucsd-nrp"] * 2
    + ["fullerton"] * 2
    + ["humboldt"] * 2
    + ["korea"]
)
SEED = 20260927


def plan() -> list:
    zones = list(ZONES)
    random.Random(SEED).shuffle(zones)  # which server goes where is not chosen
    return list(zip(SERVERS, zones))


def kubectl(*args, check=False) -> str:
    r = subprocess.run(
        ["kubectl", "-n", NS, *args], capture_output=True, text=True, timeout=120
    )
    if check and r.returncode:
        raise RuntimeError(f"kubectl {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def script_b64() -> str:
    here = Path(__file__).parent
    src = (here / "fingerprint.py").read_text(encoding="utf-8") + "\n\n"
    src += (here / "placement_scan.py").read_text(encoding="utf-8")
    return base64.b64encode(src.encode()).decode()


def descriptor(sid: int, zone: str, commit: str, ts: int) -> dict:
    run = (
        "import base64,os;exec(compile(base64.b64decode(os.environ['SCRIPT_B64']),"
        "'placement_scan.py','exec'))"
    )
    cmd = (
        "set -euo pipefail\n"
        "export PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1\n"
        'echo "T_START $(date +%s.%N)"\n'
        "pip install -q --no-cache-dir numpy\n"
        'echo "T_PIP $(date +%s.%N)"\n'
        f"git init -q /repo && git -C /repo remote add origin {REPO}\n"
        f"git -C /repo fetch -q --depth 1 origin {commit}\n"
        "git -C /repo checkout -q FETCH_HEAD\n"
        "git -C /repo log -1 --format='repo %H'\n"
        'echo "T_CLONE $(date +%s.%N)"\n'
        "export PYTHONPATH=/repo\n"
        f'python -c "{run}"\n'
        'echo "T_END $(date +%s.%N)"\n'
    )
    return {
        "name": f"tqp-place-{sid}-{zone}-{ts}"[:63],
        "image": IMAGE,
        "command": ["/bin/bash", "-lc", cmd],
        "args": [],
        "env": {"TQP_SERVER_ID": str(sid), "SCRIPT_B64": script_b64()},
        "resources": {
            "cpu": "1",
            "memory": "2Gi",
            "gpu": 0,
            "ephemeral_storage": "3Gi",
        },
        "labels": {"atlas.io/batch": BATCH},
        "backoff_limit": 0,
        "node_selector": {ZONE_LABEL: zone},
        "volumes": [
            {
                "name": "idx",
                "mount_path": "/idx",
                "claim_name": f"tqp-fleet-1t-{sid}",
                "read_only": True,
            },
            {
                "name": "shared",
                "mount_path": "/shared",
                "claim_name": "tqp-fleet-shared",
                "read_only": True,
            },
        ],
    }


def preflight(d: dict) -> list:
    bad = []
    r = d["resources"]
    if r["cpu"] != "1" or r["memory"] != "2Gi":
        bad.append("not the exempt class the 1T pilot measured (1 CPU / 2Gi)")
    if int(r.get("gpu") or 0):
        bad.append("requests a GPU")
    if not r.get("ephemeral_storage"):
        bad.append("no ephemeral storage declared")
    if any(not v.get("read_only") for v in d["volumes"]):
        bad.append("a volume is not read-only")
    if not d["volumes"][0]["claim_name"].startswith("tqp-fleet-1t-"):
        bad.append("not a 1T claim")
    src = base64.b64decode(d["env"]["SCRIPT_B64"]).decode()
    if "sleep" in (d["command"][-1] + src).replace("# ", ""):
        bad.append("the command or the script sleeps")
    if "fetch -q --depth 1 origin " not in d["command"][-1]:
        bad.append("the clone is not pinned to a commit")
    if not d["name"].rsplit("-", 1)[-1].isdigit():
        bad.append("Job name not timestamped")
    return bad


def claim_in_use(claim: str) -> list:
    try:
        items = json.loads(kubectl("get", "pods", "-o", "json")).get("items", [])
    except ValueError:
        return ["(pod list unreadable)"]
    return [
        p["metadata"]["name"]
        for p in items
        if p["status"].get("phase") not in ("Succeeded", "Failed")
        or p["metadata"].get("deletionTimestamp")
        for v in p["spec"].get("volumes") or []
        if (v.get("persistentVolumeClaim") or {}).get("claimName") == claim
    ]


async def _submit(d: dict) -> list:
    import nats

    nc = await nats.connect("nats://localhost:4222", name="tqp-placement-submit")
    got: list = []

    async def on_status(msg):
        if msg.subject.endswith(d["name"]):
            got.append(json.loads(msg.data))

    await nc.subscribe("burst.status.>", cb=on_status)
    await nc.flush()
    await nc.publish(
        "burst.submit", json.dumps({"job_id": d["name"], "descriptor": d}).encode()
    )
    await nc.flush()
    end = time.monotonic() + 60
    while not got and time.monotonic() < end:
        await asyncio.sleep(0.5)
    await nc.drain()
    return got


MIN_SUBMIT_GAP_S = 120.0  # no burst of Job creations in the shared namespace
FOREIGN_CLAIMS = ("tqp-rbq-data",)  # another campaign's data: never mount it


def delete_finished_job(job: str, log) -> bool:
    """Delete ``job`` only once it has succeeded or failed; refuse otherwise."""
    raw = kubectl("get", "job", job, "-o", "json")
    st = json.loads(raw).get("status", {}) if raw else {}
    if raw and not (st.get("succeeded") or st.get("failed")):
        log(f"NOT deleting unfinished {job}")
        return False
    if raw:
        kubectl("delete", "job", job, "--ignore-not-found", "--wait=true")
    end = time.monotonic() + 600
    while kubectl("get", "pods", "-l", f"job-name={job}", "-o", "name"):
        if time.monotonic() > end:
            log(f"NOTE {job} pod still terminating after 600 s")
            break
        time.sleep(5)
    return True


def parse_log(log: str) -> dict:
    out = {"marks": {}, "fingerprint": None, "repo": None}
    for line in log.splitlines():
        if line.startswith("T_"):
            k, v = line.split(" ", 1)
            out["marks"][k] = float(v)
        elif line.startswith("FINGERPRINT "):
            out["fingerprint"] = json.loads(line.split(" ", 1)[1])
        elif line.startswith("repo "):
            out["repo"] = line.split(" ", 1)[1]
    return out


def run(
    out: Path, commit: str, maxpar: int, pend_s: int, max_s: int, todo: list
) -> dict:
    active: dict = {}
    results: dict = {}
    last_submit = 0.0

    def log(msg):
        print(f"=== {time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)

    while todo or active:
        while todo and len(active) < maxpar:
            sid, zone = todo.pop(0)
            claim = f"tqp-fleet-1t-{sid}"
            if claim_in_use(claim):
                log(f"SKIP {claim}: in use")
                results[sid] = {"sid": sid, "zone": zone, "phase": "skipped: in use"}
                continue
            d = descriptor(sid, zone, commit, int(time.time()))
            if any(
                (v.get("claim_name") or "").startswith(FOREIGN_CLAIMS)
                for v in d["volumes"]
            ):
                raise SystemExit("refusing another campaign's claim")
            wait = last_submit + MIN_SUBMIT_GAP_S - time.time()
            if wait > 0:
                time.sleep(wait)
            last_submit = time.time()
            status = asyncio.run(_submit(d))
            active[sid] = {
                "sid": sid,
                "zone": zone,
                "job": d["name"],
                "t_submit": time.time(),
                "status": status,
                "node": None,
            }
            log(f"SUBMIT {d['name']} status={[s.get('state') for s in status]}")
        time.sleep(30)
        for sid, a in list(active.items()):
            st = kubectl(
                "get",
                "job",
                a["job"],
                "-o",
                "jsonpath={.status.succeeded}/{.status.failed}",
            )
            if a["node"] is None:
                node = kubectl(
                    "get",
                    "pods",
                    "-l",
                    f"job-name={a['job']}",
                    "-o",
                    "jsonpath={.items[0].spec.nodeName}",
                ).strip()
                if node:
                    a["node"] = node
                    a["t_placed"] = time.time()
                    log(f"PLACED {a['job']} on {node}")
            age = time.time() - a["t_submit"]
            if st.startswith("1/"):
                phase = "succeeded"
            elif st.endswith("/1"):
                phase = "failed"
            elif age > max_s:
                # Never delete an unfinished Job: another campaign's breaker counts
                # an ACTIVE Job disappearing as a disturbance. Leave it and stop
                # submitting; a person decides.
                if not a.get("overdue"):
                    a["overdue"] = True
                    log(
                        f"OVERDUE {a['job']} ({int(age)} s, node {a['node']}): left "
                        "running, not deleted; no further submissions"
                    )
                    todo.clear()
                continue
            else:
                continue
            a["phase"] = phase
            a.update(parse_log(kubectl("logs", f"job/{a['job']}")))
            a["t_done"] = time.time()
            delete_finished_job(a["job"], log)
            log(f"{phase.upper()} {a['job']} on {a['node']}")
            results[sid] = a
            (out / f"server-{sid}.json").write_text(json.dumps(a, indent=2), "utf-8")
            del active[sid]
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--commit", required=True, help="the commit every clone pins")
    p.add_argument("--maxpar", type=int, default=8)
    p.add_argument("--pend", type=int, default=2700, help="unscheduled after (s)")
    p.add_argument("--max", type=int, default=5 * 3600, help="timeout per job (s)")
    p.add_argument(
        "--pilot",
        help="SID:ZONE, one job on a server outside the plan, to check the "
        "pipeline end to end; its numbers are not part of the scored set",
    )
    p.add_argument("--submit", action="store_true")
    a = p.parse_args(argv)
    if not (len(a.commit) == 40 and all(c in "0123456789abcdef" for c in a.commit)):
        print(
            "--commit must be a full 40-character id: GitHub serves a fetch by "
            "commit only for the full id (the pilot's short one failed)",
            file=sys.stderr,
        )
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bad = set()
    for sid, zone in plan():
        b = preflight(descriptor(sid, zone, a.commit, 1790000000))
        bad |= set(b)
    shown = descriptor(*plan()[0], a.commit, 1790000000)
    shown["env"]["SCRIPT_B64"] = f"<{len(shown['env']['SCRIPT_B64'])} chars>"
    print(
        json.dumps(
            {"plan": plan(), "example": shown, "preflight": sorted(bad) or "PASS"},
            indent=1,
        )
    )
    if bad:
        return 2
    if not a.submit:
        print("dry run: nothing created (pass --submit)", file=sys.stderr)
        return 0
    todo = plan()
    if a.pilot:
        sid, zone = a.pilot.split(":")
        if int(sid) in SERVERS:
            print("the pilot must use a server outside the plan", file=sys.stderr)
            return 2
        todo = [(int(sid), zone)]
    res = run(out, a.commit, a.maxpar, a.pend, a.max, todo)
    print(json.dumps({s: r.get("phase") for s, r in sorted(res.items())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

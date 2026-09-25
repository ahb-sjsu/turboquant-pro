"""Pilot ONE exempt-class reference job (1 CPU, 2 GiB, TQP_REF_BLOCK) and meter it.

Runs on Atlas from /home/claude/tqp_fleet. Picks the lowest server id not in the driver's
checkpoint, submits desc_ref(sid) through the burst controller (never kubectl apply), then
samples `kubectl top pod` every 20 s until the Job succeeds or fails, and prints the peak
and mean CPU and memory against the request, the exit, and the pod's last log lines.
It never resubmits and never touches any other Job.

    python3 pilot_ref_exempt.py            # submit + meter one reference job
    python3 pilot_ref_exempt.py --phase ivf  # one IVF job on a server whose reference is done
    python3 pilot_ref_exempt.py --meter-only <job-name>
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time

sys.path.insert(0, "/home/claude/tqp_fleet")
import driver1t_post as d  # noqa: E402

NS = d.NS
REQ_CPU_M, REQ_MEM_MI = 1000, 2048


def sh(*args: str, timeout: int = 40) -> str:
    r = subprocess.run(list(args), capture_output=True, text=True, timeout=timeout)
    return r.stdout


def kget(kind: str, name: str):
    out = sh("kubectl", "-n", NS, "get", kind, name, "-o", "json")
    return json.loads(out) if out.strip() else None


def parse_cpu(s: str) -> float:
    return float(s[:-1]) if s.endswith("m") else float(s) * 1000


def parse_mem(s: str) -> float:
    m = re.match(r"(\d+)(Ki|Mi|Gi)?", s)
    v, u = float(m.group(1)), m.group(2)
    return v / 1024 if u == "Ki" else v * 1024 if u == "Gi" else v


def meter(name: str) -> None:
    t0 = time.time()
    cpu, mem = [], []
    pod = None
    while True:
        job = kget("job", name)
        st = (job or {}).get("status", {})
        pods = sh(
            "kubectl", "-n", NS, "get", "pods", "-l", f"job-name={name}", "-o", "json"
        )
        items = json.loads(pods).get("items", []) if pods.strip() else []
        if items:
            pod = items[-1]["metadata"]["name"]
            phase = items[-1]["status"].get("phase")
        else:
            phase = "-"
        top = sh(
            "kubectl", "-n", NS, "top", "pod", "-l", f"job-name={name}", "--no-headers"
        )
        for line in top.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                cpu.append(parse_cpu(parts[1]))
                mem.append(parse_mem(parts[2]))
        el = int(time.time() - t0)
        last = f"cpu {cpu[-1]:.0f}m mem {mem[-1]:.0f}Mi" if cpu else "no metrics yet"
        print(
            f"t+{el:5d}s phase={phase} succeeded={st.get('succeeded', 0)} failed={st.get('failed', 0)} {last}",
            flush=True,
        )
        if st.get("succeeded") or st.get("failed"):
            break
        if el > 4 * 3600:
            print(
                "PILOT giving up after 4 h (job still running); not deleting it",
                flush=True,
            )
            break
        time.sleep(20)
    if cpu:
        n = len(cpu)
        print(
            f"samples {n}  CPU mean {sum(cpu)/n:.0f}m peak {max(cpu):.0f}m ({100*sum(cpu)/n/REQ_CPU_M:.0f}% / {100*max(cpu)/REQ_CPU_M:.0f}% of {REQ_CPU_M}m)"
        )
        print(
            f"            MEM mean {sum(mem)/n:.0f}Mi peak {max(mem):.0f}Mi trough {min(mem):.0f}Mi ({100*sum(mem)/n/REQ_MEM_MI:.0f}% / {100*max(mem)/REQ_MEM_MI:.0f}% of {REQ_MEM_MI}Mi)"
        )
    if pod:
        print("--- pod log tail:")
        print(sh("kubectl", "-n", NS, "logs", pod, "--tail=8", timeout=60))
        p = kget("pod", pod)
        cs = (
            (p or {})
            .get("status", {})
            .get("containerStatuses", [{}])[0]
            .get("state", {})
        )
        print("container state:", json.dumps(cs)[:300])
    print("PILOT_DONE" if st.get("succeeded") else "PILOT_FAILED", flush=True)


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--meter-only":
        meter(sys.argv[2])
        return
    phase = (
        "ivf"
        if "--phase" in sys.argv and sys.argv[sys.argv.index("--phase") + 1] == "ivf"
        else "ref"
    )
    state = json.load(open(d.STATE_PATH))
    done = set(state.get(phase, {}).get("done", []))
    ref_done = set(state.get("ref", {}).get("done", []))
    prefix = f"aqx-{phase}1t-"
    make = d.desc_ivf if phase == "ivf" else d.desc_ref
    order = sorted(ref_done) if phase == "ivf" else list(reversed(range(d.N_SERVERS)))
    # First server not done whose Job name is free, or whose old Job is finished and can be
    # deleted before reuse (a policy rule); a Job still running is left alone and skipped.
    sid = name = None
    # Highest ids first: never submitted before, so no stale Job object and no partial on the
    # shared volume, which makes the pilot a full-length measurement rather than a skip.
    for i in order:
        if i in done:
            continue
        cand = f"{prefix}{i}"
        job = kget("job", cand)
        if job is None:
            sid, name = i, cand
            break
        st = job.get("status", {})
        if st.get("succeeded") or st.get("failed") or not st.get("active"):
            print(
                f"{cand}: finished Job object from the earlier run, deleting before reuse",
                flush=True,
            )
            sh(
                "kubectl",
                "-n",
                NS,
                "delete",
                "job",
                cand,
                "--ignore-not-found",
                "--wait=false",
                timeout=60,
            )
            for _ in range(20):
                time.sleep(6)
                pods = sh(
                    "kubectl",
                    "-n",
                    NS,
                    "get",
                    "pods",
                    "-l",
                    f"job-name={cand}",
                    "-o",
                    "json",
                )
                if not (json.loads(pods).get("items") if pods.strip() else []):
                    break
            sid, name = i, cand
            break
        print(f"{cand}: still active on the cluster, skipping", flush=True)
    if sid is None:
        raise SystemExit("no free server to pilot")
    desc = make(sid)
    print(f"PILOT submit {name}: resources {desc.resources} env {desc.env}", flush=True)
    with d.Client() as client:
        res = client.submit(desc)
    print(f"PILOT job_id={res.job_id}", flush=True)
    meter(name)


if __name__ == "__main__":
    main()

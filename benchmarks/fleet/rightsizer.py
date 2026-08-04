# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Resident right-sizer for NRP pods: measure what workloads use, then size them.

NRP deletes the Job object of any pod whose usage falls outside its request for
too long. The bands are CPU 20-200% and memory 20-150% of request, and a pod is
exempt entirely if it asks for at most 1 CPU and 2 GiB. On 2026-08-04 a 1T fleet
launch lost a night to this: builds asked for 6 CPU while using 0.4, and for
8 GiB while sitting at 143 Mi during their resume-skip phase. Every fix was a
guess because nobody was measuring.

This watches the namespace, records a peak-and-trough profile per workload, and
turns that into the request window that is actually legal:

    request >= peak          because requests == limits, so exceeding it OOMs
    request <= trough / 0.20 because below that the floor is violated

Those two can conflict. A workload whose peak is more than five times its
trough has NO legal request, and no amount of tuning will find one -- the shape
of its memory use has to change. Saying that out loud is the point of this
tool; it is the conclusion that took a night to reach by hand.

Modes:
  watch    sample forever, update the profile store, warn on live violations
  advise   print the legal window and a recommendation for a workload
  check    validate a rendered manifest before you submit it

metrics-server lags roughly 30s and reports a smoothed value, so a single
reading is not a peak. This keeps running extremes and needs a workload to run
a few minutes before its advice is worth anything; `samples` in the output says
how much it has seen. Trusting one early reading is exactly the mistake that
led to a 3Gi request and an OOM at exit 137.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

NS = os.environ.get("TQP_NS", "ssu-atlas-ai")
STORE = os.environ.get("TQP_RIGHTSIZER_STORE", "/home/claude/tqp_fleet/rightsizer.json")

# NRP utilization policy. Usage outside these fractions of request is a
# violation; at or under the ignored thresholds nothing is monitored at all.
CPU_LO, CPU_HI = 0.20, 2.00
MEM_LO, MEM_HI = 0.20, 1.50
IGNORED_CPU = 1.0
IGNORED_MEM = 2.0 * 1024**3

_SUFFIX = re.compile(r"-[a-z0-9]{5,10}$")


def workload(pod: str) -> str:
    """Strip the pod's random suffix so replicas share one profile."""
    return _SUFFIX.sub("", pod)


def parse_cpu(s: str) -> float:
    s = s.strip()
    if s.endswith("m"):
        return float(s[:-1]) / 1000.0
    if s.endswith("n"):
        return float(s[:-1]) / 1e9
    return float(s or 0)


def parse_mem(s: str) -> float:
    s = s.strip()
    for unit, mult in (("Ki", 1024), ("Mi", 1024**2), ("Gi", 1024**3), ("Ti", 1024**4)):
        if s.endswith(unit):
            return float(s[: -len(unit)]) * mult
    return float(re.sub(r"[^0-9.]", "", s) or 0)


def kubectl(*args: str) -> str:
    r = subprocess.run(
        ["kubectl", "-n", NS, *args], capture_output=True, text=True, timeout=60
    )
    return r.stdout if r.returncode == 0 else ""


def load() -> dict:
    try:
        with open(STORE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save(d: dict) -> None:
    tmp = STORE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1, sort_keys=True)
    os.replace(tmp, STORE)


def requests_of(pod: str) -> tuple[float, float]:
    """The pod's own cpu/memory request, so violations can be judged live."""
    out = kubectl(
        "get",
        "pod",
        pod,
        "-o",
        "jsonpath={.spec.containers[0].resources.requests.cpu}"
        "|{.spec.containers[0].resources.requests.memory}",
    )
    if "|" not in out:
        return 0.0, 0.0
    c, m = out.split("|", 1)
    return parse_cpu(c or "0"), parse_mem(m or "0")


def sample(store: dict) -> list[str]:
    """One sampling pass. Returns human-readable warnings."""
    warn: list[str] = []
    top = kubectl("top", "pod", "--no-headers")
    if not top.strip():
        return warn
    for line in top.strip().splitlines():
        f = line.split()
        if len(f) < 3:
            continue
        pod, cpu, mem = f[0], parse_cpu(f[1]), parse_mem(f[2])
        # metrics-server reports 0 for a pod it has not scraped yet; recording
        # that as a trough would make every fresh workload look non-compliant.
        if cpu == 0 and mem == 0:
            continue
        w = workload(pod)
        e = store.setdefault(
            w,
            {
                "cpu_min": cpu,
                "cpu_max": cpu,
                "mem_min": mem,
                "mem_max": mem,
                "samples": 0,
                "last_seen": 0,
            },
        )
        e["cpu_min"] = min(e["cpu_min"], cpu)
        e["cpu_max"] = max(e["cpu_max"], cpu)
        e["mem_min"] = min(e["mem_min"], mem)
        e["mem_max"] = max(e["mem_max"], mem)
        e["samples"] += 1
        e["last_seen"] = int(time.time())

        rc, rm = requests_of(pod)
        e["req_cpu"], e["req_mem"] = rc, rm
        if rc <= IGNORED_CPU and rm <= IGNORED_MEM:
            continue  # exempt, nothing to warn about
        if rc and not (CPU_LO <= cpu / rc <= CPU_HI):
            warn.append(f"{pod}: cpu {cpu:.2f} is {cpu / rc:.0%} of request {rc:.2f}")
        if rm and not (MEM_LO <= mem / rm <= MEM_HI):
            warn.append(
                f"{pod}: mem {mem / 2**20:.0f}Mi is {mem / rm:.0%} of request "
                f"{rm / 2**20:.0f}Mi"
            )
    return warn


def advise(e: dict) -> dict:
    """Turn a profile into the legal request window, or say there is none."""
    peak_m, trough_m = e["mem_max"], e["mem_min"]
    peak_c, trough_c = e["cpu_max"], e["cpu_min"]
    out: dict = {"samples": e["samples"]}

    if peak_m <= IGNORED_MEM and peak_c <= IGNORED_CPU:
        out["verdict"] = "exempt"
        out["recommend"] = {"cpu": "1", "memory": "2Gi"}
        out["why"] = (
            "peak fits the ignored range (cpu<=1, mem<=2Gi), so the pod is not "
            "monitored at all -- the safest place to be"
        )
        return out

    lo_m, hi_m = peak_m, trough_m / MEM_LO
    lo_c, hi_c = peak_c, trough_c / CPU_LO
    out["mem_window_mib"] = [round(lo_m / 2**20), round(hi_m / 2**20)]
    out["cpu_window"] = [round(lo_c, 2), round(hi_c, 2)]

    if lo_m > hi_m:
        out["verdict"] = "IMPOSSIBLE"
        out["why"] = (
            f"memory peak {peak_m / 2**20:.0f}Mi is {peak_m / max(trough_m, 1):.1f}x "
            f"its trough {trough_m / 2**20:.0f}Mi. The bands allow at most 5x "
            f"(1.0/0.20), so no request is legal: size for the peak and the "
            f"troughs fall under the floor, size for the troughs and it OOMs. "
            f"The workload's memory shape has to change -- reuse a buffer, "
            f"stream, or move intermediates off the heap."
        )
        return out

    pick_m = min(hi_m, peak_m * 1.25)
    pick_c = min(hi_c, max(peak_c * 1.25, 1.0))
    out["verdict"] = "ok"
    out["recommend"] = {
        "cpu": f"{max(1, int(round(pick_c)))}",
        "memory": f"{int(pick_m // 2**20)}Mi",
    }
    out["why"] = "peak fits, and the trough still clears the 20% floor"
    return out


def cmd_watch(args: argparse.Namespace) -> int:
    store = load()
    print(f"right-sizer watching {NS} every {args.interval}s -> {STORE}", flush=True)
    while True:
        try:
            for w in sample(store):
                print(f"VIOLATION {w}", flush=True)
            save(store)
        except Exception as exc:  # a transient API error must not kill the daemon
            print(f"sample failed: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(args.interval)


def cmd_advise(args: argparse.Namespace) -> int:
    store = load()
    keys = [k for k in store if args.workload in k] if args.workload else list(store)
    if not keys:
        print("no profile yet; run `watch` while the workload runs")
        return 1
    for k in sorted(keys):
        e = store[k]
        a = advise(e)
        print(f"\n{k}  ({e['samples']} samples)")
        print(f"  cpu  {e['cpu_min']:.2f} .. {e['cpu_max']:.2f}")
        print(f"  mem  {e['mem_min'] / 2**20:.0f}Mi .. {e['mem_max'] / 2**20:.0f}Mi")
        print(f"  verdict: {a['verdict']}")
        if "recommend" in a:
            print(f"  recommend: {a['recommend']}")
        print(f"  {a['why']}")
        if e["samples"] < 10:
            print("  NOTE: few samples, and metrics-server lags ~30s. Treat the")
            print("  peak as a lower bound until this has watched a full cycle.")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Validate a rendered manifest against policy and any known profile."""
    text = open(args.manifest, encoding="utf-8").read()
    mc = re.search(r"requests:\s*\{[^}]*cpu:\s*\"?([0-9.]+)", text)
    mm = re.search(r"requests:\s*\{[^}]*memory:\s*([0-9]+[KMGT]i)", text)
    if not (mc and mm):
        print("could not find requests in the manifest")
        return 2
    rc, rm = float(mc.group(1)), parse_mem(mm.group(1))
    print(f"manifest asks cpu={rc} memory={rm / 2**20:.0f}Mi")
    if rc <= IGNORED_CPU and rm <= IGNORED_MEM:
        print("VERDICT ok: inside the ignored range, not monitored")
        return 0
    store = load()
    hit = [k for k in store if args.workload and args.workload in k]
    if not hit:
        print("VERDICT unknown: monitored, and no measured profile to judge it by.")
        print("Run it once under `watch` before trusting these numbers.")
        return 0
    e = store[hit[0]]
    bad = []
    if e["mem_max"] > rm:
        bad.append(f"peak {e['mem_max'] / 2**20:.0f}Mi exceeds request -> OOM")
    if e["mem_min"] / rm < MEM_LO:
        bad.append(f"trough is {e['mem_min'] / rm:.0%} of request -> swept")
    if e["cpu_max"] and e["cpu_max"] / rc > CPU_HI:
        bad.append(f"cpu peak is {e['cpu_max'] / rc:.0%} of request")
    if e["cpu_min"] / rc < CPU_LO:
        bad.append(f"cpu trough is {e['cpu_min'] / rc:.0%} of request -> swept")
    if bad:
        print("VERDICT REJECT:")
        for b in bad:
            print(f"  - {b}")
        print(f"  {advise(e)['why']}")
        return 1
    print("VERDICT ok against the measured profile")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch")
    w.add_argument("--interval", type=int, default=20)
    w.set_defaults(fn=cmd_watch)
    a = sub.add_parser("advise")
    a.add_argument("workload", nargs="?", default="")
    a.set_defaults(fn=cmd_advise)
    c = sub.add_parser("check")
    c.add_argument("manifest")
    c.add_argument("--workload", default="")
    c.set_defaults(fn=cmd_check)
    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

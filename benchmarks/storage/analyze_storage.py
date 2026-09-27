"""Tabulate the storage probes and score the predictions in RESULTS_storage_probe.md.

    python analyze_storage.py DIR [DIR ...] [--out analysis.json]

Each DIR holds the driver's records (one JSON per probe). Probes are tabulated
by class and zone. For the location sweep (records whose ``zone`` is set and
that share one PVC), the predictions are scored:

- P1 statvfs used space rises by the bytes written (within 1 MiB) and returns to
  its baseline after the delete (within 1 MiB);
- P2 on a ceph mount, kernel ``read_bytes`` is 0 while ``rchar`` covers the
  bytes read;
- P3 fdatasync p50, random-read p50 and unlink latency are each correlated with
  the location coordinate r = mean create latency (Pearson > 0.9);
- P4 the cluster metrics observer's CPU is within 3x of the pod's lifetime mean.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

MIB = 2**20


def load(dirs) -> list:
    recs = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            with open(f, encoding="utf-8") as fh:
                r = json.load(fh)
            if isinstance(r, dict) and "probe" in r:
                r["_file"] = f
                recs.append(r)
    return recs


def phase(p, name):
    return next((x for x in p["phases"] if x["phase"] == name), None)


def row(r) -> dict:
    p = r.get("probe") or {}
    out = {
        "sweep": "location" if "-loc-" in (r.get("pvc") or "") else "class",
        "class": r.get("class"),
        "zone": r.get("zone"),
        "node": r.get("node"),
        "phase": r.get("phase"),
        "error": p.get("error"),
    }
    if not p.get("phases"):
        return out
    w, f, sr, rr, md = (
        phase(p, n) for n in ("seq_write", "fsync", "seq_read", "rand_read", "metadata")
    )
    cl = phase(p, "cleanup")
    u0 = p["fs_before"]["used_bytes"]
    create_ms = 1e3 / md["app"]["create_per_s"] if md["app"]["create_per_s"] else None
    out.update(
        write_mb_s=w["app"]["mb_s"],
        write_mib=w["app"]["bytes"] / MIB,
        fsync_p50_ms=f["app"]["ms_p50"],
        fsync_p99_ms=f["app"]["ms_p99"],
        read_mb_s=sr["app"]["mb_s"],
        read_mib=sr["app"]["bytes"] / MIB,
        rand_p50_ms=rr["app"]["ms_p50"],
        rand_ops=rr["app"]["ops"],
        create_ms=create_ms,
        unlink_ms=(
            1e3 / md["app"]["unlink_per_s"] if md["app"]["unlink_per_s"] else None
        ),
        # P1: what the filesystem counted against what was written
        used_after_write_mib=(w["fs_after"]["used_bytes"] - u0) / MIB,
        used_after_cleanup_mib=(cl["fs_after"]["used_bytes"] - u0) / MIB,
        # P2: what the kernel's block counters saw of the reads
        read_bytes_kernel=sr["kernel_io"].get("read_bytes"),
        rchar=sr["kernel_io"].get("rchar"),
        read_bytes_app=sr["app"]["bytes"],
        write_bytes_kernel=w["kernel_io"].get("write_bytes"),
        fstype=(p.get("mount") or {}).get("fstype"),
        cpu_s=(p.get("usage") or {}).get("cpu_s"),
        life_s=p["t_end"] - p["t_start"],
    )
    cm = [m["cpu_cores"] for m in r.get("cluster_metrics") or []]
    out["metrics_cpu_mean"] = sum(cm) / len(cm) if cm else None
    out["metrics_samples"] = len(cm)
    out["self_cpu_mean"] = out["cpu_s"] / out["life_s"] if out["cpu_s"] else None
    return out


def pearson(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pts)
    if n < 3:
        return None, n
    mx, my = sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    syy = sum((y - my) ** 2 for _, y in pts)
    if sxx == 0 or syy == 0:
        return None, n
    return sxy / math.sqrt(sxx * syy), n


def score(rows) -> dict:
    ok = [r for r in rows if r.get("write_mib")]
    p1 = all(
        abs(r["used_after_write_mib"] - r["write_mib"]) <= 1
        and abs(r["used_after_cleanup_mib"]) <= 1
        for r in ok
    )
    ceph = [r for r in ok if "ceph" in (r.get("fstype") or "")]
    p2 = all(
        r["read_bytes_kernel"] == 0 and r["rchar"] >= r["read_bytes_app"] for r in ceph
    )
    rx = [r["create_ms"] for r in ok]
    p3 = {
        k: dict(zip(("pearson", "n"), pearson(rx, [r[k] for r in ok])))
        for k in ("fsync_p50_ms", "rand_p50_ms", "unlink_ms")
    }
    p3_pass = all(v["pearson"] is not None and v["pearson"] > 0.9 for v in p3.values())
    ratios = [
        r["metrics_cpu_mean"] / r["self_cpu_mean"]
        for r in ok
        if r.get("metrics_cpu_mean") and r.get("self_cpu_mean")
    ]
    p4 = bool(ratios) and all(1 / 3 <= x <= 3 for x in ratios)
    return {
        "n_probes": len(ok),
        "P1_invariance": p1,
        "P2_ceph_block_counter_blind": p2 if ceph else None,
        "P3_one_coordinate": {"per_observable": p3, "pass": p3_pass},
        "P4_metrics_within_3x": {"ratios": ratios, "pass": p4},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rows = [row(r) for r in load(a.dirs)]
    # The location sweep is the probes that shared one volume ("-loc-" PVCs);
    # the class sweep's rook-cephfs probe, also pinned to a zone, is not one.
    loc = [r for r in rows if r["sweep"] == "location"]
    doc = {"rows": rows, "location_sweep": score(loc) if loc else None}
    text = json.dumps(doc, indent=2)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

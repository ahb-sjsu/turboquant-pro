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


# ---- rehabilitation (R1, R4) and 1T reuse (L1-L4), registered in 10b0508 ------


def _spearman(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pts) < 3:
        return None, len(pts)

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(order):
            r[i] = float(k)
        return r

    return pearson(ranks([p[0] for p in pts]), ranks([p[1] for p in pts]))


def _parse_ts(ts):
    from datetime import datetime

    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def _window_s(w):
    """Seconds in a Kubernetes duration ("3m0s", "30s", "1.5s")."""
    import re

    if not w:
        return None
    units = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 1e-3}
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", w)
    return sum(float(n) * units[u] for n, u in parts) if parts else None


def _c_at(timeline, t):
    """The pod's own CPU seconds at time t, linearly between recorded marks."""
    if not timeline or t < timeline[0][0] or t > timeline[-1][0]:
        return None
    for (t0, c0), (t1, c1) in zip(timeline, timeline[1:]):
        if t0 <= t <= t1:
            return c0 if t1 == t0 else c0 + (c1 - c0) * (t - t0) / (t1 - t0)
    return timeline[-1][1]


def score_r4(recs) -> dict:
    rows, inside_burn = [], []
    for r in recs:
        p = r.get("probe") or {}
        tl = [(t, c) for t, c in p.get("cpu_timeline") or [] if c is not None]
        burn = next((x for x in p.get("phases", []) if x["phase"] == "cpu_burn"), None)
        b1 = p.get("t_end")
        b0 = b1 - burn["wall_s"] if burn and b1 else None
        for m in r.get("cluster_metrics") or []:
            ts, w = _parse_ts(m.get("timestamp")), _window_s(m.get("window"))
            if ts is None or not w:
                continue
            c1, c0 = _c_at(tl, ts), _c_at(tl, ts - w)
            if c1 is None or c0 is None:
                continue  # the window is not inside the pod's recorded life
            model = (c1 - c0) / w
            row = {
                "zone": r.get("zone"),
                "ts": ts,
                "window_s": w,
                "metrics": m["cpu_cores"],
                "model": model,
                "err": m["cpu_cores"] - model,
            }
            rows.append(row)
            if b0 is not None and ts - w >= b0 and ts <= b1:
                inside_burn.append(row)
    # the API repeats one sample until the next scrape: score each (zone, ts) once
    uniq = {(x["zone"], x["ts"]): x for x in rows}.values()
    ub = {(x["zone"], x["ts"]): x for x in inside_burn}.values()
    return {
        "R4a": {
            "n": len(uniq),
            "max_abs_err": max((abs(x["err"]) for x in uniq), default=None),
            "pass": bool(uniq) and all(abs(x["err"]) <= 0.1 for x in uniq),
        },
        "R4b": {
            "n": len(ub),
            "readings": [x["metrics"] for x in ub],
            "pass": bool(ub) and all(abs(x["metrics"] - 1.0) <= 0.1 for x in ub),
        },
        "samples": list(uniq),
    }


def score_r1(recs, rows) -> dict:
    out = []
    for r in recs:
        p = r.get("probe") or {}
        w, d = p.get("settle_after_write"), p.get("settle_after_delete")
        if w is None:
            continue
        out.append({"zone": r.get("zone"), "write": w, "delete": d})
    r1a = bool(out) and all(
        o["write"]["settled"] and o["delete"]["settled"] for o in out
    )
    rz = {x["zone"]: x.get("create_ms") for x in rows}
    xs = [rz.get(o["zone"]) for o in out]
    ys = [o["write"]["settle_s"] for o in out]
    rho, n = _spearman(xs, ys)
    return {
        "R1a": {"pass": r1a, "per_zone": out},
        "R1b": {
            "spearman_settle_vs_r": rho,
            "n": n,
            "pass": rho is not None and abs(rho) < 0.6,
        },
    }


def ro_row(r) -> dict:
    p = r.get("probe") or {}
    ph = {x["phase"]: x for x in p.get("phases", [])}
    if "first_block" not in ph:
        return {
            "claim": r.get("pvc"),
            "zone": r.get("zone"),
            "phase": r.get("phase"),
            "error": p.get("error"),
        }
    sr, rr = ph["seq_read"], ph["rand_read"]
    return {
        "claim": r.get("pvc"),
        "zone": r.get("zone"),
        "node": r.get("node"),
        "phase": r.get("phase"),
        "r_blk_ms": ph["first_block"]["app"]["ms_p50"],
        "rand_p50_ms": rr["app"]["ms_p50"],
        "rand_ops": rr["app"]["ops"],
        "seq_mb_s": sr["app"]["mb_s"],
        "seq_mib": sr["app"]["bytes"] / MIB,
        "kernel_over_app_seq": (
            sr["kernel_io"].get("read_bytes", 0) / max(sr["app"]["bytes"], 1)
        ),
        "used_bytes": p["fs_before"]["used_bytes"],
    }


def score_l(rows) -> dict:
    ok = [x for x in rows if "r_blk_ms" in x]
    l1 = {}
    for claim in sorted({x["claim"] for x in ok}):
        v = [x for x in ok if x["claim"] == claim]
        pr, n = pearson([x["r_blk_ms"] for x in v], [x["rand_p50_ms"] for x in v])
        sp, _ = _spearman([x["r_blk_ms"] for x in v], [x["seq_mb_s"] for x in v])
        l1[claim] = {
            "n": n,
            "pearson_rand_vs_r": pr,
            "spearman_seq_vs_r": sp,
            "pass": pr is not None and pr > 0.9 and sp is not None and sp < -0.8,
        }
    by_zone = {}
    for x in ok:
        by_zone.setdefault(x["zone"], []).append(x)
    obj_ratio, keys = [], ("r_blk_ms", "rand_p50_ms", "seq_mb_s")
    for v in by_zone.values():
        if len(v) == 2:
            for k in keys:
                a, b = v[0][k], v[1][k]
                if a and b:
                    obj_ratio.append(max(a, b) / min(a, b))
    spans = {}
    for k in keys:
        vals = [x[k] for x in ok if x[k]]
        spans[k] = max(vals) / min(vals) if vals else None
    l2 = {
        "max_object_ratio_at_a_zone": max(obj_ratio, default=None),
        "zone_span": spans,
        "pass": bool(obj_ratio)
        and max(obj_ratio) < 1.3
        and all(s and s > 3 for s in spans.values()),
    }
    used = {}
    for x in ok:
        used.setdefault(x["claim"], set()).add(x["used_bytes"])
    l3 = {
        "distinct_used_per_claim": {c: len(u) for c, u in used.items()},
        "pass": bool(used) and all(len(u) == 1 for u in used.values()),
    }
    ratios = [x["kernel_over_app_seq"] for x in ok]
    l4 = {
        "ratios": ratios,
        "pass": bool(ratios) and all(0.95 <= q <= 1.05 for q in ratios),
    }
    return {"L1": l1, "L2": l2, "L3": l3, "L4": l4}


def rehab_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="score R1, R4 and L1-L4")
    ap.add_argument("--rehab", required=True, help="dir of the rehabilitation sweep")
    ap.add_argument("--onet", required=True, help="dir of the 1T read-only probes")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rrecs, orecs = load([a.rehab]), load([a.onet])
    rows = [row(r) for r in rrecs]
    lrows = [ro_row(r) for r in orecs]
    doc = {
        "rehab_rows": rows,
        "R1": score_r1(rrecs, rows),
        "R4": score_r4(rrecs),
        "onet_rows": lrows,
        "L": score_l(lrows),
    }
    text = json.dumps(doc, indent=2)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(rehab_main(sys.argv[2:]) if sys.argv[1:2] == ["rehab"] else main())

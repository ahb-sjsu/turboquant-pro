"""Check what ``tqp fabric`` observed against the traffic ``leaf_echo`` sent.

Inputs: the recorder's JSON-lines file (``tqp fabric --record``) and the
responder's output (which carries the client's own counts). The observed path is
either the NRP leaf link (``--path leaf``) or the responder's own connection on the hub
(``--path responder:NAME --exact``, for the rehearsal on Atlas, where that
connection carries nothing else).

Every check is written as a prediction with the tolerance it is held to, decided
before looking at the numbers:

- messages: exact. The server's message counters must move by exactly what the
  client sent and received, plus the messages this protocol adds (the result,
  and on the leaf the one ``burst.submit`` that launched the pod), plus whatever
  unrelated traffic crossed the path in the window, which is reported, not
  hidden.
- bytes: the server counts payload bytes; the check reports observed / sent.
- round-trip time: the server's figure (hub to leaf pod) cannot exceed the
  client's median (client pod to Atlas and back, which includes that hop).
- rates: the peak derived rate over one poll interval cannot exceed the client's
  own burst rate, and should reach burst_n / interval when the burst fits in one
  interval.

    python benchmarks/fabric/analyze.py record.jsonl responder.json --path leaf
"""

from __future__ import annotations

import argparse
import json
import sys


def load(record: str) -> list:
    with open(record, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


_K = ("msgs", "bytes")


def counters(doc: dict, path: str) -> dict | None:
    """The observed path's counters in one snapshot, oriented along the
    experiment: ``fwd`` is client toward responder, ``back`` the replies.

    ``leaf``: the hub's counters for the leaf link. ``in`` is what the hub got
    over the link (the pod's messages: fwd), ``out`` what it sent (back).
    ``responder:NAME``: the responder's own connection on the hub. The server's
    ``out`` is what it delivered to the responder (fwd) and ``in`` what the
    responder published (back). None when the path is absent from the snapshot."""
    if path == "leaf":
        found = doc.get("leafs") or []
        if len(found) != 1:
            return None
        c, fwd, back = found[0], "in", "out"
    else:
        name = path.split(":", 1)[1]
        found = [c for c in doc.get("connections") or [] if c.get("name") == name]
        if len(found) != 1:
            return None
        c, fwd, back = found[0], "out", "in"
    return {
        "fwd_msgs": c[f"{fwd}_msgs"],
        "fwd_bytes": c[f"{fwd}_bytes"],
        "back_msgs": c[f"{back}_msgs"],
        "back_bytes": c[f"{back}_bytes"],
        "fwd_msgs_per_s": c["rates"].get(f"{fwd}_msgs_per_s"),
        "rtt_ms": c.get("rtt_ms"),
    }


def analyze(
    snaps: list, resp: dict, path: str, extra_fwd: int, extra_back: int, exact: bool
) -> dict:
    cl = resp.get("client") or {}
    if not cl:
        raise SystemExit("the responder never received the client's result")
    t0, t1 = cl["t_start"], cl["t_end"]
    seen = [(s, counters(s, path)) for s in snaps if s.get("reachable")]
    seen = [(s, c) for s, c in seen if c is not None]
    before = [c for s, c in seen if s["t"] <= t0]
    after = [c for s, c in seen if s["t"] >= t1]
    if not before or not after:
        raise SystemExit(
            f"need a snapshot of {path} before the client started and after it "
            f"ended; have {len(before)} before, {len(after)} after"
        )
    b, a = before[-1], after[0]
    d = {f"{w}_{k}": a[f"{w}_{k}"] - b[f"{w}_{k}"] for w in ("fwd", "back") for k in _K}

    sent, recv, ph = cl["sent"], cl["received"], cl["phases"]
    result_bytes = len(json.dumps(cl))  # the result message's payload, near enough
    want = {
        "fwd_msgs": sent["msgs"] + 1 + extra_fwd,  # + the result message
        "back_msgs": recv["msgs"] + extra_back,
        "fwd_bytes": sent["payload_bytes"] + result_bytes,
        "back_bytes": recv["payload_bytes"],
    }

    def count_check(key):
        obs, exp = d[key], want[key]
        return {
            "observed": obs,
            "expected": exp,
            "unrelated": obs - exp,
            "pass": obs == exp if exact else obs >= exp,
        }

    window = [c for s, c in seen if t0 - 5 <= s["t"] <= t1 + 5]
    rates = [c["fwd_msgs_per_s"] for c in window if c["fwd_msgs_per_s"] is not None]
    peak = max(rates) if rates else None
    rtts = sorted({c["rtt_ms"] for c in window if c["rtt_ms"] is not None})
    interval = next((s["interval_s"] for s in snaps if s.get("interval_s")), None)
    burst = ph["burst"]
    n_req = ph["rtt"]["n"] + sum(r["n"] for r in ph["sizes"])
    checks = {
        "fwd_msgs": count_check("fwd_msgs"),
        "back_msgs": count_check("back_msgs"),
        "fwd_bytes_ratio": d["fwd_bytes"] / max(want["fwd_bytes"], 1),
        "back_bytes_ratio": d["back_bytes"] / max(want["back_bytes"], 1),
        "rtt": {
            "server_ms": rtts,
            "client_p50_ms": ph["rtt"]["ms"]["p50"],
            "pass": (
                all(r <= ph["rtt"]["ms"]["p50"] * 1.05 for r in rtts) if rtts else None
            ),
        },
        "burst_rate": {
            "client_msgs_per_s": burst["msgs_per_s"],
            "client_wall_s": burst["wall_s"],
            "observed_peak_fwd_msgs_per_s": peak,
            "poll_interval_s": interval,
            "expected_peak_if_in_one_interval": (
                (burst["n"] + 1) / interval if interval else None
            ),
            "pass": peak is not None and peak <= burst["msgs_per_s"] * 1.05,
        },
        "responder": {
            "echo_msgs": resp["counts"]["echo_msgs"],
            "sink_msgs": resp["counts"]["sink_msgs"],
            "lossless": resp["counts"]["sink_msgs"] == burst["n"]
            and resp["counts"]["echo_msgs"] == n_req,
        },
    }
    return {
        "path": path,
        "exact": exact,
        "window": {"t_start": t0, "t_end": t1, "snapshots": len(seen)},
        "delta": d,
        "client": {"sent": sent, "received": recv, "phases": ph, "usage": cl["usage"]},
        "checks": checks,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("record")
    p.add_argument("responder")
    p.add_argument("--path", default="leaf", help="leaf, or responder:NAME")
    p.add_argument("--extra-fwd", type=int, default=0)
    p.add_argument("--extra-back", type=int, default=0)
    p.add_argument(
        "--exact",
        action="store_true",
        help="the path carries nothing else, so counts must match exactly",
    )
    p.add_argument("--out")
    args = p.parse_args(argv)
    with open(args.responder, encoding="utf-8") as f:
        resp = json.load(f)
    rep = analyze(
        load(args.record), resp, args.path, args.extra_fwd, args.extra_back, args.exact
    )
    text = json.dumps(rep, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
